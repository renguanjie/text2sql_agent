"""No-follow filesystem primitives for semantic-model storage."""

from __future__ import annotations

import os
import secrets
import stat
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterator

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised by explicit fail-closed test
    _fcntl = None


_TRANSACTION_LOCK_NAME = ".semantic-compile.lock"
_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.RLock] = {}


class SemanticStorageError(OSError):
    """Raised when a semantic storage path cannot be trusted."""


@dataclass(frozen=True)
class FileIdentity:
    """Stable identity of an opened filesystem object."""

    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class VerifiedRead:
    """Bytes read from one pinned regular file plus a revalidation token."""

    content: bytes
    path: Path
    parent_identity: FileIdentity
    file_identity: FileIdentity
    content_sha256: str
    anchor: Path | None

    def verify(self) -> None:
        """Re-read and reject any metadata or content identity change."""
        try:
            with _open_directory(self.path.parent, anchor=self.anchor) as (
                parent_descriptor,
                _parent_path,
            ):
                if _identity(os.fstat(parent_descriptor)) != self.parent_identity:
                    raise SemanticStorageError("语义存储目录在读取期间已被替换")
                metadata = _lstat_at(parent_descriptor, self.path.name)
                if metadata is None or stat.S_ISLNK(metadata.st_mode):
                    raise SemanticStorageError(
                        "语义存储文件在读取期间已被替换或成为符号链接"
                    )
                if not stat.S_ISREG(metadata.st_mode):
                    raise SemanticStorageError("语义存储读取目标不是普通文件")
                if _identity(metadata) != self.file_identity:
                    raise SemanticStorageError("语义存储文件在读取期间已被替换")

                flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
                descriptor = os.open(
                    self.path.name,
                    flags,
                    dir_fd=parent_descriptor,
                )
                try:
                    content, identity = _read_regular_descriptor(descriptor)
                    if identity != self.file_identity:
                        raise SemanticStorageError(
                            "语义存储文件在读取期间已被替换"
                        )
                    if sha256(content).hexdigest() != self.content_sha256:
                        raise SemanticStorageError(
                            "语义存储文件内容在读取期间已被替换"
                        )
                    current = _lstat_at(parent_descriptor, self.path.name)
                    if current is None or _identity(current) != self.file_identity:
                        raise SemanticStorageError(
                            "语义存储文件在读取期间已被替换"
                        )
                finally:
                    os.close(descriptor)
                if _identity(os.fstat(parent_descriptor)) != self.parent_identity:
                    raise SemanticStorageError("语义存储目录在读取期间已被替换")
        except SemanticStorageError:
            raise
        except (NotImplementedError, OSError) as exc:
            raise SemanticStorageError("无法安全验证语义存储文件") from exc


def _identity(metadata: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
    )


def _same_file_object(left: FileIdentity, right: FileIdentity) -> bool:
    return (left.device, left.inode) == (right.device, right.inode)


def _read_regular_descriptor(descriptor: int) -> tuple[bytes, FileIdentity]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise SemanticStorageError("语义存储读取目标不是普通文件")
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    after = os.fstat(descriptor)
    if _identity(after) != _identity(before):
        raise SemanticStorageError("语义存储文件在读取期间已被替换")
    return b"".join(chunks), _identity(after)


def _require_no_follow_support() -> None:
    required = (
        os.name == "posix",
        hasattr(os, "O_NOFOLLOW"),
        hasattr(os, "O_DIRECTORY"),
        os.open in os.supports_dir_fd,
        os.mkdir in os.supports_dir_fd,
        os.stat in os.supports_dir_fd,
        os.unlink in os.supports_dir_fd,
    )
    if not all(required):
        raise SemanticStorageError(
            "当前平台不支持安全的 no-follow 语义存储操作"
        )


def lexical_absolute(path: Path) -> Path:
    """Return an absolute normalized path without resolving symbolic links."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    if sys.platform != "darwin":
        return absolute

    for alias in (Path("/var"), Path("/tmp")):
        try:
            relative = absolute.relative_to(alias)
            alias_metadata = alias.lstat()
            parent_metadata = alias.parent.stat()
        except (ValueError, OSError):
            continue
        parent_is_protected = (
            parent_metadata.st_uid == 0
            and stat.S_IMODE(parent_metadata.st_mode) & 0o022 == 0
        )
        if (
            stat.S_ISLNK(alias_metadata.st_mode)
            and alias_metadata.st_uid == 0
            and parent_is_protected
        ):
            return Path(os.path.realpath(alias)) / relative
    return absolute


def validate_no_symlink_path(path: Path, *, anchor: Path | None = None) -> Path:
    """Validate each existing lexical component without following symlinks."""
    absolute = lexical_absolute(path)
    if anchor is not None:
        absolute_anchor = lexical_absolute(anchor)
        try:
            absolute.relative_to(absolute_anchor)
        except ValueError:
            raise SemanticStorageError("语义存储路径越过可信根") from None

    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise SemanticStorageError(f"无法验证语义存储路径: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise SemanticStorageError(f"语义存储路径不得包含符号链接: {current}")
    return absolute


def _directory_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    return flags | getattr(os, "O_CLOEXEC", 0)


@contextmanager
def _open_directory(
    path: Path, *, anchor: Path | None = None, create: bool = False
) -> Iterator[tuple[int, Path]]:
    """Open a lexical directory one component at a time without following links."""
    _require_no_follow_support()
    absolute = validate_no_symlink_path(path, anchor=anchor)
    if absolute.anchor != os.sep:
        raise SemanticStorageError("当前平台不支持安全的语义存储路径锚定")

    descriptor = os.open(os.sep, _directory_flags())
    try:
        for part in absolute.parts[1:]:
            try:
                child = os.open(part, _directory_flags(), dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, _directory_flags(), dir_fd=descriptor)
            except (NotADirectoryError, OSError) as exc:
                raise SemanticStorageError(
                    f"语义存储目录不安全: {absolute}"
                ) from exc
            os.close(descriptor)
            descriptor = child
        yield descriptor, absolute
    except FileNotFoundError as exc:
        raise SemanticStorageError(f"语义存储目录不存在: {absolute}") from exc
    finally:
        os.close(descriptor)


def _verify_directory_binding(path: Path, descriptor: int) -> None:
    expected = _identity(os.fstat(descriptor))
    with _open_directory(path) as (current_descriptor, _absolute):
        if _identity(os.fstat(current_descriptor)) != expected:
            raise SemanticStorageError("语义存储目录在操作期间已被替换")


def _lstat_at(descriptor: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _validate_regular_target(descriptor: int, name: str) -> None:
    metadata = _lstat_at(descriptor, name)
    if metadata is None:
        return
    if stat.S_ISLNK(metadata.st_mode):
        raise SemanticStorageError(f"语义存储文件不得是符号链接: {name}")
    if not stat.S_ISREG(metadata.st_mode):
        raise SemanticStorageError(f"语义存储目标不是普通文件: {name}")


def _thread_lock_for(path: Path) -> threading.RLock:
    key = os.fspath(path)
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def semantic_transaction_lock(
    directory: Path, *, anchor: Path | None = None
) -> Iterator[None]:
    """Serialize one semantic publish transaction through a no-follow lock fd."""
    _require_no_follow_support()
    if _fcntl is None or not all(
        hasattr(_fcntl, name) for name in ("flock", "LOCK_EX", "LOCK_UN")
    ):
        raise SemanticStorageError("当前平台不支持安全的 flock 文件锁")

    absolute = validate_no_symlink_path(directory, anchor=anchor)
    with _thread_lock_for(absolute):
        with _open_directory(absolute, anchor=anchor, create=True) as (
            directory_descriptor,
            directory_path,
        ):
            lock_descriptor: int | None = None
            locked = False
            try:
                _verify_directory_binding(directory_path, directory_descriptor)
                metadata = _lstat_at(directory_descriptor, _TRANSACTION_LOCK_NAME)
                if metadata is not None and stat.S_ISLNK(metadata.st_mode):
                    raise SemanticStorageError("语义存储文件锁不得是符号链接")
                if metadata is not None and not stat.S_ISREG(metadata.st_mode):
                    raise SemanticStorageError("语义存储文件锁不是普通文件")

                flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
                flags |= getattr(os, "O_CLOEXEC", 0)
                lock_descriptor = os.open(
                    _TRANSACTION_LOCK_NAME,
                    flags,
                    0o600,
                    dir_fd=directory_descriptor,
                )
                lock_identity = _identity(os.fstat(lock_descriptor))
                if not stat.S_ISREG(lock_identity.mode):
                    raise SemanticStorageError("语义存储文件锁不是普通文件")
                current = _lstat_at(directory_descriptor, _TRANSACTION_LOCK_NAME)
                if (
                    current is None
                    or stat.S_ISLNK(current.st_mode)
                    or not _same_file_object(_identity(current), lock_identity)
                ):
                    raise SemanticStorageError("语义存储文件锁绑定不安全")

                _fcntl.flock(lock_descriptor, _fcntl.LOCK_EX)
                locked = True
                _verify_directory_binding(directory_path, directory_descriptor)
                current = _lstat_at(directory_descriptor, _TRANSACTION_LOCK_NAME)
                if (
                    current is None
                    or stat.S_ISLNK(current.st_mode)
                    or not stat.S_ISREG(current.st_mode)
                    or not _same_file_object(_identity(current), lock_identity)
                ):
                    raise SemanticStorageError("语义存储文件锁在等待期间已被替换")

                yield

                _verify_directory_binding(directory_path, directory_descriptor)
                current = _lstat_at(directory_descriptor, _TRANSACTION_LOCK_NAME)
                if (
                    current is None
                    or stat.S_ISLNK(current.st_mode)
                    or not stat.S_ISREG(current.st_mode)
                    or not _same_file_object(_identity(current), lock_identity)
                ):
                    raise SemanticStorageError("语义存储文件锁在事务期间已被替换")
            except SemanticStorageError:
                raise
            except (NotImplementedError, OSError) as exc:
                raise SemanticStorageError("无法安全获取语义存储 flock 文件锁") from exc
            finally:
                if lock_descriptor is not None:
                    try:
                        if locked:
                            _fcntl.flock(lock_descriptor, _fcntl.LOCK_UN)
                    finally:
                        os.close(lock_descriptor)


def remove_file_no_follow(
    destination: Path, *, anchor: Path | None = None
) -> None:
    """Remove one regular file through a pinned parent without following links."""
    absolute = validate_no_symlink_path(destination, anchor=anchor)
    with _open_directory(absolute.parent, anchor=anchor) as (
        parent_descriptor,
        parent_path,
    ):
        try:
            _verify_directory_binding(parent_path, parent_descriptor)
            metadata = _lstat_at(parent_descriptor, absolute.name)
            if metadata is None:
                return
            if stat.S_ISLNK(metadata.st_mode):
                raise SemanticStorageError(
                    f"语义存储待移除文件不得是符号链接: {absolute.name}"
                )
            if not stat.S_ISREG(metadata.st_mode):
                raise SemanticStorageError(
                    f"语义存储待移除目标不是普通文件: {absolute.name}"
                )
            expected = _identity(metadata)
            flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(absolute.name, flags, dir_fd=parent_descriptor)
            try:
                if _identity(os.fstat(descriptor)) != expected:
                    raise SemanticStorageError("语义存储待移除文件已被替换")
            finally:
                os.close(descriptor)
            current = _lstat_at(parent_descriptor, absolute.name)
            if current is None or _identity(current) != expected:
                raise SemanticStorageError("语义存储待移除文件已被替换")
            os.unlink(absolute.name, dir_fd=parent_descriptor)
            if _lstat_at(parent_descriptor, absolute.name) is not None:
                raise SemanticStorageError("语义存储文件移除失败")
            _verify_directory_binding(parent_path, parent_descriptor)
        except SemanticStorageError:
            raise
        except (NotImplementedError, OSError) as exc:
            raise SemanticStorageError("无法安全移除语义存储文件") from exc


def atomic_write_bytes(
    destination: Path, content: bytes, *, anchor: Path | None = None
) -> None:
    """Atomically publish bytes through one pinned, no-follow parent directory."""
    absolute = validate_no_symlink_path(destination, anchor=anchor)
    temporary_name = f".{absolute.name}.{secrets.token_hex(12)}.tmp"
    temporary_created = False

    with _open_directory(absolute.parent, anchor=anchor, create=True) as (
        parent_descriptor,
        parent_path,
    ):
        try:
            _verify_directory_binding(parent_path, parent_descriptor)
            _validate_regular_target(parent_descriptor, absolute.name)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            flags |= getattr(os, "O_CLOEXEC", 0)
            temporary_descriptor = os.open(
                temporary_name,
                flags,
                0o600,
                dir_fd=parent_descriptor,
            )
            temporary_created = True
            try:
                view = memoryview(content)
                while view:
                    written = os.write(temporary_descriptor, view)
                    view = view[written:]
                os.fsync(temporary_descriptor)
                temporary_identity = _identity(os.fstat(temporary_descriptor))
            finally:
                os.close(temporary_descriptor)

            _verify_directory_binding(parent_path, parent_descriptor)
            _validate_regular_target(parent_descriptor, absolute.name)
            os.replace(
                temporary_name,
                absolute.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            temporary_created = False
            published = _lstat_at(parent_descriptor, absolute.name)
            if (
                published is None
                or not stat.S_ISREG(published.st_mode)
                or not _same_file_object(_identity(published), temporary_identity)
            ):
                raise SemanticStorageError("语义存储文件在发布期间已被替换")
            _verify_directory_binding(parent_path, parent_descriptor)
        except SemanticStorageError:
            raise
        except (NotImplementedError, OSError) as exc:
            raise SemanticStorageError("无法安全写入语义存储文件") from exc
        finally:
            if temporary_created:
                try:
                    os.unlink(temporary_name, dir_fd=parent_descriptor)
                except FileNotFoundError:
                    pass


def read_bytes_no_follow(
    path: Path, *, anchor: Path | None = None
) -> VerifiedRead:
    """Read one regular file without following links and retain its identity."""
    absolute = validate_no_symlink_path(path, anchor=anchor)
    with _open_directory(absolute.parent, anchor=anchor) as (
        parent_descriptor,
        parent_path,
    ):
        try:
            _verify_directory_binding(parent_path, parent_descriptor)
            flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(absolute.name, flags, dir_fd=parent_descriptor)
            try:
                content, file_identity = _read_regular_descriptor(descriptor)
            finally:
                os.close(descriptor)
            token = VerifiedRead(
                content=content,
                path=absolute,
                parent_identity=_identity(os.fstat(parent_descriptor)),
                file_identity=file_identity,
                content_sha256=sha256(content).hexdigest(),
                anchor=lexical_absolute(anchor) if anchor is not None else None,
            )
            token.verify()
            return token
        except SemanticStorageError:
            raise
        except (NotImplementedError, OSError) as exc:
            raise SemanticStorageError("无法安全读取语义存储文件") from exc
