"""No-follow filesystem primitives for semantic-model storage."""

from __future__ import annotations

import os
import secrets
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


class SemanticStorageError(OSError):
    """Raised when a semantic storage path cannot be trusted."""


@dataclass(frozen=True)
class FileIdentity:
    """Stable identity of an opened filesystem object."""

    device: int
    inode: int


@dataclass(frozen=True)
class VerifiedRead:
    """Bytes read from one pinned regular file plus a revalidation token."""

    content: bytes
    path: Path
    parent_identity: FileIdentity
    file_identity: FileIdentity
    anchor: Path | None

    def verify(self) -> None:
        """Reject replacement of either the file or any containing directory."""
        with _open_directory(self.path.parent, anchor=self.anchor) as (
            parent_descriptor,
            _parent_path,
        ):
            if _identity(os.fstat(parent_descriptor)) != self.parent_identity:
                raise SemanticStorageError("语义存储目录在读取期间已被替换")
            metadata = _lstat_at(parent_descriptor, self.path.name)
            if metadata is None or stat.S_ISLNK(metadata.st_mode):
                raise SemanticStorageError("语义存储文件在读取期间已被替换或成为符号链接")
            if not stat.S_ISREG(metadata.st_mode):
                raise SemanticStorageError("语义存储读取目标不是普通文件")
            if _identity(metadata) != self.file_identity:
                raise SemanticStorageError("语义存储文件在读取期间已被替换")


def _identity(metadata: os.stat_result) -> FileIdentity:
    return FileIdentity(metadata.st_dev, metadata.st_ino)


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
                or _identity(published) != temporary_identity
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
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise SemanticStorageError("语义存储读取目标不是普通文件")
                chunks: list[bytes] = []
                while chunk := os.read(descriptor, 1024 * 1024):
                    chunks.append(chunk)
                after = os.fstat(descriptor)
                if _identity(after) != _identity(before):
                    raise SemanticStorageError("语义存储文件在读取期间已被替换")
            finally:
                os.close(descriptor)
            token = VerifiedRead(
                content=b"".join(chunks),
                path=absolute,
                parent_identity=_identity(os.fstat(parent_descriptor)),
                file_identity=_identity(before),
                anchor=lexical_absolute(anchor) if anchor is not None else None,
            )
            token.verify()
            return token
        except SemanticStorageError:
            raise
        except (NotImplementedError, OSError) as exc:
            raise SemanticStorageError("无法安全读取语义存储文件") from exc
