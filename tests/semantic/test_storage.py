import os
import select
import subprocess
import sys
from pathlib import Path

import pytest

from core.semantic import storage
from core.semantic.storage import SemanticStorageError, read_bytes_no_follow


def _rewrite_in_place(path: Path, content: bytes) -> None:
    with path.open("r+b", buffering=0) as handle:
        handle.seek(0)
        handle.write(content)
        handle.truncate()
        os.fsync(handle.fileno())


def test_verified_read_rejects_in_place_rewrite_on_same_inode(tmp_path: Path):
    """Break caught: identity checks that trust only device and inode."""
    target = tmp_path / "snapshot.json"
    target.write_bytes(b'{"safe":true}')
    verified = read_bytes_no_follow(target)
    original_inode = target.stat().st_ino

    _rewrite_in_place(target, b'{"evil":true}')

    assert target.stat().st_ino == original_inode
    with pytest.raises(SemanticStorageError, match="读取期间"):
        verified.verify()


def test_transaction_lock_rejects_symlink_without_touching_target(tmp_path: Path):
    """Break caught: opening a path-following lock file outside the storage root."""
    root = tmp_path / ".compiled"
    root.mkdir()
    outside = tmp_path / "outside-lock"
    outside.write_bytes(b"unchanged")
    (root / ".semantic-compile.lock").symlink_to(outside)
    transaction_lock = getattr(storage, "semantic_transaction_lock", None)

    assert transaction_lock is not None
    with pytest.raises(SemanticStorageError, match="锁|符号链接|安全"):
        with transaction_lock(root):
            pytest.fail("unsafe lock unexpectedly acquired")

    assert outside.read_bytes() == b"unchanged"


def test_transaction_lock_serializes_independent_processes(tmp_path: Path):
    """Break caught: a process-local-only lock allowing concurrent publishers."""
    root = tmp_path / ".compiled"
    child_code = """
import sys
from pathlib import Path
from core.semantic.storage import semantic_transaction_lock

root = Path(sys.argv[1])
print("attempting", flush=True)
with semantic_transaction_lock(root):
    print("acquired", flush=True)
"""
    process = None
    try:
        with storage.semantic_transaction_lock(root):
            process = subprocess.Popen(
                [sys.executable, "-c", child_code, os.fspath(root)],
                cwd=Path(__file__).parents[2],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert process.stdout is not None
            readable, _, _ = select.select([process.stdout], [], [], 5)
            assert readable
            assert process.stdout.readline() == "attempting\n"
            readable, _, _ = select.select([process.stdout], [], [], 0.2)
            assert readable == []

        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stderr
        assert stdout == "acquired\n"
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
