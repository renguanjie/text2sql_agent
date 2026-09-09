"""Thread-safe loading of validated compiled semantic snapshots."""

from __future__ import annotations

import json
import re
import threading
from hashlib import sha256
from pathlib import Path
from typing import Any

import jsonschema
from pydantic import ValidationError

from .models import SemanticModel, SemanticSnapshot


_SHA256_HEX = re.compile(r"[0-9a-fA-F]{64}")
_DATASOURCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_SEMVER = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)")


class SemanticRegistryError(ValueError):
    """Raised when the active compiled semantic snapshot is not trustworthy."""


class SemanticRegistry:
    def __init__(self, compiled_root: Path, datasource: str):
        if not isinstance(datasource, str) or _DATASOURCE.fullmatch(datasource) is None:
            raise SemanticRegistryError("数据源不是安全标识符")
        compiled_root = Path(compiled_root)
        if compiled_root.is_symlink():
            raise SemanticRegistryError("编译目录不得是符号链接")
        self._compiled_root = compiled_root
        self._datasource = datasource
        self._lock = threading.RLock()
        self._snapshot: SemanticSnapshot | None = None

    def load_active(self) -> SemanticSnapshot:
        """Atomically replace the active snapshot only after complete validation."""
        candidate = self._read_active_candidate()
        with self._lock:
            self._snapshot = candidate
            return candidate

    def get_snapshot(self) -> SemanticSnapshot:
        """Return the exact immutable snapshot previously activated by this registry."""
        with self._lock:
            if self._snapshot is None:
                raise SemanticRegistryError("语义模型尚未加载")
            return self._snapshot

    @property
    def compiled_root(self) -> Path:
        return self._compiled_root

    def _read_active_candidate(self) -> SemanticSnapshot:
        manifest_path = self._compiled_root / "manifest.json"
        self._reject_symlink(manifest_path, "清单")
        manifest = self._read_json(manifest_path, "清单")
        version = self._active_version(manifest)
        path = self._snapshot_path(version)
        payload = self._read_json(path, "快照")
        snapshot = self._parse_snapshot(payload)

        if snapshot.datasource != self._datasource:
            raise SemanticRegistryError("快照数据源与请求的数据源不一致")
        if snapshot.model_version != version:
            raise SemanticRegistryError("快照版本与清单版本不一致")
        if snapshot.dialect != "mysql":
            raise SemanticRegistryError("快照方言不受支持")
        if _SHA256_HEX.fullmatch(snapshot.source_sha256) is None:
            raise SemanticRegistryError("source_sha256 格式无效")
        if snapshot.content_sha256 != self._content_sha256(snapshot):
            raise SemanticRegistryError("content_sha256 校验失败")
        return snapshot

    def _active_version(self, manifest: Any) -> str:
        if not isinstance(manifest, dict):
            raise SemanticRegistryError("清单必须是对象")
        active = manifest.get("active")
        if not isinstance(active, dict):
            raise SemanticRegistryError("清单缺少 active 对象")
        version = active.get(self._datasource)
        if not isinstance(version, str) or _SEMVER.fullmatch(version) is None:
            raise SemanticRegistryError(f"清单未指定数据源 {self._datasource} 的活动版本")
        return version

    def _snapshot_path(self, version: str) -> Path:
        root_path = self._compiled_root.absolute()
        datasource_path = root_path / self._datasource
        snapshot_path = datasource_path / f"{version}.json"
        self._reject_symlink(root_path, "编译目录")
        self._reject_symlink(datasource_path, "数据源目录")
        self._reject_symlink(snapshot_path, "快照")
        root = root_path.resolve(strict=False)
        datasource_dir = datasource_path.resolve(strict=False)
        path = snapshot_path.resolve(strict=False)
        try:
            datasource_dir.relative_to(root)
            path.relative_to(datasource_dir)
        except ValueError:
            raise SemanticRegistryError("清单中的数据源或版本不是安全的快照位置")
        if path.parent != datasource_dir:
            raise SemanticRegistryError("清单中的数据源或版本不是安全的快照位置")
        if not path.is_file():
            raise SemanticRegistryError(f"活动快照不存在: {path}")
        return path

    @staticmethod
    def _reject_symlink(path: Path, label: str) -> None:
        if path.is_symlink():
            raise SemanticRegistryError(f"{label}不得是符号链接")

    @staticmethod
    def _read_json(path: Path, label: str) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SemanticRegistryError(f"无法读取{label}: {exc}") from exc

    @staticmethod
    def _parse_snapshot(payload: Any) -> SemanticSnapshot:
        try:
            jsonschema.validate(payload, SemanticSnapshot.model_json_schema())
            return SemanticSnapshot.model_validate(payload)
        except (jsonschema.ValidationError, ValidationError):
            raise SemanticRegistryError("快照格式无效") from None

    @staticmethod
    def _content_sha256(snapshot: SemanticSnapshot) -> str:
        semantic_payload = {
            name: getattr(snapshot, name) for name in SemanticModel.model_fields
        }
        canonical = json.dumps(
            SemanticModel.model_validate(semantic_payload).model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(canonical).hexdigest()
