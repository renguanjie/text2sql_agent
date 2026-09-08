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


class SemanticRegistryError(ValueError):
    """Raised when the active compiled semantic snapshot is not trustworthy."""


class SemanticRegistry:
    def __init__(self, compiled_root: Path, datasource: str):
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
        manifest = self._read_json(self._compiled_root / "manifest.json", "清单")
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
        if not isinstance(version, str) or not version:
            raise SemanticRegistryError(f"清单未指定数据源 {self._datasource} 的活动版本")
        return version

    def _snapshot_path(self, version: str) -> Path:
        root = self._compiled_root.resolve()
        datasource_dir = (root / self._datasource).resolve()
        path = (datasource_dir / f"{version}.json").resolve()
        if path.parent != datasource_dir or datasource_dir.parent != root:
            raise SemanticRegistryError("清单中的数据源或版本不是安全的快照位置")
        if not path.is_file():
            raise SemanticRegistryError(f"活动快照不存在: {path}")
        return path

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
