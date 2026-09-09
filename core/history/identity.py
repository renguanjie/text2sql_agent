"""Immutable identity for governed SQL history and Few-Shot isolation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_CONTENT_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class GovernedHistoryIdentity:
    datasource: str
    dialect: str
    semantic_version: str
    content_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.datasource
            or len(self.datasource) > 191
            or not self.dialect
            or len(self.dialect) > 32
            or not self.semantic_version
            or len(self.semantic_version) > 64
            or _CONTENT_SHA256.fullmatch(self.content_sha256) is None
        ):
            raise ValueError("invalid governed history identity")

    @classmethod
    def from_snapshot(cls, snapshot: Any) -> "GovernedHistoryIdentity":
        return cls(
            datasource=snapshot.datasource,
            dialect=snapshot.dialect,
            semantic_version=snapshot.model_version,
            content_sha256=snapshot.content_sha256,
        )

    def storage_fields(self, *, validated: bool) -> dict[str, object]:
        return {
            "semantic_datasource": self.datasource,
            "semantic_dialect": self.dialect,
            "semantic_version": self.semantic_version,
            "semantic_content_sha256": self.content_sha256,
            "governance_validated": validated,
        }

    def retrieval_fields(self) -> dict[str, object]:
        return self.storage_fields(validated=True)

    def matches(self, record: Mapping[str, Any]) -> bool:
        validated = record.get("governance_validated")
        validation_is_true = validated is True or (
            type(validated) is int and validated == 1
        )
        return bool(
            validation_is_true
            and record.get("semantic_datasource") == self.datasource
            and record.get("semantic_dialect") == self.dialect
            and record.get("semantic_version") == self.semantic_version
            and record.get("semantic_content_sha256") == self.content_sha256
        )

    def matches_audit_identity(self, record: Mapping[str, Any]) -> bool:
        """Match persisted identity fields without treating validation as proof."""
        return bool(
            record.get("semantic_datasource") == self.datasource
            and record.get("semantic_dialect") == self.dialect
            and record.get("semantic_version") == self.semantic_version
            and record.get("semantic_content_sha256") == self.content_sha256
        )


__all__ = ["GovernedHistoryIdentity"]
