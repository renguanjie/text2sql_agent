"""Short-lived process-local proofs for governed history mutations."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from hmac import compare_digest
from hmac import digest as hmac_digest
from secrets import token_bytes, token_hex
from threading import Lock
from time import monotonic
from typing import Callable

_NONCE = re.compile(r"[0-9a-f]{32}")
_MAC = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class GovernedHistorySave:
    """The complete, immutable payload of one governed audit insert."""

    session_id: str | None
    user_query: str
    generated_sql: str
    matched_knowledge_json: str | None
    retrieval_query: str | None
    validation_status: str
    validation_error: str | None
    semantic_datasource: str
    semantic_dialect: str
    semantic_version: str
    semantic_content_sha256: str


@dataclass(frozen=True, slots=True)
class GovernedHistoryFinalization:
    """The immutable CAS payload for one governed execution attempt."""

    history_id: int
    session_id: str
    user_query: str
    generated_sql: str
    execution_status: str
    execution_error: str | None
    execution_result_json: str | None
    semantic_datasource: str
    semantic_dialect: str
    semantic_version: str
    semantic_content_sha256: str


@dataclass(frozen=True, slots=True)
class HistoryMutationProof:
    """Opaque HMAC capability tied to exactly one immutable command."""

    nonce: str
    expires_at: float
    mac: str


def _proof_payload(command: object, nonce: str, expires_at: float) -> bytes:
    envelope = {
        "command_type": type(command).__name__,
        "command": asdict(command),
        "expires_at": expires_at.hex(),
        "nonce": nonce,
    }
    return json.dumps(
        envelope,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class HistoryProofIssuer:
    """Signs commands; only the governed pipeline service receives this object."""

    def __init__(
        self,
        secret: bytes,
        *,
        ttl_seconds: float,
        clock: Callable[[], float],
    ) -> None:
        self._secret = secret
        self._ttl_seconds = ttl_seconds
        self._clock = clock

    def issue(
        self,
        command: GovernedHistorySave | GovernedHistoryFinalization,
    ) -> HistoryMutationProof:
        if type(command) not in (GovernedHistorySave, GovernedHistoryFinalization):
            raise TypeError("unsupported governed history command")
        nonce = token_hex(16)
        expires_at = self._clock() + self._ttl_seconds
        mac = hmac_digest(
            self._secret,
            _proof_payload(command, nonce, expires_at),
            "sha256",
        ).hex()
        return HistoryMutationProof(nonce=nonce, expires_at=expires_at, mac=mac)


class HistoryProofVerifier:
    """Verifies and atomically consumes process-local command proofs."""

    def __init__(self, secret: bytes, *, clock: Callable[[], float]) -> None:
        self._secret = secret
        self._clock = clock
        self._consumed: dict[str, float] = {}
        self._lock = Lock()

    def consume(
        self,
        command: GovernedHistorySave | GovernedHistoryFinalization,
        proof: HistoryMutationProof,
    ) -> bool:
        if (
            type(command) not in (GovernedHistorySave, GovernedHistoryFinalization)
            or type(proof) is not HistoryMutationProof
            or type(proof.nonce) is not str
            or _NONCE.fullmatch(proof.nonce) is None
            or type(proof.expires_at) is not float
            or not math.isfinite(proof.expires_at)
            or proof.expires_at <= self._clock()
            or type(proof.mac) is not str
            or _MAC.fullmatch(proof.mac) is None
        ):
            return False
        try:
            payload = _proof_payload(command, proof.nonce, proof.expires_at)
        except Exception:
            return False
        expected = hmac_digest(self._secret, payload, "sha256").hex()
        if not compare_digest(expected, proof.mac):
            return False
        with self._lock:
            now = self._clock()
            if proof.expires_at <= now:
                return False
            expired = [
                nonce
                for nonce, expires_at in self._consumed.items()
                if expires_at <= now
            ]
            for nonce in expired:
                self._consumed.pop(nonce, None)
            if proof.nonce in self._consumed:
                return False
            self._consumed[proof.nonce] = proof.expires_at
        return True


def create_history_proof_pair(
    *,
    ttl_seconds: float = 30.0,
    clock: Callable[[], float] = monotonic,
) -> tuple[HistoryProofIssuer, HistoryProofVerifier]:
    """Return a per-process signer/verifier pair with no persisted key material."""
    if (
        isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, (int, float))
        or not math.isfinite(float(ttl_seconds))
        or ttl_seconds <= 0
        or ttl_seconds > 300
    ):
        raise ValueError("history proof ttl must be between 0 and 300 seconds")
    if not callable(clock):
        raise TypeError("history proof clock must be callable")
    secret = token_bytes(32)
    return (
        HistoryProofIssuer(
            secret,
            ttl_seconds=float(ttl_seconds),
            clock=clock,
        ),
        HistoryProofVerifier(secret, clock=clock),
    )


__all__ = [
    "GovernedHistoryFinalization",
    "GovernedHistorySave",
    "HistoryMutationProof",
    "HistoryProofIssuer",
    "HistoryProofVerifier",
    "create_history_proof_pair",
]
