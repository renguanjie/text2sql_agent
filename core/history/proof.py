"""Short-lived process-local proofs for governed history mutations."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from hmac import compare_digest
from hmac import digest as hmac_digest
from secrets import token_bytes, token_hex
from threading import Lock
from time import monotonic
from typing import Callable

_NONCE = re.compile(r"[0-9a-f]{32}")
_MAC = re.compile(r"[0-9a-f]{64}")
_DEFAULT_MAX_OUTSTANDING_PROOFS = 1_024
_MAX_OUTSTANDING_PROOFS = 4_096


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


@dataclass(frozen=True, slots=True)
class _OutstandingProof:
    """The issuer-side facts a verifier must see before accepting a proof."""

    expires_at: float
    payload_digest: bytes


class _OutstandingProofStore:
    """Bounded, shared proof state with indexed expiry-heap removal."""

    def __init__(self, max_outstanding: int) -> None:
        self._max_outstanding = max_outstanding
        self._records: dict[str, _OutstandingProof] = {}
        self._expiry_heap: list[tuple[float, str]] = []
        self._heap_positions: dict[str, int] = {}
        self._lock = Lock()

    def register(
        self,
        nonce: str,
        expires_at: float,
        payload_digest: bytes,
        *,
        now: float,
    ) -> bool:
        """Register a new proof, or fail closed when the store is full."""
        with self._lock:
            self._purge_expired(now)
            if len(self._records) >= self._max_outstanding:
                raise RuntimeError("history proof outstanding capacity exhausted")
            if nonce in self._records:
                return False
            self._records[nonce] = _OutstandingProof(expires_at, payload_digest)
            self._heap_push(expires_at, nonce)
            return True

    def verify_and_consume(
        self,
        command: GovernedHistorySave | GovernedHistoryFinalization,
        proof: HistoryMutationProof,
        *,
        secret: bytes,
        clock: Callable[[], float],
    ) -> bool:
        """Validate and delete exactly one issued proof under the shared lock."""
        with self._lock:
            now = clock()
            self._purge_expired(now)
            if proof.expires_at <= now:
                return False
            try:
                payload = _proof_payload(command, proof.nonce, proof.expires_at)
                payload_digest = sha256(payload).digest()
                expected = hmac_digest(secret, payload, "sha256").hex()
            except Exception:
                return False
            if not compare_digest(expected, proof.mac):
                return False

            # HMAC work can span the expiry boundary, so check it once more
            # before removing the outstanding record.
            now = clock()
            self._purge_expired(now)
            if proof.expires_at <= now:
                return False
            outstanding = self._records.get(proof.nonce)
            if (
                outstanding is None
                or outstanding.expires_at != proof.expires_at
                or not compare_digest(outstanding.payload_digest, payload_digest)
            ):
                return False
            self._records.pop(proof.nonce)
            self._heap_remove(proof.nonce)
            return True

    def size(self, *, now: float) -> int:
        """Return live entries and reclaim any expired records on observation."""
        with self._lock:
            self._purge_expired(now)
            return len(self._records)

    def _purge_expired(self, now: float) -> None:
        while self._expiry_heap and self._expiry_heap[0][0] <= now:
            _expires_at, nonce = self._expiry_heap[0]
            self._heap_remove(nonce)
            self._records.pop(nonce, None)

    def _heap_push(self, expires_at: float, nonce: str) -> None:
        index = len(self._expiry_heap)
        self._expiry_heap.append((expires_at, nonce))
        self._heap_positions[nonce] = index
        self._sift_up(index)

    def _heap_remove(self, nonce: str) -> None:
        index = self._heap_positions.pop(nonce)
        last = self._expiry_heap.pop()
        if index == len(self._expiry_heap):
            return
        self._expiry_heap[index] = last
        self._heap_positions[last[1]] = index
        parent = (index - 1) // 2
        if index and self._expiry_heap[index] < self._expiry_heap[parent]:
            self._sift_up(index)
        else:
            self._sift_down(index)

    def _sift_up(self, index: int) -> None:
        while index:
            parent = (index - 1) // 2
            if self._expiry_heap[parent] <= self._expiry_heap[index]:
                return
            self._heap_swap(index, parent)
            index = parent

    def _sift_down(self, index: int) -> None:
        heap_size = len(self._expiry_heap)
        while True:
            left = 2 * index + 1
            right = left + 1
            smallest = index
            if (
                left < heap_size
                and self._expiry_heap[left] < self._expiry_heap[smallest]
            ):
                smallest = left
            if (
                right < heap_size
                and self._expiry_heap[right] < self._expiry_heap[smallest]
            ):
                smallest = right
            if smallest == index:
                return
            self._heap_swap(index, smallest)
            index = smallest

    def _heap_swap(self, left: int, right: int) -> None:
        self._expiry_heap[left], self._expiry_heap[right] = (
            self._expiry_heap[right],
            self._expiry_heap[left],
        )
        self._heap_positions[self._expiry_heap[left][1]] = left
        self._heap_positions[self._expiry_heap[right][1]] = right


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
        outstanding_store: _OutstandingProofStore,
    ) -> None:
        self._secret = secret
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._outstanding_store = outstanding_store

    def issue(
        self,
        command: GovernedHistorySave | GovernedHistoryFinalization,
    ) -> HistoryMutationProof:
        if type(command) not in (GovernedHistorySave, GovernedHistoryFinalization):
            raise TypeError("unsupported governed history command")
        now = self._clock()
        expires_at = now + self._ttl_seconds
        for _ in range(8):
            nonce = token_hex(16)
            payload = _proof_payload(command, nonce, expires_at)
            if not self._outstanding_store.register(
                nonce,
                expires_at,
                sha256(payload).digest(),
                now=now,
            ):
                continue
            mac = hmac_digest(self._secret, payload, "sha256").hex()
            return HistoryMutationProof(nonce=nonce, expires_at=expires_at, mac=mac)
        raise RuntimeError("could not allocate a unique history proof nonce")

    @property
    def outstanding_size(self) -> int:
        """The number of live, issued proofs retained in this process."""
        return self._outstanding_store.size(now=self._clock())


class HistoryProofVerifier:
    """Verifies and atomically consumes process-local command proofs."""

    def __init__(
        self,
        secret: bytes,
        *,
        clock: Callable[[], float],
        outstanding_store: _OutstandingProofStore,
    ) -> None:
        self._secret = secret
        self._clock = clock
        self._outstanding_store = outstanding_store

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
            or type(proof.mac) is not str
            or _MAC.fullmatch(proof.mac) is None
        ):
            return False
        return self._outstanding_store.verify_and_consume(
            command,
            proof,
            secret=self._secret,
            clock=self._clock,
        )

    @property
    def outstanding_size(self) -> int:
        """The number of live, issued proofs retained in this process."""
        return self._outstanding_store.size(now=self._clock())


def create_history_proof_pair(
    *,
    ttl_seconds: float = 30.0,
    max_outstanding: int = _DEFAULT_MAX_OUTSTANDING_PROOFS,
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
    if (
        isinstance(max_outstanding, bool)
        or not isinstance(max_outstanding, int)
        or not 1 <= max_outstanding <= _MAX_OUTSTANDING_PROOFS
    ):
        raise ValueError(
            "history proof outstanding capacity must be an integer between 1 and 4096"
        )
    if not callable(clock):
        raise TypeError("history proof clock must be callable")
    secret = token_bytes(32)
    outstanding_store = _OutstandingProofStore(max_outstanding)
    return (
        HistoryProofIssuer(
            secret,
            ttl_seconds=float(ttl_seconds),
            clock=clock,
            outstanding_store=outstanding_store,
        ),
        HistoryProofVerifier(
            secret,
            clock=clock,
            outstanding_store=outstanding_store,
        ),
    )


__all__ = [
    "GovernedHistoryFinalization",
    "GovernedHistorySave",
    "HistoryMutationProof",
    "HistoryProofIssuer",
    "HistoryProofVerifier",
    "create_history_proof_pair",
]
