"""Authenticated loopback telemetry for the isolated runtime audit."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import socket
import socketserver
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

RUNTIME_AUDIT_TELEMETRY_SCHEMA = "goldenexperience.runtime_audit_telemetry.v1"
RUNTIME_AUDIT_TELEMETRY_MAX_BYTES = 1024 * 1024


class RuntimeAuditTelemetryError(RuntimeError):
    """Raised when authenticated audit telemetry is missing or malformed."""


class RuntimeAuditTelemetryEmitter:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        nonce: str,
        secret_hex: str,
        connect_timeout_s: float = 5.0,
    ) -> None:
        if host not in {"127.0.0.1", "localhost"}:
            raise RuntimeAuditTelemetryError("runtime telemetry must use loopback")
        if type(port) is not int or not 0 < port <= 65535:
            raise RuntimeAuditTelemetryError("runtime telemetry port is invalid")
        if not nonce or not _is_hex_secret(secret_hex):
            raise RuntimeAuditTelemetryError("runtime telemetry identity is invalid")
        if not _finite_positive(connect_timeout_s):
            raise RuntimeAuditTelemetryError("runtime telemetry timeout is invalid")
        self.host = host
        self.port = port
        self.nonce = nonce
        self.secret = bytes.fromhex(secret_hex)
        self.connect_timeout_s = connect_timeout_s
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()

    def emit(self, *, request_id: str, kind: str, evidence: Mapping[str, Any]) -> None:
        if not request_id or not kind:
            raise RuntimeAuditTelemetryError("runtime telemetry event identity is incomplete")
        payload = {
            "schema_version": RUNTIME_AUDIT_TELEMETRY_SCHEMA,
            "nonce": self.nonce,
            "request_id": request_id,
            "kind": kind,
            "evidence": dict(evidence),
        }
        raw_payload = _canonical_json_bytes(payload)
        envelope = {
            "payload": payload,
            "hmac_sha256": hmac.new(self.secret, raw_payload, hashlib.sha256).hexdigest(),
        }
        encoded = _canonical_json_bytes(envelope)
        if len(encoded) > RUNTIME_AUDIT_TELEMETRY_MAX_BYTES:
            raise RuntimeAuditTelemetryError("runtime telemetry event exceeds its size bound")
        with self._lock:
            for attempt in range(2):
                try:
                    self._connected_socket().sendall(encoded)
                    return
                except OSError as exc:
                    self._close_unlocked()
                    if attempt:
                        raise RuntimeAuditTelemetryError(
                            "runtime telemetry delivery failed"
                        ) from exc

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()

    def _connected_socket(self) -> socket.socket:
        if self._socket is None:
            client = socket.create_connection(
                (self.host, self.port),
                timeout=self.connect_timeout_s,
            )
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._socket = client
        return self._socket

    def _close_unlocked(self) -> None:
        if self._socket is not None:
            with suppress(OSError):
                self._socket.shutdown(socket.SHUT_RDWR)
            self._socket.close()
            self._socket = None


class RuntimeAuditTelemetryCollector:
    def __init__(self, *, nonce: str, secret_hex: str) -> None:
        if not nonce or not _is_hex_secret(secret_hex):
            raise RuntimeAuditTelemetryError("runtime telemetry identity is invalid")
        self.nonce = nonce
        self.secret = bytes.fromhex(secret_hex)
        self._events: list[dict[str, Any]] = []
        self._seen: set[tuple[str, str]] = set()
        self._errors: list[str] = []
        self._condition = threading.Condition()
        self._server = _TelemetryServer(("127.0.0.1", 0), _TelemetryHandler, self)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="golden-runtime-telemetry",
            daemon=True,
        )
        self._thread.start()

    @property
    def host(self) -> str:
        return "127.0.0.1"

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def wait_for(
        self,
        *,
        request_id: str,
        kind: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        if not request_id or not kind or not _finite_positive(timeout_s):
            raise RuntimeAuditTelemetryError("runtime telemetry wait is invalid")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._errors:
                    raise RuntimeAuditTelemetryError(self._errors[0])
                for index, event in enumerate(self._events):
                    if event["request_id"] == request_id and event["kind"] == kind:
                        return self._events.pop(index)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeAuditTelemetryError(
                        f"runtime telemetry timed out for {request_id}:{kind}"
                    )
                self._condition.wait(remaining)

    def assert_drained(self) -> None:
        """Require every authenticated event to have been consumed by the audit."""

        with self._condition:
            if self._errors:
                raise RuntimeAuditTelemetryError(self._errors[0])
            if self._events:
                identities = sorted(
                    f"{event['request_id']}:{event['kind']}" for event in self._events
                )
                raise RuntimeAuditTelemetryError(
                    "runtime telemetry has unconsumed events: " + ", ".join(identities)
                )

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeAuditTelemetryError("runtime telemetry server did not stop")

    def __enter__(self) -> RuntimeAuditTelemetryCollector:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _accept(self, envelope: Any) -> None:
        try:
            if not isinstance(envelope, dict) or set(envelope) != {"payload", "hmac_sha256"}:
                raise RuntimeAuditTelemetryError("runtime telemetry envelope is malformed")
            payload = envelope["payload"]
            signature = envelope["hmac_sha256"]
            if not isinstance(payload, dict) or not isinstance(signature, str):
                raise RuntimeAuditTelemetryError("runtime telemetry envelope types are invalid")
            expected = hmac.new(
                self.secret,
                _canonical_json_bytes(payload),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise RuntimeAuditTelemetryError("runtime telemetry authentication failed")
            if (
                set(payload) != {"schema_version", "nonce", "request_id", "kind", "evidence"}
                or payload.get("schema_version") != RUNTIME_AUDIT_TELEMETRY_SCHEMA
                or payload.get("nonce") != self.nonce
                or not isinstance(payload.get("request_id"), str)
                or not payload["request_id"]
                or not isinstance(payload.get("kind"), str)
                or not payload["kind"]
                or not isinstance(payload.get("evidence"), dict)
            ):
                raise RuntimeAuditTelemetryError("runtime telemetry payload is malformed")
            _canonical_json_bytes(payload["evidence"])
        except (RuntimeAuditTelemetryError, TypeError, ValueError) as exc:
            with self._condition:
                self._errors.append(str(exc))
                self._condition.notify_all()
            return
        identity = (payload["request_id"], payload["kind"])
        with self._condition:
            if identity in self._seen:
                self._errors.append(
                    f"runtime telemetry event was duplicated: {identity[0]}:{identity[1]}"
                )
                self._condition.notify_all()
                return
            self._seen.add(identity)
            self._events.append(payload)
            self._condition.notify_all()


class _TelemetryServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = False
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[socketserver.StreamRequestHandler],
        collector: RuntimeAuditTelemetryCollector,
    ) -> None:
        self.collector = collector
        super().__init__(address, handler)


class _TelemetryHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            line = self.rfile.readline(RUNTIME_AUDIT_TELEMETRY_MAX_BYTES + 1)
            if not line:
                return
            if len(line) > RUNTIME_AUDIT_TELEMETRY_MAX_BYTES or not line.endswith(b"\n"):
                self.server.collector._accept(None)  # type: ignore[attr-defined]
                return
            try:
                envelope = json.loads(line, object_pairs_hook=_unique_json_object)
            except (UnicodeError, json.JSONDecodeError, ValueError):
                envelope = None
            self.server.collector._accept(envelope)  # type: ignore[attr-defined]


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("runtime telemetry JSON contains a duplicate key")
        result[key] = value
    return result


def _is_hex_secret(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _finite_positive(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )
