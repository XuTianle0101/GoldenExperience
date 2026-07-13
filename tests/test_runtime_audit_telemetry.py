from __future__ import annotations

import hashlib
import json
import socket

import pytest

from goldenexperience.runtime.runtime_audit_telemetry import (
    RUNTIME_AUDIT_TELEMETRY_SCHEMA,
    RuntimeAuditTelemetryCollector,
    RuntimeAuditTelemetryEmitter,
    RuntimeAuditTelemetryError,
)


def _secret() -> str:
    return hashlib.sha256(b"runtime-telemetry-test").hexdigest()


def test_runtime_telemetry_round_trips_authenticated_events() -> None:
    with RuntimeAuditTelemetryCollector(nonce="audit", secret_hex=_secret()) as collector:
        emitter = RuntimeAuditTelemetryEmitter(
            host=collector.host,
            port=collector.port,
            nonce="audit",
            secret_hex=_secret(),
        )
        try:
            emitter.emit(
                request_id="request-1",
                kind="gate",
                evidence={"accepted": False, "decision": "predicted_unsafe"},
            )
            emitter.emit(
                request_id="request-1",
                kind="execution",
                evidence={"source_chunks_read": 0},
            )
            gate = collector.wait_for(request_id="request-1", kind="gate", timeout_s=1.0)
            execution = collector.wait_for(
                request_id="request-1",
                kind="execution",
                timeout_s=1.0,
            )
            collector.assert_drained()
        finally:
            emitter.close()

    assert gate["schema_version"] == RUNTIME_AUDIT_TELEMETRY_SCHEMA
    assert gate["evidence"]["decision"] == "predicted_unsafe"
    assert execution["evidence"]["source_chunks_read"] == 0


def test_runtime_telemetry_rejects_an_unauthenticated_event() -> None:
    with RuntimeAuditTelemetryCollector(nonce="audit", secret_hex=_secret()) as collector:
        envelope = {
            "payload": {
                "schema_version": RUNTIME_AUDIT_TELEMETRY_SCHEMA,
                "nonce": "audit",
                "request_id": "forged",
                "kind": "gate",
                "evidence": {},
            },
            "hmac_sha256": "0" * 64,
        }
        with socket.create_connection((collector.host, collector.port), timeout=1.0) as client:
            client.sendall((json.dumps(envelope) + "\n").encode("utf-8"))

        with pytest.raises(RuntimeAuditTelemetryError, match="authentication failed"):
            collector.wait_for(request_id="forged", kind="gate", timeout_s=1.0)


def test_runtime_telemetry_requires_loopback_and_a_strong_secret() -> None:
    with pytest.raises(RuntimeAuditTelemetryError, match="loopback"):
        RuntimeAuditTelemetryEmitter(
            host="0.0.0.0",
            port=1,
            nonce="audit",
            secret_hex=_secret(),
        )
    with pytest.raises(RuntimeAuditTelemetryError, match="identity"):
        RuntimeAuditTelemetryCollector(nonce="audit", secret_hex="short")


def test_runtime_telemetry_rejects_duplicate_event_identities() -> None:
    with RuntimeAuditTelemetryCollector(nonce="audit", secret_hex=_secret()) as collector:
        emitter = RuntimeAuditTelemetryEmitter(
            host=collector.host,
            port=collector.port,
            nonce="audit",
            secret_hex=_secret(),
        )
        try:
            emitter.emit(request_id="duplicate", kind="gate", evidence={"attempt": 1})
            emitter.emit(request_id="duplicate", kind="gate", evidence={"attempt": 2})
            with pytest.raises(RuntimeAuditTelemetryError, match="duplicated"):
                collector.wait_for(request_id="never", kind="gate", timeout_s=1.0)
        finally:
            emitter.close()


def test_runtime_telemetry_exposes_unconsumed_events() -> None:
    with RuntimeAuditTelemetryCollector(nonce="audit", secret_hex=_secret()) as collector:
        emitter = RuntimeAuditTelemetryEmitter(
            host=collector.host,
            port=collector.port,
            nonce="audit",
            secret_hex=_secret(),
        )
        try:
            emitter.emit(request_id="pending", kind="gate", evidence={})
            emitter.emit(request_id="sentinel", kind="gate", evidence={})
            collector.wait_for(request_id="sentinel", kind="gate", timeout_s=1.0)
            with pytest.raises(RuntimeAuditTelemetryError, match="unconsumed"):
                collector.assert_drained()
            collector.wait_for(request_id="pending", kind="gate", timeout_s=1.0)
            collector.assert_drained()
        finally:
            emitter.close()
