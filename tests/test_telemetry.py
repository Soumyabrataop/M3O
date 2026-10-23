"""
Test suite for M3O Dual Telemetry Logging Pipeline.

Verifies:
1. Canonical schema payload validity (layer_name, alpha, frobenius_drift, cos_sim, status, timestamp).
2. Hot path MongoDB Atlas mock logging and fail-safe error handling.
3. Cold path Snowflake buffer mechanics and micro-batch flush.
4. Automatic fallback mode when environment variables are omitted.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from telemetry.logger import (
    TelemetryLogger,
    LayerMetricRecord,
    DEFAULT_SNOWFLAKE_TABLE,
)


def test_schema_payload_validity():
    """Verify LayerMetricRecord schema contains all required fields and correct data types."""
    now_iso = datetime.now(timezone.utc).isoformat()
    record = LayerMetricRecord(
        layer_name="model.language_model.layers.0.self_attn.q_proj.weight",
        alpha=0.5,
        frobenius_drift=0.00382,
        cos_sim=0.99998,
        status="STABLE",
        timestamp=now_iso,
    )

    payload = record.to_dict()
    required_keys = {"layer_name", "alpha", "frobenius_drift", "cos_sim", "status", "timestamp"}
    assert set(payload.keys()) == required_keys

    assert isinstance(payload["layer_name"], str)
    assert isinstance(payload["alpha"], float)
    assert isinstance(payload["frobenius_drift"], float)
    assert isinstance(payload["cos_sim"], float)
    assert isinstance(payload["status"], str)
    assert isinstance(payload["timestamp"], str)

    sf_tuple = record.to_snowflake_tuple()
    assert len(sf_tuple) == 6
    assert sf_tuple[0] == record.layer_name
    assert sf_tuple[1] == 0.5
    assert sf_tuple[4] == "STABLE"


def test_hot_path_mock_logging():
    """Verify hot-path captures document records per layer without crashing."""
    logger = TelemetryLogger(mock_mode=True)

    record = logger.log_layer_metric(
        layer_name="model.language_model.layers.0.mlp.gate_proj.weight",
        alpha=1.8,
        frobenius_drift=0.0125,
        cos_sim=0.99995,
        status="STABLE",
    )

    assert len(logger.mongo_mock_records) == 1
    stored = logger.mongo_mock_records[0]

    assert stored["layer_name"] == "model.language_model.layers.0.mlp.gate_proj.weight"
    assert stored["alpha"] == 1.8
    assert stored["frobenius_drift"] == 0.0125
    assert stored["cos_sim"] == 0.99995
    assert stored["status"] == "STABLE"
    assert "timestamp" in stored
    logger.close()


def test_cold_path_buffering_and_flush():
    """Verify Snowflake in-memory buffering and micro-batch flush execution."""
    logger = TelemetryLogger(mock_mode=True, auto_flush_threshold=100)

    # Ingest 5 metrics
    for i in range(5):
        logger.log_layer_metric(
            layer_name=f"layer_{i}",
            alpha=0.5,
            frobenius_drift=0.001 * i,
            cos_sim=1.0 - (0.0001 * i),
            status="STABLE",
        )

    # Check buffer contains 5 records before flush
    assert len(logger.snowflake_buffer) == 5
    assert len(logger.snowflake_flushed_mock) == 0

    # Execute micro-batch flush
    flushed_count = logger.flush_to_snowflake()
    assert flushed_count == 5
    assert len(logger.snowflake_buffer) == 0
    assert len(logger.snowflake_flushed_mock) == 5

    logger.close()


def test_auto_flush_threshold():
    """Verify micro-batches trigger automatically when auto_flush_threshold is reached."""
    batch_size = 4
    logger = TelemetryLogger(mock_mode=True, auto_flush_threshold=batch_size)

    # Ingest 9 records (should trigger 2 auto-flushes of 4, leaving 1 in buffer)
    for i in range(9):
        logger.log_layer_metric(
            layer_name=f"layer_{i}",
            alpha=1.0,
            frobenius_drift=0.005,
            cos_sim=0.9999,
            status="STABLE",
        )

    assert len(logger.snowflake_flushed_mock) == 8
    assert len(logger.snowflake_buffer) == 1

    # Closing flushes remaining buffer
    logger.close()
    assert len(logger.snowflake_flushed_mock) == 9
    assert len(logger.snowflake_buffer) == 0


def test_fallback_when_env_omitted(monkeypatch):
    """Verify pipeline runs completely cleanly even when all DB env vars are absent."""
    # Ensure env vars are cleared
    for var in [
        "MONGODB_URI",
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "M3O_TELEMETRY_MOCK",
    ]:
        monkeypatch.delenv(var, raising=False)

    logger = TelemetryLogger()
    assert not logger.is_mongo_active
    assert not logger.is_snowflake_active

    # Log metrics with no environment configured
    rec = logger.log_layer_metric(
        layer_name="test.fallback.layer",
        alpha=0.7,
        frobenius_drift=0.002,
        cos_sim=0.98,
        status="STABLE",
    )

    assert rec.status == "STABLE"
    assert len(logger.mongo_mock_records) == 1
    assert len(logger.snowflake_buffer) == 1

    flushed = logger.flush_to_snowflake()
    assert flushed == 1
    assert len(logger.snowflake_buffer) == 0
    logger.close()


def test_fail_safe_invalid_mongo_connection(monkeypatch):
    """Verify MongoDB connection attempt to an unreachable host falls back within timeout."""
    # Point to an unreachable host with 2000ms serverSelectionTimeoutMS
    invalid_uri = "mongodb://127.0.0.1:29999/?serverSelectionTimeoutMS=2000"
    logger = TelemetryLogger(mongo_uri=invalid_uri)

    # Verify fallback to mock mode occurred without throwing exception
    assert not logger.is_mongo_active

    rec = logger.log_layer_metric(
        layer_name="unreachable.test.layer",
        alpha=0.5,
        frobenius_drift=0.001,
        cos_sim=1.0,
        status="STABLE",
    )
    assert rec.layer_name == "unreachable.test.layer"
    assert len(logger.mongo_mock_records) == 1
    logger.close()


if __name__ == "__main__":
    test_schema_payload_validity()
    test_hot_path_mock_logging()
    test_cold_path_buffering_and_flush()
    test_auto_flush_threshold()
    test_fallback_when_env_omitted(pytest.MonkeyPatch())
    test_fail_safe_invalid_mongo_connection(pytest.MonkeyPatch())
    print("\n[+] All Telemetry tests completed successfully!")
