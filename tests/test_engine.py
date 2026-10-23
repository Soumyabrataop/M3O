"""
Test suite for M3O Weight Surgery Engine.

Streams safetensors checkpoints, verifies matrix vector math, heuristics classifications,
and benchmarks execution speed for lambda_val=0.5 and lambda_val=1.8.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pytest

# Ensure repository root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.merger import (
    WeightSurgeryEngine,
    compute_task_vector,
    compute_merged_layer,
    compute_heuristics,
    classify_status,
    STATUS_STABLE,
    STATUS_DEGRADED,
    STATUS_CRITICAL,
)


def test_matrix_vector_calculations():
    """Verify task vector and merged layer math on controlled numerical arrays."""
    w_base = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    w_tuned = np.array([[1.5, 2.5], [2.0, 5.0]], dtype=np.float32)

    # 1. Task vector: tau = W_tuned - W_base
    tau = compute_task_vector(w_tuned, w_base)
    expected_tau = np.array([[0.5, 0.5], [-1.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(tau, expected_tau, rtol=1e-5)

    # 2. Merged layer: W_merged = W_base + (lambda_val * tau)
    # Test lambda = 0.5
    w_merged_05 = compute_merged_layer(w_base, tau, lambda_val=0.5, preserve_dtype=True)
    expected_05 = w_base + (0.5 * expected_tau)
    np.testing.assert_allclose(w_merged_05, expected_05, rtol=1e-5)

    # Test lambda = 1.8
    w_merged_18 = compute_merged_layer(w_base, tau, lambda_val=1.8, preserve_dtype=True)
    expected_18 = w_base + (1.8 * expected_tau)
    np.testing.assert_allclose(w_merged_18, expected_18, rtol=1e-5)


def test_collapse_heuristic_thresholds():
    """Verify status classification against threshold boundaries (0.85 and 0.65)."""
    assert classify_status(0.95) == STATUS_STABLE
    assert classify_status(0.85) == STATUS_STABLE
    assert classify_status(0.8499) == STATUS_DEGRADED
    assert classify_status(0.70) == STATUS_DEGRADED
    assert classify_status(0.65) == STATUS_DEGRADED
    assert classify_status(0.6499) == STATUS_CRITICAL
    assert classify_status(0.10) == STATUS_CRITICAL
    assert classify_status(-0.5) == STATUS_CRITICAL


def test_synthetic_collapse_heuristics():
    """Test compute_heuristics function outputs expected cosine similarity and frobenius drift."""
    w_base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    w_merged = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cos_sim, frob_drift, status = compute_heuristics(w_base, w_merged)

    assert pytest.approx(cos_sim, abs=1e-5) == 1.0
    assert pytest.approx(frob_drift, abs=1e-5) == 0.0
    assert status == STATUS_STABLE


def run_streaming_speed_benchmark():
    """Stream weights from disk, execute passes for lambda=0.5 and lambda=1.8, and benchmark speed."""
    base_path = REPO_ROOT / "weights" / "gemma4_base.safetensors"
    model_path = REPO_ROOT / "weights" / "model.safetensors"

    if not base_path.exists() or not model_path.exists():
        pytest.skip(f"Safetensors weight files missing: {base_path} or {model_path}")

    print("\n" + "=" * 70)
    print("M3O WEIGHT SURGERY ENGINE BENCHMARK")
    print("=" * 70)
    print(f"Base Checkpoint : {base_path} ({base_path.stat().st_size / (1024**3):.2f} GB)")
    print(f"Tuned Checkpoint: {model_path} ({model_path.stat().st_size / (1024**3):.2f} GB)")

    engine = WeightSurgeryEngine(str(base_path), str(model_path))

    for lam in [0.5, 1.8]:
        print(f"\n---> Executing Streaming Merge Pass (lambda = {lam})...")
        t_start = time.perf_counter()
        summary = engine.run_merge(lambda_val=lam, return_tensors=False)
        t_duration = time.perf_counter() - t_start

        print(f"  * Layers Processed        : {summary.total_layers}")
        print(f"  * Elapsed Wall Time       : {summary.elapsed_seconds:.2f} s")
        print(f"  * Streaming Throughput    : {summary.throughput_layers_per_sec:.2f} layers/sec")
        print(f"  * Mean Cosine Similarity  : {summary.mean_cos_sim:.6f}")
        print(f"  * Min / Max Cosine Sim    : {summary.min_cos_sim:.6f} / {summary.max_cos_sim:.6f}")
        print(f"  * Mean Frobenius Drift    : {summary.mean_frob_drift:.6f}")
        print(f"  * Integrity Breakdown     : {summary.stable_count} STABLE | {summary.degraded_count} DEGRADED | {summary.critical_count} CRITICAL")

        # Sanity checks
        assert summary.total_layers == 140, f"Expected 140 target layers, got {summary.total_layers}"
        assert summary.stable_count == 140, "All layers should be STABLE given subtle perturbation shift"
        assert summary.elapsed_seconds < 60.0, f"Streaming pass took too long: {summary.elapsed_seconds:.2f}s"
        assert summary.throughput_layers_per_sec > 2.0, f"Throughput too low: {summary.throughput_layers_per_sec:.2f} layers/s"

    print("\n" + "=" * 70)
    print("[+] All streaming passes and benchmarks completed successfully!")
    print("=" * 70)


def test_streaming_engine_execution():
    """Pytest test case for streaming engine execution."""
    run_streaming_speed_benchmark()


if __name__ == "__main__":
    test_matrix_vector_calculations()
    test_collapse_heuristic_thresholds()
    test_synthetic_collapse_heuristics()
    run_streaming_speed_benchmark()
