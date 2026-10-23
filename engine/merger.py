"""
Low-Level Weight Surgery Engine for M3O (Mechanistic Model Merger & Optimizer).

Provides memory-efficient streaming ingestion from Safetensors archives, task vector
surgery mathematics, and real-time model collapse & integrity heuristics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Sequence
import time
import numpy as np
import safetensors

try:
    import ml_dtypes  # noqa: F401
except ImportError:
    pass


# Status constants based on Cosine Similarity
STATUS_STABLE = "STABLE"
STATUS_DEGRADED = "DEGRADED"
STATUS_CRITICAL = "CRITICAL"


@dataclass
class LayerMergeResult:
    """Telemetry and metrics for an individually merged tensor layer."""
    tensor_name: str
    lambda_val: float
    cos_sim: float
    frob_drift: float
    status: str
    shape: Tuple[int, ...]
    dtype: str
    merged_tensor: Optional[np.ndarray] = None


@dataclass
class MergeEngineSummary:
    """Aggregated summary of a model surgery merge execution pass."""
    lambda_val: float
    total_layers: int
    stable_count: int
    degraded_count: int
    critical_count: int
    mean_cos_sim: float
    min_cos_sim: float
    max_cos_sim: float
    mean_frob_drift: float
    elapsed_seconds: float
    throughput_layers_per_sec: float
    results: List[LayerMergeResult] = field(default_factory=list)


def classify_status(cos_sim: float) -> str:
    """
    Classify structural integrity status based on cosine similarity:
    - STABLE:   cos_sim >= 0.85
    - DEGRADED: 0.65 <= cos_sim < 0.85
    - CRITICAL: cos_sim < 0.65
    """
    if cos_sim >= 0.85:
        return STATUS_STABLE
    elif cos_sim >= 0.65:
        return STATUS_DEGRADED
    else:
        return STATUS_CRITICAL


def compute_task_vector(w_tuned: np.ndarray, w_base: np.ndarray) -> np.ndarray:
    """
    Calculate the task vector: tau = W_tuned - W_base.
    Computes difference in original precision.
    """
    return w_tuned - w_base


def compute_merged_layer(
    w_base: np.ndarray,
    tau: np.ndarray,
    lambda_val: float,
    preserve_dtype: bool = True,
) -> np.ndarray:
    """
    Calculate merged layer tensor: W_merged = W_base + (lambda_val * tau).
    Optionally casts back to W_base dtype to maintain precision parity.
    """
    # Compute in float32 for linear combination stability
    w_merged = w_base.astype(np.float32) + (float(lambda_val) * tau.astype(np.float32))
    if preserve_dtype:
        return w_merged.astype(w_base.dtype)
    return w_merged


def compute_heuristics(w_base: np.ndarray, w_merged: np.ndarray) -> Tuple[float, float, str]:
    """
    Compute cosine similarity and Frobenius norm drift between flattened base and merged tensors:
      cos_sim = dot(W_base, W_merged) / (norm(W_base) * norm(W_merged) + 1e-9)
      frob_drift = norm(W_merged - W_base)
    
    Returns:
      (cos_sim, frob_drift, status)
    """
    base_flat = w_base.astype(np.float32).ravel()
    merged_flat = w_merged.astype(np.float32).ravel()

    dot_val = np.dot(base_flat, merged_flat)
    norm_base = np.linalg.norm(base_flat)
    norm_merged = np.linalg.norm(merged_flat)

    cos_sim = float(dot_val / (norm_base * norm_merged + 1e-9))
    frob_drift = float(np.linalg.norm(merged_flat - base_flat))
    status = classify_status(cos_sim)

    return cos_sim, frob_drift, status


class WeightSurgeryEngine:
    """
    Incremental streaming weight surgery engine.
    Reads tensor pairs one-by-one from base and tuned checkpoints without loading
    full model checkpoints into host memory.
    """

    def __init__(
        self,
        base_path: str = "./weights/gemma4_base.safetensors",
        tuned_path: str = "./weights/model.safetensors",
    ):
        self.base_path = Path(base_path)
        self.tuned_path = Path(tuned_path)

        if not self.base_path.exists():
            raise FileNotFoundError(f"Base safetensors not found at: {self.base_path.resolve()}")
        if not self.tuned_path.exists():
            raise FileNotFoundError(f"Tuned safetensors not found at: {self.tuned_path.resolve()}")

    def stream_merge(
        self,
        lambda_val: float,
        target_keys: Optional[Sequence[str]] = None,
        return_tensors: bool = False,
        preserve_dtype: bool = True,
    ) -> Generator[LayerMergeResult, None, None]:
        """
        Streamingly yield merged layer results one tensor at a time.
        
        Args:
            lambda_val: Scaling coefficient for the task vector.
            target_keys: Optional subset of keys to process. If None, uses all keys in base_path.
            return_tensors: If True, attaches the merged tensor array to LayerMergeResult.
            preserve_dtype: If True, preserves base tensor dtype.
        """
        with safetensors.safe_open(str(self.base_path), framework="numpy") as f_base, \
             safetensors.safe_open(str(self.tuned_path), framework="numpy") as f_tuned:

            keys_to_process = target_keys if target_keys is not None else list(f_base.keys())

            for key in keys_to_process:
                # Incremental single-tensor load
                w_base = f_base.get_tensor(key)
                w_tuned = f_tuned.get_tensor(key)

                # Vector surgery math
                tau = compute_task_vector(w_tuned, w_base)
                w_merged = compute_merged_layer(w_base, tau, lambda_val, preserve_dtype=preserve_dtype)

                # Real-time integrity metrics
                cos_sim, frob_drift, status = compute_heuristics(w_base, w_merged)

                yield LayerMergeResult(
                    tensor_name=key,
                    lambda_val=lambda_val,
                    cos_sim=cos_sim,
                    frob_drift=frob_drift,
                    status=status,
                    shape=w_base.shape,
                    dtype=str(w_base.dtype),
                    merged_tensor=w_merged if return_tensors else None,
                )

    def run_merge(
        self,
        lambda_val: float,
        target_keys: Optional[Sequence[str]] = None,
        return_tensors: bool = False,
    ) -> MergeEngineSummary:
        """
        Execute a full streaming merge pass across all target layers and return an aggregated summary.
        """
        t0 = time.perf_counter()
        results: List[LayerMergeResult] = []

        stable_count = 0
        degraded_count = 0
        critical_count = 0
        cos_sims: List[float] = []
        frob_drifts: List[float] = []

        for res in self.stream_merge(lambda_val=lambda_val, target_keys=target_keys, return_tensors=return_tensors):
            results.append(res)
            cos_sims.append(res.cos_sim)
            frob_drifts.append(res.frob_drift)

            if res.status == STATUS_STABLE:
                stable_count += 1
            elif res.status == STATUS_DEGRADED:
                degraded_count += 1
            else:
                critical_count += 1

        elapsed = time.perf_counter() - t0
        total_layers = len(results)
        throughput = (total_layers / elapsed) if elapsed > 0 else 0.0

        mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0
        min_cos = float(np.min(cos_sims)) if cos_sims else 0.0
        max_cos = float(np.max(cos_sims)) if cos_sims else 0.0
        mean_frob = float(np.mean(frob_drifts)) if frob_drifts else 0.0

        return MergeEngineSummary(
            lambda_val=lambda_val,
            total_layers=total_layers,
            stable_count=stable_count,
            degraded_count=degraded_count,
            critical_count=critical_count,
            mean_cos_sim=mean_cos,
            min_cos_sim=min_cos,
            max_cos_sim=max_cos,
            mean_frob_drift=mean_frob,
            elapsed_seconds=elapsed,
            throughput_layers_per_sec=throughput,
            results=results,
        )
