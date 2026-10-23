"""M3O Engine - Task vector math and tensor streaming logic."""

from engine.merger import (
    WeightSurgeryEngine,
    LayerMergeResult,
    MergeEngineSummary,
    compute_task_vector,
    compute_merged_layer,
    compute_heuristics,
    classify_status,
    STATUS_STABLE,
    STATUS_DEGRADED,
    STATUS_CRITICAL,
)

__all__ = [
    "WeightSurgeryEngine",
    "LayerMergeResult",
    "MergeEngineSummary",
    "compute_task_vector",
    "compute_merged_layer",
    "compute_heuristics",
    "classify_status",
    "STATUS_STABLE",
    "STATUS_DEGRADED",
    "STATUS_CRITICAL",
]
