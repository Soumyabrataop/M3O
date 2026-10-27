"""
Real Stress Testing Module for M3O (Mechanistic Model Merger & Optimizer).

Provides comprehensive stress testing that:
1. Tests multiple alpha values across the spectrum
2. Identifies collapse boundaries empirically
3. Measures performance under load
4. Validates numerical stability
5. Tests edge cases (negative alpha, extreme values)
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from engine.merger import (
    WeightSurgeryEngine,
    STATUS_STABLE,
    STATUS_DEGRADED,
    STATUS_CRITICAL,
    LayerMergeResult,
    classify_status,
)


@dataclass
class StressTestResult:
    """Results from a single stress test pass with a specific alpha value."""
    alpha: float
    total_layers: int
    stable_count: int
    degraded_count: int
    critical_count: int
    mean_cos_sim: float
    min_cos_sim: float
    mean_frob_drift: float
    max_frob_drift: float
    elapsed_seconds: float
    throughput: float
    status: str
    
    def to_dict(self) -> Dict:
        return {
            "alpha": self.alpha,
            "total_layers": self.total_layers,
            "stable_count": self.stable_count,
            "degraded_count": self.degraded_count,
            "critical_count": self.critical_count,
            "mean_cos_sim": self.mean_cos_sim,
            "min_cos_sim": self.min_cos_sim,
            "mean_frob_drift": self.mean_frob_drift,
            "max_frob_drift": self.max_frob_drift,
            "elapsed_seconds": self.elapsed_seconds,
            "throughput": self.throughput,
            "status": self.status,
        }


@dataclass
class StressTestReport:
    """Comprehensive stress test report across multiple alpha values."""
    results: List[StressTestResult] = field(default_factory=list)
    collapse_threshold_degraded: Optional[float] = None  # Alpha where first DEGRADED appears
    collapse_threshold_critical: Optional[float] = None  # Alpha where first CRITICAL appears
    optimal_alpha: Optional[float] = None  # Alpha with best stability/performance ratio
    performance_baseline: Optional[float] = None  # Baseline throughput (layers/sec)
    
    def to_dict(self) -> Dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "collapse_threshold_degraded": self.collapse_threshold_degraded,
            "collapse_threshold_critical": self.collapse_threshold_critical,
            "optimal_alpha": self.optimal_alpha,
            "performance_baseline": self.performance_baseline,
        }


def run_comprehensive_stress_test(
    base_path: str,
    tuned_path: str,
    alpha_values: Optional[List[float]] = None,
    layer_limit: Optional[int] = None,
    verbose: bool = True,
) -> StressTestReport:
    """
    Run a comprehensive stress test across multiple alpha values.
    
    Tests the following scenarios:
    1. Negative alpha (inverse task vector - subtracts fine-tuning)
    2. Zero alpha (pure base model - identity operation)
    3. Small positive alpha (subtle merge)
    4. Standard alpha values (0.5, 0.7, 1.0)
    5. Aggressive alpha (> 1.0 - amplifies task vector)
    6. Extreme alpha (tests numerical stability)
    
    Returns a report with collapse thresholds and performance metrics.
    """
    base_file = Path(base_path)
    tuned_file = Path(tuned_path)
    
    if not base_file.exists():
        raise FileNotFoundError(f"Base weights not found: {base_file}")
    if not tuned_file.exists():
        raise FileNotFoundError(f"Tuned weights not found: {tuned_file}")
    
    # Default alpha test spectrum
    if alpha_values is None:
        alpha_values = [
            -1.0,   # Inverse task vector
            -0.5,   # Partial inverse
            0.0,    # Identity (base only)
            0.3,    # Conservative merge
            0.5,    # Standard merge
            0.7,    # Default
            1.0,    # Full fine-tuned
            1.5,    # Aggressive
            2.0,    # Double task vector
            3.0,    # Extreme
            5.0,    # Very extreme
            10.0,   # Collapse territory
        ]
    
    engine = WeightSurgeryEngine(base_path=str(base_file), tuned_path=str(tuned_file))
    
    # Get target keys
    import safetensors
    with safetensors.safe_open(str(base_file), framework="numpy") as f_base:
        all_keys = list(f_base.keys())
    
    if layer_limit and layer_limit > 0:
        all_keys = all_keys[:layer_limit]
    
    if verbose:
        print(f"\n{'='*70}")
        print("M3O COMPREHENSIVE STRESS TEST")
        print(f"{'='*70}")
        print(f"Base: {base_file.name} ({base_file.stat().st_size / (1024**3):.2f} GB)")
        print(f"Tuned: {tuned_file.name} ({tuned_file.stat().st_size / (1024**3):.2f} GB)")
        print(f"Layers to test: {len(all_keys)}")
        print(f"Alpha values: {len(alpha_values)} test points")
        print(f"{'='*70}\n")
    
    report = StressTestReport()
    
    for alpha in alpha_values:
        if verbose:
            print(f"Testing α = {alpha:+.2f}...", end=" ", flush=True)
        
        t_start = time.perf_counter()
        results: List[LayerMergeResult] = []
        
        stable_count = 0
        degraded_count = 0
        critical_count = 0
        cos_sims: List[float] = []
        frob_drifts: List[float] = []
        
        try:
            for res in engine.stream_merge(lambda_val=alpha, target_keys=all_keys):
                results.append(res)
                cos_sims.append(res.cos_sim)
                frob_drifts.append(res.frob_drift)
                
                if res.status == STATUS_STABLE:
                    stable_count += 1
                elif res.status == STATUS_DEGRADED:
                    degraded_count += 1
                else:
                    critical_count += 1
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            continue
        
        t_elapsed = time.perf_counter() - t_start
        throughput = len(results) / t_elapsed if t_elapsed > 0 else 0.0
        
        mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0
        min_cos = float(np.min(cos_sims)) if cos_sims else 0.0
        mean_frob = float(np.mean(frob_drifts)) if frob_drifts else 0.0
        max_frob = float(np.max(frob_drifts)) if frob_drifts else 0.0
        
        # Determine overall status for this alpha
        if critical_count > 0:
            status = STATUS_CRITICAL
        elif degraded_count > 0:
            status = STATUS_DEGRADED
        else:
            status = STATUS_STABLE
        
        result = StressTestResult(
            alpha=alpha,
            total_layers=len(results),
            stable_count=stable_count,
            degraded_count=degraded_count,
            critical_count=critical_count,
            mean_cos_sim=mean_cos,
            min_cos_sim=min_cos,
            mean_frob_drift=mean_frob,
            max_frob_drift=max_frob,
            elapsed_seconds=t_elapsed,
            throughput=throughput,
            status=status,
        )
        report.results.append(result)
        
        # Track collapse thresholds
        if degraded_count > 0 and report.collapse_threshold_degraded is None:
            report.collapse_threshold_degraded = alpha
        if critical_count > 0 and report.collapse_threshold_critical is None:
            report.collapse_threshold_critical = alpha
        
        # Track performance baseline (using alpha=0 as reference)
        if alpha == 0.0:
            report.performance_baseline = throughput
        
        if verbose:
            status_color = "\033[92m" if status == STATUS_STABLE else ("\033[93m" if status == STATUS_DEGRADED else "\033[91m")
            reset_color = "\033[0m"
            print(f"{status_color}{status}{reset_color} | Cos: {mean_cos:.4f} (min: {min_cos:.4f}) | "
                  f"Frob: {mean_frob:.4f} | {stable_count}S/{degraded_count}D/{critical_count}C | "
                  f"{throughput:.1f} lyr/s")
    
    # Find optimal alpha (highest mean cos_sim among stable configurations)
    stable_results = [r for r in report.results if r.status == STATUS_STABLE]
    if stable_results:
        best = max(stable_results, key=lambda r: r.mean_cos_sim)
        report.optimal_alpha = best.alpha
    
    if verbose:
        print(f"\n{'='*70}")
        print("STRESS TEST SUMMARY")
        print(f"{'='*70}")
        print(f"Degraded threshold: α = {report.collapse_threshold_degraded or 'Not reached'}")
        print(f"Critical threshold: α = {report.collapse_threshold_critical or 'Not reached'}")
        print(f"Optimal alpha: α = {report.optimal_alpha or 'N/A'} (best stable config)")
        print(f"Performance baseline: {report.performance_baseline:.1f} layers/sec" if report.performance_baseline else "Performance baseline: N/A")
        print(f"{'='*70}\n")
    
    return report


def run_simulated_collapse_test(
    num_layers: int = 140,
    alpha_values: Optional[List[float]] = None,
    verbose: bool = True,
) -> StressTestReport:
    """
    Run a simulated collapse test with synthetic weights that demonstrate
    real DEGRADED and CRITICAL behavior.
    
    This is useful for:
    1. Demonstrating the collapse detection system
    2. Testing without large weight files
    3. Understanding how alpha affects different layer types
    
    Unlike the real stress test which uses actual weights, this simulates
    realistic task vector scenarios where:
    - Different layers have different sensitivity to task vector scaling
    - Some layers collapse earlier than others
    - The relationship mimics real model surgery scenarios
    """
    if alpha_values is None:
        alpha_values = [
            0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0
        ]
    
    if verbose:
        print(f"\n{'='*70}")
        print("M3O SIMULATED COLLAPSE TEST")
        print(f"{'='*70}")
        print(f"Simulating {num_layers} layers with realistic collapse behavior")
        print(f"Alpha values: {len(alpha_values)} test points")
        print(f"{'='*70}\n")
    
    report = StressTestReport()
    
    for alpha in alpha_values:
        if verbose:
            print(f"Testing α = {alpha:+.2f}...", end=" ", flush=True)
        
        stable_count = 0
        degraded_count = 0
        critical_count = 0
        cos_sims: List[float] = []
        frob_drifts: List[float] = []
        
        for layer_idx in range(num_layers):
            # Simulate realistic collapse behavior:
            # - Different layers have different "sensitivity" to alpha scaling
            # - Attention layers (first ~35) are more sensitive
            # - MLP layers are more robust
            # - Some layers have orthogonal task vectors (collapse faster)
            
            # Determine layer type based on index pattern
            proj_type = layer_idx % 4  # 0=q_proj, 1=v_proj, 2=gate_proj, 3=down_proj
            layer_depth = layer_idx // 4
            
            # Base cosine similarity without scaling (represents task vector alignment)
            # Task vectors can be: well-aligned (high cos), orthogonal (cos~0), or anti-aligned (negative)
            if proj_type == 0:  # q_proj - often most sensitive
                base_alignment = 0.92 - (layer_depth * 0.003)  # Degrades with depth
                sensitivity = 0.15
            elif proj_type == 1:  # v_proj - moderate sensitivity
                base_alignment = 0.95 - (layer_depth * 0.002)
                sensitivity = 0.10
            elif proj_type == 2:  # gate_proj - can be very sensitive
                base_alignment = 0.88 - (layer_depth * 0.004)
                sensitivity = 0.20
            else:  # down_proj - usually robust
                base_alignment = 0.97 - (layer_depth * 0.001)
                sensitivity = 0.08
            
            # Simulate cosine similarity degradation with alpha scaling
            # cos_sim decreases as alpha moves away from optimal (around 0.5-1.0)
            alpha_deviation = abs(alpha - 0.7)  # Optimal around 0.7
            
            # Non-linear degradation: small alpha changes are fine, large ones cause collapse
            if alpha_deviation < 0.5:
                degradation = 0.0
            elif alpha_deviation < 1.5:
                degradation = sensitivity * (alpha_deviation - 0.5) * 0.1
            elif alpha_deviation < 3.0:
                degradation = sensitivity * 0.1 + sensitivity * (alpha_deviation - 1.5) * 0.15
            else:
                degradation = sensitivity * 0.25 + sensitivity * (alpha_deviation - 3.0) * 0.2
            
            # Add some noise for realism
            noise = np.random.normal(0, 0.01)
            
            cos_sim = max(-1.0, min(1.0, base_alignment - degradation + noise))
            cos_sims.append(cos_sim)
            
            # Frobenius drift scales roughly linearly with alpha magnitude
            frob_base = 0.001 + (layer_depth * 0.0001)
            frob_drift = frob_base * (abs(alpha) + 0.5) * (1 + sensitivity)
            frob_drifts.append(frob_drift)
            
            # Classify status
            if cos_sim >= 0.85:
                stable_count += 1
            elif cos_sim >= 0.65:
                degraded_count += 1
            else:
                critical_count += 1
        
        t_elapsed = 0.01 * num_layers  # Simulated time
        throughput = num_layers / t_elapsed if t_elapsed > 0 else 0.0
        
        mean_cos = float(np.mean(cos_sims))
        min_cos = float(np.min(cos_sims))
        mean_frob = float(np.mean(frob_drifts))
        max_frob = float(np.max(frob_drifts))
        
        if critical_count > 0:
            status = STATUS_CRITICAL
        elif degraded_count > 0:
            status = STATUS_DEGRADED
        else:
            status = STATUS_STABLE
        
        result = StressTestResult(
            alpha=alpha,
            total_layers=num_layers,
            stable_count=stable_count,
            degraded_count=degraded_count,
            critical_count=critical_count,
            mean_cos_sim=mean_cos,
            min_cos_sim=min_cos,
            mean_frob_drift=mean_frob,
            max_frob_drift=max_frob,
            elapsed_seconds=t_elapsed,
            throughput=throughput,
            status=status,
        )
        report.results.append(result)
        
        # Track collapse thresholds
        if degraded_count > 0 and report.collapse_threshold_degraded is None:
            report.collapse_threshold_degraded = alpha
        if critical_count > 0 and report.collapse_threshold_critical is None:
            report.collapse_threshold_critical = alpha
        
        # Track performance baseline
        if alpha == 0.0:
            report.performance_baseline = throughput
        
        if verbose:
            status_color = "\033[92m" if status == STATUS_STABLE else ("\033[93m" if status == STATUS_DEGRADED else "\033[91m")
            reset_color = "\033[0m"
            print(f"{status_color}{status}{reset_color} | Cos: {mean_cos:.4f} (min: {min_cos:.4f}) | "
                  f"Frob: {mean_frob:.4f} | {stable_count}S/{degraded_count}D/{critical_count}C | "
                  f"{throughput:.1f} lyr/s")
    
    # Find optimal alpha
    stable_results = [r for r in report.results if r.status == STATUS_STABLE]
    if stable_results:
        best = max(stable_results, key=lambda r: r.mean_cos_sim)
        report.optimal_alpha = best.alpha
    
    if verbose:
        print(f"\n{'='*70}")
        print("SIMULATED TEST SUMMARY")
        print(f"{'='*70}")
        print(f"Degraded threshold: α = {report.collapse_threshold_degraded or 'Not reached'}")
        print(f"Critical threshold: α = {report.collapse_threshold_critical or 'Not reached'}")
        print(f"Optimal alpha: α = {report.optimal_alpha or 'N/A'} (best stable config)")
        print(f"{'='*70}\n")
    
    return report


def find_collapse_boundary(
    base_path: str,
    tuned_path: str,
    search_range: Tuple[float, float] = (0.5, 10.0),
    tolerance: float = 0.1,
    layer_limit: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """
    Binary search to find the exact alpha value where collapse occurs.
    
    Uses binary search between search_range to find:
    1. First alpha where DEGRADED appears
    2. First alpha where CRITICAL appears
    
    Returns dict with boundary values.
    """
    base_file = Path(base_path)
    tuned_file = Path(tuned_path)
    
    if not base_file.exists() or not tuned_file.exists():
        raise FileNotFoundError("Weight files not found")
    
    engine = WeightSurgeryEngine(base_path=str(base_file), tuned_path=str(tuned_file))
    
    import safetensors
    with safetensors.safe_open(str(base_file), framework="numpy") as f_base:
        all_keys = list(f_base.keys())
    
    if layer_limit:
        all_keys = all_keys[:layer_limit]
    
    def check_alpha(alpha: float) -> Tuple[int, int]:
        """Returns (degraded_count, critical_count) for given alpha."""
        degraded = 0
        critical = 0
        for res in engine.stream_merge(lambda_val=alpha, target_keys=all_keys):
            if res.status == STATUS_DEGRADED:
                degraded += 1
            elif res.status == STATUS_CRITICAL:
                critical += 1
        return degraded, critical
    
    if verbose:
        print("\nBinary searching for collapse boundaries...")
    
    # Find DEGRADED boundary
    degraded_boundary = None
    low, high = search_range
    while high - low > tolerance:
        mid = (low + high) / 2
        degraded, critical = check_alpha(mid)
        if degraded > 0 or critical > 0:
            high = mid
        else:
            low = mid
    degraded_boundary = high
    
    # Find CRITICAL boundary
    critical_boundary = None
    low, high = degraded_boundary, search_range[1]
    while high - low > tolerance:
        mid = (low + high) / 2
        degraded, critical = check_alpha(mid)
        if critical > 0:
            high = mid
        else:
            low = mid
    critical_boundary = high
    
    if verbose:
        print(f"Degraded boundary: α ≈ {degraded_boundary:.2f}")
        print(f"Critical boundary: α ≈ {critical_boundary:.2f}")
    
    return {
        "degraded_boundary": degraded_boundary,
        "critical_boundary": critical_boundary,
    }
