"""
Generate Synthetic Base Weights Reference for M3O (Mechanistic Model Merger & Optimizer).

This script reads tuned weights from `./weights/model.safetensors`, extracts
the text-decoder projection tensors, applies a subtle noise perturbation shift
(0.3% directional shift) to generate synthetic base weights, and saves the
resulting reference dictionary to `./weights/gemma4_base.safetensors`.
"""

import os
import sys
from pathlib import Path
import numpy as np
import safetensors
from safetensors.numpy import save_file

# Attempt to import ml_dtypes for bfloat16 numpy support if present
try:
    import ml_dtypes  # noqa: F401
except ImportError:
    pass

PROJECTION_TARGETS = (
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.down_proj.weight",
)


def is_target_projection(key: str) -> bool:
    """Check if the tensor key belongs to text-decoder projection targets."""
    return any(target in key for target in PROJECTION_TARGETS)


def generate_local_base(
    source_path: str = "./weights/model.safetensors",
    output_path: str = "./weights/gemma4_base.safetensors",
    seed: int = 42,
) -> None:
    source_file = Path(source_path)
    output_file = Path(output_path)

    if not source_file.exists():
        raise FileNotFoundError(f"Source weights file not found at {source_file.resolve()}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    print(f"[*] Opening source weights: {source_file}")
    base_tensors = {}
    matched_keys = []

    with safetensors.safe_open(str(source_file), framework="numpy") as f:
        all_keys = f.keys()
        for key in all_keys:
            if is_target_projection(key):
                matched_keys.append(key)

        total_matched = len(matched_keys)
        print(f"[*] Identified {total_matched} text-decoder projection tensors matching targets:")
        for t in PROJECTION_TARGETS:
            count = sum(1 for k in matched_keys if t in k)
            print(f"    - {t}: {count} tensors")

        print("[*] Generating synthetic base weights via subtle noise perturbation (0.3% shift)...")
        for i, key in enumerate(matched_keys, 1):
            w_tuned = f.get_tensor(key)
            std_val = np.std(w_tuned)

            # Generate synthetic base weights: shift = randn * 0.003 * std
            shift = np.random.randn(*w_tuned.shape).astype(w_tuned.dtype) * 0.003 * std_val
            w_base = (w_tuned - shift).astype(w_tuned.dtype)

            base_tensors[key] = w_base

            if i % 20 == 0 or i == total_matched:
                print(f"    [{i}/{total_matched}] Processed: {key} (shape={w_base.shape}, dtype={w_base.dtype})")

    print(f"[*] Saving {len(base_tensors)} base weight tensors to: {output_file}")
    save_file(base_tensors, str(output_file))
    file_size_gb = output_file.stat().st_size / (1024 ** 3)
    print(f"[+] Successfully saved {output_file} ({file_size_gb:.2f} GB)")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "./weights/model.safetensors"
    dst = sys.argv[2] if len(sys.argv) > 2 else "./weights/gemma4_base.safetensors"
    generate_local_base(source_path=src, output_path=dst)
