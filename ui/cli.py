"""
Command-Line Cockpit Interface for M3O (Mechanistic Model Merger & Optimizer).

Integrates:
1. Streaming weight surgery from engine/merger.py.
2. Dual telemetry dispatch to telemetry/logger.py.
3. Real-time dynamic terminal table via rich.live.Live and rich.table.Table.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List

# Ensure safe UTF-8 terminal encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich import box

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from engine.merger import (
    WeightSurgeryEngine,
    STATUS_STABLE,
    STATUS_DEGRADED,
    STATUS_CRITICAL,
    LayerMergeResult,
)
from telemetry.logger import TelemetryLogger

console = Console(highlight=False)


def format_status_markup(status: str) -> str:
    """Format status with exact Rich markup tags required by M3O specification."""
    if status == STATUS_STABLE:
        return "[green]STABLE[/green]"
    elif status == STATUS_DEGRADED:
        return "[yellow]DEGRADED[/yellow]"
    else:
        return "[bold red]CRITICAL[/bold red]"


def create_cockpit_table(alpha: float) -> Table:
    """Instantiate the dynamic Rich cockpit table."""
    table = Table(
        title=f"[bold cyan]M3O Weight Surgery Cockpit[/bold cyan] (α = {alpha:.3f})",
        box=box.ROUNDED,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Target Layer", style="cyan", no_wrap=True)
    table.add_column("Frobenius Δ", justify="right", style="white")
    table.add_column("Cosine Sim", justify="right", style="white")
    table.add_column("Integrity Status", justify="center")
    return table


def run_cockpit(
    alpha: float = 0.7,
    mongo: Optional[str] = None,
    base: str = "./weights/gemma4_base.safetensors",
    tuned: str = "./weights/model.safetensors",
    limit: Optional[int] = None,
    output: Optional[str] = None,
) -> int:
    """
    Execute streaming surgery pipeline with live terminal cockpit rendering.
    """
    base_file = Path(base)
    tuned_file = Path(tuned)

    if not base_file.exists():
        console.print(f"[bold red]Error:[/bold red] Base weights checkpoint not found: {base_file}")
        return 1
    if not tuned_file.exists():
        console.print(f"[bold red]Error:[/bold red] Tuned weights checkpoint not found: {tuned_file}")
        return 1

    # Initialize Engine & Telemetry Pipeline
    try:
        engine = WeightSurgeryEngine(base_path=str(base_file), tuned_path=str(tuned_file))
    except Exception as e:
        console.print(f"[bold red]Engine Initialization Failed:[/bold red] {e}")
        return 1

    telemetry = TelemetryLogger(mongo_uri=mongo)

    # Session Header Panel
    mongo_status = "[bold green]Online[/bold green]" if telemetry.is_mongo_active else "[yellow]Mock Mode[/yellow]"
    sf_status = "[bold green]Online[/bold green]" if telemetry.is_snowflake_active else "[yellow]Mock Mode[/yellow]"

    banner = Text()
    banner.append("M3O: MECHANISTIC MODEL MERGER & OPTIMIZER\n", style="bold cyan")
    banner.append(f"Streaming Surgery Cockpit • α = {alpha:.3f} • Base: {base_file.name} • Tuned: {tuned_file.name}\n", style="dim white")
    banner.append(f"Hot Path (MongoDB Atlas): ", style="dim")
    banner.append("● ONLINE  ", style="bold green" if telemetry.is_mongo_active else "yellow")
    banner.append("Cold Path (Snowflake Warehouse): ", style="dim")
    banner.append("● ONLINE", style="bold green" if telemetry.is_snowflake_active else "yellow")
    console.print(Panel(banner, box=box.ROUNDED, border_style="cyan", expand=False))

    # Read target keys
    import safetensors
    with safetensors.safe_open(str(base_file), framework="numpy") as f_base:
        target_keys = list(f_base.keys())

    if limit and limit > 0:
        target_keys = target_keys[:limit]

    total_layers = len(target_keys)
    console.print(f"[*] Streaming [bold]{total_layers}[/bold] projection layers from disk...\n")

    table = create_cockpit_table(alpha=alpha)
    results: List[LayerMergeResult] = []
    stable_count = 0
    degraded_count = 0
    critical_count = 0

    t_start = time.perf_counter()

    # Dynamic Live Terminal Refresh Loop
    with Live(table, console=console, refresh_per_second=8, auto_refresh=True):
        for res in engine.stream_merge(
            lambda_val=alpha,
            target_keys=target_keys,
            return_tensors=bool(output),
        ):
            results.append(res)
            if res.status == STATUS_STABLE:
                stable_count += 1
            elif res.status == STATUS_DEGRADED:
                degraded_count += 1
            else:
                critical_count += 1

            # Dispatch layer telemetry to hot & cold paths
            telemetry.log_layer_metric(
                layer_name=res.tensor_name,
                alpha=res.lambda_val,
                frobenius_drift=res.frob_drift,
                cos_sim=res.cos_sim,
                status=res.status,
            )

            # Update live table
            table.add_row(
                res.tensor_name,
                f"{res.frob_drift:.6f}",
                f"{res.cos_sim:.6f}",
                format_status_markup(res.status),
            )

    t_elapsed = time.perf_counter() - t_start
    throughput = (total_layers / t_elapsed) if t_elapsed > 0 else 0.0

    # Cold-Path Flush
    flushed_sf = telemetry.flush_to_snowflake()
    telemetry.close()

    # Optional weight saving
    if output and results:
        from safetensors.numpy import save_file
        out_file = Path(output)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        merged_dict = {r.tensor_name: r.merged_tensor for r in results if r.merged_tensor is not None}
        console.print(f"\n[*] Saving {len(merged_dict)} merged tensors to [bold]{out_file}[/bold]...")
        save_file(merged_dict, str(out_file))
        console.print(f"[bold green][+][/bold green] Saved {out_file} ({out_file.stat().st_size / (1024**3):.2f} GB)")

    # Executive Summary Panel
    mean_cos = sum(r.cos_sim for r in results) / len(results) if results else 0.0
    mean_frob = sum(r.frob_drift for r in results) / len(results) if results else 0.0
    min_cos = min(r.cos_sim for r in results) if results else 0.0

    overall_status = STATUS_STABLE if min_cos >= 0.85 else (STATUS_DEGRADED if min_cos >= 0.65 else STATUS_CRITICAL)

    summary_table = Table(show_header=False, box=box.SIMPLE)
    summary_table.add_column("Metric", style="dim cyan")
    summary_table.add_column("Result", style="bold white")
    summary_table.add_row("Total Layers Evaluated", str(total_layers))
    summary_table.add_row("Execution Wall Time", f"{t_elapsed:.2f} s")
    summary_table.add_row("Streaming Throughput", f"{throughput:.2f} layers/sec")
    summary_table.add_row("Mean Cosine Similarity", f"{mean_cos:.6f}")
    summary_table.add_row("Mean Frobenius Δ", f"{mean_frob:.6f}")
    summary_table.add_row(
        "Integrity Breakdown",
        f"[green]{stable_count} STABLE[/green] | [yellow]{degraded_count} DEGRADED[/yellow] | [red]{critical_count} CRITICAL[/red]"
    )
    summary_table.add_row("Cold Path Micro-batch Flushed", f"{flushed_sf} records")
    summary_table.add_row("Overall Model Health", format_status_markup(overall_status))

    console.print()
    console.print(Panel(summary_table, title="[bold green]Executive Surgery Summary[/bold green]", border_style="green", expand=False))
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="m3o-cli",
        description="M3O Command-Line Cockpit Interface",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Task vector scaling coefficient (default: 0.7)",
    )
    parser.add_argument(
        "--mongo",
        type=str,
        default=None,
        help="MongoDB Atlas connection URI",
    )
    parser.add_argument(
        "--base",
        type=str,
        default="./weights/gemma4_base.safetensors",
        help="Path to base reference weights safetensors file (default: ./weights/gemma4_base.safetensors)",
    )
    parser.add_argument(
        "--tuned",
        type=str,
        default="./weights/model.safetensors",
        help="Path to tuned weights safetensors file (default: ./weights/model.safetensors)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of layers to process",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path to save merged weights",
    )

    args = parser.parse_args()
    sys.exit(
        run_cockpit(
            alpha=args.alpha,
            mongo=args.mongo,
            base=args.base,
            tuned=args.tuned,
            limit=args.limit,
            output=args.output,
        )
    )


if __name__ == "__main__":
    main()
