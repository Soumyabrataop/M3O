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
from rich.markdown import Markdown
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
from engine.stress_test import run_comprehensive_stress_test, find_collapse_boundary, run_simulated_collapse_test
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
    table.add_column("Target Layer", style="cyan", ratio=4, overflow="ellipsis")
    table.add_column("Frobenius Δ", justify="right", style="white", min_width=13, no_wrap=True)
    table.add_column("Cosine Sim", justify="right", style="white", min_width=11, no_wrap=True)
    table.add_column("Integrity Status", justify="center", min_width=18, no_wrap=True)
    return table


def run_stress_test(
    base: str,
    tuned: str,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> int:
    """
    Run comprehensive stress test to find collapse boundaries and validate system.
    """
    base_file = Path(base)
    tuned_file = Path(tuned)
    
    if not base_file.exists():
        console.print(f"[bold red]Error:[/bold red] Base weights checkpoint not found: {base_file}")
        return 1
    if not tuned_file.exists():
        console.print(f"[bold red]Error:[/bold red] Tuned weights checkpoint not found: {tuned_file}")
        return 1
    
    console.print(Panel(
        "[bold cyan]M3O Stress Test Mode[/bold cyan]\n"
        "Testing multiple alpha values to find collapse boundaries\n"
        "and validate numerical stability across the spectrum.",
        box=box.ROUNDED,
        border_style="cyan",
    ))
    
    try:
        report = run_comprehensive_stress_test(
            base_path=str(base_file),
            tuned_path=str(tuned_file),
            layer_limit=limit,
            verbose=verbose,
        )
    except Exception as e:
        console.print(f"[bold red]Stress test failed:[/bold red] {e}")
        return 1
    
    return _display_stress_test_report(report)


def run_simulated_stress_test(
    limit: int = 140,
    verbose: bool = True,
) -> int:
    """
    Run simulated stress test that demonstrates collapse behavior without weight files.
    """
    console.print(Panel(
        "[bold cyan]M3O Simulated Collapse Test[/bold cyan]\n"
        "Demonstrating DEGRADED and CRITICAL collapse behavior\n"
        "with simulated task vector scenarios (no weight files needed).",
        box=box.ROUNDED,
        border_style="cyan",
    ))
    
    try:
        report = run_simulated_collapse_test(
            num_layers=limit,
            verbose=verbose,
        )
    except Exception as e:
        console.print(f"[bold red]Simulated test failed:[/bold red] {e}")
        return 1
    
    return _display_stress_test_report(report)


def _display_stress_test_report(report) -> int:
    """Display stress test report in a formatted table."""
    # Display results table
    results_table = Table(
        title="[bold]Alpha Scan Results[/bold]",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    results_table.add_column("α", justify="right", style="white")
    results_table.add_column("Status", justify="center")
    results_table.add_column("Mean Cos Sim", justify="right")
    results_table.add_column("Min Cos Sim", justify="right")
    results_table.add_column("Mean Frob Δ", justify="right")
    results_table.add_column("Breakdown", justify="center")
    results_table.add_column("Throughput", justify="right")
    
    for r in report.results:
        status_str = format_status_markup(r.status)
        breakdown = f"[green]{r.stable_count}[/green]/[yellow]{r.degraded_count}[/yellow]/[red]{r.critical_count}[/red]"
        results_table.add_row(
            f"{r.alpha:+.2f}",
            status_str,
            f"{r.mean_cos_sim:.6f}",
            f"{r.min_cos_sim:.6f}",
            f"{r.mean_frob_drift:.6f}",
            breakdown,
            f"{r.throughput:.1f} lyr/s",
        )
    
    console.print()
    console.print(results_table)
    
    # Summary panel
    summary = Table(show_header=False, box=box.SIMPLE)
    summary.add_column("Finding", style="dim cyan")
    summary.add_column("Value", style="bold white")
    
    if report.collapse_threshold_degraded:
        summary.add_row("First DEGRADED at α", f"{report.collapse_threshold_degraded:.2f}")
    else:
        summary.add_row("DEGRADED threshold", "Not reached in test range")
    
    if report.collapse_threshold_critical:
        summary.add_row("First CRITICAL at α", f"{report.collapse_threshold_critical:.2f}")
    else:
        summary.add_row("CRITICAL threshold", "Not reached in test range")
    
    if report.optimal_alpha:
        summary.add_row("Recommended α", f"{report.optimal_alpha:.2f} (best stability)")
    
    if report.performance_baseline:
        summary.add_row("Performance baseline", f"{report.performance_baseline:.1f} layers/sec")
    
    console.print()
    console.print(Panel(summary, title="[bold green]Stress Test Findings[/bold green]", border_style="green"))
    
    return 0


def run_cockpit(
    alpha: float = 0.7,
    mongo: Optional[str] = None,
    base: str = "./weights/gemma4_base.safetensors",
    tuned: str = "./weights/model.safetensors",
    limit: Optional[int] = None,
    output: Optional[str] = None,
    stress_test: bool = False,
    stress_simulate: bool = False,
    gemini_key: Optional[str] = None,
    mock_telemetry: bool = False,
) -> int:
    """
    Execute streaming surgery pipeline with live terminal cockpit rendering.
    """
    # Run simulated collapse test (no weight files needed)
    if stress_simulate:
        return run_simulated_stress_test(limit=limit or 140, verbose=True)
    
    # Run comprehensive stress test on real weights
    if stress_test:
        return run_stress_test(base=base, tuned=tuned, limit=limit, verbose=True)
    
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

    telemetry = TelemetryLogger(mongo_uri=mongo, mock_mode=mock_telemetry)

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

    # Google Gemini / Gemma 4 AI Diagnostician
    from telemetry.gemini_diagnostics import diagnose_model_surgery_with_gemini
    console.print("\n[bold cyan][*][/bold cyan] Invoking Gemma 4 AI Diagnostician ([dim]gemma-4-26b-a4b-it via Gemini API[/dim])...")
    metrics_payload = {
        "alpha": alpha,
        "total_layers": total_layers,
        "mean_cos_sim": mean_cos,
        "min_cos_sim": min_cos,
        "mean_frob_drift": mean_frob,
        "stable_count": stable_count,
        "degraded_count": degraded_count,
        "critical_count": critical_count,
    }
    report_text = diagnose_model_surgery_with_gemini(metrics_payload, api_key=gemini_key)
    console.print(Panel(Markdown(report_text), title="[bold cyan]🧠 Gemma 4 (gemma-4-26b-a4b-it) Mechanistic Assessment[/bold cyan]", border_style="cyan", expand=False))

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
        "--stress-test",
        action="store_true",
        help="Run comprehensive stress test across multiple alpha values to find collapse boundaries",
    )
    parser.add_argument(
        "--stress-simulate",
        action="store_true",
        help="Run simulated collapse test (no weight files needed) to demonstrate DEGRADED/CRITICAL behavior",
    )
    parser.add_argument(
        "--gemini-key",
        type=str,
        default=None,
        help="Google Gemini API key for gemma-4-31b-it diagnostics (or set in .env)",
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
    parser.add_argument(
        "--mock",
        action="store_true",
        dest="mock_telemetry",
        help="Force mock telemetry mode (no database connections)",
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
            stress_test=args.stress_test,
            stress_simulate=args.stress_simulate,
            gemini_key=args.gemini_key,
            mock_telemetry=args.mock_telemetry,
        )
    )


if __name__ == "__main__":
    main()
