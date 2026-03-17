#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import matplotlib.pyplot as plt


METHOD_ORDER_HINT = [
    "naive",
    "flattened",
    "overlap",
    "bucketedoverlap_1MB",
    "bucketedoverlap_10MB",
    "bucketedoverlap_100MB",
]

STAT_RE = re.compile(
    r"^\s*Rank\s*:\s*(?P<rank>\d+)\s*--\s*"
    r"(?P<metric>Mean fw pass|Mean bw pass|Mean sync time|Mean iteration time)"
    r"(?:\s+after warmup of \d+\s+steps)?\s*=\s*"
    r"(?P<value>[0-9.eE+-]+)\s*s\s*$",
    re.MULTILINE,
)

METRIC_MAP = {
    "Mean fw pass": "fw_mean_s",
    "Mean bw pass": "bw_mean_s",
    "Mean sync time": "sync_mean_s",
    "Mean iteration time": "iter_mean_s",
}

METRIC_TITLE = {
    "fw_mean_s": "Forward pass average runtime across ranks",
    "bw_mean_s": "Backward pass average runtime across ranks",
    "sync_mean_s": "Synchronization time average runtime across ranks",
    "iter_mean_s": "Iteration time average runtime across ranks",
}

METRIC_FILENAME = {
    "fw_mean_s": "forward_pass_avg_across_ranks.png",
    "bw_mean_s": "backward_pass_avg_across_ranks.png",
    "sync_mean_s": "sync_time_avg_across_ranks.png",
    "iter_mean_s": "iteration_time_avg_across_ranks.png",
}

AGGREGATED_FILENAME = {
    "fw_mean_s": "aggregated_forward_pass_avg_across_runs.png",
    "bw_mean_s": "aggregated_backward_pass_avg_across_runs.png",
    "sync_mean_s": "aggregated_sync_time_avg_across_runs.png",
    "iter_mean_s": "aggregated_iteration_time_avg_across_runs.png",
}

METRICS = ["fw_mean_s", "bw_mean_s", "sync_mean_s", "iter_mean_s"]


def method_sort_key(method: str):
    if method in METHOD_ORDER_HINT:
        return (0, METHOD_ORDER_HINT.index(method))
    return (1, method)


def parse_out_file(filepath: Path) -> List[Dict[str, Any]]:
    method = filepath.stem
    text = filepath.read_text()

    rows_by_rank: Dict[int, Dict[str, Any]] = {}

    for match in STAT_RE.finditer(text):
        rank = int(match.group("rank"))
        metric = METRIC_MAP[match.group("metric")]
        value = float(match.group("value"))

        if rank not in rows_by_rank:
            rows_by_rank[rank] = {
                "method": method,
                "rank": rank,
            }

        rows_by_rank[rank][metric] = value

    return list(rows_by_rank.values())


def load_run_results(run_dir: Path) -> pd.DataFrame:
    all_rows = []

    out_files = sorted(run_dir.glob("*.out"))
    if not out_files:
        raise RuntimeError(f"No .out files found in {run_dir}")

    for fpath in out_files:
        all_rows.extend(parse_out_file(fpath))

    if not all_rows:
        raise RuntimeError(f"No parsable .out files found in {run_dir}")

    df = pd.DataFrame(all_rows)

    for col in METRICS:
        if col not in df.columns:
            df[col] = pd.NA

    return df.sort_values(["method", "rank"]).reset_index(drop=True)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("method", as_index=False)
        .agg(
            fw_mean_s=("fw_mean_s", "mean"),
            bw_mean_s=("bw_mean_s", "mean"),
            sync_mean_s=("sync_mean_s", "mean"),
            iter_mean_s=("iter_mean_s", "mean"),
        )
    )

    summary = summary.sort_values(
        by="method",
        key=lambda s: s.map(method_sort_key)
    ).reset_index(drop=True)

    return summary


def save_bar_plot(methods, values, title, ylabel, outpath):
    plt.figure(figsize=(11, 5))
    plt.bar(methods, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_aggregated_metric_plot(all_summaries: pd.DataFrame, metric: str, output_root: Path):
    pivot = all_summaries.pivot(index="run", columns="method", values=metric)

    ordered_cols = sorted(list(pivot.columns), key=method_sort_key)
    pivot = pivot[ordered_cols]

    plt.figure(figsize=(11, 6))

    for method in pivot.columns:
        plt.plot(pivot.index, pivot[method], marker="o", label=method)

    plt.title(f"Aggregated across runs: {METRIC_TITLE[metric]}")
    plt.xlabel("Run")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / AGGREGATED_FILENAME[metric], dpi=150)
    plt.close()


def process_run(run_dir: Path, output_root: Path) -> pd.DataFrame:
    print(f"\nProcessing {run_dir} ...")

    df = load_run_results(run_dir)
    summary = build_summary(df)

    run_output_dir = output_root / run_dir.name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        save_bar_plot(
            methods=summary["method"],
            values=summary[metric],
            title=f"{run_dir.name}: {METRIC_TITLE[metric]}",
            ylabel="Time (s)",
            outpath=run_output_dir / METRIC_FILENAME[metric],
        )

    summary = summary.copy()
    summary["run"] = run_dir.name
    return summary


def main():
    ddp_root = Path("ddp_outputs")
    output_root = Path("plots")
    output_root.mkdir(exist_ok=True)

    if not ddp_root.exists():
        raise RuntimeError(f"Directory not found: {ddp_root.resolve()}")

    run_dirs = sorted([p for p in ddp_root.iterdir() if p.is_dir()])

    if not run_dirs:
        raise RuntimeError(f"No run subdirectories found inside {ddp_root.resolve()}")

    all_summaries = []

    for run_dir in run_dirs:
        try:
            run_summary = process_run(run_dir, output_root)
            all_summaries.append(run_summary)
        except Exception as e:
            print(f"[ERROR] Failed processing {run_dir}: {e}")

    if not all_summaries:
        raise RuntimeError("No runs were successfully processed.")

    all_summaries_df = pd.concat(all_summaries, ignore_index=True)

    run_order = [run_dir.name for run_dir in run_dirs]
    all_summaries_df["run"] = pd.Categorical(
        all_summaries_df["run"],
        categories=run_order,
        ordered=True,
    )
    all_summaries_df = all_summaries_df.sort_values(
        ["run", "method"],
        key=lambda col: col.map(method_sort_key) if col.name == "method" else col
    ).reset_index(drop=True)

    for metric in METRICS:
        save_aggregated_metric_plot(all_summaries_df, metric, output_root)

    print(f"\nSaved per-run plots under: {output_root.resolve()}")
    print("Saved aggregated plots:")
    print(f"  {output_root / AGGREGATED_FILENAME['fw_mean_s']}")
    print(f"  {output_root / AGGREGATED_FILENAME['bw_mean_s']}")
    print(f"  {output_root / AGGREGATED_FILENAME['sync_mean_s']}")
    print(f"  {output_root / AGGREGATED_FILENAME['iter_mean_s']}")


if __name__ == "__main__":
    main()
