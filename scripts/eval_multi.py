#!/usr/bin/env python3
"""
Batch evaluation helper.
Runs `main_eval_acc.py` over multiple prediction directories (e.g. A/B/C folds or runs),
parses the reported average metrics tuple, then computes the overall mean across runs.

Example:
    python scripts/eval_multi.py \
        --gt_path /root/autodl-tmp/Data/GT_Meshes/FAUST \
        --pred_paths results/FAUST_runA results/FAUST_runB results/FAUST_runC

Prerequisites:
    Each pred_path must contain .ply files whose basenames match those under gt_path.
    `main_eval_acc.py` expects arguments: --gt_path, --pred_path, [--sample_num].
Output:
    Per-run metrics and global mean printed to stdout. Optionally writes a CSV if --csv_out is given.
"""
import argparse
import subprocess
import re
import statistics
import sys
from pathlib import Path
from typing import List, Tuple
import json

PAREN_RE = re.compile(r"\(([-+0-9eE\.,\s]+)\)")

METRIC_NAMES = ["CD1","CD2","F1","NC","NR","ECD1","EF1"]


def run_single(main_eval_py: Path, gt_path: Path, pred_path: Path, sample_num: int) -> Tuple[str, List[float]]:
    """Invoke main_eval_acc.py for one prediction directory and parse the average metrics line.
    Returns (run_name, metrics_list). Raises RuntimeError if parsing fails.
    """
    if not pred_path.exists():
        raise FileNotFoundError(f"Pred path not found: {pred_path}")
    cmd = [sys.executable, str(main_eval_py), "--gt_path", str(gt_path), "--pred_path", str(pred_path), "--sample_num", str(sample_num)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Eval failed for {pred_path} (code {proc.returncode}):\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}")
    stdout = proc.stdout
    # Find the tuple line
    match = PAREN_RE.search(stdout)
    if not match:
        raise RuntimeError(f"Could not parse metrics from output for {pred_path}. Output:\n{stdout}")
    raw = match.group(1)
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    if len(parts) != len(METRIC_NAMES):
        raise RuntimeError(f"Expected {len(METRIC_NAMES)} metrics, got {len(parts)} in: {raw}")
    try:
        metrics = [float(x) for x in parts]
    except ValueError as e:
        raise RuntimeError(f"Float conversion error in metrics '{raw}': {e}")
    run_name = pred_path.name
    return run_name, metrics


def format_row(name: str, metrics: List[float]) -> str:
    return name + ": " + ", ".join(f"{m:.4f}" for m in metrics)


def main():
    ap = argparse.ArgumentParser(description="Batch evaluate multiple prediction directories and average metrics.")
    ap.add_argument('--gt_path', required=True, type=Path, help='Ground truth mesh directory')
    ap.add_argument('--pred_paths', required=True, nargs='+', type=Path, help='One or more prediction directories')
    ap.add_argument('--sample_num', type=int, default=100000, help='Point sample count forwarded to main_eval_acc.py')
    ap.add_argument('--main_eval', type=Path, default=Path('main_eval_acc.py'), help='Path to main_eval_acc.py')
    ap.add_argument('--csv_out', type=Path, help='Optional path to write per-run + mean metrics as CSV')
    ap.add_argument('--json_out', type=Path, help='Optional path to write metrics JSON')
    args = ap.parse_args()

    main_eval_py = args.main_eval
    if not main_eval_py.exists():
        ap.error(f"main_eval_acc.py not found at {main_eval_py}")

    all_metrics = []  # list of (name, metrics_list)
    for pred in args.pred_paths:
        try:
            name, metrics = run_single(main_eval_py, args.gt_path, pred, args.sample_num)
            all_metrics.append((name, metrics))
        except Exception as e:
            print(f"[ERROR] Skipping {pred}: {e}", file=sys.stderr)

    if not all_metrics:
        print("No successful evaluations.")
        sys.exit(1)

    # Compute mean across runs element-wise.
    # Use statistics.fmean for clarity; if any metric is NaN, propagate.
    transposed = list(zip(*(m for _, m in all_metrics)))  # list of tuples per metric across runs
    mean_metrics = [statistics.fmean(metric_group) for metric_group in transposed]

    print("\nPer-run metrics (" + "/".join(METRIC_NAMES) + "):")
    for name, metrics in all_metrics:
        print("  " + format_row(name, metrics))
    print("\nMean metrics:")
    print("  " + format_row("MEAN", mean_metrics))

    # Optional CSV
    if args.csv_out:
        try:
            import csv
            with args.csv_out.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["run"] + METRIC_NAMES)
                for name, metrics in all_metrics:
                    writer.writerow([name] + metrics)
                writer.writerow(["MEAN"] + mean_metrics)
            print(f"CSV written: {args.csv_out}")
        except Exception as e:
            print(f"[WARN] Failed to write CSV: {e}")

    if args.json_out:
        try:
            payload = {
                'runs': [{ 'name': name, 'metrics': dict(zip(METRIC_NAMES, metrics)) } for name, metrics in all_metrics],
                'mean': dict(zip(METRIC_NAMES, mean_metrics))
            }
            with args.json_out.open('w') as f:
                json.dump(payload, f, indent=2)
            print(f"JSON written: {args.json_out}")
        except Exception as e:
            print(f"[WARN] Failed to write JSON: {e}")


if __name__ == '__main__':
    main()
