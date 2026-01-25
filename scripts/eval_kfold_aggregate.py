#!/usr/bin/env python3
"""
Aggregate evaluation across all folds by running main_eval_acc.py per-fold and averaging metrics.

Assumptions:
- Each fold's predictions are stored under results/<dataset>/<fold_tag>/
- Ground-truth meshes are under Data/GT_Meshes/<dataset>/
- main_eval_acc.py prints metrics to stdout; we will parse the printed lines from MeshEvaluator.

Usage:
  python scripts/eval_kfold_aggregate.py --dataset FAUST \
      --gt_root /root/autodl-tmp/Data/GT_Meshes \
      --results_root results/FAUST \
      --sample_num 100000

It will print per-fold metrics and a final mean/std summary. It also writes a CSV.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from statistics import mean, pstdev

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = os.environ.get('PYTHON', None) or os.sys.executable

# Regex patterns to parse typical MeshEvaluator prints.
# Adjust here if your evaluator prints different keys.
# Example expected lines:
#   Accuracy: 0.001234
#   Completeness: 0.002345
#   Chamfer-L2: 0.003456
#   F-Score@1e-4: 0.7890
PATTERNS = {
    'Accuracy': re.compile(r"^\s*Accuracy\s*[:=]\s*([0-9.eE+-]+)\s*$"),
    'Completeness': re.compile(r"^\s*Completeness\s*[:=]\s*([0-9.eE+-]+)\s*$"),
    'Chamfer-L2': re.compile(r"^\s*Chamfer[- ]L2\s*[:=]\s*([0-9.eE+-]+)\s*$"),
    'F-Score@1e-4': re.compile(r"^\s*F-Score@1e-4\s*[:=]\s*([0-9.eE+-]+)\s*$"),
}


def run_eval_fold(gt_dir: Path, pred_dir: Path, sample_num: int) -> dict:
    cmd = [PY, str(REPO_ROOT / 'main_eval_acc.py'), '--gt_path', str(gt_dir), '--pred_path', str(pred_dir), '--sample_num', str(sample_num)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(REPO_ROOT))
    out_lines = []
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end='')  # passthrough for visibility
        out_lines.append(line.rstrip())
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f'Evaluation failed for {pred_dir} (rc={rc}).')

    metrics: dict[str, float] = {}
    # Try JSON line first if evaluator prints it; otherwise pattern match
    for l in out_lines:
        if l.strip().startswith('{') and l.strip().endswith('}'):
            try:
                obj = json.loads(l.strip())
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if isinstance(v, (int, float)):
                            metrics[k] = float(v)
            except Exception:
                pass
    # Fallback: regex parse common lines
    for l in out_lines:
        for key, pat in PATTERNS.items():
            m = pat.match(l)
            if m:
                try:
                    metrics[key] = float(m.group(1))
                except Exception:
                    pass
    return metrics


def aggregate_metrics(per_fold: list[dict]) -> dict:
    # union of keys
    keys = set()
    for d in per_fold:
        keys.update(d.keys())
    out = {}
    for k in sorted(keys):
        vals = [d[k] for d in per_fold if k in d]
        if not vals:
            continue
        out[f'{k}_mean'] = float(mean(vals))
        out[f'{k}_std'] = float(pstdev(vals)) if len(vals) > 1 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, type=str)
    ap.add_argument('--gt_root', required=False, type=str, default=str(REPO_ROOT / 'Data' / 'GT_Meshes'))
    ap.add_argument('--results_root', required=False, type=str, help='Root folder containing per-fold results/<dataset>/*', default=None)
    ap.add_argument('--sample_num', type=int, default=100000)
    args = ap.parse_args()

    results_root = Path(args.results_root) if args.results_root else (REPO_ROOT / 'results' / args.dataset)
    gt_dir = Path(args.gt_root) / args.dataset

    if not gt_dir.is_dir():
        raise FileNotFoundError(f'GT dir not found: {gt_dir}')
    if not results_root.is_dir():
        raise FileNotFoundError(f'Results dir not found: {results_root}')

    # Discover fold subdirs (any non-empty directory)
    fold_dirs = [p for p in results_root.iterdir() if p.is_dir()]
    fold_dirs.sort()
    if not fold_dirs:
        raise RuntimeError(f'No fold directories found under {results_root}')

    per_fold_metrics = []
    for fd in fold_dirs:
        # skip empty dirs
        has_mesh = any((fd.glob('*.ply')))
        if not has_mesh:
            # Also allow subfolder structure; if meshes are nested, user can override results_root
            print(f'[Warn] No meshes found in {fd}, skipping')
            continue
        print(f'\n=== Evaluating fold: {fd.name} ===')
        m = run_eval_fold(gt_dir, fd, args.sample_num)
        print(f'[Fold {fd.name}] metrics: {m}')
        if m:
            per_fold_metrics.append(m)

    if not per_fold_metrics:
        raise RuntimeError('No per-fold metrics parsed. Ensure main_eval_acc.py prints metrics recognizably.')

    agg = aggregate_metrics(per_fold_metrics)
    print('\n=== Aggregated metrics across folds ===')
    for k in sorted(agg.keys()):
        print(f'{k}: {agg[k]:.6f}')

    # Save CSV for convenience
    csv_path = results_root / 'kfold_metrics.csv'
    with csv_path.open('w') as f:
        # header
        # Collect all unique keys for folds
        fold_keys = sorted({kk for d in per_fold_metrics for kk in d.keys()})
        f.write('fold,' + ','.join(fold_keys) + '\n')
        for fd, md in zip([d.name for d in fold_dirs if any((d.glob("*.ply")))] , per_fold_metrics):
            row = [fd] + [str(md.get(k, '')) for k in fold_keys]
            f.write(','.join(row) + '\n')
        # Aggregates
        f.write('mean/std,' + ','.join([f"{agg.get(k+'_mean','')}/{agg.get(k+'_std','')}" for k in fold_keys]) + '\n')
    print(f"[Info] Wrote aggregated CSV to {csv_path}")

if __name__ == '__main__':
    main()
