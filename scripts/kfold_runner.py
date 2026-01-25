#!/usr/bin/env python3
"""
K-fold / LOSO-style training + reconstruction orchestrator.

Uses pre-generated splits from scripts/generate_fixed_splits.py
Expected directory structure:
  splits/<dataset>/fold_i/train_list.txt
  splits/<dataset>/fold_i/test_list.txt

Workflow per fold i:
  1. Train model on train_list (epochs configurable)
  2. Reconstruct test_list with best checkpoint

Usage example:
  python scripts/kfold_runner.py --dataset FAUST --data_root ./Data --epochs 30 --gpu 0
  (Ensure you ran generate_fixed_splits.py with --kfolds beforehand.)
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

def run(cmd, cwd=REPO_ROOT):
    print('[Run]', ' '.join(cmd))
    p = subprocess.Popen(cmd, cwd=str(cwd))
    rc = p.wait()
    return rc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, type=str)
    ap.add_argument('--data_root', required=True, type=str, help='Root containing PointClouds/<dataset>')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--splits_root', type=str, default='splits')
    ap.add_argument('--train_script', type=str, default='S2/S2_train_loon_unet.py')
    ap.add_argument('--recon_script', type=str, default='S2_reconstruct.py')
    ap.add_argument('--chunk_size', type=int, default=2000)
    ap.add_argument('--extra_train_args', type=str, default='')
    ap.add_argument('--extra_recon_args', type=str, default='')
    ap.add_argument('--use_loon_unet', action='store_true', help='Use LOON-UNet for reconstruction stage')
    ap.add_argument('--record_time', action='store_true', help='Record per-file reconstruction time and report average at end')
    # optional: limit number of folds for quick smoke tests
    ap.add_argument('--limit_folds', type=int, default=0, help='If >0, only process the first N folds')
    ap.add_argument('--resume', action='store_true', help='Skip folds that have already been completed (contain model_best.pth or similar)')
    args = ap.parse_args()

    ds_dir = Path(args.splits_root) / args.dataset
    fold_dirs = sorted([p for p in ds_dir.glob('fold_*') if p.is_dir()])
    if not fold_dirs:
        print(f'[Error] No folds found under {ds_dir}. Run generate_fixed_splits.py with --kfolds first.')
        sys.exit(1)
    print(f'[Info] Found {len(fold_dirs)} folds for dataset {args.dataset}')

    out_root = REPO_ROOT / 'runs' / 'kfold' / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    per_file_times = []
    total_files = 0
    max_folds = len(fold_dirs) if args.limit_folds <= 0 else min(args.limit_folds, len(fold_dirs))
    for fi, fold_dir in enumerate(fold_dirs[:max_folds]):
        train_list = fold_dir / 'train_list.txt'
        test_list = fold_dir / 'test_list.txt'
        val_list = fold_dir / 'val_list.txt'
        # Require all three lists
        if not train_list.is_file() or not test_list.is_file() or not val_list.is_file():
            print(f'[Warn] Missing train_list, val_list, or test_list in {fold_dir}; skipping fold')
            print(f'[Debug] train: {train_list.is_file()}, val: {val_list.is_file()}, test: {test_list.is_file()}')
            continue

        resume_ckpt = None
        if args.resume:
            # Check if this fold is already done or partially done
            # Look for any directory starting with fold{fi}_ in out_root
            candidates_dirs = sorted(list(out_root.glob(f'fold{fi}_*')), key=lambda p: p.stat().st_mtime, reverse=True)
            
            fold_done = False
            done_dir = None
            
            for d in candidates_dirs:
                if not d.is_dir(): continue
                # Check for completion markers (best model)
                if (d / 'model_best.pth').is_file() or (d / 'loon_unet_best.pth').is_file():
                    fold_done = True
                    done_dir = d
                    break
                # Check for partial progress (last model)
                if (d / 'model_last.pth').is_file():
                    resume_ckpt = d / 'model_last.pth'
                    done_dir = d
                    break
                if (d / 'last.pth').is_file():
                    resume_ckpt = d / 'last.pth'
                    done_dir = d
                    break

            if fold_done:
                print(f'[Info] Fold {fi} already completed (found best checkpoint in {done_dir.name}). Skipping.')
                continue
            
            if resume_ckpt:
                print(f'[Info] Resuming fold {fi} from checkpoint {resume_ckpt} in {done_dir.name}')
                save_dir = done_dir
                # Don't create new directory
            else:
                # Start fresh
                fold_tag = f'fold{fi}_{time.strftime("%Y%m%d_%H%M%S")}'
                save_dir = out_root / fold_tag
                save_dir.mkdir(parents=True, exist_ok=True)
        else:
            fold_tag = f'fold{fi}_{time.strftime("%Y%m%d_%H%M%S")}'
            save_dir = out_root / fold_tag
            save_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n=== Fold {fi}/{len(fold_dirs)-1}: {fold_dir.name} ===')
        # Train: pass train/val/test lists for summary completeness and validation selection
        cmd_train = [
            PY, str(REPO_ROOT / args.train_script),
            '--dataset', args.dataset,
            '--data_root', args.data_root,
            '--train_list', str(train_list),
            '--val_list', str(val_list),
            '--test_list', str(test_list),
            '--epochs', str(args.epochs),
            '--gpu', str(args.gpu),
            '--save_dir', str(save_dir),
        ]
        if resume_ckpt:
            cmd_train += ['--resume', str(resume_ckpt)]
        
        if args.extra_train_args:
            cmd_train += args.extra_train_args.split()
        rc = run(cmd_train)
        if rc != 0:
            print(f'[Error] Training failed rc={rc}; continue to next fold')
            continue
        # Best ckpt fallback: support multiple common filenames
        candidates = [
            'loon_unet_best.pth',  # expected by older pipelines
            'model_best.pth',      # saved by current trainer
            'last.pth',            # legacy last checkpoint name
            'model_last.pth',      # saved by current trainer
        ]
        best_ckpt = None
        for name in candidates:
            cand = save_dir / name
            if cand.is_file():
                best_ckpt = cand
                break
        if best_ckpt is None:
            print(f'[Warn] No checkpoint found in {save_dir}; skipping reconstruction')
            continue
        # Reconstruction: place results under split-A/B/C instead of per-fold
        split_name = None
        fname = fold_dir.name
        if fname.startswith('fold_') and '_fold_' in fname:
            try:
                # e.g., 'fold_Split-A_fold_0' -> tokens ['fold', 'Split-A', 'fold', '0']
                split_name = fname.split('_')[1]
            except Exception:
                split_name = None
        if split_name is None:
            # Fallback: inspect test_list entries to infer Split- token
            try:
                with open(test_list, 'r') as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        for token in ('Split-A', 'Split-B', 'Split-C', 'split-A', 'split-B', 'split-C'):
                            if token in s:
                                split_name = token.replace('split-', 'Split-')
                                break
                        if split_name:
                            break
            except Exception:
                split_name = None
        if not split_name:
            split_name = 'UnknownSplit'
        recon_out = str(Path('results') / args.dataset / split_name)
        cmd_recon = [PY, str(REPO_ROOT / args.recon_script), '--dataset', args.dataset, '--data_root', args.data_root, '--test_list', str(test_list), '--gpu', str(args.gpu), '--chunk_size', str(args.chunk_size), '--out_dir', recon_out]
        use_loon_unet = bool(args.use_loon_unet) or str(args.train_script).endswith('S2/S2_train_loon_unet.py') or str(args.train_script).endswith('S2_train_loon_unet.py')
        if use_loon_unet:
            # Pass both unified and legacy flag for maximum compatibility
            cmd_recon += ['--use_loon_unet', '--loon_ckpt', str(best_ckpt), '--loon_unet_ckpt', str(best_ckpt)]
        if args.extra_recon_args:
            cmd_recon += args.extra_recon_args.split()
        # If recording time, inject a simple environment flag so reconstruct script can append timings naturally (already writes timings.txt)
        before = time.time()
        rc2 = run(cmd_recon)
        duration = time.time() - before
        # Attempt to parse timings.txt inside per-fold out_dir if exists
        if args.record_time:
            # test_list entries correspond to result mesh names
            try:
                timings_path = os.path.join(recon_out, 'timings.txt')
                with open(timings_path, 'r') as f:
                    lines = [l.strip() for l in f if l.strip()]
                # First line avg, rest per-file
                for l in lines[1:]:
                    parts = l.split(',')
                    if len(parts) == 2:
                        per_file_times.append(float(parts[1]))
                total_files += len(lines) - 1
            except Exception:
                # Fall back to coarse fold duration spread across files
                try:
                    with open(test_list, 'r') as f:
                        n_test_files = sum(1 for _ in f if _.strip())
                    if n_test_files > 0:
                        per_file_times.extend([duration / n_test_files] * n_test_files)
                        total_files += n_test_files
                except Exception:
                    pass
        if rc2 != 0:
            print(f'[Error] Reconstruction failed rc={rc2}; continuing')
            continue
    if args.record_time and per_file_times:
        avg_time = sum(per_file_times) / max(1, len(per_file_times))
        print(f"\n[Info] Average reconstruction time per file across all folds: {avg_time:.4f}s (files={len(per_file_times)})")
    print('\n[Done] All folds processed.')

if __name__ == '__main__':
    main()
