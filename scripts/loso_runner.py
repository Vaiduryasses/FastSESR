#!/usr/bin/env python3
"""
LOSO orchestration script for LOON-UNet training + reconstruction.

Usage: python scripts/loso_runner.py --dataset FAUST --data_root /root/autodl-tmp/Data --epochs 30

For each subject (file or group) this script will:
 - create a train_list (all subjects except held-out) and val_list (either next group for grouped datasets like MGN, or a ratio-based subset of train for others)
 - run S2/S2_train_loon_unet.py with --train_list and --save_dir per-fold
 - after training finishes, find loon_unet_best.pth in save_dir and run S2_reconstruct.py on the held-out file

This keeps runs organized under runs/loso/<dataset>/<subject>.

MGN grouping: by default group id is filename.split('_')[0], can be customized with --mgn_group_regex.
"""
from __future__ import annotations
import argparse, os, subprocess, time, sys, re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def list_subjects(data_root: str, dataset: str, mgn_group_regex: str | None = None):
    pc_dir = Path(data_root) / 'PointClouds' / dataset
    files = sorted([p for p in pc_dir.glob('*.ply')])
    basenames = [p.name for p in files]
    if dataset.lower() == 'mgn' or mgn_group_regex:
        # group by garment id using regex or by prefix before first underscore
        groups = {}
        pattern = re.compile(mgn_group_regex) if mgn_group_regex else None
        for name in basenames:
            key = None
            if pattern:
                m = pattern.search(name)
                if m:
                    key = m.group(1)
            if key is None:
                key = name.split('_')[0]
            groups.setdefault(key, []).append(name)
        return list(groups.items())  # list of (group_id, [filenames])
    else:
        # each file is its own subject
        return [(os.path.splitext(b)[0], [b]) for b in basenames]


def run_cmd(cmd, cwd=REPO_ROOT, env=None):
    print('[Run]', ' '.join(cmd))
    p = subprocess.Popen(cmd, cwd=str(cwd), env=env)
    rc = p.wait()
    return rc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_root', type=str, default=str(./ 'Data'))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--train_script', type=str, default='S2/S2_train_loon_unet.py')
    parser.add_argument('--recon_script', type=str, default='S2_reconstruct.py')
    parser.add_argument('--chunk_size', type=int, default=5000)
    parser.add_argument('--mgn_group_regex', type=str, default='')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='For non-grouped datasets, fraction of train to use as validation')
    parser.add_argument('--val_seed', type=int, default=202405, help='Deterministic seed for picking validation subset when not using grouped-next validation')
    parser.add_argument('--extra_train_args', type=str, default='',
                        help='Extra args appended to train command')
    parser.add_argument('--extra_recon_args', type=str, default='',
                        help='Extra args appended to recon command')
    args = parser.parse_args()

    subjects = list_subjects(args.data_root, args.dataset, args.mgn_group_regex or None)
    N = len(subjects)
    print(f'[Info] Found {N} subjects/groups for dataset {args.dataset}')

    base_out = REPO_ROOT / 'runs' / 'loso' / args.dataset
    base_out.mkdir(parents=True, exist_ok=True)

    for i, (subj_id, files) in enumerate(subjects):
        print(f'\n=== LOSO fold {i+1}/{N}: holdout {subj_id} ({len(files)} files) ===')
        # remove trailing space in folder name
        fold_dir = base_out / f'{subj_id}_{time.strftime("%Y%m%d_%H%M%S")}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        # create train/val lists
        all_files = [p.name for p in sorted((Path(args.data_root) / 'PointClouds' / args.dataset).glob('*.ply'))]
        # grouped mode if dataset is MGN or regex provided
        grouped_mode = (args.dataset.lower() == 'mgn') or bool(args.mgn_group_regex)
        if grouped_mode and N >= 2:
            # Use LOSO-style: validation is the next group (cyclic)
            val_group_idx = (i + 1) % N
            val_files = subjects[val_group_idx][1]
            train_files = [f for f in all_files if (f not in files and f not in val_files)]
        else:
            # Non-grouped: pick a deterministic subset of remaining files as validation
            train_pool = [f for f in all_files if f not in files]
            # deterministic shuffle with seed
            import random
            rng = random.Random(int(args.val_seed))
            pool_sorted = sorted(train_pool)  # stable order pre-shuffle
            rng.shuffle(pool_sorted)
            k = max(1, int(round(len(pool_sorted) * float(args.val_ratio)))) if len(pool_sorted) > 0 else 0
            val_files = pool_sorted[:k]
            train_files = pool_sorted[k:]
        train_list_path = fold_dir / 'train_list.txt'
        with open(train_list_path, 'w') as f:
            for t in train_files:
                f.write(t + '\n')
        print(f'[Info] train_list with {len(train_files)} samples written to {train_list_path}')
        val_list_path = fold_dir / 'val_list.txt'
        with open(val_list_path, 'w') as f:
            for v in val_files:
                f.write(v + '\n')
        print(f'[Info] val_list with {len(val_files)} samples written to {val_list_path}')

        # also write test_list for the held-out files BEFORE training, so the train script can report test_count
        test_list_path = fold_dir / 'test_list.txt'
        with open(test_list_path, 'w') as f:
            for t in files:
                f.write(t + '\n')
        print(f'[Info] test_list with {len(files)} samples written to {test_list_path}')

        # run training
        save_dir = fold_dir / 'save'
        cmd_train = [
            PY, str(REPO_ROOT / args.train_script),
            '--dataset', args.dataset,
            '--data_root', args.data_root,
            '--train_list', str(train_list_path),
            '--val_list', str(val_list_path),
            '--test_list', str(test_list_path),  # ensure test_count is visible in dataset summary
            '--epochs', str(args.epochs),
            '--gpu', str(args.gpu),
            '--save_dir', str(save_dir)
        ]
        if args.extra_train_args:
            cmd_train += args.extra_train_args.split()
        rc = run_cmd(cmd_train)
        if rc != 0:
            print(f'[Error] Training failed for fold {subj_id} (rc={rc}); continuing to next fold')
            continue

        # find best ckpt (be tolerant on filename variations)
        candidates = [
            'loon_unet_best.pth',
            'model_best.pth',
            'last.pth',
            'model_last.pth',
        ]
        best_ckpt = None
        for name in candidates:
            cand = save_dir / name
            if cand.is_file():
                best_ckpt = cand
                break
        if best_ckpt is None:
            print(f'[Warn] No checkpoint found in {save_dir}; skipping recon for {subj_id}')
            continue

        # run reconstruction using best ckpt, target only heldout files
        cmd_recon = [
            PY, str(REPO_ROOT / args.recon_script),
            '--use_loon_unet', '--loon_unet_ckpt', str(best_ckpt),
            '--dataset', args.dataset, '--gpu', str(args.gpu),
            '--data_root', args.data_root,
            '--test_list', str(test_list_path),
            '--chunk_size', str(args.chunk_size)
        ]
        if args.extra_recon_args:
            cmd_recon += args.extra_recon_args.split()
        rc2 = run_cmd(cmd_recon)
        if rc2 != 0:
            print(f'[Error] Reconstruction failed for fold {subj_id} (rc={rc2}); continuing to next fold')
            continue

    print('\n[Done] LOSO runs completed')

if __name__ == '__main__':
    main()
