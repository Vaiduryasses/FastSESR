#!/usr/bin/env python3
"""
Generate deterministic cross-validation splits per the LOSO / stratified protocol.

Output structure (per dataset):
  splits/<dataset>/Split-A/fold_0.json  # contains {train, val, test, metadata}
  splits/<dataset>/Split-B/fold_0.json
  splits/<dataset>/Split-C/fold_0.json

MGN:
  - Group samples by garment id (regex or prefix).
  - For each split (A/B/C) use a different seed to permute groups.
  - Fold i: test = group π[i]; val = group π[(i+1) mod G]; train = rest.

Other datasets (excluding ABC & MGN):
  - For each split (A/B/C) shuffle samples with a seed, partition into 10 buckets (~equal size).
  - Fold f: test = bucket f (~10%), val = buckets (f+1) & (f+2) (~20%), train = remaining 7 buckets (~70%).

Each JSON stores the seed, split name, fold index, and bucket/group assignments to guarantee reproducibility.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_SPLIT_NAMES = ("Split-A", "Split-B", "Split-C")
DEFAULT_SEEDS = (202401, 202402, 202403)


def list_files(data_root: Path, dataset: str) -> List[str]:
    pc_dir = data_root / 'PointClouds' / dataset
    if not pc_dir.is_dir():
        raise FileNotFoundError(f'PointCloud directory not found: {pc_dir}')
    files = sorted([p.name for p in pc_dir.glob('*.ply')])
    if not files:
        raise RuntimeError(f'No .ply files found for dataset {dataset} in {pc_dir}')
    return files


def group_mgn(files: Sequence[str], regex: str | None) -> List[Tuple[str, List[str]]]:
    pattern = re.compile(regex) if regex else None
    groups: Dict[str, List[str]] = {}
    for f in files:
        if pattern:
            match = pattern.search(f)
            key = match.group(1) if match else f.split('_')[0]
        else:
            key = f.split('_')[0]
        groups.setdefault(key, []).append(f)
    grouped = [(g, sorted(members)) for g, members in sorted(groups.items(), key=lambda kv: kv[0])]
    return grouped


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def chunk_evenly(items: List[str], k: int) -> List[List[str]]:
    n = len(items)
    base = n // k
    rem = n % k
    buckets: List[List[str]] = []
    idx = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        buckets.append(items[idx: idx + size])
        idx += size
    return buckets


def generate_mgn_splits(files: List[str], out_root: Path, seeds: Sequence[int], split_names: Sequence[str], regex: str | None):
    groups = group_mgn(files, regex)
    if not groups:
        raise RuntimeError('No garment groups derived for MGN; check regex or file naming.')
    G = len(groups)
    print(f"  [Info] MGN groups: {G}")
    for split_name, seed in zip(split_names, seeds):
        rng = random.Random(seed)
        perm = groups.copy()
        rng.shuffle(perm)
        split_dir = out_root / split_name
        for fold_idx in range(G):
            test_gid, test_files = perm[fold_idx]
            val_gid, val_files = perm[(fold_idx + 1) % G]
            train_groups = [item for j, item in enumerate(perm) if j not in {fold_idx, (fold_idx + 1) % G}]
            train_files = [f for _, lst in train_groups for f in lst]
            payload = {
                'dataset': 'MGN',
                'split': split_name,
                'fold': fold_idx,
                'seed': seed,
                'protocol': 'garment-group-losov',
                'groups': {
                    'test': test_gid,
                    'val': val_gid,
                    'train': [gid for gid, _ in train_groups],
                    'permutation': [gid for gid, _ in perm],
                },
                'train': train_files,
                'val': list(val_files),
                'test': list(test_files),
            }
            write_json(split_dir / f'fold_{fold_idx}.json', payload)
        print(f"  [Split {split_name}] seed={seed} -> {split_dir}")


def generate_bucket_splits(dataset: str, files: List[str], out_root: Path, seeds: Sequence[int], split_names: Sequence[str]):
    if len(files) < 10:
        raise RuntimeError(f"Dataset {dataset} has {len(files)} samples; need >=10 for 10-way bucket split.")
    for split_name, seed in zip(split_names, seeds):
        rng = random.Random(seed)
        shuffled = files.copy()
        rng.shuffle(shuffled)
        buckets = chunk_evenly(shuffled, 10)
        split_dir = out_root / split_name
        bucket_meta = {f'B{i}': buckets[i] for i in range(10)}
        write_json(split_dir / 'buckets.json', {
            'dataset': dataset,
            'split': split_name,
            'seed': seed,
            'buckets': bucket_meta,
        })
        for fold_idx in range(10):
            test_bucket = fold_idx
            val_bucket1 = (fold_idx + 1) % 10
            val_bucket2 = (fold_idx + 2) % 10
            test_files = buckets[test_bucket]
            val_files = buckets[val_bucket1] + buckets[val_bucket2]
            train_files = [f for i, bucket in enumerate(buckets) if i not in {test_bucket, val_bucket1, val_bucket2} for f in bucket]
            payload = {
                'dataset': dataset,
                'split': split_name,
                'fold': fold_idx,
                'seed': seed,
                'protocol': 'bucket-70-20-10',
                'buckets': {
                    'test': test_bucket,
                    'val': [val_bucket1, val_bucket2],
                    'train': [i for i in range(10) if i not in {test_bucket, val_bucket1, val_bucket2}],
                },
                'train': train_files,
                'val': val_files,
                'test': test_files,
            }
            write_json(split_dir / f'fold_{fold_idx}.json', payload)
        print(f"  [Split {split_name}] seed={seed} -> {split_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True, help='Root data directory containing PointClouds/<dataset>')
    ap.add_argument('--datasets', type=str, nargs='+', required=True, help='Datasets to process')
    ap.add_argument('--split_names', type=str, nargs='*', default=list(DEFAULT_SPLIT_NAMES), help='Names for each split (default Split-A/B/C)')
    ap.add_argument('--seeds', type=int, nargs='*', default=list(DEFAULT_SEEDS), help='Seeds for each split (default 202401/202402/202403)')
    ap.add_argument('--mgn_group_regex', type=str, default='', help='Regex with capture group for MGN garment grouping (e.g. "(^[^_]+)")')
    args = ap.parse_args()

    split_names = args.split_names or list(DEFAULT_SPLIT_NAMES)
    seeds = args.seeds or list(DEFAULT_SEEDS)
    if len(split_names) != len(seeds):
        raise ValueError('Number of split names must match number of seeds.')

    data_root = Path(args.data_root)
    out_root = Path('splits')

    for ds in args.datasets:
        files = list_files(data_root, ds)
        ds_out = out_root / ds
        ds_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[Dataset] {ds}: {len(files)} samples")
        if ds.lower() == 'mgn':
            generate_mgn_splits(files, ds_out, seeds, split_names, args.mgn_group_regex or None)
        elif ds.lower() == 'abc':
            print('  [Warn] ABC skipped (dataset excluded from stratified protocol).')
        else:
            generate_bucket_splits(ds, files, ds_out, seeds, split_names)

    print('\n[Done] Split JSON files generated.')


if __name__ == '__main__':
    main()
