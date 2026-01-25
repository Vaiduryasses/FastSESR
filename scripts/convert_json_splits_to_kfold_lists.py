#!/usr/bin/env python3
"""Convert split JSONs to fold directories with train_list.txt, val_list.txt, test_list.txt

Usage: python3 scripts/convert_json_splits_to_kfold_lists.py --dataset FAUST
Writes into splits/<dataset>/fold_<SplitName>_fold_<i>/
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, type=str)
args = parser.parse_args()

root = Path('splits') / args.dataset
if not root.exists():
    raise SystemExit(f'splits for dataset not found: {root}')

# find all jsons under splits/<dataset>/**/fold_*.json
jsons = sorted(root.rglob('fold_*.json'))
if not jsons:
    print('No fold json files found under', root)
    raise SystemExit(1)

for p in jsons:
    payload = json.loads(p.read_text(encoding='utf8'))
    train = payload.get('train', []) or []
    val = payload.get('val', []) or []
    test = payload.get('test', []) or []
    # target folder: splits/<dataset>/fold_{splitname}_{foldname}
    split_name = p.parent.name
    folder = root / f'fold_{split_name}_{p.stem}'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'train_list.txt').write_text('\n'.join(train) + ('\n' if train else ''), encoding='utf8')
    (folder / 'val_list.txt').write_text('\n'.join(val) + ('\n' if val else ''), encoding='utf8')
    (folder / 'test_list.txt').write_text('\n'.join(test) + ('\n' if test else ''), encoding='utf8')
    print('wrote', folder)

print('Done.')
