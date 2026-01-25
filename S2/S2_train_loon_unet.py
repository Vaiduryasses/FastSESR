#!/usr/bin/env python3
"""Train LOON-UNet with Chamfer-distance validation and deterministic splits."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split
from torch.utils import checkpoint as _ckpt

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for environments without tqdm
    def tqdm(iterable=None, total=None, desc=None, leave=True, unit=None):
        return iterable if iterable is not None else range(total or 0)

try:
    from pytorch3d.loss import chamfer_distance as _p3d_chamfer
except Exception:  # pytorch3d is optional at train time
    _p3d_chamfer = None

try:
    from pytorch3d.ops import (
        knn_points as _p3d_knn_points,
        knn_gather as _p3d_knn_gather,
        sample_farthest_points as _p3d_sample_farthest_points,
    )
except Exception:
    _p3d_knn_points, _p3d_knn_gather, _p3d_sample_farthest_points = None, None, None


def _chunk_ranges(total: int, chunk: int):
    chunk = max(1, int(chunk))
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        yield start, end


def _compute_logits_chunked(
    tri_net: torch.nn.Module,
    feat: torch.Tensor,
    off: torch.Tensor,
    chunk: int,
    with_grad: bool = True,
    use_ckpt: bool = True,
) -> torch.Tensor:
    """Chunked forward for triangle net with optional gradient checkpointing.

    Args:
        tri_net: frozen triangle network used for logits.
        feat, off: [Nflat, K, 3].
        chunk: max rows per forward.
        with_grad: whether to preserve gradient graph for logits.
        use_ckpt: when with_grad, wrap each chunk with checkpoint to reduce memory.
    """
    if feat.shape[0] <= chunk:
        if with_grad and use_ckpt:
            return _ckpt.checkpoint(lambda a, b: tri_net(a, b), feat, off, use_reentrant=False)
        return tri_net(feat, off)
    outputs = []
    for start, end in _chunk_ranges(feat.shape[0], chunk):
        f = feat[start:end]
        o = off[start:end]
        if with_grad and use_ckpt:
            out = _ckpt.checkpoint(lambda a, b: tri_net(a, b), f, o, use_reentrant=False)
        else:
            out = tri_net(f, o)
        outputs.append(out)
    return torch.cat(outputs, dim=0)


def _knn_points_chunked(Q_bnc: torch.Tensor, P_bnc: torch.Tensor, K: int, chunk_q: int):
    """Chunked kNN query along the query dimension S to limit peak memory.

    Args:
        Q_bnc: [B, S, 3] queries
        P_bnc: [B, N, 3] database
        K: neighbors
        chunk_q: max queries per knn_points call; if <=0, process all at once
    Returns:
        idx [B, S, K] matching pytorch3d.ops.knn_points(...).idx
    """
    if _p3d_knn_points is None:
        raise ImportError("pytorch3d.ops.knn_points 未安装，无法计算 kNN。")
    B, S, _ = Q_bnc.shape
    if chunk_q is None or int(chunk_q) <= 0 or S <= int(chunk_q):
        return _p3d_knn_points(Q_bnc, P_bnc, K=int(K)).idx
    idx_list = []
    for s, e in _chunk_ranges(S, int(chunk_q)):
        out = _p3d_knn_points(Q_bnc[:, s:e], P_bnc, K=int(K))
        idx_list.append(out.idx)
    return torch.cat(idx_list, dim=1)


def _pseudo_rowwise_top2_flat(logits: torch.Tensor, thresh: float):
    """Build row-wise top-2 pseudo labels on-device.

    Returns:
        idx_dev: LongTensor [Nflat, K_, 2]
        mask_dev: FloatTensor [Nflat, K_]
        pos_count: int (number of positive elements)
        elem_count: int (total elements in logits matrix)
        K_: int (K-1)
    """
    if logits.ndim != 2:
        raise ValueError("Logits must be 2-D tensor [Nflat,(K-1)^2].")
    Nflat, T = logits.shape
    K_ = int(round(T ** 0.5))
    if K_ * K_ != T:
        raise ValueError(f"Logits dim {T} is not a perfect square.")
    logits_m = logits.view(Nflat, K_, K_)
    top2_vals, top2_idx = torch.topk(logits_m, k=2, dim=-1)
    mask = (torch.sigmoid(top2_vals[..., :1]) > thresh).to(logits.dtype).squeeze(-1)  # [Nflat, K_]
    pos_count = int((mask.sum() * 2).item())
    elem_count = int(logits_m.numel())
    return top2_idx, mask, pos_count, elem_count, K_


def _build_pseudo_from_topk(shape, idx_dev: torch.Tensor, mask_dev: torch.Tensor) -> torch.Tensor:
    pseudo = torch.zeros(shape, device=idx_dev.device, dtype=torch.float32)
    pseudo.scatter_(2, idx_dev, 1.0)
    return pseudo * mask_dev.unsqueeze(-1)


def set_seed(seed: int) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _normalize_basename(name: str) -> str:
    if not name:
        return ""
    base = os.path.basename(str(name).strip())
    if not base:
        return ""
    return base if base.lower().endswith(".ply") else f"{base}.ply"


def _read_name_list(path: str) -> List[str]:
    names: List[str] = []
    with open(path, "r") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for token in stripped.replace("\t", " ").split():
                norm = _normalize_basename(token)
                if norm:
                    names.append(norm)
                    break
    return names


def _indices_for_names(dataset, names: Set[str]) -> List[int]:
    if not hasattr(dataset, "pairs"):
        raise AttributeError("Dataset must expose a 'pairs' attribute with pointcloud/gt paths.")
    lookup = {os.path.basename(pc): idx for idx, (pc, _) in enumerate(dataset.pairs)}
    return sorted(idx for name in names if (idx := lookup.get(name)) is not None)


def _build_subset(
    dataset_default,
    dataset_special,
    target_names: Set[str],
    special_names: Set[str],
) -> Optional[torch.utils.data.Dataset]:
    if not target_names:
        return None
    target = {_normalize_basename(n) for n in target_names if _normalize_basename(n)}
    if not target:
        return None
    parts: List[torch.utils.data.Dataset] = []
    default_names = target - special_names
    if default_names:
        idx = _indices_for_names(dataset_default, default_names)
        if idx:
            parts.append(Subset(dataset_default, idx))
    if special_names and dataset_special is not None:
        spec = target & special_names
        if spec:
            idx = _indices_for_names(dataset_special, spec)
            if idx:
                parts.append(Subset(dataset_special, idx))
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def load_split_definition(path: str) -> Tuple[Set[str], Set[str], Set[str], Dict[str, object]]:
    with open(path, "r") as handle:
        payload = json.load(handle)
    train = {_normalize_basename(n) for n in payload.get("train", [])}
    val = {_normalize_basename(n) for n in payload.get("val", [])}
    test = {_normalize_basename(n) for n in payload.get("test", [])}
    meta = {k: v for k, v in payload.items() if k not in {"train", "val", "test"}}
    return train, val, test, meta


def gather_env_info(device: torch.device) -> Dict[str, object]:
    cuda_available = torch.cuda.is_available()
    env = {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", "unknown"),
        "device": str(device),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
    }
    if cuda_available:
        try:
            env["cuda_version"] = getattr(torch.version, "cuda", None)
            env["cuda_device_name"] = torch.cuda.get_device_name(device)  # type: ignore[arg-type]
        except Exception:
            pass
    return env


def defensive_load_state_dict(module: torch.nn.Module, ckpt_path: Path, map_location: torch.device) -> Dict[str, object]:
    report: Dict[str, object] = {"ckpt_path": str(ckpt_path), "loaded": False, "missing_keys": [], "unexpected_keys": []}
    if not ckpt_path.is_file():
        report["warning"] = "checkpoint_not_found"
        return report
    try:
        state = torch.load(str(ckpt_path), map_location=map_location)
    except Exception as exc:  # pragma: no cover - file IO
        report["warning"] = f"load_failed: {exc}"
        return report
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "net", "module", "model"):
            inner = state.get(key)
            if isinstance(inner, dict):
                state = inner
                break
    try:
        missing, unexpected = module.load_state_dict(state, strict=False)
        report["missing_keys"] = list(missing)
        report["unexpected_keys"] = list(unexpected)
        report["loaded"] = True
    except Exception as exc:  # pragma: no cover
        report["warning"] = f"state_dict_error: {exc}"
    return report


def chamfer_distance_batched(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError("Chamfer expects tensors shaped [B, N, 3] and [B, M, 3].")
    if _p3d_chamfer is not None:
        loss, _ = _p3d_chamfer(pred, target, batch_reduction="mean", point_reduction="mean")
        return loss
    dists = torch.cdist(pred, target, p=2)
    loss_ab = dists.min(dim=2).values.mean(dim=1)
    loss_ba = dists.min(dim=1).values.mean(dim=1)
    return (loss_ab + loss_ba).mean()


def _ensure_dataset_ready(
    dataset_obj,
    dataset_cls,
    data_root: str,
    dataset_name: str,
    delta: float,
    rescale_delta: bool,
) -> torch.utils.data.Dataset:
    if len(dataset_obj) > 0:
        return dataset_obj
    attempted: List[str] = []
    root_path = Path(data_root)
    if "_" in dataset_name:
        prefix = dataset_name.split("_")[0]
        alt_gt_dir = root_path / prefix / "test"
        attempted.append(str(alt_gt_dir))
        if alt_gt_dir.is_dir():
            target = root_path / "GT_Meshes" / dataset_name
            target.mkdir(parents=True, exist_ok=True)
            for ply in alt_gt_dir.glob("*.ply"):
                dest = target / ply.name
                if dest.exists():
                    continue
                try:
                    os.symlink(ply.resolve(), dest)
                except OSError:
                    shutil.copy(ply, dest)
            retry = dataset_cls(data_root=data_root, dataset=dataset_name, delta=delta, rescale_delta=rescale_delta)
            if len(retry) > 0:
                return retry
    raise RuntimeError(
        f"No paired pointcloud/gt found for dataset '{dataset_name}'."
        f" Checked {data_root}/PointClouds/{dataset_name} and attempted {attempted}."
    )


def prepare_datasets(
    args: argparse.Namespace,
    dataset_cls,
    project_root: Path,
) -> Tuple[torch.utils.data.Dataset, Optional[torch.utils.data.Dataset], Dict[str, object]]:
    data_root = args.data_root or str(project_root / "Data")
    dataset_default = dataset_cls(data_root=data_root, dataset=args.dataset, delta=args.delta, rescale_delta=args.rescale_delta)
    dataset_default = _ensure_dataset_ready(dataset_default, dataset_cls, data_root, args.dataset, args.delta, args.rescale_delta)

    available_names = {_normalize_basename(os.path.basename(pc)) for pc, _ in dataset_default.pairs}

    special_names: Set[str] = set()
    if args.special_list and os.path.isfile(args.special_list):
        special_names = {_normalize_basename(n) for n in _read_name_list(args.special_list)} & available_names
    dataset_special = dataset_cls(data_root=data_root, dataset=args.dataset, delta=args.delta_special, rescale_delta=args.rescale_delta) if special_names else None

    train_names: Set[str] = set()
    val_names: Set[str] = set()
    test_names: Set[str] = set()
    split_meta: Dict[str, object] = {}
    split_source = "random"

    if args.split_config:
        train_names, val_names, test_names, split_meta = load_split_definition(args.split_config)
        split_source = "split_config"
        args.split_name = str(split_meta.get("split", ""))
        args.split_fold = split_meta.get("fold")
        args.split_seed = split_meta.get("seed")

    if args.train_list and os.path.isfile(args.train_list):
        train_names = {_normalize_basename(n) for n in _read_name_list(args.train_list)}
        if split_source == "random":
            split_source = "train_list"
    if getattr(args, "val_list", "") and os.path.isfile(args.val_list):
        val_names = {_normalize_basename(n) for n in _read_name_list(args.val_list)}
        split_source = "explicit_lists"
    if getattr(args, "test_list", "") and os.path.isfile(args.test_list):
        test_names = {_normalize_basename(n) for n in _read_name_list(args.test_list)}
        split_source = "explicit_lists"

    missing = (train_names | val_names | test_names) - available_names
    if missing:
        preview = sorted(list(missing))[:10]
        raise RuntimeError(f"Split references missing samples: {preview} (total {len(missing)})")

    if train_names & val_names:
        raise RuntimeError("Train and validation splits overlap; please verify split definition.")
    if train_names & test_names:
        raise RuntimeError("Train and test splits overlap; please verify split definition.")
    if val_names & test_names:
        raise RuntimeError("Validation and test splits overlap; please verify split definition.")

    dataset_full = _build_subset(dataset_default, dataset_special, available_names, special_names)
    if dataset_full is None:
        raise RuntimeError("Failed to construct dataset subset from available samples.")

    if train_names:
        train_set = _build_subset(dataset_default, dataset_special, train_names, special_names)
        if train_set is None:
            raise RuntimeError("Training subset is empty after filtering; check train list.")
    else:
        train_set = dataset_full

    # Always produce a validation set: prefer explicit val_list; else derive from dataset_full deterministically
    val_set: Optional[torch.utils.data.Dataset]
    if val_names:
        val_set = _build_subset(dataset_default, dataset_special, val_names, special_names)
        if val_set is None:
            raise RuntimeError("Validation subset is empty after filtering; check val list.")
    else:
        total_len = len(dataset_full)
        if total_len < 2:
            raise RuntimeError("Dataset too small to derive a validation set.")
        n_val = max(1, int(round(total_len * float(args.val_ratio))))
        if n_val >= total_len:
            n_val = max(1, total_len - 1)
        n_train = total_len - n_val
        generator = torch.Generator()
        generator.manual_seed(int(args.seed))
        splits = random_split(dataset_full, [n_train, n_val], generator=generator)
        train_set, val_set = splits[0], splits[1]
        split_source = "random_split"

    summary = {
        "dataset": args.dataset,
        "data_root": data_root,
        "total_available": len(available_names),
        "train_count": len(train_set),
        "val_count": len(val_set) if val_set is not None else 0,
        "test_count": len(test_names),
        "special_count": len(special_names),
        "split_source": split_source,
        "train_list_path": args.train_list or None,
        "val_list_path": getattr(args, "val_list", None) or None,
        "test_list_path": getattr(args, "test_list", None) or None,
        "split_config": args.split_config or None,
        "split_meta": split_meta,
        "train_names_preview": sorted(list(train_names))[:10],
        "val_names_preview": sorted(list(val_names))[:10],
        "test_names_preview": sorted(list(test_names))[:10],
    }

    return train_set, val_set, summary


def build_model(args: argparse.Namespace, device: torch.device, project_root: Path) -> Tuple[torch.nn.Module, Dict[str, object]]:
    import importlib

    try:
        mod_unet = importlib.import_module("S2.LoonUNet")
    except ImportError:
        mod_unet = importlib.import_module("LoonUNet")
    DGCNNEncoder = getattr(mod_unet, "DGCNNEncoder")
    LoonBottleneck = getattr(mod_unet, "LoonBottleneck")
    # DecoderHead is no longer used inside LoonUNet (fp10 acts as decoder)
    LoonUNet = getattr(mod_unet, "LoonUNet")

    try:
        mod_recon = importlib.import_module("S2.ReconNet")
    except ImportError:
        mod_recon = importlib.import_module("ReconNet")
    S2ReconNet = getattr(mod_recon, "S2ReconNet")

    encoder = DGCNNEncoder(k=int(args.unet_k), ratio1=float(args.unet_ratio1), ratio2=float(args.unet_ratio2)).to(device)
    try:
        f_dim = int(getattr(getattr(encoder, "l4"), "conv").out_channels)
    except Exception:
        f_dim = 0
    use_q_as_feat = not bool(getattr(args, 'disable_q_as_feat', False))
    bottleneck = LoonBottleneck(
        hidden=int(args.unet_hidden),
        T=int(args.unet_T),
        K=int(args.unet_K),
        chunk_size=max(1, int(args.chunk_size)),
        f_dim=f_dim,
        use_q_as_feat=use_q_as_feat,
        share_offset_former=not bool(getattr(args, 'no_share_offset_former', False)),
    ).to(device)
    c_skip = int(getattr(getattr(encoder, "l1"), "conv").out_channels) if hasattr(encoder, "l1") else 32

    cin = 3 + 6 * int(args.Lembed)
    # IMPORTANT: ReconNet sequence length must match LOON bottleneck K
    triangle = S2ReconNet(Cin=cin, knn=int(args.unet_K), Lembed=int(args.Lembed)).to(device)
    load_report = defensive_load_state_dict(triangle, project_root / "trained_models" / f"model_knn{int(args.unet_K)}.pth", device)
    triangle.eval()
    for param in triangle.parameters():
        param.requires_grad_(False)

    # LoonUNet no longer accepts decoder head; pass encoder, bottleneck, triangle
    model = LoonUNet(encoder, bottleneck, triangle).to(device)
    return model, load_report


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
    device_type: str,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    def _worker_init(worker_id: int) -> None:
        base_seed = int(seed) + worker_id * 9973
        random.seed(base_seed)
        np.random.seed(base_seed % (2**32 - 1))
        torch.manual_seed(base_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device_type == "cuda"),
        drop_last=False,
        worker_init_fn=_worker_init,
        generator=generator,
    )


def run_step(
    model: torch.nn.Module,
    batch,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: Optional[optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    train: bool = True,
) -> Dict[str, float]:
    points, gt = batch
    if isinstance(points, torch.Tensor) and points.ndim == 3 and points.size(0) == 1:
        points = points[0]
    if isinstance(gt, torch.Tensor) and gt.ndim == 3 and gt.size(0) == 1:
        gt = gt[0]
    if not isinstance(points, torch.Tensor) or not isinstance(gt, torch.Tensor):
        raise TypeError("Dataset must return (points, gt) tensors.")

    points = points.to(device)
    gt = gt.to(device)
    P0_b3n = points.transpose(0, 1).unsqueeze(0).contiguous()
    gt_b1m3 = gt.unsqueeze(0).contiguous()

    grad_context = nullcontext() if train else torch.no_grad()
    with grad_context:
        if device.type == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp))
        else:
            autocast_ctx = nullcontext()
        with autocast_ctx:
            unsup_val = None
            if train and getattr(args, "unsup", False):
                # 无监督训练：模型前向仅返回 offsets；随后用冻结的 tri_net 计算 logits 并构造伪标签做 BCE
                if _p3d_knn_points is None or _p3d_knn_gather is None:
                    raise ImportError("pytorch3d.ops (knn_points/knn_gather) 未安装，无法计算无监督三角损失。")

                # 1) 在 P3 上做 FPS，再通过多层索引映射回 P0，得到 sel
                # P0_b3n: [B,3,N]
                P0_bn3 = P0_b3n.permute(0, 2, 1).contiguous()  # [B,N,3]
                _, N, _ = P0_bn3.shape
                max_points = int(getattr(args, "unsup_max_points", 10000))

                # 若 P0 点数不超过上限，则不做任何子采样，直接用全部点参与无监督损失
                if max_points <= 0 or N <= max_points:
                    sel = torch.arange(N, device=P0_bn3.device)
                else:
                    # N > max_points: 根据开关选择在 P0 还是 P3 上做 FPS
                    if bool(getattr(args, "unsup_fps_on_p3", False)):
                        # 在 P3 上做 FPS，再映射回 P0
                        with torch.no_grad():
                            enc = getattr(model, "enc", None)
                            if enc is None:
                                raise RuntimeError("model 没有 enc 编码器，无法在 P3 上做采样。")

                            # 通过 encoder 得到 P3 以及多层索引
                            (P0_enc, F0), (P1, F1, idx1), (P2, F2, idx2), (P3, F3, idx3) = enc(P0_b3n)

                            # P3: [B,3,N3] -> [B,N3,3]
                            P3_bn3 = P3.permute(0, 2, 1).contiguous()
                            _, N3, _ = P3_bn3.shape

                            if N3 == 0:
                                raise RuntimeError("encoder 输出的 P3 为空，无法进行无监督采样。")

                            # 先在 P3 上选出索引 idx_p3 (大小 S_P3)，然后映射到 P0
                            if N3 > max_points:
                                # 在 P3 上做 FPS，若不可用则退化为随机
                                if _p3d_sample_farthest_points is not None:
                                    pts_1 = P3_bn3[:1]  # [1,N3,3]
                                    _, fps_idx = _p3d_sample_farthest_points(pts_1, K=max_points)
                                    idx_p3 = fps_idx[0]
                                else:
                                    idx_p3 = torch.randperm(N3, device=P3_bn3.device)[:max_points]
                            else:
                                idx_p3 = torch.arange(N3, device=P3_bn3.device)

                            # 构建 P3 -> P0 的映射：P1 = P0[idx1]，P2 = P1[idx2]，P3 = P2[idx3]
                            # 只需 batch=0 的索引，然后共享到整个 batch（与原实现一致）
                            idx1_b = idx1[0]        # [N1]
                            idx2_b = idx2[0]        # [N2]
                            idx3_b = idx3[0]        # [N3]

                            map1 = idx1_b                               # P1 -> P0
                            map2 = map1[idx2_b]                          # P2 -> P0
                            map3 = map2[idx3_b]                          # P3 -> P0，形状 [N3]

                            sel0 = map3[idx_p3]                          # 选中的 P0 索引（未排序）
                            sel = sel0.sort().values                     # [S]
                    else:
                        # 在 P0 上直接做 FPS（或随机），不经过 encoder
                        if _p3d_sample_farthest_points is not None:
                            pts_1 = P0_bn3[:1]  # [1,N,3]
                            _, fps_idx = _p3d_sample_farthest_points(pts_1, K=max_points)
                            sel = fps_idx[0].sort().values
                        else:
                            sel = torch.randperm(N, device=P0_bn3.device)[:max_points]
                            sel = sel.sort().values

                # 2) 对子集做前向，得到子集偏移 offsets_sub
                P0_sub_bn3 = P0_bn3[:, sel]                           # [B,S,3]
                P0_sub_b3s = P0_sub_bn3.permute(0, 2, 1).contiguous() # [B,3,S]
                offsets_sub = model(P0_sub_b3s)                        # [B,3,S]

                # 3) 仅在子集中进行 kNN 和三角伪标签构建
                Pp_sub_bn3 = P0_sub_b3s.permute(0, 2, 1).contiguous() + offsets_sub.permute(0, 2, 1).contiguous()  # [B,S,3]
                dP_sub_bn3 = offsets_sub.permute(0, 2, 1).contiguous()                                              # [B,S,3]

                # 以子集自身作为数据库，进一步降低内存使用
                Q_bnc = Pp_sub_bn3  # queries [B,S,3]
                P_bnc = Pp_sub_bn3  # database [B,S,3]

                # Chunked kNN to reduce temporary distance tensor peak
                knn_q_chunk = int(getattr(args, 'knn_chunk_size', 0) or args.chunk_size)
                idx_bnk = _knn_points_chunked(Q_bnc, P_bnc, K=int(args.unet_K), chunk_q=knn_q_chunk)  # [B,S,K]

                cur_bn3 = Q_bnc
                off_bn3 = dP_sub_bn3
                nbr_pts = _p3d_knn_gather(P_bnc, idx_bnk)     # [B,S,K,3]
                nbr_off = _p3d_knn_gather(dP_sub_bn3, idx_bnk)    # [B,S,K,3]
                rel_pts = nbr_pts - cur_bn3.unsqueeze(2)      # [B,S,K,3]
                rel_off = nbr_off - off_bn3.unsqueeze(2)      # [B,S,K,3]

                feat = rel_pts.reshape(-1, int(args.unet_K), 3)
                off = rel_off.reshape(-1, int(args.unet_K), 3)

                tri = getattr(model, 'tri', None)
                if tri is None:
                    raise RuntimeError("model 没有 tri 子模块，无法计算无监督三角损失。")
                tri.eval()
                chunk = max(1, int(args.chunk_size))
                thresh = float(getattr(args, 'unsup_p1_thresh', 0.5))

                if getattr(args, 'unsup_two_pass', False):
                    # Pass-1: no-grad to build pseudo labels (on-device) without CPU sync
                    pseudo_meta = []
                    total_pos = 0
                    total_elem = 0
                    with torch.no_grad():
                        for start, end in _chunk_ranges(feat.shape[0], chunk):
                            logits_chunk = _compute_logits_chunked(tri, feat[start:end], off[start:end], chunk=end-start, with_grad=False)
                            idx_dev, mask_dev, pos_cnt, elem_cnt, K_ = _pseudo_rowwise_top2_flat(logits_chunk, thresh)
                            pseudo_meta.append((start, end, idx_dev, mask_dev, K_))
                            total_pos += pos_cnt
                            total_elem += elem_cnt
                    if total_elem == 0:
                        raise RuntimeError("未能生成无监督 logits，可能是子采样过小。")
                    pos_weight = (total_elem - total_pos) / max(total_pos, 1) if total_pos > 0 else 1.0
                    loss_sum = None
                    pos_weight_tensor = None
                    for start, end, idx_dev, mask_dev, K_ in pseudo_meta:
                        logits_chunk = _compute_logits_chunked(tri, feat[start:end], off[start:end], chunk=end-start, with_grad=True, use_ckpt=True)
                        logits_m = logits_chunk.view(-1, K_, K_)
                        if pos_weight_tensor is None:
                            pos_weight_tensor = torch.tensor(pos_weight, device=logits_m.device, dtype=logits_m.dtype)
                        if loss_sum is None:
                            loss_sum = torch.zeros((), device=logits_m.device, dtype=logits_m.dtype)
                        pseudo_chunk = _build_pseudo_from_topk(logits_m.shape, idx_dev, mask_dev)
                        loss_chunk = F.binary_cross_entropy_with_logits(
                            logits_m,
                            pseudo_chunk,
                            reduction='sum',
                            pos_weight=pos_weight_tensor,
                        )
                        loss_sum = loss_sum + loss_chunk
                    unsup_loss = loss_sum / float(total_elem)
                else:
                    # Single-pass: compute logits with grad, but build pseudo from logits.detach()
                    logits = _compute_logits_chunked(tri, feat, off, chunk, with_grad=True, use_ckpt=True)
                    logits_detached = logits.detach()
                    idx_dev, mask_dev, pos_cnt, elem_cnt, K_ = _pseudo_rowwise_top2_flat(logits_detached, thresh)
                    logits_m = logits.view(-1, K_, K_)
                    pseudo = _build_pseudo_from_topk(logits_m.shape, idx_dev, mask_dev)
                    pos_weight = (elem_cnt - pos_cnt) / max(pos_cnt, 1) if pos_cnt > 0 else 1.0
                    unsup_loss = F.binary_cross_entropy_with_logits(
                        logits_m,
                        pseudo,
                        reduction='sum',
                        pos_weight=torch.tensor(pos_weight, device=logits_m.device, dtype=logits_m.dtype),
                    ) / float(elem_cnt)

                reg_loss = offsets_sub.pow(2).mean() * float(args.lambda_reg)
                total_loss = unsup_loss + reg_loss
                cd_loss = torch.tensor(float('nan'), device=device)
                unsup_val = unsup_loss.detach()
            else:
                # 验证或监督训练（--sup）：计算 Chamfer + reg
                offsets = model(P0_b3n)
                pred = (P0_b3n + offsets).permute(0, 2, 1).contiguous()
                cd_loss = chamfer_distance_batched(pred, gt_b1m3)
                reg_loss = offsets.pow(2).mean() * float(args.lambda_reg)
                total_loss = cd_loss + reg_loss

    if train:
        if optimizer is None:
            raise ValueError("Optimizer required for training step.")
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

    # 训练阶段无监督时 cd_loss 为 NaN，占位；返回时仍提供字段但供上层跳过聚合
    cd_val = float(cd_loss.detach().cpu()) if cd_loss.numel() == 1 else float('nan')
    out = {
        "loss": float(total_loss.detach().cpu()),
        "cd": cd_val,
        "reg": float(reg_loss.detach().cpu()),
    }
    if unsup_val is not None:
        out["unsup"] = float(unsup_val.cpu())
    return out


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: Optional[optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    train: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    desc = f"Epoch [{'train' if train else 'val'}]"
    total_loss = 0.0
    total_cd = 0.0
    count_cd = 0
    total_reg = 0.0
    total_unsup = 0.0
    n_batches = 0
    start_time = time.time()
    iterator = tqdm(loader, desc=desc, unit="batch", leave=False)

    for batch_idx, batch in enumerate(iterator, start=1):
        metrics = run_step(model, batch, device, args, optimizer=optimizer, scaler=scaler, train=train)
        total_loss += metrics["loss"]
        # 聚合 cd（跳过 NaN）
        try:
            if metrics.get("cd", float("nan")) == metrics.get("cd", float("nan")):
                total_cd += metrics["cd"]
                count_cd += 1
        except Exception:
            pass
        total_reg += metrics["reg"]
        n_batches += 1
        post = {"loss": f"{metrics['loss']:.4f}", "reg": f"{metrics['reg']:.4f}"}
        if metrics.get('cd', float('nan')) == metrics.get('cd', float('nan')):
            post["cd"] = f"{metrics['cd']:.4f}"
        if 'unsup' in metrics:
            total_unsup += metrics['unsup']
            post["unsup"] = f"{metrics['unsup']:.4f}"
        iterator.set_postfix(post)
        if train and args.log_batches and logger and (batch_idx % max(1, int(args.log_interval)) == 0):
            if 'unsup' in metrics:
                logger.info(
                    "batch=%d loss=%.6f cd=%.6f reg=%.6f unsup=%.6f",
                    batch_idx,
                    metrics["loss"], metrics["cd"], metrics["reg"], metrics["unsup"],
                )
            else:
                logger.info(
                    "batch=%d loss=%.6f cd=%.6f reg=%.6f",
                    batch_idx,
                    metrics["loss"], metrics["cd"], metrics["reg"],
                )

    iterator.close()
    if n_batches == 0:
        return {"loss": 0.0, "cd": 0.0, "reg": 0.0, "time": 0.0}

    elapsed = time.time() - start_time
    out = {
        "loss": total_loss / n_batches,
        "reg": total_reg / n_batches,
        "time": elapsed,
    }
    if count_cd > 0:
        out["cd"] = total_cd / count_cd
    else:
        out["cd"] = float('nan')
    if total_unsup > 0 and getattr(args, 'unsup', False):
        out["unsup"] = total_unsup / n_batches
    return out


def append_metrics(csv_path: Path, headers: Sequence[str], row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a") as handle:
        if write_header:
            handle.write(",".join(headers) + "\n")
        handle.write(",".join(str(row.get(h, "")) for h in headers) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser("Train LOON-UNet end-to-end (unsupervised train, Chamfer for validation)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="ABC")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--train_list", type=str, default="")
    parser.add_argument("--val_list", type=str, default="")
    parser.add_argument("--test_list", type=str, default="")
    parser.add_argument("--split_config", type=str, default="")
    parser.add_argument("--special_list", type=str, default="")
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--delta_special", type=float, default=0.005)
    parser.add_argument("--rescale_delta", action="store_true")
    parser.add_argument("--lambda_reg", type=float, default=1e-3)
    parser.add_argument("--unet_k", type=int, default=16)
    parser.add_argument("--unet_ratio1", type=float, default=0.25)
    parser.add_argument("--unet_ratio2", type=float, default=0.25)
    parser.add_argument("--unet_hidden", type=int, default=64)
    parser.add_argument("--unet_T", type=int, default=3)
    parser.add_argument("--unet_K", type=int, default=50)
    parser.add_argument("--chunk_size", type=int, default=5000)
    parser.add_argument("--knn_chunk_size", type=int, default=0, help="Queries per knn_points call; 0 uses chunk_size")
    parser.add_argument("--knn", type=int, default=50)
    parser.add_argument("--Lembed", type=int, default=8)
    # Ablation switch: q as feature
    parser.add_argument("--disable_q_as_feat", dest="disable_q_as_feat", action="store_true", help="Do not concatenate q statistics into OffsetFormer features")
    # Compatible with old name --disable_q_feat
    parser.add_argument("--disable_q_feat", dest="disable_q_as_feat", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no_share_offset_former", action="store_true", help="Use distinct OffsetFormerCell parameters at each iteration instead of sharing")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--log_batches", action="store_true")
    parser.add_argument("--log_interval", type=int, default=30)
    parser.add_argument("--notes", type=str, default="")
    # 无监督/监督开关（默认无监督）
    parser.add_argument("--unsup", dest="unsup", action="store_true")
    parser.add_argument("--sup", dest="unsup", action="store_false")
    # 无监督伪标签阈值与两阶段开关（两阶段可选，默认单阶段即可）
    parser.add_argument("--unsup_p1_thresh", type=float, default=0.5)
    parser.add_argument("--unsup_two_pass", action="store_true")
    parser.add_argument("--unsup_max_points", type=int, default=10000)
    parser.add_argument("--unsup_fps_on_p3", action="store_true", help="If set, do FPS on P3 then map back to P0; otherwise FPS directly on P0")
    parser.set_defaults(unsup=True)

    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]

    sys.path.insert(0, str(project_root))
    try:
        from dataset.pc_recon_with_gt import PCReconWithGT  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("Failed to import PCReconWithGT from dataset.pc_recon_with_gt") from exc

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = project_root / "runs" / "S2_train"
        if args.split_config:
            split_name = Path(args.split_config).stem.replace("fold", "fold-")
            save_dir = save_dir / split_name
        elif args.train_list:
            save_dir = save_dir / Path(args.train_list).stem
        save_dir = save_dir / time.strftime("%Y%m%d-%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(save_dir / "train.log"), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("train")

    logger.info("Starting training on device %s", device)
    logger.info("Arguments: %s", json.dumps(vars(args), indent=2, default=str))

    train_set, val_set, split_summary = prepare_datasets(args, PCReconWithGT, project_root)
    logger.info("Dataset summary: %s", json.dumps(split_summary, indent=2, default=str))

    model, recon_report = build_model(args, device, project_root)
    logger.info("Recon subnetwork load report: %s", json.dumps(recon_report, indent=2, default=str))

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_file():
            checkpoint = torch.load(str(resume_path), map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            logger.warning("Resume checkpoint %s not found. Continuing from scratch.", resume_path)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")

    train_loader = create_dataloader(train_set, args.batch_size, args.num_workers, seed=args.seed + 1, shuffle=True, device_type=device.type)
    val_loader = None
    if val_set is not None:
        val_loader = create_dataloader(val_set, args.batch_size, args.num_workers, seed=args.seed + 2, shuffle=False, device_type=device.type)

    env_info = gather_env_info(device)
    with open(save_dir / "env.json", "w") as f_env:
        json.dump(env_info, f_env, indent=2, default=str)
    with open(save_dir / "args.json", "w") as f_args:
        json.dump(vars(args), f_args, indent=2, default=str)
    with open(save_dir / "split.json", "w") as f_split:
        json.dump(split_summary, f_split, indent=2, default=str)

    best_val = float("inf")  # track best validation criterion (now total loss)
    best_epoch = -1

    metrics_headers = ["epoch", "phase", "loss", "cd", "reg", "time"]
    if getattr(args, "unsup", False):
        metrics_headers.insert(3, "unsup")
    history_json: List[Dict[str, object]] = []

    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        torch.manual_seed(args.seed + epoch)
        train_metrics = run_epoch(model, train_loader, device, args, optimizer=optimizer, scaler=scaler, train=True, logger=logger)
        logger.info("Epoch %d train metrics: %s", epoch, train_metrics)
        history_json.append({"epoch": epoch, "phase": "train", **train_metrics})
        append_metrics(save_dir / "metrics_epoch.csv", metrics_headers, {"epoch": epoch, "phase": "train", **train_metrics})

        val_metrics: Optional[Dict[str, float]] = None
        if val_loader is not None:
            val_metrics = run_epoch(model, val_loader, device, args, optimizer=None, scaler=None, train=False, logger=logger)
            logger.info("Epoch %d val metrics: %s", epoch, val_metrics)
            history_json.append({"epoch": epoch, "phase": "val", **val_metrics})
            append_metrics(save_dir / "metrics_epoch.csv", metrics_headers, {"epoch": epoch, "phase": "val", **val_metrics})

            # Use total validation loss as selection criterion (not only Chamfer)
            val_crit = float(val_metrics.get("loss", float("inf")))
            if val_crit + float(args.min_delta) < best_val:
                best_val = val_crit
                best_epoch = epoch
                patience_counter = 0
                # 在 checkpoint 中嵌入必要 meta（便于重建阶段读取 delta 等）
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": {
                        "dataset": args.dataset,
                        "delta": float(args.delta),
                        "rescale_delta": bool(args.rescale_delta),
                        "unet_k": int(args.unet_k),
                        "unet_ratio1": float(args.unet_ratio1),
                        "unet_ratio2": float(args.unet_ratio2),
                        "unet_hidden": int(args.unet_hidden),
                        "unet_T": int(args.unet_T),
                        "unet_K": int(args.unet_K),
                        "use_point_transformer": bool(getattr(args, 'use_point_transformer', False)),
                        "use_gat": bool(getattr(args, 'use_gat', False)),
                        "gat_heads": int(getattr(args, 'gat_heads', 4)),
                        "no_pyramid": bool(args.no_pyramid),
                        "Lembed": int(args.Lembed),
                        "use_q_as_feat": not bool(getattr(args, 'disable_q_as_feat', False)),
                        "use_q_for_modulation": not bool(getattr(args, 'disable_q_modulation', False)),
                        # backward compatibility keys
                        "enable_quality_modulator": not bool(getattr(args, 'disable_q_modulation', False)),
                        "enable_q_feat": not bool(getattr(args, 'disable_q_as_feat', False)),
                        "share_offset_former": not bool(getattr(args, 'no_share_offset_former', False)),
                    },
                }
                torch.save(ckpt, save_dir / "model_best.pth")
                logger.info("Saved new best checkpoint with val loss %.6f at epoch %d", best_val, epoch)
            else:
                patience_counter += 1
        else:
            ckpt_last = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "args": {
                    "dataset": args.dataset,
                    "delta": float(args.delta),
                    "rescale_delta": bool(args.rescale_delta),
                    "unet_k": int(args.unet_k),
                    "unet_ratio1": float(args.unet_ratio1),
                    "unet_ratio2": float(args.unet_ratio2),
                    "unet_hidden": int(args.unet_hidden),
                    "unet_T": int(args.unet_T),
                    "unet_K": int(args.unet_K),
                    "use_point_transformer": bool(getattr(args, 'use_point_transformer', False)),
                    "use_gat": bool(getattr(args, 'use_gat', False)),
                    "gat_heads": int(getattr(args, 'gat_heads', 4)),
                    "no_pyramid": bool(args.no_pyramid),
                    "Lembed": int(args.Lembed),
                    "use_q_as_feat": not bool(getattr(args, 'disable_q_as_feat', False)),
                    "use_q_for_modulation": not bool(getattr(args, 'disable_q_modulation', False)),
                    "enable_quality_modulator": not bool(getattr(args, 'disable_q_modulation', False)),
                    "enable_q_feat": not bool(getattr(args, 'disable_q_as_feat', False)),
                    "share_offset_former": not bool(getattr(args, 'no_share_offset_former', False)),
                },
            }
            torch.save(ckpt_last, save_dir / "model_last.pth")

        with open(save_dir / "history.json", "w") as f_hist:
            json.dump(history_json, f_hist, indent=2, default=str)

        if val_loader is not None:
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    final_path = save_dir / "model_last.pth"
    if not final_path.exists():
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val": best_val,
            "args": {
                "dataset": args.dataset,
                "delta": float(args.delta),
                "rescale_delta": bool(args.rescale_delta),
                "unet_k": int(args.unet_k),
                "unet_ratio1": float(args.unet_ratio1),
                "unet_ratio2": float(args.unet_ratio2),
                "unet_hidden": int(args.unet_hidden),
                "unet_T": int(args.unet_T),
                "unet_K": int(args.unet_K),
                "use_point_transformer": bool(getattr(args, 'use_point_transformer', False)),
                "use_gat": bool(getattr(args, 'use_gat', False)),
                "gat_heads": int(getattr(args, 'gat_heads', 4)),
                "no_pyramid": bool(args.no_pyramid),
                "Lembed": int(args.Lembed),
                "use_q_as_feat": not bool(getattr(args, 'disable_q_as_feat', False)),
                "use_q_for_modulation": not bool(getattr(args, 'disable_q_modulation', False)),
                "enable_quality_modulator": not bool(getattr(args, 'disable_q_modulation', False)),
                "enable_q_feat": not bool(getattr(args, 'disable_q_as_feat', False)),
                "share_offset_former": not bool(getattr(args, 'no_share_offset_former', False)),
            },
        }, final_path)
    logger.info("Training completed. Best epoch: %d, best val loss: %.6f", best_epoch, best_val)


if __name__ == "__main__":
    main()
