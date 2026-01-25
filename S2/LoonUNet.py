import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
try:  # Safe import of checkpoint, inference environment may lack this submodule
    from torch.utils.checkpoint import checkpoint as _ckpt
except Exception:
    _ckpt = None

try:
    from pytorch3d.ops import knn_points, knn_gather, sample_farthest_points
except Exception:
    knn_points = None
    knn_gather = None
    sample_farthest_points = None


def _assert_p3d():
    if knn_points is None or knn_gather is None:
        raise ImportError("pytorch3d is required for LoonUNet (knn_points/knn_gather). Please install pytorch3d.")


def _three_nn(p_fine_b3n: torch.Tensor, p_coarse_b3n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find 3-NN of fine points in coarse set.
    Returns (dist [B,Nf,3], idx [B,Nf,3])."""
    _assert_p3d()
    Q = p_fine_b3n.transpose(1, 2).contiguous()   # [B,Nf,3]
    P = p_coarse_b3n.transpose(1, 2).contiguous() # [B,Nc,3]
    out = knn_points(Q, P, K=3)  # dists: [B,Nf,3], idx: [B,Nf,3]
    return out.dists, out.idx


def _interpolate_three(dP_coarse_b3n: torch.Tensor, idx_bnf3: torch.Tensor, w_bnf3: torch.Tensor) -> torch.Tensor:
    """Weighted interpolate coarse vectors onto fine points via 3NN.
    dP_coarse: [B,3,Nc], idx: [B,Nf,3], w: [B,Nf,3] -> [B,3,Nf]
    """
    _assert_p3d()
    B, _, Nc = dP_coarse_b3n.shape
    dP_nc3 = dP_coarse_b3n.transpose(1, 2).contiguous()  # [B,Nc,3]
    gathered = knn_gather(dP_nc3, idx_bnf3)  # [B,Nf,3,3]
    w_exp = w_bnf3.unsqueeze(-1)             # [B,Nf,3,1]
    dP_fine = (gathered * w_exp).sum(dim=2)  # [B,Nf,3]
    return dP_fine.transpose(1, 2).contiguous()  # [B,3,Nf]


def upsample_offsets(P_coarse_b3n: torch.Tensor, P_fine_b3n: torch.Tensor, dP_coarse_b3n: torch.Tensor) -> torch.Tensor:
    """3NN upsample offsets: coarse -> fine.
    P_*: [B,3,N*], dP_coarse: [B,3,Nc] -> [B,3,Nf]
    """
    dists, idx = _three_nn(P_fine_b3n, P_coarse_b3n)  # [B,Nf,3]
    w = 1.0 / (dists.clamp_min(1e-8))                 # [B,Nf,3]
    w = w / w.sum(dim=-1, keepdim=True)
    return _interpolate_three(dP_coarse_b3n, idx, w)


class EdgeConvBlock(nn.Module):
    """EdgeConv: concat(x_j - x_i, x_i) -> Conv/GN/ReLU -> max over K.
    Features are taken from the key set, centered by the query set.
    """
    def __init__(self, c_in: int, c_out: int, num_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(2 * c_in, c_out, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(num_groups, c_out), num_channels=c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, Fq_bcn: torch.Tensor, Fk_bcn: torch.Tensor, Pq_b3n: torch.Tensor, Pk_b3n: torch.Tensor, k: int) -> torch.Tensor:
        _assert_p3d()
        # Build graph with coordinates
        Q = Pq_b3n.transpose(1, 2).contiguous()  # [B,Nq,3]
        P = Pk_b3n.transpose(1, 2).contiguous()  # [B,Nk,3]
        knn = knn_points(Q, P, K=k)
        idx = knn.idx  # [B,Nq,K]

        # Gather features from key set and center by query features
        Fk_bnc = Fk_bcn.transpose(1, 2).contiguous()      # [B,Nk,C]
        nbr = knn_gather(Fk_bnc, idx)                     # [B,Nq,K,C]
        Fi_bnc = Fq_bcn.transpose(1, 2).contiguous()      # [B,Nq,C]
        Fi_rep = Fi_bnc.unsqueeze(2).expand_as(nbr)       # [B,Nq,K,C]
        pair = torch.cat([nbr - Fi_rep, Fi_rep], dim=-1)  # [B,Nq,K,2C]
        pair = pair.permute(0, 3, 1, 2).contiguous()      # [B,2C,Nq,K]
        out = self.conv(pair)
        out = self.gn(out)
        out = self.act(out)
        out = out.max(dim=-1)[0]                          # [B,C_out,Nq]
        return out


class DGCNNEncoder(nn.Module):
    def __init__(self, k: int = 16, c0: int = 8, c1: int = 32, c2: int = 64, c3: int = 64, c4: int = 128,
                 ratio1: float = 0.25, ratio2: float = 0.25):
        super().__init__()
        self.k = int(k)
        self.ratio1 = float(ratio1)
        self.ratio2 = float(ratio2)
        self.stem = nn.Conv1d(3, c0, 1)
        self.l1 = EdgeConvBlock(c0, c1)
        self.l2 = EdgeConvBlock(c1, c2)
        self.l3 = EdgeConvBlock(c2, c3)
        self.l4 = EdgeConvBlock(c3, c4)

    @staticmethod
    def _gather_bcn(x_bcn: torch.Tensor, idx_bm: torch.Tensor) -> torch.Tensor:
        # x: [B,C,N], idx: [B,M] -> [B,C,M]
        B, C, N = x_bcn.shape
        Bm = idx_bm.shape[1]
        idx_exp = idx_bm.unsqueeze(1).expand(B, C, Bm)
        return torch.gather(x_bcn, dim=2, index=idx_exp)

    def _fps(self, P_b3n: torch.Tensor, m: int) -> torch.Tensor:
        _assert_p3d()
        # sample_farthest_points returns (new_xyz[B,M,3], idx[B,M]) on P in [B,N,3]
        P_bnc = P_b3n.transpose(1, 2).contiguous()
        _, idx = sample_farthest_points(P_bnc, K=m)
        return idx

    def forward(self, xyz_b3n: torch.Tensor):
        _assert_p3d()
        B, _, N0 = xyz_b3n.shape
        P0 = xyz_b3n
        F0 = self.stem(xyz_b3n)                # [B,c0,N0]
        # Use checkpoint to wrap EdgeConv to reduce memory peak (effective during training/with gradients)
        if _ckpt is not None and torch.is_grad_enabled():
            try:
                F0a = _ckpt(lambda a, b, c, d: self.l1(a, b, c, d, k=self.k), F0, F0, P0, P0, use_reentrant=False)
            except TypeError:
                F0a = _ckpt(lambda a, b, c, d: self.l1(a, b, c, d, k=self.k), F0, F0, P0, P0)
        else:
            F0a = self.l1(F0, F0, P0, P0, k=self.k)   # [B,c1,N0]

        # FPS: P0 -> P1
        N1 = max(1, int(N0 * self.ratio1))
        idx1 = self._fps(P0, m=N1)             # [B,N1]
        P1 = self._gather_bcn(P0, idx1)        # [B,3,N1]
        F1_sk = self._gather_bcn(F0a, idx1)    # [B,c1,N1]
        if _ckpt is not None and torch.is_grad_enabled():
            try:
                F1a = _ckpt(lambda a, b, c, d: self.l2(a, b, c, d, k=self.k), F1_sk, F0a, P1, P0, use_reentrant=False)
            except TypeError:
                F1a = _ckpt(lambda a, b, c, d: self.l2(a, b, c, d, k=self.k), F1_sk, F0a, P1, P0)
        else:
            F1a = self.l2(F1_sk, F0a, P1, P0, k=self.k)  # cross-set EdgeConv

        # FPS: P1 -> P2
        N2 = max(1, int(N1 * self.ratio2))
        idx2 = self._fps(P1, m=N2)             # [B,N2]
        P2 = self._gather_bcn(P1, idx2)        # [B,3,N2]
        F2_sk = self._gather_bcn(F1a, idx2)    # [B,c2,N2]
        if _ckpt is not None and torch.is_grad_enabled():
            try:
                F2_mid = _ckpt(lambda a, b, c, d: self.l3(a, b, c, d, k=self.k), F2_sk, F1a, P2, P1, use_reentrant=False)
            except TypeError:
                F2_mid = _ckpt(lambda a, b, c, d: self.l3(a, b, c, d, k=self.k), F2_sk, F1a, P2, P1)
            try:
                F2a = _ckpt(lambda a, b, c, d: self.l4(a, b, c, d, k=self.k), F2_mid, F1a, P2, P1, use_reentrant=False)
            except TypeError:
                F2a = _ckpt(lambda a, b, c, d: self.l4(a, b, c, d, k=self.k), F2_mid, F1a, P2, P1)
        else:
            F2_mid = self.l3(F2_sk, F1a, P2, P1, k=self.k)
            F2a = self.l4(F2_mid, F1a, P2, P1, k=self.k)

        return (P0, F0a), (P1, F1a, idx1), (P2, F2a, idx2)


def _stats_from_logits_simple(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    p_max, _ = probs.max(dim=-1, keepdim=True)
    p_mean = probs.mean(dim=-1, keepdim=True)
    eps = 1e-6
    ent = -(probs * (probs + eps).log() + (1 - probs) * (1 - probs + eps).log()).mean(dim=-1, keepdim=True)
    top2 = torch.topk(logits, k=2, dim=-1).values
    gap = (top2[:, :1] - top2[:, 1:2])
    return torch.cat([p_max, p_mean, ent, gap], dim=-1)


def _pseudo_rowwise_top2_flat(logits: torch.Tensor, thresh: float):
    """Build pseudo-labels by taking top-2 per row (consistent with S2_train_loon_unet logic).

    Args:
        logits: [Nflat, (K-1)^2]
        thresh: Confidence threshold (in sigmoid space)
    Returns:
        idx_dev: LongTensor [Nflat, K_, 2]
        mask_dev: FloatTensor [Nflat, K_]
        pos_count: int  Number of positive samples (multiplied by 2)
        elem_count: int Total number of elements
        K_: int       Number of neighbors - 1
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


class GeometryGuidedAttention(nn.Module):
    """Point Transformer style attention."""
    def __init__(self, dim: int, k: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.k = k
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Linear projection
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Relative position encoding MLP: [dx, dy, dz, dist] -> dim
        self.pos_mlp = nn.Sequential(
            nn.Linear(4, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

        # Attention weight computation MLP (outputs scalar weight)
        # Input: head_dim (result of Q-K+Pos) -> Output: head_dim -> sum -> scalar
        self.attn_mlp = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.head_dim, 1) # Output scalar Score
        )

        # Output projection
        self.to_out = nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor, P: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        h: [B, N, C] Node features
        P: [B, 3, N] Coordinates
        idx: [B, N, K] Neighborhood indices
        """
        _assert_p3d()
        B, N, C = h.shape

        # 1. Relative geometric position encoding
        P_trans = P.transpose(1, 2).contiguous() # [B, N, 3]
        P_nbr = knn_gather(P_trans, idx)         # [B, N, K, 3]
        rel_pos = P_nbr - P_trans.unsqueeze(2)   # [B, N, K, 3]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True) # [B, N, K, 1]
        geom_feat = torch.cat([rel_pos, dist], dim=-1)   # [B, N, K, 4]

        # pos_enc: [B, N, K, Heads, HeadDim]
        pe = self.pos_mlp(geom_feat).view(B, N, self.k, self.num_heads, self.head_dim)

        # 2. Prepare Q, K, V
        Q = self.to_q(h).view(B, N, 1, self.num_heads, self.head_dim) # [B, N, 1, H, D]
        
        K_feat = self.to_k(h)
        K_nbr = knn_gather(K_feat, idx).view(B, N, self.k, self.num_heads, self.head_dim)
        
        V_feat = self.to_v(h)
        V_nbr = knn_gather(V_feat, idx).view(B, N, self.k, self.num_heads, self.head_dim)

        # 3. Subtraction Attention
        # Relation = Q - K + Pos
        # This approach better captures geometric matching relationships in point clouds
        relation = Q - K_nbr + pe # [B, N, K, H, D]

        # Calculate Attention Scores
        attn_logits = self.attn_mlp(relation) # [B, N, K, H, 1]
        attn_weights = F.softmax(attn_logits, dim=2) # Normalize along K dimension [B, N, K, H, 1]

        # 4. Aggregation
        # Value = V_feat + PosEncoding (explicitly inject geometric information)
        V_geom = V_nbr + pe

        agg = (attn_weights * V_geom).sum(dim=2) # [B, N, H, D]
        agg = agg.view(B, N, C)

        return self.to_out(agg)


class OffsetFormerCell(nn.Module):
    """Loon bottleneck offset refinement cell."""
    def __init__(self, in_dim: int, hidden_dim: int = 64, k: int = 16, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection (fuse h, q, F2, offset)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Core attention
        self.attn = GeometryGuidedAttention(
            dim=hidden_dim,
            k=k,
            num_heads=4,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Output head
        self.offset_head = nn.Linear(hidden_dim, 3)

        # Very small initial step size to prevent destroying geometric structure in early training
        self.step_scale = nn.Parameter(torch.tensor(1e-4, dtype=torch.float32))

    def forward(self, features: torch.Tensor, P: torch.Tensor, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Embedding
        x = self.input_proj(features)
        x = self.norm1(x)

        # 2. Attention Block (Residual)
        h_attn = self.attn(x, P, idx)
        x = self.norm2(x + self.dropout(h_attn))

        # 3. FFN Block (Residual)
        h_ffn = self.ffn(x)
        h_new = self.norm3(x + h_ffn)

        # 4. Delta Prediction
        delta_bn3 = self.offset_head(h_new) * self.step_scale
        delta_b3n = delta_bn3.transpose(1, 2).contiguous()

        return delta_b3n, h_new


class LoonBottleneck(nn.Module):
    def __init__(self, hidden: int = 64, T: int = 2, K: int = 32,
                 q_dim: int = 4, chunk_size: int = 2000,
                 f_dim: int = 0,
                 use_q_as_feat: bool = True,
                 share_offset_former: bool = True):
        super().__init__()
        self.hidden = int(hidden)
        self.T = int(T)
        self.K = int(K)
        self.chunk = int(chunk_size)
        self.q_dim = int(q_dim)
        self.f_dim = int(f_dim)
        self.use_q_as_feat = bool(use_q_as_feat)
        self.share_offset_former = bool(share_offset_former)

        # Input fusion dimension: hidden(h) + [q_dim(q)] + f_dim(F2) + 3(dP)
        # When q_feat is disabled, q_dim should not be counted in input_proj's input dimension, otherwise it will cause linear layer dimension mismatch
        self.pt_in_dim = self.hidden + (self.q_dim if self.use_q_as_feat else 0) + self.f_dim + 3

        def _make_cell():
            return OffsetFormerCell(
                in_dim=self.pt_in_dim,
                hidden_dim=self.hidden,
                k=self.K,
                dropout=0.1,
            )

        if self.share_offset_former:
            self.offset_former = _make_cell()
        else:
            self.offset_former = nn.ModuleList(_make_cell() for _ in range(self.T))

        # Initialize h using nearest neighbor scale
        self.init_mlp = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden)
        )

        # Checkpoint switch
        self.use_ckpt = True

    def _triangle_logits_chunked(self, tri_net: nn.Module, feat: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
        with_grad = torch.is_grad_enabled()
        use_ckpt = self.use_ckpt and with_grad
        if feat.shape[0] <= self.chunk:
            if use_ckpt and _ckpt is not None:
                try:
                    return _ckpt(lambda a, b: tri_net(a, b), feat, off, use_reentrant=False)
                except TypeError:
                    return _ckpt(lambda a, b: tri_net(a, b), feat, off)
            return tri_net(feat, off)
        outs = []
        for i in range(0, feat.shape[0], self.chunk):
            sl = slice(i, min(i + self.chunk, feat.shape[0]))
            f = feat[sl]
            o = off[sl]
            if use_ckpt and _ckpt is not None:
                try:
                    out = _ckpt(lambda a, b: tri_net(a, b), f, o, use_reentrant=False)
                except TypeError:
                    out = _ckpt(lambda a, b: tri_net(a, b), f, o)
            else:
                out = tri_net(f, o)
            outs.append(out)
        return torch.cat([outs_i for outs_i in outs], dim=0)

    def forward(self, P2_b3n: torch.Tensor, F2_bcn: Optional[torch.Tensor], tri_net: nn.Module):
        _assert_p3d()
        B, _, N2 = P2_b3n.shape
        device = P2_b3n.device

        offsets = torch.zeros(B, 3, N2, device=device)
        # PointTransformer branch does not need z sequence

        # init hidden from nearest neighbor scale
        d = knn_points(P2_b3n.transpose(1, 2).contiguous(), P2_b3n.transpose(1, 2).contiguous(), K=2)
        d0 = torch.sqrt(d.dists[..., 1:2].clamp_min(1e-12))  # [B,N2,1]
        geom = torch.cat([d0, d0 ** 2, torch.zeros(B, N2, 4, device=device)], dim=-1)[..., :6]  # [B,N2,6]
        h = self.init_mlp(geom.reshape(-1, 6)).reshape(B, N2, -1)  # [B,N2,H]

        for t in range(self.T):
            Pp = P2_b3n + offsets  # [B,3,N2]
            Q = Pp.transpose(1, 2).contiguous()  # [B,N2,3]
            P = Q
            d = knn_points(Q, P, K=self.K)
            idx = d.idx  # [B,N2,K]

            # Build triangle_net inputs: features & offsets_features (relative to self)
            cur = Pp.transpose(1, 2).contiguous()        # [B,N2,3]
            off = offsets.transpose(1, 2).contiguous()   # [B,N2,3]
            nbr_pts = knn_gather(cur, idx)               # [B,N2,K,3]
            nbr_off = knn_gather(off, idx)               # [B,N2,K,3]
            rel_pts = nbr_pts - cur.unsqueeze(2)         # [B,N2,K,3]
            rel_off = nbr_off - off.unsqueeze(2)         # [B,N2,K,3]

            feat_flat = rel_pts.reshape(-1, self.K, 3)
            off_flat = rel_off.reshape(-1, self.K, 3)
            logits = self._triangle_logits_chunked(tri_net, feat_flat, off_flat)  # [(B*N2),(K-1)^2]
            q = _stats_from_logits_simple(logits).reshape(B, N2, -1)             # [B,N2,q_dim]

            # PointTransformerCell offset refinement
            dP_bn3 = offsets.transpose(1, 2)  # [B,N2,3]
            if self.f_dim > 0:
                if F2_bcn is not None:
                    F2_bnC = F2_bcn.transpose(1, 2)
                else:
                    F2_bnC = torch.zeros(B, N2, self.f_dim, device=device, dtype=dP_bn3.dtype)
            else:
                F2_bnC = torch.zeros(B, N2, 0, device=device, dtype=dP_bn3.dtype)

            # Decide whether to concatenate q into features based on the switch, and use it as attention condition
            feat_list = [h]
            if self.use_q_as_feat:
                feat_list.append(q)
            feat_list.append(F2_bnC)
            feat_list.append(dP_bn3)
            inp_feat = torch.cat(feat_list, dim=-1)

            # Call OffsetFormerCell
            if self.share_offset_former or not isinstance(self.offset_former, nn.ModuleList):
                cell = self.offset_former
            else:
                cell_idx = min(t, len(self.offset_former) - 1)
                cell = self.offset_former[cell_idx]
            delta_b3n, h_new = cell(inp_feat, Pp, idx)

            offsets = offsets + delta_b3n
            h = h_new  # Update loop hidden state

        return offsets  # [B,3,N2]


class PN2FP_Offsets(nn.Module):
    """
    PointNet++ style learnable Feature Propagation (FP) for offset upsampling:
    1) Perform k=3 kNN from coarse point set to fine point set, use p=2 inverse distance weighting (IDW) to interpolate coarse offsets to fine;
    2) Concatenate the interpolated offsets with fine layer skip features, pass through shared MLP to produce refined (residual) increment.

    Input:
        P_coarse_b3n: [B,3,Nc]
        P_fine_b3n:   [B,3,Nf]
        dP_coarse_b3n:[B,3,Nc]  (predicted offsets at coarse layer)
        F_skip_bcn:    [B,C_skip,Nf] (skip features at fine layer)
    Output:
        dP_fine_b3n:  [B,3,Nf]  (fine layer offsets after learnable interpolation + refinement)
    """
    def __init__(self, c_skip: int, hidden: int = 64, k: int = 3, p: int = 2, use_residual: bool = True):
        super().__init__()
        self.k, self.p, self.use_residual = int(k), int(p), bool(use_residual)
        self.mlp = nn.Sequential(
            nn.Conv1d(3 + c_skip, hidden, 1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden), num_channels=hidden),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, hidden, 1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden), num_channels=hidden),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, 3, 1)  # Output refined increment for interpolated offsets
        )

    def forward(self, P_coarse_b3n: torch.Tensor, P_fine_b3n: torch.Tensor,
                dP_coarse_b3n: torch.Tensor, F_skip_bcn: torch.Tensor) -> torch.Tensor:
        _assert_p3d()
        # 1) kNN + inverse distance weighting (IDW) interpolate coarse offsets to fine
        Q = P_fine_b3n.transpose(1, 2).contiguous()   # [B,Nf,3]
        P = P_coarse_b3n.transpose(1, 2).contiguous() # [B,Nc,3]
        knn = knn_points(Q, P, K=self.k)
        idx, d2 = knn.idx, knn.dists                  # [B,Nf,k], [B,Nf,k] (squared distance)
        d = d2.clamp_min(1e-12).sqrt()                # Actual distance
        w = 1.0 / (d ** self.p)                       # IDW weights
        w = w / w.sum(dim=-1, keepdim=True)           # Normalize -> [B,Nf,k]

        dP_nc3 = dP_coarse_b3n.transpose(1, 2).contiguous()  # [B,Nc,3]
        nbr = knn_gather(dP_nc3, idx)                         # [B,Nf,k,3]
        dP_interp = (nbr * w.unsqueeze(-1)).sum(dim=2)        # [B,Nf,3]
        dP_interp = dP_interp.transpose(1, 2).contiguous()    # [B,3,Nf]

        # 2) Concatenate with fine layer skip features + MLP refinement (residual increment)
        x = torch.cat([dP_interp, F_skip_bcn], dim=1)         # [B,3+c_skip,Nf]
        delta = self.mlp(x)                                   # [B,3,Nf]
        return dP_interp + delta if self.use_residual else delta


class DecoderHead(nn.Module):
    def __init__(self, c_skip: int, c_mid: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(3 + c_skip, c_mid, 1)
        self.gn1 = nn.GroupNorm(num_groups=min(8, c_mid), num_channels=c_mid)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv1d(c_mid, 3, 1)

    def forward(self, dP0_b3n: torch.Tensor, F0_bcn: torch.Tensor) -> torch.Tensor:
        x = torch.cat([dP0_b3n, F0_bcn], dim=1)
        x = self.act1(self.gn1(self.conv1(x)))
        x = self.conv2(x)
        return x


class LoonUNet(nn.Module):
    def __init__(self, encoder: DGCNNEncoder, bottleneck: LoonBottleneck,
                 tri_net: nn.Module,
                 use_old_upsample: bool = False):
        super().__init__()
        self.enc = encoder
        self.bottleneck = bottleneck
        self.tri = tri_net
        self.use_old_upsample = bool(use_old_upsample)
        # Two-level learnable FP: P2->P1 (fuse F1), P1->P0 (fuse F0)
        # According to DGCNNEncoder channels: F0=c1=32, F1=c2=64
        self.fp21 = PN2FP_Offsets(c_skip=64, hidden=64)  # coarse=P2 -> fine=P1
        self.fp10 = PN2FP_Offsets(c_skip=32, hidden=64)  # coarse=P1 -> fine=P0

        for p in self.tri.parameters():
            p.requires_grad_(False)
        self.tri.eval()

    def forward(self, P0_b3n: torch.Tensor):
        (P0, F0), (P1, F1, _), (P2, F2, _) = self.enc(P0_b3n)
        # Wrap bottleneck to reduce memory peak (only during training/with gradients enabled).
        if _ckpt is not None and torch.is_grad_enabled():
            try:
                dP2 = _ckpt(lambda a, b, c: self.bottleneck(a, b, c), P2, F2, self.tri, use_reentrant=False)
            except TypeError:
                dP2 = _ckpt(lambda a, b, c: self.bottleneck(a, b, c), P2, F2, self.tri)
        else:
            dP2 = self.bottleneck(P2, F2, self.tri)
        if self.use_old_upsample:
            # Compatible with old 3NN upsampling
            dP1 = upsample_offsets(P2, P1, dP2)
            dP0 = upsample_offsets(P1, P0, dP1)
        else:
            # Learnable upsampling (PointNet++ style FP)
            if _ckpt is not None and torch.is_grad_enabled():
                try:
                    dP1 = _ckpt(lambda a, b, c, d: self.fp21(a, b, c, d), P2, P1, dP2, F1, use_reentrant=False)
                except TypeError:
                    dP1 = _ckpt(lambda a, b, c, d: self.fp21(a, b, c, d), P2, P1, dP2, F1)
                try:
                    dP0 = _ckpt(lambda a, b, c, d: self.fp10(a, b, c, d), P1, P0, dP1, F0, use_reentrant=False)
                except TypeError:
                    dP0 = _ckpt(lambda a, b, c, d: self.fp10(a, b, c, d), P1, P0, dP1, F0)
            else:
                dP1 = self.fp21(P2, P1, dP2, F1)                         # P2->P1
                dP0 = self.fp10(P1, P0, dP1, F0)                         # P1->P0
        return dP0
