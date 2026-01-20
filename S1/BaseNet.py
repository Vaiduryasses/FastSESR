import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

def knn_feature(x, k):
    """KNN in feature space with correct Euclidean distance formula.
    
    Args:
        x: [B, K, C] feature tensor
        k: number of neighbors
    Returns:
        idx: [B, K, k] neighbor indices (excluding self)
    """
    inner = -2 * torch.matmul(x, x.transpose(2, 1))  # [B, K, K]
    xx = torch.sum(x ** 2, dim=-1, keepdim=True)     # [B, K, 1]
    # Correct Euclidean distance squared: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i^T*x_j
    pairwise_distance = xx + xx.transpose(1, 2) + inner  # [B, K, K]
    
    # Mask diagonal (avoid picking itself as neighbor)
    K = x.size(1)
    diag = torch.eye(K, device=x.device, dtype=torch.bool).unsqueeze(0)  # [1, K, K]
    pairwise_distance = pairwise_distance.masked_fill(diag, 1e9)
    
    idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]  # [B, K, k]
    return idx


class DynamicGraphAttentionBlock(nn.Module):
    """GNN Layer - Pre-Norm 
    """
    def __init__(self, dim, k_neighbor=8):
        super().__init__()
        self.k_neighbor = k_neighbor
        self.norm = nn.LayerNorm(dim) # Pre-Norm
        self.attn_mlp = nn.Sequential(nn.Linear(dim * 2, dim), nn.LeakyReLU(0.2), nn.Linear(dim, 1))
        self.feature_mlp = nn.Sequential(nn.Linear(dim * 2, dim), nn.LeakyReLU(0.2))
        self.drop_path = DropPath(0.1)

    def forward(self, x, pos_embed, rel_pos_coords=None):
        """
        Args:
            x: [B, K, C] feature tensor
            pos_embed: [B, K, C] position embedding
            rel_pos_coords: [B, K, 3] relative position coordinates (for stable KNN)
        """
        # Pre-Norm
        x_norm = self.norm(x)
        x_with_pos = x_norm + pos_embed
        
        B, K, C = x_with_pos.shape
        k_effective = min(self.k_neighbor, K - 1)
        if k_effective <= 0: return self.drop_path(torch.zeros_like(x))

        # Use geometric coordinates for KNN (more stable than feature space)
        if rel_pos_coords is not None:
            # KNN in coordinate space (stable, doesn't change with features)
            neighbor_idx = knn_feature(rel_pos_coords, k=k_effective)  # [B, K, k]
        else:
            # Fallback: use feature+position space (less stable but backward compatible)
            neighbor_idx = knn_feature(x_with_pos, k=k_effective)
        
        idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * K
        idx_flat = (neighbor_idx + idx_base).view(-1)
        
        x_flat = x_with_pos.reshape(B * K, C)
        neighbor_feat_flat = x_flat[idx_flat, :]
        neighbor_feat = neighbor_feat_flat.view(B, K, k_effective, C)

        self_feat = x_with_pos.unsqueeze(2).expand(-1, -1, k_effective, -1)
        
        attn_input = torch.cat([self_feat, neighbor_feat], dim=-1)
        attn_logits = self.attn_mlp(attn_input)
        attn_weights = F.softmax(attn_logits, dim=2)

        feature_input = torch.cat([self_feat, neighbor_feat - self_feat], dim=-1)
        transformed_neighbor_feat = self.feature_mlp(feature_input)

        agg_feat = torch.sum(attn_weights * transformed_neighbor_feat, dim=2)
        
        return self.drop_path(agg_feat)

class SelfAttention(nn.Module):
    
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(dim) # Pre-Norm
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(0.1)

    def forward(self, x, pos_embed, rel_pos_bias):
        x_norm = self.norm(x)
        x_with_pos = x_norm + pos_embed
        
        B, K, C = x_with_pos.shape
        qkv = self.qkv(x_with_pos).reshape(B, K, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # rel_pos_bias [B, K, K]
        attn = attn + rel_pos_bias.unsqueeze(1)
        
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, K, C)
        out = self.proj(out)
        return self.drop_path(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim) # Pre-Norm
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
        )
        self.drop_path = DropPath(0.1)

    def forward(self, x):
        return self.drop_path(self.net(self.norm(x)))

class TransGNNBlock(nn.Module):
    def __init__(self, dim, k_neighbor=8, num_heads=4, ff_dim_multiplier=4):
        super().__init__()
        self.gnn_layer = DynamicGraphAttentionBlock(dim, k_neighbor)
        self.attn_layer = SelfAttention(dim, num_heads)
        self.ffn = FeedForward(dim, dim * ff_dim_multiplier)

    def forward(self, x, pos_embed, rel_pos_bias, rel_pos_coords=None):
        """
        Args:
            x: [B, K, C] feature tensor
            pos_embed: [B, K, C] position embedding
            rel_pos_bias: [B, K, K] relative position bias
            rel_pos_coords: [B, K, 3] relative position coordinates (for stable KNN in GNN)
        """
        # Pass rel_pos_coords to GNN for stable geometric KNN
        x = x + self.gnn_layer(x, pos_embed, rel_pos_coords)
        x = x + self.attn_layer(x, pos_embed, rel_pos_bias)
        x = x + self.ffn(x)
        return x

class S1BaseNet(nn.Module):
    """TransGNN  with Edge-Factorized Parameterization: S_jk = s0_j + s0_k + s_jk
    
    Version 1: Analytic sjk (pair_alpha, pair_bias)
    Version 2: Learnable low-rank sjk (use_pair_lowrank=True, pair_rank)
    """
    def __init__(self, Cin, knn=50, embed_dim=64, num_blocks=5, k_neighbor=16, num_heads=4,
                 # Edge-factorized parameters
                 use_edge_factorized=True,
                 pair_alpha=0.5, pair_bias=0.0,
                 use_pair_lowrank=False, pair_rank=32):
        super().__init__()
        self.K = knn
        self.Km = knn - 1
        self.use_edge_factorized = use_edge_factorized
        
        # Pair term parameters (for analytic version)
        # Both pair_alpha and pair_bias are now learnable to let the model find optimal scale and bias
        self.pair_alpha = nn.Parameter(torch.tensor(float(pair_alpha), dtype=torch.float32))
        self.pair_bias = nn.Parameter(torch.tensor(float(pair_bias), dtype=torch.float32))
        
        # Low-rank pair term parameters (for learnable version)
        self.use_pair_lowrank = use_pair_lowrank
        self.pair_rank = pair_rank if use_pair_lowrank else None

        self.input_proj = nn.Linear(Cin, embed_dim)
        self.pos_embed = nn.Linear(3, embed_dim)
        
        self.rel_pos_bias_mlp = nn.Sequential(
            nn.Linear(3, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )
        
        self.blocks = nn.ModuleList([
            TransGNNBlock(embed_dim, k_neighbor, num_heads)
            for _ in range(num_blocks)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)
        
        if self.use_edge_factorized:
            # === Edge-factorized heads ===
            # 1) center-to-neighbor edge logits s0_j: [B, K-1]
            self.edge0_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, 1)
            )
            
            # 2) Optional low-rank pair term: sjk = uu^T (symmetric low-rank)
            if self.use_pair_lowrank:
                self.pair_u = nn.Linear(embed_dim, pair_rank, bias=False)
                self.pair_scale = 1.0 / (pair_rank ** 0.5)
                self.pair_alpha_lr = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
                self.pair_bias_lr = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        else:
            # Original direct parameterization
            self.prediction_head = nn.Sequential(
                nn.Linear(self.K * embed_dim, 1024), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(1024, (self.K - 1) * (self.K - 1))
            )
    
    def forward(self, in_feats):
        rel_pos_coords = in_feats[..., :3] # [B, K, 3]

        x = self.input_proj(in_feats)
        pos_embedding = self.pos_embed(rel_pos_coords)

        p_i = rel_pos_coords.unsqueeze(2) # [B, K, 1, 3]
        p_j = rel_pos_coords.unsqueeze(1) # [B, 1, K, 3]
        rel_pos_matrix = p_i - p_j        # [B, K, K, 3]
        rel_pos_bias = self.rel_pos_bias_mlp(rel_pos_matrix).squeeze(-1) # [B, K, K]

        for block in self.blocks:
            # Pass rel_pos_coords to blocks for stable geometric KNN in GNN layers
            x = block(x, pos_embedding, rel_pos_bias, rel_pos_coords)

        x = self.final_norm(x)  # [B, K, D]
        
        if self.use_edge_factorized:
            # === Edge-factorized parameterization ===
            # 1) s0: center-to-neighbor edge logits [B, K-1]
            x_nei = x[:, 1:, :]  # neighbor tokens [B, K-1, D]
            s0 = self.edge0_head(x_nei).squeeze(-1)  # [B, K-1]
            
            # 2) sjk: pair term between neighbors
            if self.use_pair_lowrank:
                # Version 2: Learnable low-rank sjk = uu^T (symmetric, rank<=r)
                u = self.pair_u(x_nei)  # [B, K-1, r]
                sjk = torch.matmul(u, u.transpose(1, 2)) * self.pair_scale  # symmetric, rank<=r
                alpha = F.softplus(self.pair_alpha_lr)  # enforce positive
                sjk = alpha * sjk + self.pair_bias_lr
            else:
                # Version 1: Analytic sjk from geometry
                # Use relative positions between neighbors: ||(p_j - p_k)||^2
                p_nei = rel_pos_coords[:, 1:, :]  # [B, K-1, 3]
                pj = p_nei.unsqueeze(2)          # [B, K-1, 1, 3]
                pk = p_nei.unsqueeze(1)          # [B, 1, K-1, 3]
                d2 = ((pj - pk) ** 2).sum(dim=-1)  # [B, K-1, K-1]
                alpha = F.softplus(self.pair_alpha)    # enforce alpha > 0
                sjk = -alpha * d2 + self.pair_bias
            
            # 3) Compose triangle logits: S_jk = s0_j + s0_k + s_jk
            S = sjk + s0.unsqueeze(2) + s0.unsqueeze(1)  # [B, K-1, K-1]
            
            # Mask diagonal (j==k invalid)
            diag = torch.eye(self.Km, device=S.device, dtype=torch.bool).unsqueeze(0)
            S = S.masked_fill(diag, -20.0)
            
            # Enforce symmetry
            S = 0.5 * (S + S.transpose(1, 2))
            
            # Flatten to [B, (K-1)^2] for compatibility with SurfExtract
            logits = S.reshape(S.shape[0], self.Km * self.Km)
        else:
            # Original direct parameterization
            x_flat = x.flatten(start_dim=1)
            logits = self.prediction_head(x_flat)
            
            K_neighbors = self.K - 1
            logits = logits.reshape(-1, K_neighbors, K_neighbors)
            logits = (logits + logits.permute(0, 2, 1)) / 2.0
            logits = logits.reshape(-1, K_neighbors * K_neighbors)
        
        return logits

if __name__ == '__main__':
    N, K, C = 4, 50, 99
    
    print("=" * 60)
    print("Test Edge-Factorized Parameterization")
    print("=" * 60)
    
    # Test Version 1: Analytic sjk
    print("\n--- Version 1: Analytic sjk ---")
    model_v1 = S1BaseNet(Cin=C, knn=K, num_blocks=5, use_edge_factorized=True, 
                          use_pair_lowrank=False, pair_alpha=10.0)
    num_params_v1 = sum(p.numel() for p in model_v1.parameters() if p.requires_grad)
    print(f"params: {num_params_v1 / 1e6:.2f} M")
    
    input_tensor = torch.randn(N, K, C)
    logits_v1 = model_v1(input_tensor)
    print(f"input: {input_tensor.shape} -> output: {logits_v1.shape}")
    assert logits_v1.shape == (N, (K-1)**2), f"Expected {(N, (K-1)**2)}, got {logits_v1.shape}"
    print("✓ Version 1")
    
    # Test Version 2: Learnable low-rank sjk
    print("\n--- Version 2: Learnable low-rank sjk ---")
    model_v2 = S1BaseNet(Cin=C, knn=K, num_blocks=5, use_edge_factorized=True,
                          use_pair_lowrank=True, pair_rank=32)
    num_params_v2 = sum(p.numel() for p in model_v2.parameters() if p.requires_grad)
    print(f"params: {num_params_v2 / 1e6:.2f} M")
    
    logits_v2 = model_v2(input_tensor)
    print(f"input: {input_tensor.shape} -> output: {logits_v2.shape}")
    assert logits_v2.shape == (N, (K-1)**2), f"Expected {(N, (K-1)**2)}, got {logits_v2.shape}"
    print("✓ Version 2 ")
    
    # Test Original (for comparison)
    print("\n--- Original: Direct parameterization ---")
    model_orig = S1BaseNet(Cin=C, knn=K, num_blocks=5, use_edge_factorized=False)
    num_params_orig = sum(p.numel() for p in model_orig.parameters() if p.requires_grad)
    print(f"params: {num_params_orig / 1e6:.2f} M")
    
    logits_orig = model_orig(input_tensor)
    print(f"input: {input_tensor.shape} -> output: {logits_orig.shape}")
    assert logits_orig.shape == (N, (K-1)**2), f"Expected {(N, (K-1)**2)}, got {logits_orig.shape}"
    print("✓ Original")
    
    print("\n" + "=" * 60)
    print("params comparison:")
    print(f"  Original:     {num_params_orig / 1e6:.2f} M")
    print(f"  Edge-Factor V1: {num_params_v1 / 1e6:.2f} M (decrease {((num_params_orig - num_params_v1) / num_params_orig * 100):.1f}%)")
    print(f"  Edge-Factor V2: {num_params_v2 / 1e6:.2f} M (decrease {((num_params_orig - num_params_v2) / num_params_orig * 100):.1f}%)")
    print("=" * 60)
    print("\nall test pass！✓")
