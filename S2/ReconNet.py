import os
import torch
from S1.BaseNet import S1BaseNet


class S2ReconNet(S1BaseNet):
    def __init__(self, Cin, knn=50, Lembed=8, load_best: bool = True, best_model_path: str = None, map_location=None, freeze_s1: bool = True,
                 # Edge-factorized parameters (passed to S1BaseNet)
                 use_edge_factorized=True, pair_alpha=10.0, pair_bias=0.0,
                 use_pair_lowrank=True, pair_rank=32,
                 # Model architecture parameters
                 embed_dim=64):
        super().__init__(Cin, knn, embed_dim=embed_dim,
                        use_edge_factorized=use_edge_factorized,
                        pair_alpha=pair_alpha, pair_bias=pair_bias,
                        use_pair_lowrank=use_pair_lowrank, pair_rank=pair_rank)
        self.Lembed = Lembed
        self.resolution = 0.01
        self.tile_rows = 16
        # Optionally load Stage-1 best checkpoint into this network
        if load_best:
            self._load_best_weights(best_model_path=best_model_path, map_location=map_location)
        # Freeze all Stage-1 parameters if requested
        if freeze_s1:
            self._freeze_s1_parameters()
        # Adapter+ path removed in S2: keep pure backbone blocks without adapters

    def _load_best_weights(self, best_model_path: str = None, map_location=None):
        """Load weights from Stage-1 best checkpoint (checkpoint dict or pure state_dict).

        Default path: S1_training/model_k{K}/best_model
        """
        # Derive default path from knn K
        default_path = f"S1_training/model_k{self.K}/best_model"
        ckpt_path = best_model_path or default_path
        if not os.path.exists(ckpt_path):
            print(f"[S2ReconNet] best_model not found at '{ckpt_path}', skip loading.")
            return
        try:
            device = map_location if map_location is not None else 'cpu'
            ckpt = torch.load(ckpt_path, map_location=device)
            # Support both full checkpoint dict and raw state_dict
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                state = ckpt['model_state_dict']
            else:
                state = ckpt
            # Strip potential DataParallel 'module.' prefix
            state = {k.replace('module.', ''): v for k, v in state.items()}
            # Report first 20 Missing/Unexpected keys (diff between model and checkpoint)
            own = self.state_dict()
            own_keys = list(own.keys())
            ckpt_keys = list(state.keys())
            missing_keys = [k for k in own_keys if k not in ckpt_keys]
            unexpected_keys = [k for k in ckpt_keys if k not in own_keys]
            # Print full diffs (no adapter.* filtering)
            if missing_keys or unexpected_keys:
                head_miss = missing_keys[:20]
                head_unexp = unexpected_keys[:20]
                print(f"[S2ReconNet] Key diff before load -> Missing:{len(missing_keys)}, Unexpected:{len(unexpected_keys)}")
                if head_miss:
                    print("  Missing (first 20):\n    " + "\n    ".join(head_miss))
                if head_unexp:
                    print("  Unexpected (first 20):\n    " + "\n    ".join(head_unexp))

            # If adapter-wrapped vs. non-wrapped keys mismatch ('.block.'), auto-remap keys
            def has_block_prefix(keys):
                return any('.block.' in k for k in keys)

            own_has_block = has_block_prefix(own_keys)
            ckpt_has_block = has_block_prefix(ckpt_keys)

            def remap_blocks_prefix(src_state, insert_block: bool):
                remapped = {}
                for k, v in src_state.items():
                    if k.startswith('blocks.'):
                        parts = k.split('.')
                        # expect parts like ['blocks', 'i', ...]
                        if len(parts) >= 3:
                            if insert_block:
                                # blocks.i.xxx -> blocks.i.block.xxx
                                if parts[2] != 'block':
                                    parts = parts[:2] + ['block'] + parts[2:]
                            else:
                                # blocks.i.block.xxx -> blocks.i.xxx
                                if len(parts) >= 4 and parts[2] == 'block':
                                    parts = parts[:2] + parts[3:]
                            new_k = '.'.join(parts)
                            remapped[new_k] = v
                            continue
                    remapped[k] = v
                return remapped

            if own_has_block and not ckpt_has_block:
                # model expects '.block.' but ckpt doesn't -> insert
                state = remap_blocks_prefix(state, insert_block=True)
                ckpt_keys = list(state.keys())
            elif ckpt_has_block and not own_has_block:
                # ckpt has '.block.' but model doesn't -> remove
                state = remap_blocks_prefix(state, insert_block=False)
                ckpt_keys = list(state.keys())

            # Load with strict=False to be robust to minor diffs
            missing, unexpected = self.load_state_dict(state, strict=False)
            # Print full diffs after loading (no adapter.* filtering)
            print(f"[S2ReconNet] Loaded S1 best weights from '{ckpt_path}'. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            if missing:
                print("  Missing after load (first 20):\n    " + "\n    ".join(missing[:20]))
            if unexpected:
                print("  Unexpected after load (first 20):\n    " + "\n    ".join(unexpected[:20]))
        except Exception as e:
            print(f"[S2ReconNet] Failed to load best_model from '{ckpt_path}': {e}")

    def _posEnc(self, x):
        # Create frequency scales in a single operation
        levels = 2. ** torch.arange(self.Lembed, dtype=x.dtype, device=x.device)
        levels = levels.view(1, 1, -1, 1)
        level_x = x[...,None,:]*levels

        # Concatenate sine and cosine functions in a vectorized way
        pe_feats = torch.stack([torch.sin(level_x), torch.cos(level_x)], dim=3)
        pe_feats = pe_feats.view(x.shape[0], x.shape[1], -1)
        pe_feats = torch.cat([x, pe_feats], dim=-1)
        return pe_feats

    def _updateFeatures(self, features):
        knn_dists = torch.linalg.norm(features, axis=-1)
        knn_scale = knn_dists[:,1:2]
        features = self.resolution*features/knn_scale[...,None]   
        features = self._posEnc(features)        # high-resolution signals
        return features

    def forward(self, features, offsets_features):
        """Forward without external neighbor indices.

        We restore dynamic graph KNN inside the backbone (S1BaseNet), so this
        method only prepares features and delegates to super().forward(in_feats).
        """
        # Prepare input features for S1BaseNet forward
        in_feats = self._updateFeatures(features + offsets_features)
        # Forward identical to S1BaseNet (triangle logits over neighbors)
        logits = super().forward(in_feats)
        return logits

    def _freeze_s1_parameters(self):
        """Freeze all parameters from the S1 backbone (this whole module)."""
        for p in self.parameters():
            p.requires_grad = False

# Adapter+ code removed entirely in S2


if __name__=='__main__':
    # Simple smoke test for shape
    N, K, C = 4, 50, 99
    model = S2ReconNet(Cin=C, knn=K, load_best=False)
    features = torch.randn(N, K, 3)
    offsets = torch.randn(N, K, 3)
    logits = model(features, offsets)
    print('logits:', logits.shape, 'expect:', (K-1)*(K-1))
    
