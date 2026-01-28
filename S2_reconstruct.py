import argparse, os
import importlib
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, total=None, desc=None, leave=True, unit=None, position=None):
        return iterable if iterable is not None else range(total or 0)
import numpy as np
import torch, glob, json
from torch.utils.data import DataLoader
from time import time
import open3d as o3d
# Silence Open3D warnings like "TriangleMesh appears to be a PointCloud"
try:
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
except Exception:
    pass
from dataset.pc_recon import PCReconSet
from S2.loss_unsupervised import ReconLoss
from S2.ReconNet import S2ReconNet
from S2.ExtractFace import SurfExtract


parser = argparse.ArgumentParser()
parser.add_argument('--use_loon_unet', action='store_true', help='Use LOON-UNet (encoder+bottleneck+decoder) for multi-scale offset inference')
parser.add_argument('--use_loon', action='store_true', help='Use LOON optimizer instead of hand-crafted S2 updates')
parser.add_argument('--gpu', type=int, default=0, help='GPU id')
parser.add_argument('--dataset', type=str, default='ABC', help='Dataset name')
parser.add_argument('--chunk_size', type=int, default=2000, help='Chunk size for logits/extraction')
parser.add_argument('--loon_ckpt', type=str, default='', help='Path to LOON or LOON-UNet checkpoint (preferred)')
parser.add_argument('--loon_unet_ckpt', type=str, default='', help='[Deprecated] Path to LOON-UNet checkpoint (used only if --loon_ckpt empty)')
parser.add_argument('--data_root', type=str, default='./Data', help='Root folder containing PointClouds/<dataset>')
parser.add_argument('--test_list', type=str, default='', help='Optional text file with list of pointcloud basenames (one per line) to reconstruct')
parser.add_argument('--out_dir', type=str, default='', help='Optional output directory for reconstructed meshes (overrides results/<dataset>/)')
parser.add_argument('--delta', type=float, default=None,help='Override surface voxel size delta used by PCReconSet; if set, overrides checkpoint meta')
opt = parser.parse_args()



if __name__=='__main__':
    device = torch.device('cuda:{}'.format(opt.gpu) if torch.cuda.is_available() else 'cpu')

    # Determine delta/rescale_delta from checkpoint meta (no CLI delta any more)
    def _extract_meta_dicts_quick(obj):
        metas = []
        if isinstance(obj, dict):
            for k in ['args','hparams','opt','config','model_args','hyper_parameters','meta']:
                if k in obj and isinstance(obj[k], dict):
                    metas.append(obj[k])
        return metas

    # pick ckpt depending on chosen method
    chosen_ckpt = ''
    if getattr(opt, 'use_loon_unet', False):
        chosen_ckpt = getattr(opt, 'loon_unet_ckpt', '')
    elif getattr(opt, 'use_loon', False):
        chosen_ckpt = getattr(opt, 'loon_ckpt', '')
    meta_dicts = []
    try:
        if chosen_ckpt and os.path.isfile(chosen_ckpt):
            ck = torch.load(chosen_ckpt, map_location=device)
            meta_dicts = _extract_meta_dicts_quick(ck) if isinstance(ck, dict) else []
        else:
            meta_dicts = []
    except Exception:
        meta_dicts = []

    def _find_meta_quick(keys, default=None):
        for md in meta_dicts:
            for k in keys:
                if k in md:
                    return md[k]
        return default

    # set opt.delta and opt.rescale_delta from meta if present, else defaults
    delta_meta = _find_meta_quick(['delta','voxel_size'], None)
    rescale_meta = _find_meta_quick(['rescale_delta'], False)
    cli_delta = getattr(opt, 'delta', None)
    if cli_delta is not None:
        setattr(opt, 'delta', float(cli_delta))
        print(f"[Recon] Using CLI delta override: delta={float(cli_delta)}")
    elif delta_meta is not None:
        setattr(opt, 'delta', float(delta_meta))
        print(f"[Recon] Using checkpoint delta from meta: delta={float(delta_meta)}")
    else:
        # fallback default
        setattr(opt, 'delta', 0.01)
        print('[Warn] No delta found in CLI or checkpoint meta; falling back to default delta=0.01')
    setattr(opt, 'rescale_delta', bool(rescale_meta))


    # hyper-parameter configurations
    dim, Lembed = 3, 8
    Cin = dim + dim*Lembed*2
    knn = 50
  
    # load point clouds to be reconstructed
    if getattr(opt, 'test_list', '') and os.path.isfile(opt.test_list):
        test_files = []
        with open(opt.test_list, 'r') as f:
            for line in f:
                s = line.strip()
                if not s: continue
                # if absolute path exists, use it, otherwise join data_root/PointClouds/dataset
                if os.path.isabs(s) and os.path.isfile(s):
                    test_files.append(s)
                else:
                    candidate = os.path.join(getattr(opt, 'data_root', '/root/autodl-tmp/Data'), 'PointClouds', opt.dataset, s)
                    if os.path.isfile(candidate):
                        test_files.append(candidate)
                    else:
                        # try with .ply appended
                        if not candidate.lower().endswith('.ply') and os.path.isfile(candidate + '.ply'):
                            test_files.append(candidate + '.ply')
        if len(test_files) == 0:
            print(f"[Warn] test_list provided but no files found. Falling back to directory glob.")
            test_files = glob.glob(os.path.join(getattr(opt, 'data_root', '/root/autodl-tmp/Data'), 'PointClouds', opt.dataset, '*.ply'))
    else:
        test_files = glob.glob(os.path.join(getattr(opt, 'data_root', '/root/autodl-tmp/Data'), 'PointClouds', opt.dataset, '*.ply'))
    testSet  = PCReconSet(test_files, knn=knn, delta=opt.delta, rescale_delta=opt.rescale_delta)  
    testLoader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=0)

    # store folder to reconstructed meshes
    if getattr(opt, 'out_dir', ''):
        results_folder = opt.out_dir
    else:
        results_folder = f'results/{opt.dataset}/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder, exist_ok=True)
                              
    # config model, loss function, and load the trained model
    loss_fn = ReconLoss()
    _ckpt_path = os.path.join('trained_models', 'model_knn50.pth')
    ckpt_obj = torch.load(_ckpt_path, map_location=device)
    if isinstance(ckpt_obj, dict):
        if 'model_state_dict' in ckpt_obj:
            state_dict = ckpt_obj['model_state_dict']
        elif 'state_dict' in ckpt_obj:
            state_dict = ckpt_obj['state_dict']
        else:
            state_dict = ckpt_obj
    else:
        state_dict = ckpt_obj
    if isinstance(state_dict, dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    use_pair_lowrank = False
    pair_rank = 32
    if isinstance(state_dict, dict) and 'pair_u.weight' in state_dict:
        use_pair_lowrank = True
        try:
            w = state_dict['pair_u.weight']
            if hasattr(w, 'shape') and len(w.shape) >= 2:
                pair_rank = int(w.shape[0])
        except Exception:
            pass
        print(f"[Recon][S2ReconNet] Detected lowrank checkpoint: use_pair_lowrank=True, pair_rank={pair_rank}")

    model = S2ReconNet(
        Cin=Cin,
        knn=knn,
        Lembed=Lembed,
        use_pair_lowrank=use_pair_lowrank,
        pair_rank=pair_rank,
    ).to(device)

    if isinstance(state_dict, dict):
        own = model.state_dict()
        own_keys = list(own.keys())
        ckpt_keys = list(state_dict.keys())
        missing_keys = [k for k in own_keys if k not in ckpt_keys]
        unexpected_keys = [k for k in ckpt_keys if k not in own_keys]
        if missing_keys or unexpected_keys:
            head_miss = missing_keys[:20]
            head_unexp = unexpected_keys[:20]
            print(f"[Recon][S2ReconNet] Key diff vs ckpt -> Missing:{len(missing_keys)}, Unexpected:{len(unexpected_keys)}")
            if head_miss:
                print("  Missing (first 20):\n    " + "\n    ".join(head_miss))
            if head_unexp:
                print("  Unexpected (first 20):\n    " + "\n    ".join(head_unexp))
        for k in unexpected_keys:
            state_dict.pop(k, None)
        if missing_keys:
            raise RuntimeError("S2ReconNet checkpoint is missing required parameters; please check that the architecture matches.")

    model.load_state_dict(state_dict)

    if opt.use_loon_unet:
        try:
            mod_unet = importlib.import_module('S2.LoonUNet')
        except Exception:
            try:
                mod_unet = importlib.import_module('LoonUNet')
            except Exception as e:
                raise ImportError("Could not import LoonUNet. Please ensure pytorch3d is installed and PYTHONPATH includes project root.") from e
        DGCNNEncoder = getattr(mod_unet, 'DGCNNEncoder')
        LoonBottleneck = getattr(mod_unet, 'LoonBottleneck')
        LoonUNet = getattr(mod_unet, 'LoonUNet')

        ckpt_path = getattr(opt, 'loon_unet_ckpt', '')

        raw_ckpt, state = None, None
        if ckpt_path and os.path.isfile(ckpt_path):
            try:
                raw_ckpt = torch.load(ckpt_path, map_location=device)
                state = raw_ckpt['model_state_dict'] if isinstance(raw_ckpt, dict) and 'model_state_dict' in raw_ckpt else raw_ckpt
            except Exception as e:
                print(f"[Warn] Failed to load LOON-UNet checkpoint file {ckpt_path}: {e}")
        else:
            if ckpt_path:
                print(f"[Warn] LOON-UNet checkpoint not found at {ckpt_path}. Using random init.")

        def _extract_meta_dicts(obj):
            metas = []
            if isinstance(obj, dict):
                for k in ['hparams','args','opt','config','model_args','hyper_parameters','meta']:
                    if k in obj and isinstance(obj[k], dict):
                        metas.append(obj[k])
            return metas
        meta_dicts = _extract_meta_dicts(raw_ckpt) if isinstance(raw_ckpt, dict) else []
        def _find_meta(keys, default=None):
            for md in meta_dicts:
                for k in keys:
                    if k in md:
                        return md[k]
            return default
        unet_k = int(_find_meta(['unet_k','encoder_k'], 32))
        unet_ratio1 = float(_find_meta(['unet_ratio1','ratio1'], 0.25))
        unet_ratio2 = float(_find_meta(['unet_ratio2','ratio2'], 0.25))
        # Default to True to match training-time defaults when field is absent.
        no_pyramid = bool(_find_meta(['no_pyramid'], True))
        # 从权重推断 hidden（优先）
        unet_hidden = int(_find_meta(['unet_hidden','hidden','hidden_dim'], 64))
        if isinstance(state, dict):
            for name in ['bottleneck.cell.lstm_cell.weight_hh','bottleneck.lstm_cell.weight_hh','cell.lstm_cell.weight_hh']:
                if name in state and hasattr(state[name], 'shape'):
                    try:
                        unet_hidden = int(state[name].shape[1])
                        break
                    except Exception:
                        pass
        unet_T = int(_find_meta(['unet_T'], 2))
        unet_K = int(_find_meta(['unet_K'], 50))
        use_q_as_feat = bool(_find_meta(['use_q_as_feat', 'enable_q_feat'], True))
        share_offset_former = bool(_find_meta(['share_offset_former'], True))

        target_c4 = None
        if isinstance(state, dict) and 'enc.l4.conv.weight' in state:
            try:
                target_c4 = int(state['enc.l4.conv.weight'].shape[0])
            except Exception:
                target_c4 = None
        if target_c4 is not None:
            enc = DGCNNEncoder(k=unet_k, ratio1=unet_ratio1, ratio2=unet_ratio2, c4=target_c4).to(device)
        else:
            enc = DGCNNEncoder(k=unet_k, ratio1=unet_ratio1, ratio2=unet_ratio2).to(device)
        # Determine F2 feature dim from encoder's top layer
        try:
            f_dim = int(getattr(getattr(enc, 'l4'), 'conv').out_channels)
        except Exception:
            f_dim = 0
            if isinstance(state, dict) and 'enc.l4.conv.weight' in state:
                try:
                    f_dim = int(state['enc.l4.conv.weight'].shape[0])
                except Exception:
                    f_dim = 0
        # Strict alignment with FastSESR's LoonBottleneck signature.
        bottleneck = LoonBottleneck(
            hidden=unet_hidden,
            T=unet_T,
            K=unet_K,
            chunk_size=max(1, int(opt.chunk_size)),
            f_dim=f_dim,
            use_q_as_feat=use_q_as_feat,
            share_offset_former=share_offset_former,
        ).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # Strict alignment with FastSESR's LoonUNet signature.
        loon_unet = LoonUNet(enc, bottleneck, model).to(device)
        try:
            setattr(loon_unet, 'no_pyramid', bool(no_pyramid))
        except Exception:
            pass

        if isinstance(state, dict):
            try:
                loon_unet.load_state_dict(state)
                print(f"[Info] Loaded LOON-UNet weights from {ckpt_path}")
            except Exception as e:
                print(f"[Warn] Strict load failed: {e}\n       Falling back to strict=False...")
                try:
                    missing, unexpected = loon_unet.load_state_dict(state, strict=False)
                    enable_qm = bool(_find_meta(['use_q_for_modulation', 'enable_quality_modulator'], True))
                    only_qm_missing = False
                    if missing:
                        try:
                            only_qm_missing = all(('q_mod.' in k or k.endswith('.q_mod') or '.attn.q_mod.' in k) for k in missing)
                        except Exception:
                            only_qm_missing = False
                    if missing:
                        if (not enable_qm) and only_qm_missing:
                            print(f"[Info] quality_modulator disabled in ckpt; missing q_mod params: {len(missing)}")
                        else:
                            print(f"[Warn] missing keys: {len(missing)} (show up to 10): {missing[:10]}")
                    if unexpected:
                        print(f"[Warn] unexpected keys: {len(unexpected)} (show up to 10): {unexpected[:10]}")
                except Exception as e2:
                    print(f"[Error] Failed to load LOON-UNet weights even with strict=False: {e2}")
        try:
            delta_meta = _find_meta(['delta'], opt.delta)
            print(f"[Info] Using LOON-UNet params: unet_k={unet_k}, no_pyramid={no_pyramid}, unet_ratio1={unet_ratio1}, unet_ratio2={unet_ratio2}, unet_hidden={unet_hidden}, unet_T={unet_T}, unet_K={unet_K}, f_dim={f_dim}, use_q_as_feat={use_q_as_feat}, share_offset_former={share_offset_former}, delta={delta_meta}")
        except Exception:
            pass

        try:
            print("[Info] LOON-UNet parameter snapshot:")
            try:
                fdim = getattr(bottleneck, 'f_dim', None)
                print(f"  detected bottleneck.f_dim = {fdim}")
            except Exception:
                pass
            named = dict(loon_unet.named_parameters())
            def _ps(n):
                if n in named:
                    p = named[n]
                    print(f"  {n}: shape={tuple(p.shape)}, requires_grad={p.requires_grad}")
            for key in ['bottleneck.cell.lstm_cell.weight_ih', 'bottleneck.cell.lstm_cell.weight_hh', 'bottleneck.cell.lstm_cell.bias_ih', 'bottleneck.cell.lstm_cell.bias_hh', 'enc.l4.conv.weight']:
                _ps(key)
            total = sum(p.numel() for p in loon_unet.parameters())
            trainable = sum(p.numel() for p in loon_unet.parameters() if p.requires_grad)
            print(f"  total_params={total:,}, trainable_params={trainable:,}")
            if isinstance(state, dict):
                ck_keys = set(state.keys())
                model_keys = set(loon_unet.state_dict().keys())
                missing = model_keys - ck_keys
                unexpected = ck_keys - model_keys
                print(f"  checkpoint_keys={len(ck_keys)}, model_keys={len(model_keys)}, missing_in_ckpt={len(missing)}, unexpected_in_ckpt={len(unexpected)}")
                if len(missing) > 0:
                    print("   missing (up to 10):", list(missing)[:10])
                if len(unexpected) > 0:
                    print("   unexpected (up to 10):", list(unexpected)[:10])
        except Exception as e:
            print(f"[Warn] Failed to print parameter snapshot: {e}")
        # ensure LOON-UNet and triangle net are in eval mode and frozen for inference
        loon_unet.eval()
        for p in loon_unet.parameters():
            p.requires_grad_(False)
        OffsetOPTer = None
        loon = None
    elif not opt.use_loon:
        # set the OffsetOPT optimizer
        if 'ABC' in opt.dataset:
            # switch off offset optimization, and the proposed offset initialization
            OffsetOPTer = OffsetOPT(model, loss_fn, device, maxIter=1, zero_init=True)
        else:
            OffsetOPTer = OffsetOPT(model, loss_fn, device, maxIter=100, zero_init=False)
        loon = None
        loon_unet = None
    else:
        try:
            mod = importlib.import_module('S2.LOONOptimizer')
        except Exception:
            try:
                mod = importlib.import_module('LOONOptimizer')
            except Exception as e:
                raise ImportError("Could not import LOONOptimizer. Ensure 'S2' is a package and PYTHONPATH includes project root.") from e
        _LOON = getattr(mod, 'LOONOptimizer')
        loon_hidden = 64
        loon_T = 20
        loon_k = 50
        ckpt_path = getattr(opt, 'loon_ckpt', '')
        raw_ckpt, state = None, None
        if ckpt_path and os.path.isfile(ckpt_path):
            try:
                raw_ckpt = torch.load(ckpt_path, map_location=device)
                state = raw_ckpt['model_state_dict'] if isinstance(raw_ckpt, dict) and 'model_state_dict' in raw_ckpt else raw_ckpt
            except Exception as e:
                print(f"[Warn] Failed to load LOON checkpoint file {ckpt_path}: {e}")
        def _extract_meta_dicts(obj):
            metas = []
            if isinstance(obj, dict):
                for k in ['hparams','args','opt','config','model_args','hyper_parameters','meta']:
                    if k in obj and isinstance(obj[k], dict):
                        metas.append(obj[k])
            return metas
        meta_dicts = _extract_meta_dicts(raw_ckpt) if isinstance(raw_ckpt, dict) else []
        def _find_meta(keys, default=None):
            for md in meta_dicts:
                for k in keys:
                    if k in md:
                        return md[k]
            return default
        loon_hidden = int(_find_meta(['loon_hidden','hidden','hidden_dim'], loon_hidden) or loon_hidden)
        loon_T = int(_find_meta(['loon_T','T'], loon_T) or loon_T)
        loon_k = int(_find_meta(['loon_k','k'], loon_k) or loon_k)

        loon = _LOON(hidden_dim=loon_hidden, T=loon_T, k=loon_k).to(device)
        try:
            setattr(loon, 'chunk_size', int(opt.chunk_size))
        except Exception:
            pass
        if state:
            try:
                missing, unexpected = loon.load_state_dict(state, strict=False)
                if missing:
                    print(f"[Warn] LOON load_state_dict missing keys: {len(missing)} (showing up to 10): {missing[:10]}")
                if unexpected:
                    print(f"[Warn] LOON load_state_dict unexpected keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}")
                print(f"[Info] Loaded LOON weights from {ckpt_path}")
            except Exception as e:
                print(f"[Warn] Failed to load LOON state_dict from {ckpt_path}: {e}")
        try:
            print(f"[Info] Using LOON params: hidden={loon_hidden}, T={loon_T}, k={loon_k}, delta={opt.delta}")
        except Exception:
            pass
        loon.eval()
        for p in loon.parameters():
            p.requires_grad_(False)
        try:
            fixed_knn_sets = {"ScanNet", "CARLA_1M", "Thingi10k"}
            cpu_knn_sets = {"ScanNet", "Thingi10k", "CARLA_1M", "Matterport3D", "Stanford3D"}
            if opt.dataset in fixed_knn_sets:
                setattr(loon, 'fixed_knn', True)
            if opt.dataset in cpu_knn_sets:
                setattr(loon, 'knn_on_cpu', True)
        except Exception:
            pass
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        OffsetOPTer = None
        loon_unet = None
    runtime = np.zeros(len(test_files))
    point_counts = np.zeros(len(test_files), dtype=np.int64)
    stats_file = None
    stats_path = os.path.join(results_folder, 'reconstruction_stats.txt')
    try:
        stats_file = open(stats_path, 'w')
        stats_file.write("filename,num_points,time_seconds\n")
    except Exception as e:
        print(f"[Warn] Failed to open stats file {stats_path}: {e}")
        stats_file = None
    idx = 0
    pbar = tqdm(total=len(test_files), desc=f"Reconstruct {opt.dataset}", unit="file")
    for data in testLoader:
        data = [item[0] for item in data]
        points, knn_indices, center, scale = data
        try:
            num_points = int(points.shape[0])
        except Exception:
            try:
                num_points = int(len(points))
            except Exception:
                num_points = -1
        if 0 <= idx < len(point_counts):
            point_counts[idx] = num_points
        start_time = time()
        if opt.use_loon_unet:
            points_dev = points.to(device)
            P0_b3n = points_dev.t().unsqueeze(0).contiguous()   # [1,3,N]
            with torch.no_grad():
                dP0_b3n = loon_unet(P0_b3n)                     # [1,3,N]
            final_offsets = dP0_b3n.squeeze(0).t().contiguous() # [N,3]

            knn_idx_dev = knn_indices.to(device)
            extractor = SurfExtract()
            N_total = points_dev.shape[0]
            step = max(1, opt.chunk_size)
            all_tris = []
            with torch.no_grad():
                for i in range(0, N_total, step):
                    sl = slice(i, min(i + step, N_total))
                    feat_sl = points_dev[knn_idx_dev[sl]] - points_dev[sl, None, :]
                    off_sl = final_offsets[knn_idx_dev[sl]] - final_offsets[knn_idx_dev[sl, 0], None, :]
                    logits_sl = model(feat_sl, off_sl)
                    tris_sl = extractor(points_dev, logits_sl, knn_idx_dev[sl])
                    if tris_sl.numel() > 0:
                        all_tris.append(tris_sl)
            if len(all_tris) > 0:
                pred_triangles = torch.cat(all_tris, dim=0)
                pred_triangles = torch.sort(pred_triangles, dim=-1).values
                pred_triangles = torch.unique(pred_triangles, dim=0)
            else:
                pred_triangles = torch.empty((0, 3), dtype=torch.long, device='cpu')

            pts_world = (points_dev.detach().cpu() * scale + center).numpy()
            tris_np = pred_triangles.detach().cpu().numpy() if pred_triangles.numel() > 0 else None
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(pts_world)
            mesh.triangles = o3d.utility.Vector3iVector(tris_np) if tris_np is not None else o3d.utility.Vector3iVector([])
            recon_mesh = mesh
        elif not opt.use_loon:
            recon_mesh = OffsetOPTer(points, knn_indices, center, scale, req_inter_logits=False)
        else:
            points_dev = points.to(device)
            final_offsets = loon(points_dev, model, create_graph=False)  # [N, 3]

            knn_idx_dev = knn_indices.to(device)
            extractor = SurfExtract()
            N_total = points_dev.shape[0]
            step = max(1, opt.chunk_size)
            all_tris = []
            with torch.no_grad():
                for i in range(0, N_total, step):
                    sl = slice(i, min(i + step, N_total))
                    feat_sl = points_dev[knn_idx_dev[sl]] - points_dev[sl, None, :]
                    off_sl = final_offsets[knn_idx_dev[sl]] - final_offsets[knn_idx_dev[sl, 0], None, :]
                    logits_sl = model(feat_sl, off_sl)
                    tris_sl = extractor(points_dev, logits_sl, knn_idx_dev[sl])
                    if tris_sl.numel() > 0:
                        all_tris.append(tris_sl)
            if len(all_tris) > 0:
                pred_triangles = torch.cat(all_tris, dim=0)
                pred_triangles = torch.sort(pred_triangles, dim=-1).values
                pred_triangles = torch.unique(pred_triangles, dim=0)
            else:
                pred_triangles = torch.empty((0, 3), dtype=torch.long, device='cpu')

            pts_world = (points_dev.detach().cpu() * scale + center).numpy()
            tris_np = pred_triangles.detach().cpu().numpy() if pred_triangles.numel() > 0 else None
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(pts_world)
            mesh.triangles = o3d.utility.Vector3iVector(tris_np) if tris_np is not None else o3d.utility.Vector3iVector([])
            recon_mesh = mesh
        runtime[idx] = time()-start_time
        try:
            if stats_file is not None:
                fname_log = os.path.basename(test_files[idx]) if idx < len(test_files) else f"file_{idx}"
                stats_file.write(f"{fname_log},{num_points},{float(runtime[idx]):.6f}\n")
                stats_file.flush()
        except Exception:
            pass
        try:
            method = 'LOON-UNet' if opt.use_loon_unet else ('LOON' if opt.use_loon else 'OffsetOPT')
            pbar.set_postfix({"time": f"{runtime[idx]:.2f}s", "method": method})
        except Exception:
            pass
        pbar.update(1)

        out_path = os.path.join(results_folder, '%s'%test_files[idx].split('/')[-1])
        try:
            has_tris = (hasattr(recon_mesh, 'triangles') and len(recon_mesh.triangles) > 0)
        except Exception:
            has_tris = False
        if has_tris:
            o3d.io.write_triangle_mesh(out_path, recon_mesh)
        else:
            pc = o3d.geometry.PointCloud(recon_mesh.vertices)
            pc_out = out_path.replace('.ply', '_pc.ply') if out_path.lower().endswith('.ply') else (out_path + '.pc.ply')
            o3d.io.write_point_cloud(pc_out, pc)
            print(f"[Warn] No triangles extracted for {os.path.basename(out_path)}; wrote point cloud to {os.path.basename(pc_out)} instead.")
        idx += 1
    try:
        pbar.close()
    except Exception:
        pass
    try:
        if stats_file is not None:
            stats_file.close()
            print(f"[Info] Wrote per-file stats to {stats_path}")
    except Exception:
        pass
    try:
        n_done = idx
        if n_done > 0:
            avg_time = float(runtime[:n_done].mean())
        else:
            avg_time = 0.0
        print(f"[Info] Reconstructed {n_done} files. Average time per file: {avg_time:.2f}s")
        max_points = None
        max_points_time = None
        max_points_file = None
        if n_done > 0:
            try:
                valid_points = point_counts[:n_done]
                max_idx = int(valid_points.argmax())
                max_points = int(valid_points[max_idx])
                max_points_time = float(runtime[max_idx])
                max_points_file = os.path.basename(test_files[max_idx]) if max_idx < len(test_files) else f"file_{max_idx}"
                print(f"[Info] Max points sample: {max_points_file}, num_points={max_points}, time={max_points_time:.2f}s")
            except Exception as _e_max:
                print(f"[Warn] Failed to compute max points statistics: {_e_max}")
        try:
            timings_path = os.path.join(results_folder, 'timings.txt')
            with open(timings_path, 'w') as f:
                f.write(f"avg_time_per_file_seconds,{avg_time:.6f}\n")
                if max_points is not None and max_points_time is not None and max_points_file is not None:
                    f.write(f"max_points,{max_points},max_points_file,{max_points_file},max_points_time_seconds,{max_points_time:.6f}\n")
                f.write("filename,num_points,time_seconds\n")
                for i in range(n_done):
                    fname = os.path.basename(test_files[i]) if i < len(test_files) else f"file_{i}"
                    pts = int(point_counts[i]) if i < len(point_counts) else -1
                    f.write(f"{fname},{pts},{float(runtime[i]):.6f}\n")
        except Exception:
            pass
    except Exception:
        pass

        
