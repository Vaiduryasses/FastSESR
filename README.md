# FastSESR: Fast Scene-level Explicit Surface Reconstruction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.6.0+-orange.svg" alt="PyTorch">
</p>

FastSESR is a deep learning-based point cloud surface reconstruction framework that employs a two-stage (S1/S2) training strategy to achieve efficient mesh reconstruction from point clouds. The project supports multiple datasets and provides complete training, evaluation, and reconstruction pipelines.

## 📋 Table of Contents

- [Environment Setup](#-environment-setup)
- [Dependencies](#-dependencies)
- [Dataset Preparation](#-dataset-preparation)
- [Stage 1 Training](#-stage-1-training)
- [Stage 2 Training](#-stage-2-training)
- [Pre-trained Models](#-pre-trained-models)
- [Reconstruction & Evaluation](#-reconstruction--evaluation)
- [Project Structure](#-project-structure)

---

## 🛠 Environment Setup
### 1. Create Conda Virtual Environment

```bash
conda create -n fastsesr python=3.10
conda activate fastsesr
```

### 2. Install PyTorch

Install PyTorch according to your CUDA version (PyTorch 2.6.0+ recommended). We use cuda 11.8 with Pytorch 2.6.0:

```bash
# CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install PyTorch3D

PyTorch3D is required for LOON-UNet's kNN operations and Chamfer distance computation:

```bash
# Using conda (recommended)
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

# Or install from source
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### 4. Install Other Dependencies

```bash
pip install -r requirements.txt
```

---

## 📁 Dataset Preparation

### Dataset Directory Structure

The expected data directory structure is as follows:

```
Data/
├── ABC/
│   ├── train/                    # ABC training set (PLY format)
│   │   ├── 00000001.ply
│   │   ├── 00000002.ply
│   │   └── ...
│   └── test/                     # ABC test set
│       ├── 00000501.ply
│       └── ...
├── PointClouds/
│   ├── FAUST/                    # FAUST dataset point clouds
│   │   ├── tr_reg_000.ply
│   │   └── ...
│   ├── MGN/                      # MGN dataset point clouds
│   └── <other_datasets>/
└── GT_Meshes/
    ├── FAUST/                    # FAUST Ground Truth meshes
    │   ├── tr_reg_000.ply
    │   └── ...
    └── <other_datasets>/
```

### Dataset Download

The pre-processed datasets can be obtained from the [OffsetOPT](https://github.com/EnyaHermite/OffsetOPT) repository (CVPR 2025).

| Dataset | Description | Source |
|---------|-------------|--------|
| **ABC** | CAD model dataset for S1 training | [OffsetOPT](https://github.com/EnyaHermite/OffsetOPT) |
| **FAUST** | Human body scan dataset | [OffsetOPT](https://github.com/EnyaHermite/OffsetOPT) |
| **MGN** | Multi-garment human dataset | [OffsetOPT](https://github.com/EnyaHermite/OffsetOPT) |

Please refer to the OffsetOPT repository for detailed download instructions and data preparation guidelines.

### Data Preprocessing

Ensure all point cloud files are in `.ply` format. For format conversion, you can use Open3D:

```python
import open3d as o3d

mesh = o3d.io.read_triangle_mesh("input.obj")
pcd = mesh.sample_points_uniformly(number_of_points=100000)
o3d.io.write_point_cloud("output.ply", pcd)
```

---

## 🎯 Stage 1 Training

Stage 1 trains the base triangle classification network using the ABC dataset.

### Training Command

```bash
python S1_train.py \
    --gpu 0 \
    --max_epoch 301 \
    --use_pair_lowrank 1 \
    --pair_rank 32 \
    --pair_alpha 0.5 \
    --pair_bias 0.0
```

### Parameter Description

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gpu` | 0 | GPU device ID |
| `--max_epoch` | 301 | Maximum number of training epochs |
| `--ckpt_path` | None | Checkpoint path to resume training |
| `--use_pair_lowrank` | 0 | Whether to use low-rank pair decomposition (0/1) |
| `--pair_rank` | 32 | Rank for low-rank decomposition |
| `--pair_alpha` | 0.5 | Initial pair_alpha value |
| `--pair_bias` | 0.0 | Initial pair_bias value |

### Training Output

Trained models and logs are saved in:

```
S1_training/
└── model_k50/
    ├── log_train.txt           # Training log
    ├── best_model              # Best model checkpoint
    └── ckpt_epoch_*.pth        # Epoch checkpoints
```

### Resume Training from Checkpoint

```bash
python S1_train.py \
    --gpu 0 \
    --max_epoch 301 \
    --ckpt_path S1_training/model_k50/ckpt_epoch_100.pth
```

---

## 🚀 Stage 2 Training

Stage 2 uses LOON-UNet for multi-scale offset learning.

### Dataset Splitting

#### Step 1: Generate Fixed Split Configurations

Use `generate_fixed_splits.py` to generate reproducible K-fold splits:

```bash
python scripts/generate_fixed_splits.py \
    --data_root /path/to/Data \
    --datasets FAUST MGN \
    --split_names Split-A Split-B Split-C \
    --seeds 202401 202402 202403
```

This will generate split configuration files under the `splits/<dataset>/` directory.

#### Step 2: Convert JSON Splits to File Lists

```bash
python scripts/convert_json_splits_to_kfold_lists.py --dataset FAUST
```

Generated directory structure:

```
splits/
└── FAUST/
    ├── Split-A/
    │   ├── fold_0.json
    │   └── ...
    ├── fold_Split-A_fold_0/
    │   ├── train_list.txt
    │   ├── val_list.txt
    │   └── test_list.txt
    └── ...
```

### K-Fold Cross-Validation Training

```bash
python scripts/kfold_runner.py \
    --dataset FAUST \
    --data_root /path/to/Data \
    --epochs 30 \
    --gpu 0 \
    --splits_root splits \
    --train_script S2/S2_train_loon_unet.py \
    --chunk_size 2000
```

#### Parameter Description

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | (required) | Dataset name (e.g., FAUST, MGN) |
| `--data_root` | (required) | Data root directory |
| `--epochs` | 30 | Training epochs per fold |
| `--gpu` | 0 | GPU device ID |
| `--splits_root` | splits | Split configuration directory |
| `--chunk_size` | 2000 | Chunk size (reduces memory usage) |
| `--use_loon_unet` | False | Use LOON-UNet for reconstruction |
| `--resume` | False | Skip completed folds |

### LOSO (Leave-One-Subject-Out) Training

Suitable for scenarios requiring leave-one-out validation:

```bash
python scripts/loso_runner.py \
    --dataset FAUST \
    --data_root /path/to/Data \
    --epochs 30 \
    --gpu 0 \
    --val_ratio 0.2
```

### Direct Training with S2_train_loon_unet.py

For the ABC dataset or custom training, you can directly call the training script:

```bash
python S2/S2_train_loon_unet.py \
    --dataset ABC \
    --data_root /path/to/Data \
    --train_list /path/to/train_list.txt \
    --val_list /path/to/val_list.txt \
    --epochs 30 \
    --gpu 0 \
    --batch_size 1 \
    --lr 0.001 \
    --save_dir runs/ABC_train
```

#### Full Parameter List

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | (required) | Dataset name |
| `--data_root` | Data | Data root directory |
| `--train_list` | "" | Training sample list file |
| `--val_list` | "" | Validation sample list file |
| `--test_list` | "" | Test sample list file |
| `--split_config` | "" | JSON format split configuration file |
| `--epochs` | 30 | Number of training epochs |
| `--batch_size` | 1 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--weight_decay` | 0.0 | Weight decay |
| `--gpu` | 0 | GPU device ID |
| `--seed` | 42 | Random seed |
| `--delta` | 0.0 | Surface voxel size |
| `--rescale_delta` | False | Whether to rescale delta based on model scale |
| `--unet_k` | 16 | DGCNN encoder K nearest neighbors |
| `--unet_hidden` | 64 | Bottleneck hidden dimension |
| `--unet_T` | 3 | LOON iteration steps |
| `--unet_K` | 50 | Triangle network KNN count |
| `--save_dir` | "" | Model save directory |
| `--amp` | False | Enable mixed precision training |

---

## 💾 Pre-trained Models

Pre-trained Stage 1 models are saved in the `trained_models/` directory:

```
trained_models/
└── model_knn50.pth              # Pre-trained model with KNN=50
```

### Model Loading

Stage 2 training will automatically load pre-trained weights from `trained_models/model_knn{K}.pth`. Ensure this file exists:

```python
# Check pre-trained model
import os
assert os.path.exists('trained_models/model_knn50.pth'), "Pre-trained model not found!"
```

### Using Self-trained S1 Model

If using your own trained S1 model, S2 will automatically look for `S1_training/model_k{knn}/best_model`:

```bash
# After S1 training completes, the model is located at
S1_training/model_k50/best_model
```

---

## 🔍 Reconstruction & Evaluation

### Reconstruction with LOON-UNet

```bash
python S2_reconstruct.py \
    --dataset FAUST \
    --data_root /path/to/Data \
    --use_loon_unet \
    --loon_unet_ckpt runs/kfold/FAUST/fold_0/save/loon_unet_best.pth \
    --gpu 0 \
    --chunk_size 2000 \
    --out_dir results/FAUST
```

### Reconstruction with OffsetOPT (Traditional Method)

```bash
python S2_reconstruct.py \
    --dataset ABC \
    --data_root /path/to/Data \
    --gpu 0 \
    --out_dir results/ABC
```

### Evaluate Reconstruction Quality

```bash
python main_eval_acc.py \
    --gt_path /path/to/Data/GT_Meshes/FAUST \
    --pred_path results/FAUST \
    --sample_num 100000
```

### Batch Evaluation for Multiple Folds

```bash
python scripts/eval_multi.py \
    --gt_path /path/to/Data/GT_Meshes/FAUST \
    --pred_paths results/FAUST_fold0 results/FAUST_fold1 results/FAUST_fold2 \
    --csv_out results/metrics.csv
```

---

## 📂 Project Structure

```
FastSESR/
├── S1/                           # Stage 1 modules
│   ├── BaseNet.py                # S1 base network (DGCNN + GNN)
│   ├── loss_supervised.py        # Supervised loss function
│   └── fitModel.py               # Training utility class
├── S2/                           # Stage 2 modules
│   ├── LoonUNet.py               # LOON-UNet network architecture
│   ├── ReconNet.py               # Reconstruction network (inherits from S1)
│   ├── ExtractFace.py            # Triangle face extraction
│   ├── offset_opt.py             # Offset optimizer
│   ├── loss_unsupervised.py      # Unsupervised loss
│   └── S2_train_loon_unet.py     # S2 training script
├── dataset/                      # Dataset loaders
│   ├── mesh_train.py             # Mesh training dataset
│   ├── pc_recon.py               # Point cloud reconstruction dataset
│   └── pc_recon_with_gt.py       # Point cloud dataset with GT
├── scripts/                      # Utility scripts
│   ├── generate_fixed_splits.py  # Generate split configurations
│   ├── convert_json_splits_to_kfold_lists.py  # Convert split format
│   ├── kfold_runner.py           # K-fold training orchestration
│   ├── loso_runner.py            # LOSO training orchestration
│   └── eval_multi.py             # Batch evaluation
├── trained_models/               # Pre-trained models
│   └── model_knn50.pth
├── utils/                        # Utility functions
│   └── augmentor.py              # Data augmentation
├── eval/                         # Evaluation tools
├── S1_train.py                   # Stage 1 training entry
├── S2_reconstruct.py             # Reconstruction entry
├── main_eval_acc.py              # Evaluation entry
└── README.md
```

---

## 🙏 Acknowledgements

This project is based on the work of [OffsetOPT](https://github.com/EnyaHermite/OffsetOPT) (CVPR 2025). We thank the authors for providing the datasets and baseline implementation.

