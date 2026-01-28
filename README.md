# FastSESR: Fast Scene-level Explicit Surface Reconstruction
<img src="./framework.png" alt="主要示意图" width="700"/>

FastSESR is a point cloud surface reconstruction framework that employs a two-stage  training strategy to achieve efficient mesh reconstruction from point clouds. The project supports multiple datasets and provides complete training, evaluation, and reconstruction pipelines.

## 📋 Table of Contents

- [Environment Setup](#-environment-setup)
- [Dataset Preparation](#-dataset-preparation)
- [Training](#-training)
- [Pre-trained Models](#-pre-trained-models)
- [Reconstruction & Evaluation](#-reconstruction--evaluation)

---

## 🛠 Environment Setup
### 1. Create Conda Virtual Environment

```bash
conda create -n fastsesr python=3.10
conda activate fastsesr
```

### 2. Install PyTorch

Install PyTorch for your CUDA version (PyTorch 2.6.0+ is recommended). We use cuda 11.8 with Pytorch 2.6.0:

```bash
# CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install PyTorch3D

PyTorch3D is required for kNN operations and Chamfer distance computation:

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

Please refer to the OffsetOPT repository for detailed download instructions and data preparation guidelines. Note that you should replace CARLA with CARLA_1M in GT_Meshes

---
## 🚀 Training
### 1.Triangle Candidate Network (TCN)

We train TCN with the ABC dataset:

```bash
python S1_train.py \
    --gpu 0 \
    --max_epoch 300 \
    --use_pair_lowrank 1 \
```

Trained models and logs are saved in:

```
S1_training/
└── model_k50/
    ├── log_train.txt           # Training log
    ├── best_model              # Best model checkpoint
    └── ckpt_epoch_*.pth        # Epoch checkpoints
```
---

### 2.Offset Optimization Network (OON)


#### Dataset Splitting

##### Step 1: Generate Fixed Split Configurations

Use `generate_fixed_splits.py` to generate reproducible K-fold splits:

```bash
python scripts/generate_fixed_splits.py  --data_root ./Data --datasets <DATASET>
```

This will generate split configuration files under the `splits/<dataset>/` directory.

##### Step 2: Convert JSON Splits to File Lists

```bash
python scripts/convert_json_splits_to_kfold_lists.py --dataset <DATASET>
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

#### K-Fold Cross-Validation Training


```bash
#ABC
python S2/S2_train_loon_unet.py --delta 0.0 --dataset ABC_train --data_root ./Data --gpu <GPU> --epochs 1 --lr 1e-3 --unet_T 2 --unsup 
#CARLA
python scripts/kfold_runner.py   --dataset CARLA_1M   --data_root ./Data   --epochs 30   --gpu <GPU>   --use_loon_unet   --extra_train_args " --unet_k 16 --unsup --unsup_max_points 20000 --chunk_size 3000 --unet_T 2 --delta 0.1"
#Matterport3D
python scripts/kfold_runner.py   --dataset Matterport3D   --data_root ./Data   --epochs 30   --gpu <GPU>   --use_loon_unet   --extra_train_args " --unet_k 16 --unsup --unsup_max_points 20000 --chunk_size 3000 --unet_T 2 --delta 0.1"
#ScanNet
python3 scripts/kfold_runner.py   --dataset ScanNet   --data_root ./Data   --epochs 30   --gpu <GPU>   --use_loon_unet   --extra_train_args " --unet_k 16 --unsup --unsup_max_points 20000 --chunk_size 3000 --unet_T 2 --delta 0.02”
```

---

## 💾 Pre-trained Models

Pre-trained TCN models are saved in the `trained_models/` directory:

```
trained_models/
└── model_knn50.pth              # Pre-trained model with KNN=50
```

If using your own trained TCN model, S2 will automatically look for `S1_training/model_k{knn}/best_model`. We also provide pretrained OON models onthe  ABC dataset in `runs/S2_train/model_best.pth`

---

## 🔍 Reconstruction & Evaluation

### Reconstruction 
For shape-level datasets, you need to reconstruct the surface with the commands below; for scene-level datasets, `\kfold_runer.py`can reconstruct the surface automatically:
```bash
#ABC
python S2_reconstruct.py --use_loon_unet --loon_unet_ckpt runs/S2_train/<TIME>/model_best.pth   --dataset ABC_test  --gpu <GPU>
#FAUST
python S2_reconstruct.py --use_loon_unet --loon_unet_ckpt runs/S2_train/<TIME>/model_best.pth   --dataset FAUST  --gpu <GPU>
#MGN
python S2_reconstruct.py --use_loon_unet --loon_unet_ckpt runs/S2_train/<TIME>/model_best.pth   --dataset MGN  --gpu <GPU>
```

### Evaluate Reconstruction Quality

  - shape evaluation (sample 100K points): 
    ```bash
    # ABC: 
    python main_eval_acc.py --gt_path=./Data/ABC/test --pred_path=./results/ABC_test

    # FAUST:
    python main_eval_acc.py --gt_path=./Data/GT_Meshes/FAUST --pred_path=./results/FAUST

    # MGN:
    python main_eval_acc.py --gt_path=./Data/GT_Meshes/MGN --pred_path=./results/MGN
    ```

  - scene evaluation (sample 1 Million points): 
    ```bash
    # ScanNet:
    python main_eval_acc.py --gt_path=./Data/GT_Meshes/ScanNet --pred_path=./results/ScanNet/Split-A --sample_num=1000000 

    # Matterport3D:
    python main_eval_acc.py --gt_path=./Data/GT_Meshes/Matterport3D --pred_path=./results/Matterport3D/Split-A --sample_num=1000000 

    # CARLA:
    python main_eval_acc.py --gt_path=./Data/GT_Meshes/CARLA_1M --pred_path=./results/CARLA_1M/Split-A --sample_num=1000000 
    ```

---

## 🙏 Acknowledgements

This project is based on the work of [OffsetOPT](https://github.com/EnyaHermite/OffsetOPT) (CVPR 2025). We thank the authors for providing the datasets and baseline implementation.

