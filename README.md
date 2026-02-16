# TensorStream

TensorStream is a PyTorch + C++ extension project for accelerating LiDAR point-cloud preprocessing. The repository focuses on two core operations implemented in C++ and exposed to Python:

- Loading KITTI `.bin` point-cloud files into tensors.
- Voxel-grid downsampling on CPU.

These ops are integrated into a training/evaluation pipeline built around a PointNet classifier.

---

## What this repository contains

### Core C++ extension
The extension is defined in `setup.py` and implemented in `src/voxel_ops.cpp`.

It exposes:

- `tensorstream_ops.load_kitti_bin(path)` → returns `(N, 4)` float tensor (`x, y, z, intensity`).
- `tensorstream_ops.voxel_downsample(points_xyz, voxel_size)` → returns `(M, 3)` tensor of voxel centroids.

### Python training stack
- `dataset.py`: `Kitti3D` dataset class that uses the C++ extension for loading + downsampling and parses KITTI labels into a binary car/non-car target.
- `pointnet.py`: PointNet model implementation used for classification.
- `train_full.py`: training loop, batching/collation logic, checkpoint saving.
- `evaluate.py`: validation/inference metrics (accuracy, classification report, confusion matrix, latency per sample).
- `benchmark.py`: compares NumPy voxel downsampling against the C++ extension.

### Frontend prototype
`frontend/index.html` is a standalone UI prototype for uploading `.bin` files and visualizing processing stats. It expects a `/process_lidar` backend endpoint, which is **not included** in this repository.

---

## Project layout

```text
TensorStream/
├── src/
│   └── voxel_ops.cpp
├── frontend/
│   └── index.html
├── dataset.py
├── pointnet.py
├── train_full.py
├── evaluate.py
├── benchmark.py
├── setup.py
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.9+
- PyTorch
- A C++17-capable compiler
  - Linux: `g++`/`clang++`
  - Windows: MSVC Build Tools
- `ninja` (recommended for faster extension builds)

Install Python dependencies:

```bash
pip install -r requirements.txt
pip install scikit-learn ninja
```

> Note: `scikit-learn` is required by `evaluate.py` but is not currently pinned in `requirements.txt`.

---

## Build the C++ extension

From the repository root:

```bash
python setup.py install
```

After a successful build, you should be able to import:

```python
import tensorstream_ops
```

---

## Dataset expectations (KITTI format)

`dataset.py` expects this structure under your dataset root:

```text
<DATA_DIR>/
├── velodyne/
│   ├── 000000.bin
│   ├── 000001.bin
│   └── ...
└── label_2/
    ├── 000000.txt
    ├── 000001.txt
    └── ...
```

Each sample is converted into a binary label:

- `1` if any object of type `Car` exists in label file.
- `0` otherwise.

The current training/evaluation scripts contain a hardcoded `DATA_DIR`; update this path before running.

---

## Run benchmark

```bash
python benchmark.py
```

This script generates synthetic 3D points and reports timing for:

- NumPy voxel downsampling (baseline)
- C++ extension voxel downsampling

---

## Train

1. Edit `DATA_DIR` in `train_full.py`.
2. Run:

```bash
python train_full.py
```

By default, checkpoints are saved as:

- `pointnet_epoch_1.pth`
- ...
- `pointnet_epoch_10.pth`

---

## Evaluate

1. Edit `DATA_DIR` in `evaluate.py`.
2. Confirm `MODEL_PATH` points to a valid checkpoint.
3. Run:

```bash
python evaluate.py
```

The script reports:

- Accuracy
- Classification report
- Confusion matrix
- Average per-sample inference latency and throughput

---

## Common issues

- **`ModuleNotFoundError: tensorstream_ops`**
  - Re-run `python setup.py install` and ensure you are in the same Python environment.

- **Compiler/toolchain build errors**
  - Verify C++ toolchain installation and PyTorch compatibility.

- **Evaluation dependency errors (`sklearn`)**
  - Install with `pip install scikit-learn`.

- **Dataset path errors**
  - Update hardcoded `DATA_DIR` values in `train_full.py` and `evaluate.py`.

---

## Current status and scope

This repository demonstrates the core acceleration path (C++ extension + PyTorch integration) and model training scripts. It does not yet include:

- A production FastAPI backend for `frontend/index.html`
- Config-driven training/evaluation (paths are currently hardcoded)
- Automated test suite/CI

---

## License

No license file is currently present in this repository. Add one before public distribution.
