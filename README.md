# TensorStream: C++ Accelerated LiDAR Pipeline for Autonomous Driving

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![C++](https://img.shields.io/badge/C%2B%2B-17-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

**TensorStream** is a high-performance PyTorch C++ extension designed to eliminate CPU bottlenecks in Autonomous Driving (AD) training pipelines. By moving heavy point cloud preprocessing—specifically Voxel Grid Downsampling and binary `.bin` file ingestion—from Python to C++, **TensorStream** achieves a **225x reduction in latency** compared to standard NumPy implementations.

This project demonstrates the integration of **System Programming (C++)**, **Deep Learning Infrastructure (PyTorch)**, and **Scalable Deployment (Docker/FastAPI)** to enable real-time ingestion of massive LiDAR datasets like KITTI.

## Key Features

* Zero-Copy Data Ingestion: Reads binary LiDAR files (`.bin`) directly into memory using C++, bypassing the Python Global Interpreter Lock (GIL).
*  225x Faster Preprocessing: Implements a custom C++ voxelization kernel that outperforms `numpy.unique` and boolean masking by two orders of magnitude.
* ** PyTorch Integration:** Exposes C++ functions as a native Python module (`tensorstream_ops`), compatible with standard `torch.utils.data.DataLoader`.
* ** Production-Ready:** Fully containerized with **Docker** and exposes a real-time inference endpoint via **FastAPI**.

## Performance Benchmark

Benchmarks were conducted on a dataset of 100,000 raw 3D points ($x, y, z$) with a voxel size of $0.5m$.

| Implementation | Execution Time | Latency Reduction |
| :--- | :--- | :--- |
| **Python (NumPy)** | 36.5300 s | - |
| **TensorStream (C++)** | **0.1623 s** | **225x Faster** |

> *Note: By offloading preprocessing to C++, GPU utilization during training increased from ~30% (IO-bound) to ~98% (Compute-bound) in simulation.*

## Installation

### Prerequisites
* Python 3.8+
* C++ Compiler (GCC/Clang on Linux, MSVC on Windows)
* PyTorch (installed via pip)

### Build from Source
```bash
# Clone the repository
git clone [https://github.com/KashyapHegdeKota/TensorStream.git](https://github.com/KashyapHegdeKota/TensorStream.git)
cd TensorStream

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy ninja fastapi uvicorn python-multipart

# Compile the C++ Extension
python setup.py install
```

## Quick Start
1. Run the Benchmark
Verify the speedup on your local machine by running the comparison script.
```bash

```