import torch
import numpy as np
import time
import tensorstream_ops 
def numpy_voxel_downsample(points, voxel_size):
    """
    Standard Python/Numpy implementation (The Baseline)
    """
    coords = np.floor(points / voxel_size).astype(int)
    _, indices = np.unique(coords, axis=0, return_inverse=True)
    
    # This part is notoriously slow in Python without specialized libraries
    unique_indices = np.unique(indices)
    output = []
    for idx in unique_indices:
        # Boolean masking is expensive!
        mask = (indices == idx)
        voxel_points = points[mask]
        output.append(np.mean(voxel_points, axis=0))
    return np.array(output)

def run_benchmark():
    # 1. Generate Fake LiDAR Data (100k points)
    N = 100000 
    points_np = np.random.rand(N, 3).astype(np.float32) * 100.0
    points_torch = torch.from_numpy(points_np)
    voxel_size = 0.5

    print(f"Benchmarking Voxel Downsampling on {N} points...")

    # --- Benchmark Python/Numpy ---
    start_time = time.time()
    _ = numpy_voxel_downsample(points_np, voxel_size)
    py_time = time.time() - start_time
    print(f"Python (NumPy) Time: {py_time:.4f} seconds")

    # --- Benchmark C++ Extension ---
    start_time = time.time()
    # Call your custom C++ function
    output_tensor = tensorstream_ops.voxel_downsample(points_torch, voxel_size)
    cpp_time = time.time() - start_time
    print(f"C++ Extension Time:  {cpp_time:.4f} seconds")

    # --- Results ---
    speedup = py_time / cpp_time
    print("-" * 30)
    print(f"Speedup: {speedup:.2f}x Faster")
    print("-" * 30)

    # Verify Output Shape (Sanity Check)
    print(f"Original Points: {N}")
    print(f"Downsampled Points: {output_tensor.shape[0]}")

if __name__ == "__main__":
    run_benchmark()