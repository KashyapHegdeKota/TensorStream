#include <torch/extension.h>
#include <vector>
#include <map>

// C++ function to perform Voxel Grid Downsampling
// Input:  points (N, 3) tensor of floats
//         voxel_size (float)
// Output: (M, 3) tensor of downsampled points
torch::Tensor voxel_downsample_cpu(torch::Tensor points, float voxel_size) {
    
    // 1. Input Validation
    auto opts = points.options();
    int num_points = points.size(0);
    
    // Accessor for fast raw data access (Generic implementation for CPU)
    auto points_acc = points.accessor<float, 2>();

    // 2. The Algorithm: Use a map to store the sum of points in each voxel grid
    // Key: Voxel Index (flattened or tuple), Value: {SumX, SumY, SumZ, Count}
    // Note: In production, we'd use a flat hash map or CUDA, but std::map is simpler for a demo.
    std::map<std::tuple<int, int, int>, std::vector<float>> grid;

    // "Performance-Critical" Loop
    for (int i = 0; i < num_points; ++i) {
        float x = points_acc[i][0];
        float y = points_acc[i][1];
        float z = points_acc[i][2];

        // Compute Voxel Coordinate
        int vx = static_cast<int>(std::floor(x / voxel_size));
        int vy = static_cast<int>(std::floor(y / voxel_size));
        int vz = static_cast<int>(std::floor(z / voxel_size));

        auto key = std::make_tuple(vx, vy, vz);

        // Aggregate stats
        if (grid.find(key) == grid.end()) {
            grid[key] = {x, y, z, 1.0f};
        } else {
            grid[key][0] += x;
            grid[key][1] += y;
            grid[key][2] += z;
            grid[key][3] += 1.0f;
        }
    }

    // 3. Compute Centroids (Mean)
    std::vector<float> downsampled_data;
    downsampled_data.reserve(grid.size() * 3);

    for (auto const& [key, val] : grid) {
        float count = val[3];
        downsampled_data.push_back(val[0] / count); // Mean X
        downsampled_data.push_back(val[1] / count); // Mean Y
        downsampled_data.push_back(val[2] / count); // Mean Z
    }

    // 4. Convert back to Tensor
    int output_size = downsampled_data.size() / 3;
    auto output_tensor = torch::from_blob(downsampled_data.data(), {output_size, 3}, torch::kFloat).clone();
    
    return output_tensor;
}

// Bindings to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("voxel_downsample", &voxel_downsample_cpu, "Voxel Grid Downsampling (CPU)");
}