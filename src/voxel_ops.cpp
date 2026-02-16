#include <torch/extension.h>
#include <vector>
#include <map>
#include <fstream>

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

torch::Tensor load_kitti_bin(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.good()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // Get file size
    input.seekg(0, std::ios::end);
    size_t size = input.tellg();
    input.seekg(0, std::ios::beg);

    // Calculate number of float32s (each point has 4 floats: x, y, z, intensity)
    size_t num_floats = size / sizeof(float);
    size_t num_points = num_floats / 4;

    // Allocate buffer and read
    std::vector<float> buffer(num_floats);
    input.read(reinterpret_cast<char*>(buffer.data()), size);
    input.close();

    // Convert to Tensor (N, 4)
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    // Clone is necessary here to own the memory in Python
    return torch::from_blob(buffer.data(), {static_cast<long>(num_points), 4}, opts).clone();
}

// Python Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("voxel_downsample", &voxel_downsample_cpu, "Voxel Downsample (CPU)");
  m.def("load_kitti_bin", &load_kitti_bin, "Load KITTI Binary File");
}