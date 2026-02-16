import torch
from torch.utils.data import Dataset
import tensorstream_ops  # Your C++ Extension
import os
import numpy as np

class Kitti3D(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root = root_dir
        self.split = split
        
        # Get all file indices (e.g., "000001")
        self.image_ids = [x.split('.')[0] for x in os.listdir(os.path.join(root_dir, 'velodyne'))]
        
        # Simple Split: First 90% train, last 10% val
        split_idx = int(len(self.image_ids) * 0.9)
        if split == 'train':
            self.image_ids = self.image_ids[:split_idx]
        else:
            self.image_ids = self.image_ids[split_idx:]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # 1. LOAD LIDAR (Via C++ TensorStream) - FAST!
        bin_path = os.path.join(self.root, 'velodyne', f'{img_id}.bin')
        points = tensorstream_ops.load_kitti_bin(bin_path)
        
        # 2. DOWNSAMPLE (Via C++) - FAST!
        # We drop intensity for geometry-only processing, or keep it if model supports it
        xyz = points[:, :3]
        processed_points = tensorstream_ops.voxel_downsample(xyz, 0.5) # 0.5m Voxel
        
        # 3. LOAD LABEL (Python is fine here, text files are tiny)
        label_path = os.path.join(self.root, 'label_2', f'{img_id}.txt')
        target = self.parse_label(label_path)
        
        return processed_points, target

    def parse_label(self, label_path):
        # KITTI Label Format: type, truncated, occluded, alpha, bbox(4), dim(3), loc(3), rot_y
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        has_car = 0
        for line in lines:
            elements = line.split()
            obj_type = elements[0]
            if obj_type == 'Car':
                has_car = 1
                break 
        
        # For a simple binary classifier (Does this scan have a car?)
        return torch.tensor(has_car, dtype=torch.float32)

# Usage
# ds = Kitti3D(root_dir='./kitti_dataset/training')