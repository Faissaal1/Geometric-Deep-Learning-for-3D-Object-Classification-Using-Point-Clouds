import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
class ModelNet40Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, split='train'):
        self.filepaths = []
        self.labels = []
        classes = sorted(os.listdir(data_path))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        
        for cls in classes:
            cls_folder = os.path.join(data_path, cls, split)
            if not os.path.isdir(cls_folder):
                continue
            for file in os.listdir(cls_folder):
                if file.endswith('.npy'):
                    self.filepaths.append(os.path.join(cls_folder, file))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        pointcloud = np.load(self.filepaths[idx])  
        label = self.labels[idx]
        return torch.from_numpy(pointcloud).float(), label
