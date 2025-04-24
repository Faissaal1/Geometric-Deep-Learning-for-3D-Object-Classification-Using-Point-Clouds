import torch
from torch.utils.data import Dataset
import numpy as np

class DummyPointCloudDataset(Dataset):
    def __init__(self, num_samples=1000, num_points=1024, num_classes=40, split='train'):
        super(DummyPointCloudDataset, self).__init__()
        self.num_samples = num_samples
        self.num_points = num_points
        self.num_classes = num_classes

        np.random.seed(42)  
        all_data = np.random.rand(num_samples, num_points, 3).astype(np.float32)
        all_labels = np.random.randint(0, num_classes, size=(num_samples, 1)).astype(np.int64)

        split_ratio = 0.8
        split_index = int(num_samples * split_ratio)

        if split == 'train':
            self.data = all_data[:split_index]
            self.labels = all_labels[:split_index]
        else:
            self.data = all_data[split_index:]
            self.labels = all_labels[split_index:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point_cloud = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(point_cloud), torch.tensor(label)
