import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetClassifier(nn.Module):
    def __init__(self, k=40):
        super(PointNetClassifier, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1) 
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        print("Input shape:", x.shape)
        assert x.shape[2] == 3, "Input must have 3 features per point"

        x = x.transpose(1, 2)
        print("Shape after transpose:", x.shape)

        x = F.relu(self.bn1(self.conv1(x)))  
        print("Shape after conv1:", x.shape)

        x = F.relu(self.bn2(self.conv2(x)))  
        print("Shape after conv2:", x.shape)

        x = F.relu(self.bn3(self.conv3(x)))  
        print("Shape after conv3:", x.shape)

        x = torch.max(x, dim=2)[0]  
        print("Shape after max pooling:", x.shape)

        x = F.relu(self.bn4(self.fc1(x))) 
        print("Shape after fc1:", x.shape)

        x = F.relu(self.bn5(self.dropout(self.fc2(x)))) 
        print("Shape after fc2:", x.shape)

        return self.fc3(x)  