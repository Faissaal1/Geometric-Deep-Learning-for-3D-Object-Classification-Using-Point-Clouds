import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import PointNetClassifier
from dataset import DummyPointCloudDataset
from eval import evaluate

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = PointNetClassifier().to(device)

    train_dataset = DummyPointCloudDataset(split='train')
    test_dataset = DummyPointCloudDataset(split='test')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for data, label in train_loader:
            print("Data shape before transpose:", data.shape)
            print("Label shape:", label.shape)

            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, label.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/10], Loss: {total_loss/len(train_loader):.4f}")
        evaluate(model, test_loader, device)
if __name__ == "__main__":
    train()
