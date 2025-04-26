import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from model import DGCNN
from dataset import ModelNet40Dataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 40
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    save_path = "./checkpoints"


    os.makedirs(save_path, exist_ok=True)

    data_path_train = r"c:\Users\faiss\Desktop\Geometric-Deep-Learning-for-3D-Object-Classification-Using-Point-Clouds-and-Meshes-\ModelNet40_npy"
    data_path_test = r"c:\Users\faiss\Desktop\Geometric-Deep-Learning-for-3D-Object-Classification-Using-Point-Clouds-and-Meshes-\ModelNet40_npy"

    train_dataset = ModelNet40Dataset(data_path=data_path_train, split='train')
    test_dataset = ModelNet40Dataset(data_path=data_path_test, split='test')    


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    model = DGCNN(k=num_classes).to(device)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    train_losses = []
    test_accuracies = []
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for points, labels in progress_bar:
            points, labels = points.to(device), labels.squeeze().to(device)  
            optimizer.zero_grad()

            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for points, labels in test_loader:
                points, labels = points.to(device), labels.squeeze().to(device)
                outputs = model(points)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)

        print(f"\nEpoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Test Accuracy: {accuracy:.2f}%")

     
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
            print(f"Saved new best model with accuracy {best_accuracy:.2f}%")

  
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
