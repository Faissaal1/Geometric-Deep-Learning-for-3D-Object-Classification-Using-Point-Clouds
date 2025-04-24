import torch
import numpy as np
import open3d as o3d
from sklearn.metrics import accuracy_score

def visualize_point_cloud(points, label=None, pred=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    color = [0, 1, 0]  
    pcd.paint_uniform_color(color)
    o3d.visualization.draw_geometries([pcd])

def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.append(predicted.cpu())
            all_labels.append(label.squeeze().cpu())

            if i == 0:
                for j in range(3):
                    pc_np = data[j].cpu().numpy()
                    gt = label[j].item()
                    pred = predicted[j].item()
                    print(f"Sample {j}: GT={gt}, Pred={pred}")
                    visualize_point_cloud(pc_np, label=gt, pred=pred)
                    visualize_point_cloud(pc_np, label=1, pred=predicted[j].item())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = accuracy_score(labels, preds)
    print(f"Test Accuracy: {acc * 100:.2f}%\n")
