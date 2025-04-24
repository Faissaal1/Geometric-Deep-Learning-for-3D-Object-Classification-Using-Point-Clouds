import torch
import open3d as o3d
import numpy as np
from model import PointNetClassifier
from eval import visualize_point_cloud

pcd = o3d.io.read_point_cloud("image/Christmas Bear.ply")
points = np.asarray(pcd.points)  
print("Shape of point cloud:", points.shape)

num_points = 1024
if points.shape[0] > num_points:
    indices = np.random.choice(points.shape[0], num_points, replace=False)
    points = points[indices]
elif points.shape[0] < num_points:
    raise ValueError("Le nuage de points contient moins de 1024 points.")

points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)  # Ajouter une dimension batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointNetClassifier().to(device)
model.eval()

points_tensor = points_tensor.to(device)
with torch.no_grad():
    outputs = model(points_tensor)
    _, predicted = torch.max(outputs, 1)

print(f"Prédiction du modèle : {predicted.item()}")
print("First 5 points:", points[:5])
visualize_point_cloud(points, label=None, pred=predicted.item())