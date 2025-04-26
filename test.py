import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from model import DGCNN
import sys

def normalize_points(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    furthest_distance = np.max(np.sqrt(np.sum(points ** 2, axis=-1)))
    points = points / furthest_distance
    return points


modelnet40_classes = [
    "airplane", "bathtub", "bed", "bench", "bookshelf",
    "bottle", "bowl", "car", "chair", "cone",
    "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
    "laptop", "mantel", "monitor", "night_stand", "person",
    "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent",
    "toilet", "tv_stand", "vase", "wardrobe", "xbox"
]

def visualize_point_cloud(points, pred=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    color_map = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1],
        [0.5, 0.5, 0.5], [1, 0.5, 0], [0.5, 0, 1],
        [0, 0.5, 1]
    ]

    if pred is not None:
        color = color_map[pred % len(color_map)]
    else:
        color = [1, 0, 0]  
    colors = np.tile(color, (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Prédiction : {modelnet40_classes[pred]} (classe {pred})")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()



pcd = o3d.io.read_point_cloud("image/Classic side table.ply")
points = np.asarray(pcd.points)
points = normalize_points(points)




num_points = 1024
if points.shape[0] > num_points:
    indices = np.random.choice(points.shape[0], num_points, replace=False)
    points = points[indices]
elif points.shape[0] < num_points:
    raise ValueError("the point cloud contains less than 1024 points")

points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DGCNN(k=40).to(device)  

checkpoint_path = "./checkpoints/best_model.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
checkpoint = torch.load('./checkpoints/best_model.pth', map_location='cpu')


model.eval()


points_tensor = points_tensor.to(device)
with torch.no_grad():
    outputs = model(points_tensor)
    _, predicted = torch.max(outputs, 1)

classe_predite = modelnet40_classes[predicted.item()]
print(f" Prediction  : {classe_predite} (classe {predicted.item()})")


visualize_point_cloud(points, pred=predicted.item())
