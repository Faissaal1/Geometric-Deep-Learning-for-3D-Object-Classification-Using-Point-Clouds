import os
import numpy as np

def read_off(filepath):
    with open(filepath, 'r') as f:
        if f.readline().strip() != 'OFF':
            raise Exception('Not a valid OFF header')
        n_verts, n_faces, _ = map(int, f.readline().strip().split(' '))
        vertices = []
        for _ in range(n_verts):
            vertices.append(list(map(float, f.readline().strip().split(' '))))
        vertices = np.array(vertices)
    return vertices

def sample_points(points, n_points=1024):
    if len(points) >= n_points:
        indices = np.random.choice(len(points), n_points, replace=False)
    else:
        indices = np.random.choice(len(points), n_points, replace=True)
    return points[indices]

def convert_off_folder_to_npy(input_folder, output_folder, n_points=1024):
    os.makedirs(output_folder, exist_ok=True)
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.off'):
                filepath = os.path.join(root, file)
                try:
                    points = read_off(filepath)
                    points = sample_points(points, n_points)
                    relative_path = os.path.relpath(root, input_folder)
                    save_dir = os.path.join(output_folder, relative_path)
                    os.makedirs(save_dir, exist_ok=True)
                    np.save(os.path.join(save_dir, file.replace('.off', '.npy')), points)
                    print(f"Saved {file} successfully.")
                except Exception as e:
                    print(f"Failed processing {file}: {e}")

convert_off_folder_to_npy(
    input_folder="C:/Users/faiss/Downloads/ModelNet40/ModelNet40",
    output_folder="C:/Users/faiss/Downloads/ModelNet40/ModelNet40_npy",
    n_points=1024
)
