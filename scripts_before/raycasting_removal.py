"""
Script for getting outer shell
"""

import open3d as o3d
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

IMPORT_PATH = "skin_model_skin.stl"
EXPORT_PATH = "sichtbare_huelle_ray.stl"

if len(sys.argv) > 1:
    IMPORT_PATH = sys.argv[1]
if len(sys.argv) > 2:
    EXPORT_PATH = sys.argv[2]
if len(sys.argv) > 3:
    WITH_VISUALIZATION = sys.argv[3] == "True"

print("Lade und bereinige Mesh...")
mesh = o3d.io.read_triangle_mesh(IMPORT_PATH)
mesh.compute_vertex_normals()

if WITH_VISUALIZATION:
    def color_points(pcd):
        axis = 1
        points = np.asarray(pcd.points)
        values = points[:, axis]
        norm_values = (values - values.min()) / (values.max() - values.min())
        cmap = plt.get_cmap("jet")
        colors = cmap(norm_values)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    R = o3d.geometry.get_rotation_matrix_from_xyz((np.radians(-90), 0, np.radians(-90)))
    mesh.rotate(R, center=mesh.get_center())
    pcd = mesh.sample_points_uniformly(number_of_points=30000)

    pcd = color_points(pcd)

    o3d.visualization.draw_geometries([mesh], width=1600, height=1200, window_name="Unbearbeiteter Kopf (Mesh)")
    o3d.visualization.draw_geometries([pcd], width=1600, height=1200, window_name="Unbearbeiteter Kopf (Pointcloud)")

mesh = mesh.remove_duplicated_vertices()
mesh = mesh.remove_non_manifold_edges()
mesh.merge_close_vertices(0.0001)

scene = o3d.t.geometry.RaycastingScene()
mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
_ = scene.add_triangles(mesh_t)

vertices = np.asarray(mesh.vertices).astype(np.float32)
center = mesh.get_center()
bbox = mesh.get_axis_aligned_bounding_box()
diameter = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())

def get_camera_positions(center, distance, count=64):
    phi = np.pi * (3. - np.sqrt(5.))
    points = []
    for i in range(count):
        y = 1 - (i / float(count - 1)) * 2
        radius_at_y = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        points.append(center + np.array([x, y, z]) * distance)
    return points

cam_radius = diameter * 1.5
camera_locations = get_camera_positions(center, cam_radius, count=16) # Weniger Kameras für Speed

visible_vertex_indices = np.zeros(len(vertices), dtype=bool)

print(f"Prüfe Sichtbarkeit von {len(vertices)} Vertices...")

for cam_pos in tqdm(camera_locations, desc="Kameras abarbeiten"):
    directions = vertices - cam_pos
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= norms # Normalisieren
    
    rays = np.hstack([np.tile(cam_pos, (len(vertices), 1)), directions]).astype(np.float32)
    rays_t = o3d.core.Tensor(rays, dtype=o3d.core.float32)
    
    # Raycasting
    ans = scene.cast_rays(rays_t)
    t_hit = ans['t_hit'].numpy()
    
    is_visible = np.abs(t_hit - norms.flatten()) < (diameter * 0.01)
    visible_vertex_indices |= is_visible

print(f"\nSichtbare Vertices gefunden: {np.sum(visible_vertex_indices)}")

if np.sum(visible_vertex_indices) == 0:
    print("FEHLER: 0 Punkte. Erhöhe die Toleranz oder prüfe die Kamera-Positionen!")
else:
    triangles = np.asarray(mesh.triangles)
    # keep all triangles with at least one visible point
    mask = visible_vertex_indices[triangles].any(axis=1)
    mesh.remove_triangles_by_mask(~mask)
    mesh.remove_unreferenced_vertices()

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    n_clusters_total = len(cluster_n_triangles)
    num_keep = 1
    largest_indices = np.argsort(cluster_n_triangles)[-num_keep:]
    mesh.remove_triangles_by_mask(~np.isin(triangle_clusters, largest_indices))
    mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(EXPORT_PATH, mesh)
    o3d.visualization.draw_geometries([mesh])

if WITH_VISUALIZATION:
    mesh = o3d.io.read_triangle_mesh(EXPORT_PATH)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=10000)
    pcd = color_points(pcd)

    o3d.visualization.draw_geometries([mesh], width=1600, height=1200, window_name="Bearbeiteter Kopf (Mesh)")
    o3d.visualization.draw_geometries([pcd], width=1600, height=1200, window_name="Bearbeiteter Kopf (Pointcloud)")