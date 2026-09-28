"""
Test mvmp Facemarker
"""

from mvmp import Facemarker

#PATH = "models/head_person2/head_p2.obj"
PATH = "models/head/head_small.obj"
#PATH = "head_depth_to_mesh_test2.obj"
# PATH = "dtm2.obj"
#PATH = "models/head_cut.stl"

marker = Facemarker(
    camera_distance_multiplier=1.0,
    allow_missing_texture=False
)
result = marker.predict(PATH)
print(result)  # FacemarkerResult(478 landmarks, 478 vertex indices)

landmarks_3d = result.landmarks_3d              # dict[int, [x, y, z]]
vertex_indices = result.closest_vertices_ids    # dict[int, int]

result.save_json("landmarks.json")

import open3d as o3d
import numpy as np

mesh = o3d.io.read_triangle_mesh(PATH, enable_post_processing=True)
mesh.compute_vertex_normals()

#o3d.visualization.draw_geometries([mesh], window_name="Mesh aus Tiefenbild")

# Landmark-Koordinaten sammeln
pts = np.array([
    result.landmarks_3d[i]
    for i in sorted(result.landmarks_3d.keys())
])

spheres = []

for p in pts:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2)#radius=0.002)
    sphere.translate(p)
    sphere.paint_uniform_color([1, 0, 0])
    spheres.append(sphere)

o3d.visualization.draw_geometries([mesh, *spheres], window_name="Mesh mit Landmarken", mesh_show_back_face=True)