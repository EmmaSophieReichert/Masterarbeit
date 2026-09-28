# Adjusted File from: https://github.com/hesom/depth_to_mesh
# 02.05.2025
# published under MIT License
# I added the ability to add color to the mesh from a pixel-aligned RGB image.

#!/usr/bin/env python

import open3d as o3d
import numpy as np
import logging
from skimage.io import imread
import cv2

DEFAULT_CAMERA = o3d.camera.PinholeCameraIntrinsic(
    width=640,
    height=480,
    fx=528.0,
    fy=528.0,
    cx=319.5,
    cy=239.5
)

logger = logging.getLogger(__name__)


def _pixel_coord_np(width, height):
    x = np.arange(width, dtype=np.int32)
    y = np.arange(height, dtype=np.int32)

    x, y = np.meshgrid(x, y)

    return np.vstack((
        x.flatten(),
        y.flatten(),
        np.ones_like(x.flatten())
    ))

"""
Makes a colored Open3D mesh from a depth image and a pixel-aligned RGB image.
"""
def depth_file_to_mesh(
    depth_image,
    rgb_image,
    cameraMatrix=None,
    minAngle=3.0,
    sun3d=False,
    depthScale=1000.0
):

    # load depth
    depth_raw = imread(depth_image).astype(np.uint16)

    height, width = depth_raw.shape[:2]

    if sun3d:
        depth_raw = np.bitwise_or(
            depth_raw >> 3,
            depth_raw << 13
        )

    depth = depth_raw.astype(np.float32)
    depth /= depthScale

    # load rgb

    rgb = imread(rgb_image)

    if rgb.shape[:2] != (height, width):
        raise ValueError(
            f"RGB und Depth haben unterschiedliche Auflösung: "
            f"RGB={rgb.shape[:2]}, Depth={(height, width)}"
        )

    if rgb.ndim == 3 and rgb.shape[2] > 3:
        rgb = rgb[:, :, :3]

    # camera

    if cameraMatrix is None:

        camera = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=528.0,
            fy=528.0,
            cx=(width - 1) / 2,
            cy=(height - 1) / 2
        )

    else:

        camera = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=cameraMatrix[0, 0],
            fy=cameraMatrix[1, 1],
            cx=cameraMatrix[0, 2],
            cy=cameraMatrix[1, 2]
        )

    return depth_to_mesh(
        depth,
        rgb,
        camera,
        minAngle
    )

"""
Makes a colored Open3D mesh from a depth image and a pixel-aligned RGB image.
"""
def depth_to_mesh(
    depth,
    rgb,
    camera=DEFAULT_CAMERA,
    minAngle=3.0
):

    # check image sizes

    height, width = depth.shape

    if rgb.shape[:2] != (height, width):
        raise ValueError(
            f"RGB und Depth müssen dieselbe Auflösung haben: "
            f"Depth={depth.shape}, RGB={rgb.shape}"
        )

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(
            f"RGB muss ein Bild mit 3 Kanälen sein: RGB={rgb.shape}"
        )

    rgb = cv2.cvtColor(
        rgb[:, :, :3],
        cv2.COLOR_BGR2RGB
    )

    # camera

    K = camera.intrinsic_matrix
    K_inv = np.linalg.inv(K)

    pixel_coords = _pixel_coord_np(
        width,
        height
    )

    # depth -> 3D

    cam_coords = (
        K_inv @ pixel_coords
        * depth.reshape(-1)
    )

    points = cam_coords.T


    colors = rgb.reshape(
        -1,
        3
    ).astype(np.float64)

    colors /= 255.0

    # Make triangles from pixels

    i, j = np.meshgrid(
        np.arange(height - 1),
        np.arange(width - 1),
        indexing="ij"
    )

    i = i.reshape(-1)
    j = j.reshape(-1)

    idx_t1 = np.stack(
        [
            i * width + j,
            (i + 1) * width + j,
            i * width + (j + 1)
        ],
        axis=-1
    )

    idx_t2 = np.stack(
        [
            i * width + (j + 1),
            (i + 1) * width + j,
            (i + 1) * width + (j + 1)
        ],
        axis=-1
    )

    indices = np.vstack(
        [
            idx_t1,
            idx_t2
        ]
    )

    verts = cam_coords[:, indices]

    v1 = (
        verts[:, :, 1]
        - verts[:, :, 0]
    )

    v2 = (
        verts[:, :, 2]
        - verts[:, :, 0]
    )

    normals = np.cross(
        v1,
        v2,
        axis=0
    )

    normal_lengths = np.linalg.norm(
        normals,
        axis=0
    )

    centers = verts.mean(axis=2)

    center_lengths = np.linalg.norm(
        centers,
        axis=0
    )

    valid = (
        (normal_lengths > 0)
        &
        (center_lengths > 0)
    )

    normals_valid = (
        normals[:, valid]
        /
        normal_lengths[valid]
    )

    centers_valid = (
        centers[:, valid]
        /
        center_lengths[valid]
    )

    angles = np.degrees(
        np.arcsin(
            np.clip(
                np.abs(
                    np.einsum(
                        "ij,ij->j",
                        normals_valid,
                        centers_valid
                    )
                ),
                0.0,
                1.0
            )
        )
    )

    angle_valid = (
        angles > minAngle
    )

    indices = indices[
        valid
    ][
        angle_valid
    ].astype(np.int32)

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(
            np.ascontiguousarray(points)
        ),
        o3d.utility.Vector3iVector(
            np.ascontiguousarray(indices)
        )
    )

    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.ascontiguousarray(colors)
    )

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    return mesh