"""
Script to try out sixdrepnet
https://github.com/Shohruh72/SixDRepNet
for head pose estimation
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sixdrepnet import SixDRepNet

torch.cuda.is_available = lambda: False
def dummy_cuda(self, *args, **kwargs):
    return self.to('cpu')
torch.nn.Module.cuda = dummy_cuda

def euler_to_rmat(pitch, yaw, roll):
    angles = np.array([pitch, yaw, roll]).flatten()

    r = R.from_euler('xyz', angles, degrees=True)
    return r.as_matrix()

def plot_head_pose(img, pitch, yaw, roll, rmat, origin_point=None):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = img_rgb.shape

    if origin_point is None:
        origin_x = img_w / 2.0
        origin_y = img_h / 2.0
    else:
        origin_x, origin_y = origin_point

    axis_length = min(img_h, img_w) * 0.15

    axes_3d = np.array([
        [axis_length, 0, 0],   # X-Achse (Rot)
        [0, axis_length, 0],   # Y-Achse (Grün)
        [0, 0, axis_length],   # Z-Achse (Blau / Blickrichtung)
    ])

    rmat_plot = rmat.copy()
    rmat_plot[1, :] *= -1

    axes_2d = np.dot(axes_3d, rmat_plot.T)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_rgb)

    ax.plot([origin_x, origin_x + axes_2d[0, 0]], [origin_y, origin_y + axes_2d[0, 1]], color="red", linewidth=3, label="X (Pitch)")

    ax.plot([origin_x, origin_x + axes_2d[1, 0]], [origin_y, origin_y + axes_2d[1, 1]], color="green", linewidth=3, label="Y (Yaw)")

    ax.plot([origin_x, origin_x + axes_2d[2, 0]], [origin_y, origin_y + axes_2d[2, 1]], color="blue", linewidth=3, label="Z (Roll)")

    ax.scatter([origin_x], [origin_y], color="yellow", s=40, zorder=5)

    p_val = float(np.squeeze(pitch))
    y_val = float(np.squeeze(yaw))
    r_val = float(np.squeeze(roll))

    info_text = f"Pitch: {p_val:.1f}°\nYaw:   {y_val:.1f}°\nRoll:  {r_val:.1f}°"

    ax.text(
        15, 35, info_text, color="white", fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.6)
    )

    plt.title("Head Pose 3D-Koordinatensystem", fontsize=14)
    plt.axis("off")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

model = SixDRepNet()

image_path = 'CameraImages_Person1/test1.PNG'
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Bild unter '{image_path}' konnte nicht geladen werden.")

pitch, yaw, roll = model.predict(img)

rmat = euler_to_rmat(pitch, yaw, roll)

plot_head_pose(img, pitch, yaw, roll, rmat)