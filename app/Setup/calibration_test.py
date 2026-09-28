"""
This script is used to validate the calibration of the camera and projector system. 
Used for Study 1
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from Setup.Camera.Calibration import Calibration

# --- CALIBRATION DATA ---
calibration = Calibration()
cam_K = calibration.cam_K
cam_dist = calibration.cam_kc
proj_K = calibration.proj_K
proj_dist = calibration.proj_kc
R = calibration.R
T = calibration.T.reshape(3,1)

# --- CONFIGURATION ---
CHESSBOARD_SIZE = (11, 7)  # inner corners
SQUARE_SIZE = 28.1         # in mm
PROJ_RES = (3840, 2160)

def validate_calibration(camera_image):
    gray = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if not ret:
        print("Schachbrett nicht gefunden! Bitte Beleuchtung prüfen.")
        return None
    else: print("Schachbrett gefunden")
    
    #------------show chessboard recognition
    # debug_view = camera_image.copy()
    # cv2.drawChessboardCorners(debug_view, CHESSBOARD_SIZE, corners, ret)
    
    # debug_view_rgb = cv2.cvtColor(debug_view, cv2.COLOR_BGR2RGB)
    
    # plt.figure(figsize=(10, 7))
    # plt.imshow(debug_view_rgb)
    # plt.title(f"Erkennung: {CHESSBOARD_SIZE} Ecken gefunden")
    # plt.axis('off')
    # plt.show(block=False) 
    # #plt.pause(3)        
    # input()
    # plt.close()
    #--------------------

    # define object points
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    _, rvec_cam, tvec_cam = cv2.solvePnP(objp, corners, cam_K, cam_dist)

    points_3d_cam, _ = cv2.projectPoints(objp, rvec_cam, tvec_cam, cam_K, cam_dist)
    
    R_cam_mat, _ = cv2.Rodrigues(rvec_cam)
    R_proj_mat = R @ R_cam_mat
    T_proj_mat = R @ tvec_cam.reshape(3, 1) + T.reshape(3, 1)
    rvec_proj, _ = cv2.Rodrigues(R_proj_mat)

    print(rvec_proj, "*", T_proj_mat)

    proj_corners_2d, _ = cv2.projectPoints(objp, rvec_proj, T_proj_mat, proj_K, proj_dist)

    validation_img = np.full((PROJ_RES[1], PROJ_RES[0], 3), 10, dtype=np.uint8) #dark grey background because projector turns off all of the time...

    for pt in proj_corners_2d:
        x, y = pt.ravel()
        if 0 <= x < PROJ_RES[0] and 0 <= y < PROJ_RES[1]:
            #cv2.circle(validation_img, (int(round(x)), int(round(y))), 1, (255, 255, 255), -1) # white points
            cv2.drawMarker(
                validation_img,
                (int(round(x)), int(round(y))),
                (255,255,255),
                markerType=cv2.MARKER_CROSS,
                markerSize=3,
                thickness=1
            )

    #cv2.imwrite("debug_projection.png", validation_img)
    return validation_img