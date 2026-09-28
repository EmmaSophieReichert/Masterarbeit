# Class by Julian Höpfinger

import Config_JH as Config
import cv2
import numpy as np
import open3d as o3d
import yaml
import CONFIG

"""
This class handles every calculation, which needs the camera calibration. This includes creating 3d points based on the
depth image or undistortion.
This class is based on the Bachelor Thesis of Maximilian Kilger.
"""
class Calibration:
    def __init__(self, shape = CONFIG.SHAPE):
        with open(Config.CALIBRATION_PATH, 'r') as ymlfile:
            calibration_yml = yaml.safe_load(ymlfile)
        self.cam_K = np.array(calibration_yml[Config.CAM_K][Config.DATA]).reshape(3, 3)
        self.cam_kc = np.array(calibration_yml[Config.CAM_KC][Config.DATA])
        self.proj_K = np.array(calibration_yml[Config.PROJ_K][Config.DATA]).reshape(3, 3)
        self.proj_kc = np.array(calibration_yml[Config.PROJ_KC][Config.DATA])
        self.R = np.asarray(calibration_yml[Config.R][Config.DATA]).reshape(3, 3)
        self.T = np.array(calibration_yml[Config.T][Config.DATA])
        self.rvec, _ = cv2.Rodrigues(self.R)

        self.depth_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        self.f_x = float(self.cam_K[0, 0])
        self.f_y = float(self.cam_K[1, 1])
        self.c_x = float(self.cam_K[0, 2])
        self.c_y = float(self.cam_K[1, 2])
        self.depth_intrinsics.set_intrinsics(shape[0], shape[1], self.f_x, self.f_y, self.c_x, self.c_y)
        print("DEPTH INTRINISCS")
        print(self.depth_intrinsics.intrinsic_matrix)

    """
    Undistorts an image.
    
    Parameters:
    img (np.ndarray): Image to be undistorted.
    
    Returns:
    np.ndarray: Undistorted image.
    """
    def undistort(self, img: np.ndarray) -> np.ndarray:
        return cv2.undistort(img, self.cam_K, self.cam_kc)

    """
    Undistorts a list of pixels.
    
    Parameters:
    pixels (np.ndarray): List of pixels to be undistorted.
    
    Returns:
    np.ndarray: Undistorted list of pixels.
    """
    def undistort_pixels(self, pixels: np.ndarray) -> np.ndarray:
        float_points = np.float32(pixels)
        undistorted_points = cv2.undistortPoints(float_points, self.cam_K, self.cam_kc)
        undistorted_pixels = cv2.convertPointsToHomogeneous(undistorted_points).reshape(-1, 3)
        undistorted_pixels = np.dot(self.cam_K, undistorted_pixels.T).T
        return np.around(undistorted_pixels[:, :2]).astype(int)

    """
    Converts an 2d point of a camera image to a 3d point.
    
    Parameters:
    pixel (np.ndarray): 2d point in an image.
    depth (np.float32): Depth value for the pixel.
    
    Returns:
    tuple: 3d point.
    """
    def convert_2d_to_3d(self, pixel: np.ndarray, depth: np.float32) -> tuple:
        # https://stackoverflow.com/questions/51272055/opencv-unproject-2d-points-to-3d-with-known-depth-z/56985565#56985565
        u = int(pixel[0])
        v = int(pixel[1])

        x = (u - self.c_x) / self.f_x * depth
        y = (v - self.c_y) / self.f_y * depth
        return round(x), round(y), round(depth)

    """
    Converts an 3d point to a 2d point in a camera image.
    
    Parameters:
    point (np.ndarray): 3d point.
    
    Returns:
    tuple: 2d point.
    """
    def convert_3d_to_2d(self, point: np.ndarray) -> tuple:
        x = int(point[0])
        y = int(point[1])
        z = int(point[2])

        u = (self.f_x * (x / z)) + self.c_x
        v = (self.f_y * (y / z)) + self.c_y
        return round(u), round(v)

    """
    Creates a point cloud from a depth image.
    
    Parameters:
    depth (np.ndarray): Depth image for the point cloud creation.
    
    Returns:
    open3d.geometry.PointCloud: Point cloud based on the depth image.
    """
    def depth_image_to_3d(self, depth: np.ndarray) -> o3d.geometry.PointCloud:
        book_pc = o3d.geometry.Image(depth)
        # The warning "Parameter 'extrinsic' unfilled" and "Parameter 'intrinsic' unfilled" seem to be an issue
        # with PyCharm because the intrinsic is set and the extrinsic is optional.
        # noinspection PyArgumentList
        book_pc = o3d.geometry.PointCloud.create_from_depth_image(book_pc,
                                                                  self.depth_intrinsics,
                                                                  depth_scale=Config.DEPTH_SCALE,
                                                                  depth_trunc=Config.DEPTH_TRUNC)
        return book_pc#.voxel_down_sample(Config.VOXEL_SIZE) #Emma

    """
    Converts 3d points to 2d points of the projection.
    
    Parameters:
    points (list): 3d points.
    
    Returns:
    np.ndarray: 2d points of the projection.
    """
    def points_to_projection(self, points: list) -> np.ndarray:
        float_points = np.asarray(points, dtype=np.float64)
        projected_points, _ = cv2.projectPoints(float_points, self.rvec, self.T, self.proj_K, self.proj_kc)
        projected_points = projected_points.reshape(-1, 2)
        # We need to multiply every point to get the correct scaling for the projection
        projected_points = np.multiply(projected_points, Config.CALIBRATION_SCALING)
        return np.around(projected_points).astype(int)

    """
    Converts 3d points to 2d points of the camera image.
    
    Parameters:
    points (np.ndarray): 3d points.
    
    Returns:
    np.ndarray: 2d points of the camera image.
    """
    def points_to_image(self, points: np.ndarray) -> np.ndarray:
        #https://answers.opencv.org/question/98929/trying-to-re-distort-image-points-using-projectpoints/
        float_points = np.asarray(points, dtype=np.float64)
        image_points, _ = cv2.projectPoints(float_points, np.zeros((3,1)), np.zeros((3,1)), self.cam_K, self.cam_kc)
        image_points = image_points.reshape(-1, 2)
        return np.around(image_points).astype(int)
