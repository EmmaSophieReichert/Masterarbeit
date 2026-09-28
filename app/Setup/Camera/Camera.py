# Class by Julian Höpfinger

import cv2
import pyk4a
#from pyk4a import PyK4A, depth_image_to_color_camera
import numpy as np
from Setup.Camera.Calibration import Calibration
from Setup.Camera.ImageDenoiser import ImageDenoiser
import open3d as o3d

"""
This class manges the camera access.  
"""
class Camera:
    def __init__(self):
        self.k4a = pyk4a.PyK4A(
            pyk4a.Config(
                camera_fps = pyk4a.FPS.FPS_30,
                depth_mode = pyk4a.DepthMode.NFOV_UNBINNED, #Emma WFOV NFOV
                color_resolution = pyk4a.ColorResolution.RES_720P#RES_1080P #Emma 3072 1080
            )
        )
        self.k4a.start()
        #EMMA
        calib = self.k4a.calibration
        intr = calib.get_camera_matrix(pyk4a.CalibrationType.COLOR)
        dist = calib.get_distortion_coefficients(pyk4a.CalibrationType.COLOR)

        extr = calib.get_extrinsic_parameters(pyk4a.CalibrationType.DEPTH, pyk4a.CalibrationType.COLOR)

        print("EXTRINSIC:\n", extr)

        print("Camera Matrix:\n", intr)
        print("Distortion:\n", dist)
        print("CALIBRATION CAMERA", intr)
        
        self.capture = self.k4a.get_capture()
        self.color_img = self.capture.color
        self.depth_img = pyk4a.depth_image_to_color_camera(self.capture.depth, self.k4a.calibration, True)

        self.calibration = Calibration(self.color_img.shape)#depth_img.shape) #EMMA!
        self.undistorted_depth_img = self.calibration.undistort(self.depth_img)

        self.denoiser = ImageDenoiser()

    """
    Capture an color, denoised depth and undistorted depth image.
    """
    def capture_image(self) -> None:
        # self.capture = self.k4a.get_capture()
        # self.color_img = self.capture.color
        # #Emma

        depth_img_list = []
        for a in range(10):
            capture = self.k4a.get_capture()
            depth_img_list.append(capture.depth)
            print("TIEFENBILD SHAPE: ", capture.depth.shape)
            self.capture = capture
            self.color_img = self.capture.color
        depth_stack = np.stack(depth_img_list)  # frames = Liste von Depth-Bildern
        depth_median = np.median(depth_stack, axis=0).astype(np.uint16)

        self.set_depth_image(pyk4a.depth_image_to_color_camera(depth_median, self.k4a.calibration, True))
        
        depth_img = pyk4a.depth_image_to_color_camera(depth_median, self.k4a.calibration, True)
        print("UNDISTORTED(?) DEPTH IMAGE SHAPE: ", depth_img.shape)
        print("DEPTH IMAGE SHAPE: ", self.depth_img.shape)
        print("Undistorted DEPTH IMAGE SHAPE: ", self.undistorted_depth_img.shape)
        print("Color img shape: ", self.color_img.shape)

        depth = self.depth_img
        num_zeros = np.sum(depth == 0)  
        total_pixels = depth.size       
        print(f"Frame end: {num_zeros} / {total_pixels} Pixel sind 0 ({num_zeros/total_pixels*100:.2f}%)")

    def stop_camera(self):
        self.k4a.stop()

    """
    Converts an pixel from an image to an 3d point based on the pixel value of the depth image.
    
    Parameters:
    pixel (np.ndarray): The pixel to convert.
    
    Returns:
    tuple: 3d point of the pixel.
    """
    def _convert_2d_to_3d(self, pixel: np.ndarray) -> tuple:
        depth = np.float32(self.depth_img[pixel[1]][pixel[0]])
        return self.calibration.convert_2d_to_3d(pixel, depth)

    """
    Converts a list of undistorted pixels to 3d point based on the pixel value of the depth image.
    
    Parameters:
    pixels (np.ndarray): The list of pixels to convert.
    
    Returns:
    list: 3d points of the pixels.
    """
    def undistorted_pixels_to_points(self, pixels: np.ndarray) -> list:
        points = []
        for pixel in pixels:
            points.append(self._convert_2d_to_3d(pixel))
        return points

    """
    Generates a point cloud of the book page based on its corners and the depth image.
    
    Parameters:
    corners (np.ndarray): Corners of the book.
    
    Returns:
    o3d.geometry.PointCloud: Point cloud of the book page.
    """
    def get_book_point_cloud(self, book_corners: np.ndarray) -> o3d.geometry.PointCloud:
        mask = np.zeros_like(self.undistorted_depth_img, dtype=np.uint8)
        cv2.fillConvexPoly(mask, book_corners, 255)
        mask = mask.astype(self.undistorted_depth_img.dtype)
        masked_image = np.where(mask == 255, self.undistorted_depth_img, 0)
        return self.calibration.depth_image_to_3d(masked_image)

    """
    Saves an undistorted and a denoised version of an depth image in this class.
    
    Parameters:
    depth (np.ndarray): Depth image  to save.
    """
    def set_depth_image(self, depth: np.ndarray) -> None:
        self.depth_img = depth
        self.undistorted_depth_img = self.calibration.undistort(self.depth_img)
        return #Emma
        self.depth_img = self.denoiser.denoise_image(depth)
        self.undistorted_depth_img = self.calibration.undistort(self.depth_img)


