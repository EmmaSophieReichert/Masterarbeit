# Class by Julian Höpfinger

import Config_JH as Config
import cv2
import numpy as np

"""
This class reduces the noise from the depth image.
"""
class ImageDenoiser:
    def __init__(self):
        pass

    """
    Filters the given image to reduce noise.
    
    Parameters:
    image (np.ndarray): The image to be filtered.
    
    Returns:
    np.ndarray: The filtered image.
    """
    @staticmethod
    def denoise_image(image: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(image.astype(np.float32), Config.FILTER_D, Config.FILTER_COLOR, Config.FILTER_SPACE)