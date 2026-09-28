import numpy as np
from Setup.Camera.Calibration import Calibration

class Projector:
    def __init__(self):
        self.calibration = Calibration()

        print("intrinsic projector",self.get_intrinsics_matrix())
        print("extrinsic projector",self.get_extrinsic_matrix())

    def get_intrinsics_matrix(self):
        return self.calibration.proj_K 

    def get_extrinsic_matrix(self):
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = self.calibration.R
        extrinsic[:3, 3] = self.calibration.T
        return extrinsic