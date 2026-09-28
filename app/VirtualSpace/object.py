import open3d as o3d
import numpy as np
import os
import copy
import CONFIG
from pathlib import Path
from scipy.spatial.transform import Rotation as R_scipy

OBJ_POS = np.array([250.0, -50.0, 1100.0]) #initial position to see the obj, when its not registered
OBJ_ROTATION = np.radians([0, 0, 0])

class ThreeDimensionalObject:

    def __init__(self):
        self.object_tracking_points = None
        self.object = self.create_object()

    """
    Create the 3D object from the mesh file
    """
    def create_object(self):
        object = self.get_coloured_mesh() 
        self.object = object
        self.path_to_albedo_img = CONFIG.MATERIAL_PATH
        return object

    """
    Load the mesh from path
    """
    def get_coloured_mesh(self):
        mesh = o3d.io.read_triangle_mesh(CONFIG.MODEL_PATH, enable_post_processing=True)

        mesh.scale(CONFIG.OBJECT_SCALE, center=mesh.get_center())

        center = mesh.get_center()
        mesh.translate(-center)

        if mesh.is_empty():
            raise RuntimeError("OBJ konnte nicht geladen werden. Pfade zu OBJ/MTL/PNG prüfen!")

        mesh.compute_vertex_normals()

        # rotate
        R_obj = mesh.get_rotation_matrix_from_xyz(OBJ_ROTATION)
        mesh.rotate(R_obj, center=mesh.get_center())

        # translate
        mesh.translate(OBJ_POS)

        #o3d.visualization.draw_geometries([mesh])

        return mesh

    """
    Get full pointcloud without any cuts
    """
    def get_pcd(self, number_of_points=25000):

        if self.object is None:
            print("Kein Objekt vorhanden zum Konvertieren!")
            return None
        
        pcd = self.object.sample_points_uniformly(number_of_points=number_of_points)
        return pcd

    """
    Apply cuts based on orientation
    """
    def _apply_orientation_cut(
        self, pcd, orientation_str, cut_ratio=0.35, center=None
    ):
        """Dreht die Punktwolke in die gewünschte Orientierung, schneidet

        entlang der Z-Achse ab und dreht sie wieder zurück.
        """
        if orientation_str == "Chin up":
            angle_nose = CONFIG.ROTATIONS.get("Nose up", CONFIG.ROTATIONS["Default"])
            angle = (angle_nose[0] - 45, angle_nose[1], angle_nose[2])
        else:
            angle = CONFIG.ROTATIONS.get(orientation_str, CONFIG.ROTATIONS["Default"])
        R = self.object.get_rotation_matrix_from_xyz(np.radians(angle))

        if center is None:
            center = pcd.get_center()

        pcd_rot = copy.deepcopy(pcd)
        pcd_rot.rotate(R, center=center)

        points = np.asarray(pcd_rot.points)
        z_min = points[:, 2].min()
        z_max = points[:, 2].max()
        z_threshold = z_min + (z_max - z_min) * cut_ratio

        mask = points[:, 2] > z_threshold
        pcd_cut = pcd_rot.select_by_index(np.where(mask)[0])

        pcd_cut.rotate(R.T, center=center)

        return pcd_cut

    """
    Get pointcloud with cuts based on orientation
    """
    def get_pcd_with_rotation_matrix(
        self, orientation_str, number_of_points=25000, cut_ratio=0.35
    ):
        o3d.utility.random.seed(42) #set same seed for reproducibility (study purposes)
        
        center = self.object.get_center()

        pcd = self.object.sample_points_uniformly(
            number_of_points=number_of_points
        )

        # Remove the hair
        pcd = self._apply_orientation_cut(
            pcd, "Nose up", cut_ratio=cut_ratio, center=center
        )

        pcd = self._apply_orientation_cut(
            pcd, "Chin up", cut_ratio=cut_ratio, center=center
        )
        
        # "Backface Culling"
        if orientation_str != "Nose up":
            pcd = self._apply_orientation_cut(
                pcd, orientation_str, cut_ratio=cut_ratio, center=center
            )

        #o3d.visualization.draw_geometries([pcd])
        return pcd