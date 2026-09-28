import open3d as o3d
import numpy as np
from VirtualSpace.object import ThreeDimensionalObject

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import CONFIG 

NEAR = 1.0
FAR = 1000.0

class VirtualSpace:
    def __init__(self, projector):
        self.intrinsic = projector.get_intrinsics_matrix()
        self.extrinsic = projector.get_extrinsic_matrix()
        self.object = ThreeDimensionalObject()

        self.material_scan = o3d.visualization.rendering.MaterialRecord()
        self.material_scan.shader = "defaultUnlit"

        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.shader = "defaultLit"
        self.material.albedo_img = o3d.io.read_image(self.object.path_to_albedo_img)

        self.window = None
        self.scene_widget = None

        self.material_pcd = o3d.visualization.rendering.MaterialRecord()
        self.material_pcd.shader = "defaultUnlit"
        self.material_pcd.base_color = [1.0, 0.0, 0.0, 1.0] # red
        self.material_pcd.point_size = 3.0

    def move_3D_object(self, extrinsic_matrix):
        if extrinsic_matrix is not None:
            object = self.object.object
            self.scene_widget.scene.set_geometry_transform("cube", extrinsic_matrix)
            self.object.object = object

    def remove_3D_object(self):
        self.scene_widget.scene.remove_geometry("cube")

    def add_3D_object(self):
        self.scene_widget.scene.add_geometry("cube", self.object.object, self.material)

    def setup_projection_environment(self, scene: o3d.visualization.rendering.Open3DScene): 
        # background
        scene.set_background(np.array([0.0, 0.0, 0.0, 1.0]))
        scene.show_skybox(False)

        # light
        scene.set_lighting(o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS, (0.577, -0.577, -0.577))
    
    def camera_settings(self, scene_widget: o3d.visualization.gui.SceneWidget, scene: o3d.visualization.rendering.Open3DScene):
        
        bounds = self.object.object.get_axis_aligned_bounding_box()
        scene_widget.setup_camera(
            self.intrinsic, 
            self.extrinsic, 
            CONFIG.WINDOW_WIDTH, 
            CONFIG.WINDOW_HEIGHT, 
            bounds
        )
        # add near far settings
        scene.camera.set_projection(
            intrinsics=self.intrinsic,
            near_plane=NEAR,
            far_plane=FAR,
            image_width=CONFIG.WINDOW_WIDTH,
            image_height=CONFIG.WINDOW_HEIGHT
        )

    def create_virtual_scene(self, app):

        window = app.create_window("Virtual Scene", CONFIG.WINDOW_WIDTH, CONFIG.WINDOW_HEIGHT)
        window.show_menu(False)
        window.os_frame = o3d.visualization.gui.Rect(0, 0, CONFIG.WINDOW_WIDTH, CONFIG.WINDOW_HEIGHT) 
        scene_widget = o3d.visualization.gui.SceneWidget()
        scene = o3d.visualization.rendering.Open3DScene(window.renderer)
        scene_widget.scene = scene
        self.setup_projection_environment(scene)

        scene.add_geometry("cube", self.object.object, self.material)

        self.camera_settings(scene_widget, scene)
        
        window.add_child(scene_widget)

        self.window = window
        self.scene_widget = scene_widget

        return window, scene_widget