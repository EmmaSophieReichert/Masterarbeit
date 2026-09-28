"""
This script provides a GUI application to interactively find the rotation of a 3D mesh or point cloud. 
The user can adjust the rotation angles around the X, Y, and Z axes using sliders, and the application will display the rotated geometry in real-time.
"""

import copy
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

MODEL_PATH = "models/head/head_small.obj"

def open_rotation_finder(mesh_or_pcd):
    app = gui.Application.instance
    app.initialize()

    window = app.create_window("Rotation Finder (0, 90, 0)", 1280, 800)

    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene.set_background([0.15, 0.15, 0.15, 1.0])

    def block_mouse(event):

        return 2 #= CONSUMED

    scene_widget.set_on_mouse(block_mouse)

    orig_geom = copy.deepcopy(mesh_or_pcd)
    center = orig_geom.get_center()

    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"

    is_mesh = isinstance(orig_geom, o3d.geometry.TriangleMesh)
    if not is_mesh and not isinstance(orig_geom, o3d.geometry.PointCloud):

        orig_geom = orig_geom.sample_points_uniformly(15000)

    bbox = orig_geom.get_axis_aligned_bounding_box()
    frame_size = bbox.get_max_extent() * 0.4
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frame_size, origin=center
    )

    scene_widget.scene.add_geometry("object", orig_geom, mat)
    scene_widget.scene.add_geometry("frame", coord_frame, mat)
    scene_widget.setup_camera(60, bbox, center)

    em = window.theme.font_size
    layout = gui.Vert(0, gui.Margins(em, em, em, em))

    label_output = gui.Label('Winkel: "(0, 0, 0)"')
    layout.add_child(label_output)

    sliders = {}
    for axis in ["X", "Y", "Z"]:
        layout.add_child(gui.Label(f"Rotation {axis}-Achse (Grad):"))
        s = gui.Slider(gui.Slider.INT)
        s.set_limits(-180, 180)
        s.int_value = 0
        sliders[axis] = s
        layout.add_child(s)

    def update_rotation(new_val=None):
        rx = int(sliders["X"].int_value)
        ry = int(sliders["Y"].int_value)
        rz = int(sliders["Z"].int_value)

        angles = (int(rx), int(ry), int(rz))
        label_output.text = f"Winkel: {angles}"
        print(f"Aktuelle Rotation: {angles}")

        R = orig_geom.get_rotation_matrix_from_xyz(np.radians(angles))
        rotated_geom = copy.deepcopy(orig_geom)
        rotated_geom.rotate(R, center=center)

        scene_widget.scene.remove_geometry("object")
        scene_widget.scene.add_geometry("object", rotated_geom, mat)

    for s in sliders.values():
        s.set_on_value_changed(update_rotation)

    window.add_child(scene_widget)
    window.add_child(layout)

    def on_layout(layout_context):
        r = window.content_rect
        panel_width = 18 * em
        scene_widget.frame = gui.Rect(
            r.x, r.y, r.width - panel_width, r.height
        )
        layout.frame = gui.Rect(
            r.width - panel_width, r.y, panel_width, r.height
        )

    window.set_on_layout(on_layout)
    app.run()

if __name__ == "__main__":
    mesh = o3d.io.read_triangle_mesh(MODEL_PATH, enable_post_processing=True)
    open_rotation_finder(mesh)