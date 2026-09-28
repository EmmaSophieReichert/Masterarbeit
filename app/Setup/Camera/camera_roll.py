import cv2
import numpy as np
from Setup.Camera.Calibration import Calibration
from Setup.Camera.ImageDenoiser import ImageDenoiser
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
#from depth_to_mesh import depth_file_to_mesh, depth_to_mesh
from Setup.Camera.depth_to_mesh_color import depth_file_to_mesh, depth_to_mesh
import CONFIG

class CameraRoll:

    def __init__(self, logger = None):
        self.color_img = None
        self.depth_img = None
        self.calibration = None
        self.roi = None
        self.denoiser = ImageDenoiser()
        self.logger = logger
    
    """
    Read image from paths
    """
    def read_image(self, color_path: str, depth_path: str):
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        else:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            
        self.color_img = cv2.imread(color_path)
        self.depth_img = depth

        print(f"Color Image Shape: {self.color_img.shape}")
        print(f"Depth Image Shape: {self.depth_img.shape}")

        self.calibration = Calibration(self.depth_img.shape)

    """
    Add image directly
    """
    def add_image(self, color_img, depth_img):
        self.color_img = color_img
        self.depth_img = depth_img
        self.calibration = Calibration(self.depth_img.shape)

    """
        Select ROI via UI
    """
    def select_roi_interactively(self):
        if self.logger:
            self.logger.log_action("start_interactive_roi")
        self.roi = None

        fig, ax = plt.subplots(figsize=(20,14))

        manager = plt.get_current_fig_manager()
        try:
            if hasattr(manager, 'window'):
                manager.window.wm_geometry("+0+0")
        except Exception:
            pass

        ax.imshow(self.color_img)
        ax.set_title("ROI auswählen: ziehen + ENTER drücken")
        ax.axis('off')

        roi_coords = {}

        def onselect(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)

            roi_coords['x'] = min(x1, x2)
            roi_coords['y'] = min(y1, y2)
            roi_coords['w'] = abs(x2 - x1)
            roi_coords['h'] = abs(y2 - y1)

            print(f"Vorschau ROI: {roi_coords}")

        def on_key(event):
            if event.key == 'enter':
                if roi_coords:
                    self.roi = (
                        roi_coords['x'],
                        roi_coords['y'],
                        roi_coords['w'],
                        roi_coords['h']
                    )
                    print(f"ROI gesetzt auf: {self.roi}")
                else:
                    self.roi = None
                    print("Keine ROI ausgewählt, nutze Standard-Filtering.")
                plt.close(fig)
                if self.logger:
                    self.logger.log_action("end_interactive_roi")

        rect_selector = RectangleSelector(
            ax,
            onselect,
            useblit=True,
            button=[1],  # left mouse button 
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True
        )

        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()

    """
    If u use a better object detection, e. g. SAM, you can use the object mask, not a rectangle ROI
    """
    def use_image_masking(self, mask_path):
        small_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if small_mask is None:
            print("Fehler: Maskenbild nicht gefunden.")
            return None

        h_target, w_target = self.depth_img.shape
    
        mask_resized = cv2.resize(small_mask, (w_target, h_target), interpolation=cv2.INTER_NEAREST)

        _, binary_mask = cv2.threshold(mask_resized, 1, 255, cv2.THRESH_BINARY)
        
        return binary_mask

    """
    Remove the bottom of the pointcloud, e.g. the table, by keeping only the highest points
    """
    def get_pointcloud_highest_points_percentage(self, pcd, percentage = 0.95):
        points = np.asarray(pcd.points)
        z_min = points[:, 2].min()
        z_max = points[:, 2].max()
        z_threshold = z_min + (z_max - z_min) * percentage

        mask = points[:, 2] < z_threshold
        pcd_cut = pcd.select_by_index(np.where(mask)[0])
        return pcd_cut

    def get_depth_img_3d(self, filtering=True, use_manual_roi=False, mask_path = None, roi=None):
        if filtering:
            mask_depth = self.depth_img.copy()
            h, w = mask_depth.shape
            
            mask_depth[mask_depth < 800] = 0 #delete everything nearer than 80cm

            final_depth = self.depth_img.copy()

            if mask_path is not None:
                mask = self.use_image_masking(mask_path)
                if mask is not None:
                    if mask.shape != final_depth.shape:
                        print(f"FORM-FEHLER: Tiefe {final_depth.shape} vs Maske {mask.shape}")
                        mask = cv2.resize(mask, (final_depth.shape[1], final_depth.shape[0]))

                    binary_mask_01 = (mask / 255).astype(final_depth.dtype)
                    final_depth = final_depth * binary_mask_01
                    print("MASK ANGEWENDET")
                    
            else:
                final_depth = np.zeros_like(mask_depth)
                if roi:
                    x_min, y_min, x_max, y_max = roi
                    y_start, y_end = max(0, int(y_min)), min(h, int(y_max))
                    x_start, x_end = max(0, int(x_min)), min(w, int(x_max))
                    
                    final_depth[y_start:y_end, x_start:x_end] = \
                        mask_depth[y_start:y_end, x_start:x_end]
                    final_color = self.color_img[y_start:y_end, x_start:x_end].copy()
                    self.roi = (
                        x_min,
                        y_min,
                        x_max - x_min,
                        y_max - y_min
                    )
                elif use_manual_roi:
                    self.select_roi_interactively()
                    rx, ry, rw, rh = self.roi
                    y_start, y_end = max(0, ry), min(h, ry + rh)
                    x_start, x_end = max(0, rx), min(w, rx + rw)
                    
                    final_depth[y_start:y_end, x_start:x_end] = \
                        mask_depth[y_start:y_end, x_start:x_end]
                    final_color = self.color_img[y_start:y_end, x_start:x_end].copy()
                else:
                    # if you know where the head is, you can always use the same ROI
                    crop_h, crop_w = 300, 300 
                    start_y = max(0, h // 2 - crop_h // 2)
                    start_x = max(0, w // 2 - crop_w // 2)
                    
                    end_y = min(h, start_y + crop_h)
                    end_x = min(w, start_x + crop_w)

                    final_depth[start_y:end_y, start_x:end_x] = \
                        mask_depth[start_y:end_y, start_x:end_x]

                    final_color = self.color_img[start_y:end_y, start_x:end_x].copy()
                    
            valid_depth = final_depth[final_depth > 0]

            if valid_depth.size > 0:
                min_z = np.min(valid_depth)
                max_z = np.max(valid_depth)
                z_diff = max_z - min_z
                print(f"Vor 3D-Konvertierung: min Z = {min_z}, max Z = {max_z}, Differenz = {z_diff}")
            else:
                print("Keine gültigen Tiefenwerte im gefilterten Bild")

            #point_cloud = self.calibration.depth_image_to_3d(final_depth) #without depth_to_mesh
            point_cloud = self.depth_to_mesh_to_pcd(final_depth, final_color, minAngle=3.0)
        else:
            #point_cloud = self.calibration.depth_image_to_3d(self.depth_img) #without depth_to_mesh
            point_cloud = self.depth_to_mesh_to_pcd(self.depth_img, self.color_img, minAngle=3.0)
            
        point_cloud.estimate_normals()
        normals = np.asarray(point_cloud.normals)
        point_cloud.normals = o3d.utility.Vector3dVector(normals * -1)

        def display_inlier_outlier(cloud, ind, radius_val):
            inlier_cloud = cloud.select_by_index(ind)
            outlier_cloud = cloud.select_by_index(ind, invert=True)

            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius_val)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color([0.1, 0.9, 0.1]) # Grün
            
            if len(outlier_cloud.points) > 0:
                first_outlier_pos = np.asarray(outlier_cloud.points)[0]
                sphere.translate(first_outlier_pos)

            print("Showing outliers (red) and inliers (gray): ")
            outlier_cloud.paint_uniform_color([1, 0, 0])
            inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, sphere],
                                            zoom=0.3412,
                                            front=[0.4257, -0.2125, -0.8795],
                                            lookat=[2.6172, 2.0475, 1.532],
                                            up=[-0.0694, -0.9768, 0.2024])

        o3d.utility.random.seed(42) #set same seed for reproducibility (study purposes)
        cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.5)

        # #show filtering
        #display_inlier_outlier(point_cloud, ind, 10.0)

        pcd_cleaned = point_cloud.select_by_index(ind)

        return self.get_pointcloud_highest_points_percentage(pcd_cleaned)
    
    def depth_to_mesh_to_pcd(self, depth, rgb, minAngle=3):
        # #visualization
        #point_cloud = self.calibration.depth_image_to_3d(self.depth_img)
        #o3d.visualization.draw_geometries([point_cloud], window_name="Tiefenbild vorher", point_show_normal=True)

        cameraMatrix = self.calibration.cam_K
        camera = o3d.camera.PinholeCameraIntrinsic(
            width=CONFIG.SHAPE[0], height=CONFIG.SHAPE[1],
            fx=cameraMatrix[0,0], fy=cameraMatrix[1,1],
            cx=cameraMatrix[0,2], cy=cameraMatrix[1,2]
        )

        mesh = depth_to_mesh(depth.astype('float32'), self.color_img, camera, minAngle)

        # #visualization
        #o3d.visualization.draw_geometries([mesh], window_name="Mesh aus Tiefenbild", mesh_show_back_face=True)

        pcd = mesh.sample_points_uniformly(number_of_points=25000)

        # #visualization
        #o3d.visualization.draw_geometries([pcd], window_name="Tiefenbild nachher", point_show_normal=True)
        
        return pcd


    """
    Used for Study 3, to evaluate the system by landmarks
    """
    def evaluate_landmarks_interactively(
        self, comparison_img, window_scale=4.0, head_pose = None
    ):
        landmarks = [
            "lm_nose",
            "lm_left_eye_outer",
            "lm_right_eye_outer",
            "lm_left_ear_outer",
            "lm_right_ear_outer",
        ]

        match head_pose:
            case "Nose up":
                landmarks = [
                    "lm_nose",
                    "lm_left_eye_outer",
                    "lm_right_eye_outer",
                ]
            case "Left ear up":
                landmarks = [
                    "lm_nose",
                    "lm_left_eye_outer",
                    "lm_left_ear_outer",
                ]
            case "Right ear up":
                landmarks = [
                    "lm_nose",
                    "lm_right_eye_outer",
                    "lm_right_ear_outer",
                ]

        h, w = self.color_img.shape[:2]
        if self.roi is not None:
            rx, ry, rw, rh = self.roi
            x_start, y_start = max(0, rx), max(0, ry)
            x_end, y_end = min(w, rx + rw), min(h, ry + rh)
        else:
            x_start, y_start, x_end, y_end = 0, 0, w, h

        crop_real = self.color_img[y_start:y_end, x_start:x_end].copy()
        crop_comp = comparison_img[y_start:y_end, x_start:x_end].copy()

        images_to_label = [
            ("real", "Kinect (Color)", crop_real, True),
            ("comp", "Vergleichsbild", crop_comp, False),
        ]

        results = {}

        win_name = "Landmark Evaluation"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        crop_h, crop_w = crop_real.shape[:2]
        cv2.resizeWindow(
            win_name, int(crop_w * window_scale), int(crop_h * window_scale)
        )

        for lm_name in landmarks:
            skip_both = False

            for key_suffix, display_name, crop_img, is_real_kinect in (
                images_to_label
            ):
                if skip_both:
                    results[f"{lm_name}_{key_suffix}"] = None
                    continue

                current_click = {"pt": None, "skip": False}

                def mouse_callback(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        current_click["pt"] = (x, y)

                cv2.setMouseCallback(win_name, mouse_callback)

                info_text = f"Landmarke: {lm_name} | Bild: {display_name}"
                help_text = "[Klick]=Punkt | [ENTER]=OK | [N]=Skip"
                print("="*40)
                print(help_text)
                print("\n")
                print(info_text)
                print("="*40)
                
                while True:
                    display = crop_img.copy()
                    
                    cv2.putText(
                        display,
                        lm_name,
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        1,
                    )

                    if current_click["pt"] is not None:
                        cx, cy = current_click["pt"]
                        display[cy, cx] = (0, 0, 255)

                    cv2.imshow(win_name, display)
                    key = cv2.waitKey(20) & 0xFF

                    if key in [13, 32]:  # Enter / Space
                        if current_click["pt"] is not None:
                            break
                    elif key in [ord("n"), ord("N")]:
                        current_click["skip"] = True
                        if is_real_kinect:
                            skip_both = True
                        break
                    elif key == 27:  # ESC
                        cv2.destroyAllWindows()
                        return

                if current_click["skip"] or current_click["pt"] is None:
                    results[f"{lm_name}_{key_suffix}"] = None
                else:
                    roi_x, roi_y = current_click["pt"]
                    full_u = x_start + roi_x
                    full_v = y_start + roi_y
                    depth_val, p3d = self._get_3d_point_from_depth(
                        full_u, full_v
                    )

                    results[f"{lm_name}_{key_suffix}"] = {
                        "u": full_u,
                        "v": full_v,
                        "point_3d": p3d,
                    }

        cv2.destroyAllWindows()

        if self.logger is not None:
            for lm_name in landmarks:
                real_data = results.get(f"{lm_name}_real")
                comp_data = results.get(f"{lm_name}_comp")

                real_u = real_data["u"] if real_data else None
                real_v = real_data["v"] if real_data else None
                comp_u = comp_data["u"] if comp_data else None
                comp_v = comp_data["v"] if comp_data else None

                if real_data and real_data["point_3d"] is not None:
                    real_x, real_y, real_z = [
                        float(v) for v in real_data["point_3d"]
                    ]
                else:
                    real_x, real_y, real_z = None, None, None

                if comp_data and comp_data["point_3d"] is not None:
                    comp_x, comp_y, comp_z = [
                        float(v) for v in comp_data["point_3d"]
                    ]
                else:
                    comp_x, comp_y, comp_z = None, None, None

                dist_3d_mm = None
                if None not in (
                    real_x,
                    real_y,
                    real_z,
                    comp_x,
                    comp_y,
                    comp_z,
                ):
                    p_real = np.array([real_x, real_y, real_z])
                    p_comp = np.array([comp_x, comp_y, comp_z])
                    dist_3d_mm = float(np.linalg.norm(p_real - p_comp))
                    p_comp_2d = np.array([comp_x, comp_y, real_z])
                    dist_2d_mm = float(np.linalg.norm(p_real - p_comp_2d))

                self.logger.log_real_head_data(
                    landmark_name=lm_name,
                    color_u=real_u,
                    color_v=real_v,
                    color_x=real_x,
                    color_y=real_y,
                    color_z=real_z,
                    comp_u=comp_u,
                    comp_v=comp_v,
                    comp_x=comp_x,
                    comp_y=comp_y,
                    comp_z=comp_z,
                    distance_3d_mm=dist_3d_mm,
                    distance_2d_mm=dist_2d_mm,
                )

            print("Landmarken-Daten erfolgreich einzeln geloggt.")

    """
    Get the 3D point from the depth image at pixel coordinates (u, v) with optional window size for median filtering.
    """
    def _get_3d_point_from_depth(
        self,
        u: float,
        v: float,
        window_size: int = 3,
    ):
        depth_img = self.depth_img
        cam_k = self.calibration.cam_K
        cam_kc = self.calibration.cam_kc

        h, w = depth_img.shape[:2]
        u_int, v_int = int(round(u)), int(round(v))

        if u_int < 0 or u_int >= w or v_int < 0 or v_int >= h:
            return None, None

        half_w = window_size // 2
        y_min, y_max = max(0, v_int - half_w), min(h, v_int + half_w + 1)
        x_min, x_max = max(0, u_int - half_w), min(w, u_int + half_w + 1)

        patch = depth_img[y_min:y_max, x_min:x_max]
        valid_depths = patch[
            (patch > 0) & np.isfinite(patch)
        ]

        if valid_depths.size == 0:
            return None, None

        depth_val = float(np.median(valid_depths))
        cam_k = np.asarray(cam_k, dtype=np.float64)

        pixel_pt = np.array([[[u, v]]], dtype=np.float32)

        if cam_kc is not None and len(cam_kc) > 0:
            cam_kc = np.asarray(cam_kc, dtype=np.float64)
            undistorted_norm = cv2.undistortPoints(pixel_pt, cam_k, cam_kc)
            x_norm, y_norm = undistorted_norm[0, 0]
            p_3d = np.array([x_norm * depth_val, y_norm * depth_val, depth_val], dtype=np.float64)
        else:
            fx, fy = cam_k[0, 0], cam_k[1, 1]
            cx, cy = cam_k[0, 2], cam_k[1, 2]
            x_3d = (u - cx) * depth_val / fx
            y_3d = (v - cy) * depth_val / fy
            p_3d = np.array([x_3d, y_3d, depth_val], dtype=np.float64)

        return depth_val, p_3d