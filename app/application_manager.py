import sys
import os
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["OPEN3D_CPU_RENDERING"] = "true"

import cv2
import open3d as o3d
import open3d.visualization.gui as gui

from Setup.projector import Projector
from VirtualSpace.virtual_space import VirtualSpace
from Tracking.automatic_registration import AutomaticRegistration
from Setup.Camera.camera_roll import CameraRoll
#from Setup.Camera.Camera import Camera #TODO: fix this
from Setup.Camera.Calibration import Calibration
from Projection.image_projection import ImageProjection
from Setup.calibration_test import *
from Tracking.FaceDetection.pose_detector import PoseDetector
from DataCollection.experiment_logger import ExperimentLogger
import CONFIG
import time
import numpy as np
import copy

class ApplicationManager:
    def __init__(self, realtime=False):
        self.realtime = realtime

        self.projector = Projector()
        self.virtual_space = VirtualSpace(self.projector)
        self.calibration = Calibration()
        self.automatic_registration = AutomaticRegistration()
        self.camera_roll = CameraRoll()
        self.image_projection = ImageProjection(self.calibration)
        self.pose_detector = PoseDetector()

        self.camera = None
        if self.realtime:
            from Setup.Camera.Camera import Camera
            self.camera = Camera()

        self.window = None

    """
    Image taking mode for collecting images for the evaluation study 3
    """
    def collect_camera_images(self):
        if not self.camera:
            from Setup.Camera.Camera import Camera
            self.camera = Camera()
        folder = "CameraImages_Person9"
        os.makedirs(folder, exist_ok=True)
        for i in range(10):
            print("Press Enter for img")
            input()
            self.camera.capture_image()
            cv2.imwrite(f"{folder}/rgb_{i:02d}.png", self.camera.color_img)
            cv2.imwrite(f"{folder}/depth_view_{i:02d}.png", self.camera.depth_img)
            cv2.imwrite(f"{folder}/depth_undistorted_view_{i:02d}.png", self.camera.undistorted_depth_img)
            depth = self.camera.depth_img.astype('uint16')
            depth_undistorted = self.camera.depth_img.astype('uint16')
            cv2.imwrite(f"{folder}/depth_{i:02d}.png", depth)
            cv2.imwrite(f"{folder}/depth_undistorted_{i:02d}.png", depth_undistorted)
            np.save(f"{folder}/depth_np_{i:02d}.npy", self.camera.depth_img)
            np.save(f"{folder}/depth_undistorted_np_{i:02d}.npy", self.camera.undistorted_depth_img)

    """
    Run the test calibration mode for study 1
    """
    def run_test_calibration(self):
        folder = "1_ProCamCalib_Study"
        os.makedirs(folder, exist_ok=True)
        print("Press Enter for first image")
        input()
        for i in range(100):
            self.camera.capture_image()
            img = self.camera.color_img
            proj_map = validate_calibration(img)
            if proj_map is not None:
                self.image_projection.show_img_fullscreen(proj_map, True)
                self.camera.capture_image()
                cv2.imwrite(f"{folder}/rgb_{i:02d}.png", self.camera.color_img)

    """
    Evaluation mode for study 2 with printed head
    """
    def run_with_evaluation_study_2(self):
        logger = ExperimentLogger("DataCollection/Study2_printed_head_performance.csv", "DataCollection/Study2_printed_head_accuracy.csv")
        self.image_projection = ImageProjection(self.calibration, logger)
        self.camera_roll = CameraRoll(logger=logger)
        logger.open()

        for voxel_size in [20, 18, 16, 14, 12, 10, 8]:
            CONFIG.VOXEL_SIZE = voxel_size
            print(f"Voxel size set to: {CONFIG.VOXEL_SIZE}")
            for head_pos in ["Nose up", "Right ear up", "Left ear up", "Nose down"]:
                print(f"Please pose head in position: \n \n {head_pos} \n")

                logger.set_metadata(CONFIG.VOXEL_SIZE, head_pos)

                logger.log_action("start_logging")

                if self.realtime:
                    print("Press Enter to capture image")
                    input()
                    logger.log_action("start_image_capture")
                    self.camera.capture_image()
                    logger.log_action("end_image_capture")
                    logger.log_action("start_preprocessing_image")
                    self.camera_roll.add_image(self.camera.color_img, self.camera.undistorted_depth_img) #self.camera.depth_img)
                    scan = self.camera_roll.get_depth_img_3d(filtering=True, use_manual_roi=True)
                    cv2.destroyAllWindows()
                    logger.log_action("end_preprocessing_image")
                else:
                    self.camera_roll.read_image(CONFIG.RGB_IMG_PATH, CONFIG.DEPTH_IMG_PATH)
                    scan = self.camera_roll.get_depth_img_3d(filtering=True, use_manual_roi=True)

                logger.log_action("start_preprocessing_model")   
                pcd_3d_model = self.virtual_space.object.get_pcd_with_rotation_matrix(head_pos)
                logger.log_action("end_preprocessing_model")

                logger.log_action("start_registration_ppf")
                registrations = self.automatic_registration.get_registration_matrix_with_ppf(
                    pcd_3d_model,
                    scan,
                    with_icp=True,
                    voxel_size=CONFIG.VOXEL_SIZE,
                )

                logger.log_action("end_registration_ppf")
                
                logger.log_action("start_save_img_for_projection")

                counter = 0

                best_registration_matrix = registrations[0][0]
                obj_to_show = self.virtual_space.object.object
                obj_to_show_copy = copy.deepcopy(obj_to_show)
                obj_to_show_copy.transform(best_registration_matrix)

                logger.log_result(f"icp_{counter}", registrations[counter][3], registrations[counter][4])
                
                self.image_projection.create_mesh_projection(obj_to_show_copy)
                logger.log_action("end_save_img_for_projection")
                logger.log_action("start_show_img_on_projector")

                def on_image_shown():
                    nonlocal counter, obj_to_show
                    logger.log_action("end_show_img_on_projector")
                    logger.log_action("end_logging")
                    if self.realtime:
                        print("Press ENTER to capture image of projection")
                        input()
                        self.camera.capture_image()
                        folder = "2_Printed_Head_Study"
                        cv2.imwrite(f"{folder}/rgb_{CONFIG.VOXEL_SIZE}_{head_pos}_{counter}.png", self.camera.color_img)

                self.image_projection.show_img_fullscreen(on_image_shown = on_image_shown)

        logger.close()

    """
        Evaluation mode for study 3 with real heads
    """
    def run_with_evaluation_study_3(self):
        logger = ExperimentLogger("DataCollection/Study3_real_head_performance.csv", "DataCollection/Study3_real_head_accuracy.csv", "DataCollection/Study3_real_head_evaluation.csv")
        self.image_projection = ImageProjection(self.calibration, logger)
        self.camera_roll = CameraRoll(logger=logger)
        logger.open()
        logger.set_metadata(voxel_size=CONFIG.VOXEL_SIZE)

        for num_head in range(1, 10):
            if num_head == 1:
                CONFIG.ROTATIONS = {
                    "Nose up": (0, -90, 0),
                    "Left ear up": (180, 0 , 0),
                    "Right ear up": (0, 0, 0),
                    "Default": (0, 0, 0),
                }
            elif num_head == 5 or num_head == 7:
                CONFIG.ROTATIONS = {
                    "Nose up": (0, -30, 0),
                    "Left ear up": (0, -120, 0),
                    "Right ear up": (0, 60, 0),
                    "Default": (0, 0, 0)
                }
            else:
                CONFIG.ROTATIONS = {
                    "Nose up": (0, 0, 0),
                    "Left ear up": (0, -90, 0),
                    "Right ear up": (0, 90, 0),
                    "Default": (0, 0, 0),
                }
            CONFIG.MODEL_PATH = f"models/head_person{num_head}/head_p{num_head}.obj"
            CONFIG.MATERIAL_PATH = f"models/head_person{num_head}/head_p{num_head}.jpg"
            self.virtual_space.object.create_object() #update the paths from init
            logger.set_metadata(participant_id=num_head)
            for num_img in range(0, 3):
                logger.set_metadata(CONFIG.VOXEL_SIZE, "None")
                logger.log_action("start_logging")

                rgb_img_path = f"CameraImages_Person{num_head}/rgb_{num_img:02d}.png"
                depth_img_path = f"CameraImages_Person{num_head}/depth_np_{num_img:02d}.npy"

                logger.log_action("start_face_detection")
                pose_result = self.pose_detector.detect_faces(rgb_img_path)
                logger.log_action("end_face_detection")

                logger.log_action("start_preprocessing_image")

                self.camera_roll.read_image(rgb_img_path, depth_img_path)
                
                if not pose_result:
                    scan = self.camera_roll.get_depth_img_3d(filtering=True, use_manual_roi=True)
        
                    print("Please select the head position!")
                    print("  0 - Nose up \n  1 - Right ear up \n  2 - Left ear up")
                    head_pos_nr = int(input())
                    head_pos = "Nose up"
                    match head_pos_nr:
                        case 0:
                            head_pos = "Nose up"
                        case 1:
                            head_pos = "Right ear up"
                        case 2:
                            head_pos = "Left ear up"
                else:
                    crop_img, bbox = self.pose_detector.get_roi(margin_top = CONFIG.MARGIN_TOP, margin_bottom = CONFIG.MARGIN_BOTTOM, margin_sides = CONFIG.MARGIN_SIDES)
                    scan = self.camera_roll.get_depth_img_3d(filtering=True, roi=bbox)
                    head_pos = self.pose_detector.get_face_rotation()

                logger.log_action("end_preprocessing_image")

                logger.set_metadata(CONFIG.VOXEL_SIZE, head_pos)

                logger.log_action("start_preprocessing_model")   
                pcd_3d_model = self.virtual_space.object.get_pcd_with_rotation_matrix(head_pos)
                logger.log_action("end_preprocessing_model")

                if pcd_3d_model.has_normals():
                    print("PCD Normals Checksum:", np.sum(np.asarray(pcd_3d_model.normals)[:5]))

                logger.log_action("start_registration_ppf")
                registrations = self.automatic_registration.get_registration_matrix_with_ppf(
                    pcd_3d_model,
                    scan,
                    with_icp=True,
                    voxel_size=CONFIG.VOXEL_SIZE,
                )
                best_registration_matrix = registrations[0][0]
                obj_to_show = self.virtual_space.object.object
                obj_to_show_copy = copy.deepcopy(obj_to_show)
                obj_to_show_copy.transform(best_registration_matrix)
                logger.log_result(f"icp_{0}", registrations[0][3], registrations[0][4])
                logger.log_action("end_registration_ppf")
                
                logger.log_action("start_save_img_for_virtual_camera")
                overlay_img = self.show_overlay_img_camera(self.camera_roll.color_img, obj_to_show_copy, True)
                logger.log_action("end_save_img_for_virtual_camera")

                logger.log_action("end_logging")
                self.camera_roll.evaluate_landmarks_interactively(overlay_img, head_pose=head_pos)

        logger.close()

    """
    Normal mode for running the application 
    pick tracking for realtime tracking
    """
    def run_with_pictures(self, realtime_tracking=False):

        camera_roll = self.camera_roll
        if self.realtime:
            print("Press Enter to capture image")
            input()
            time.sleep(10)
            self.camera.capture_image()
            camera_roll.add_image(self.camera.color_img, self.camera.undistorted_depth_img) #self.camera.depth_img)
        else:
            camera_roll.read_image(CONFIG.RGB_IMG_PATH, CONFIG.DEPTH_IMG_PATH)
        
        cv2.destroyAllWindows()

        pose_result = self.pose_detector.detect_faces(CONFIG.RGB_IMG_PATH)

        if not pose_result:
            print("No face detected by Mediapipe. Select ROI manually.")
            scan = camera_roll.get_depth_img_3d(filtering=True, use_manual_roi=True)
            print("Roi selected")
            cv2.destroyAllWindows()

            print("Please select the head position!")
            print("  0 - Nose up \n  1 - Right ear up \n  2 - Left ear up")
            
            head_pos_nr = int(sys.stdin.readline())

            head_pos = "Nose up"
            match head_pos_nr:
                case 0:
                    head_pos = "Nose up"
                case 1:
                    head_pos = "Right ear up"
                case 2:
                    head_pos = "Left ear up"
        else:
            crop_img, bbox = self.pose_detector.get_roi(margin_top = CONFIG.MARGIN_TOP, margin_bottom = CONFIG.MARGIN_BOTTOM, margin_sides = CONFIG.MARGIN_SIDES)
            scan = camera_roll.get_depth_img_3d(filtering=True, roi=bbox)
            head_pos = self.pose_detector.get_face_rotation()
        
        app = o3d.visualization.gui.Application.instance #VIEW
        app.initialize() #VIEW
        
        window, scene_widget = self.virtual_space.create_virtual_scene(app) #VIEW
        self.window = window #VIEW

        #Show scan
        self.virtual_space.scene_widget.scene.add_geometry("scan", scan, self.virtual_space.material_scan) #VIEW

        pcd = self.virtual_space.object.get_pcd_with_rotation_matrix(head_pos)

        registrations = self.automatic_registration.get_registration_matrix_with_ppf(pcd, scan, with_icp=True)

        # VIEW-----
        
        current_index = {"i": 0, "y": 0}

        def move_object(pos):
            print("Index: ", current_index["i"], "/", len(registrations)-1)
            print("Confidence: ", registrations[current_index["i"]][2])
            print("Moving object to position: \n", pos)
            self.virtual_space.move_3D_object(pos)

        def on_key(event):
            if event.type == gui.KeyEvent.DOWN:
                if event.key == gui.KeyName.N:
                    current_index["i"] = (current_index["i"] + 1) % len(registrations)
                    move_object(registrations[current_index["i"]][0])
                if event.key == gui.KeyName.R:
                    print("R key pressed")
                    current_index["y"] += 1
                    if current_index["y"] % 2 == 1:
                        self.virtual_space.remove_3D_object()
                        print("3D Object removed")
                    else:
                        self.virtual_space.add_3D_object()
                        move_object(registrations[current_index["i"]][0])
                        print("3D Object added")

        window.set_on_key(on_key)

        move_object(registrations[0][0])

        # ---VIEW

        obj_to_show = self.virtual_space.object.object
        obj_to_show_copy = copy.deepcopy(obj_to_show)
        obj_to_show_copy.transform(registrations[0][0])

        self.image_projection.create_mesh_projection(obj_to_show_copy)
        #self.image_projection.create_pointcloud_projection(scan) #if you want to display the scan

        #self.show_overlay_img_camera(camera_roll.color_img, obj_to_show_copy) #if you want to see transparent mode

        #--------FOR IMAGES FOR THESIS
        # self.virtual_space.scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        # self.virtual_space.scene_widget.scene.set_lighting(
        #     o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS,
        #     [0, 0, 0]
        # )
        # self.virtual_space.scene_widget.scene.remove_geometry("cube")
        # self.virtual_space.scene_widget.scene.add_geometry("pcd", pcd, self.virtual_space.material_pcd)
        # self.virtual_space.scene_widget.scene.set_geometry_transform("pcd", registrations[0][0])
        #--------FOR IMAGES FOR THESIS

        app.run()  #VIEW

        registration =registrations[0][0]

        """
        Method for getting ROI for realtime tracking mode
        Calculates the projected bounding box of the 3D object in the camera image
        """
        def get_projected_roi(
            mesh,
            registration,
            image_shape,
            margin=50
        ):

            h, w = image_shape[:2]

            bbox = mesh.get_axis_aligned_bounding_box()

            points = np.asarray(
                bbox.get_box_points()
            )

            R = registration[:3, :3]
            t = registration[:3, 3]

            points = points @ R.T + t

            projected = self.calibration.points_to_image(
                points
            )

            projected = np.asarray(projected)

            x_min = int(np.min(projected[:, 0])) - margin
            y_min = int(np.min(projected[:, 1])) - margin

            x_max = int(np.max(projected[:, 0])) + margin
            y_max = int(np.max(projected[:, 1])) + margin

            # Clamping
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w - 1, x_max)
            y_max = min(h - 1, y_max)

            return (x_min, y_min, x_max, y_max)

        """
        Method for realtime tracking mode
        """
        def on_image_loop():
            nonlocal registration
            if(realtime_tracking):
                # time.sleep(1)
                self.camera.capture_image()
                print("Captured image for realtime tracking")
                camera_roll.add_image(self.camera.color_img, self.camera.undistorted_depth_img) #self.camera.depth_img)
                roi = get_projected_roi(
                    self.virtual_space.object.object,
                    registration,
                    CONFIG.SHAPE,
                    margin=80
                )
                scan = camera_roll.get_depth_img_3d(filtering=True, roi=roi) 
                registration, fitness, rmse = self.automatic_registration.refine_registration_icp(pcd, scan, registration, threshold=CONFIG.VOXEL_SIZE ) 
                obj_to_show = self.virtual_space.object.object
                obj_to_show_copy = copy.deepcopy(obj_to_show)
                obj_to_show_copy.transform(registration)
                img = self.image_projection.create_mesh_projection(obj_to_show_copy, save_img=False) 
                return img

        if realtime_tracking:
            self.image_projection.show_img_fullscreen(on_image_loop = on_image_loop)
        else:
            self.image_projection.show_img_fullscreen()

    """
    Not necessary for the application, but can be used to show the overlay image of the virtual camera and the real camera image
    """
    def show_overlay_img_camera(self, img, obj, evaluation_mode=False):

        self.image_projection.create_mesh_projection(obj ,True, height=img.shape[0], width=img.shape[1])

        render_image = cv2.imread("Projection/output_overlay.png")
        if render_image is None:
            raise FileNotFoundError("Die Datei 'output_overlay.png' konnte nicht geladen werden.")

        if evaluation_mode:
            return render_image
            
        virtual_cam_img_bgr = np.asarray(render_image)
        real_cam_img = img

        real_h, real_w = real_cam_img.shape[:2]
        virt_h, virt_w = virtual_cam_img_bgr.shape[:2]

        if (real_h, real_w) != (virt_h, virt_w):
            print(f"Größenunterschied erkannt! Anpassung von {virt_w}x{virt_h} auf {real_w}x{real_h}")
            virtual_cam_img_bgr = cv2.resize(virtual_cam_img_bgr, (real_w, real_h), interpolation=cv2.INTER_LINEAR)

        overlay_img = cv2.addWeighted(real_cam_img, 0.7, virtual_cam_img_bgr, 0.3, 0)

        print("\n" + "="*40)
        print("STEUERUNG FÜR DAS ANZEIGEFENSTER:")
        print("  [1] - Nur das Original-Kamerabild anzeigen")
        print("  [2] - Nur das virtuelle 3D-Overlay anzeigen")
        print("  [3] - Halb-transparentes Overlay anzeigen (Standard)")
        print("  [ESC] oder [q] - Fenster schließen und weiter im Code")
        print("="*40 + "\n")

        current_mode = "3" 

        while True:
            if current_mode == "1":
                display_img = real_cam_img
            elif current_mode == "2":
                display_img = virtual_cam_img_bgr
            else:
                display_img = overlay_img

            cv2.imshow("Halb-transparentes Overlay", display_img)
            
            key = cv2.waitKey(0) & 0xFF

            if key == ord('1'):
                current_mode = "1"
                print("Modus gewechselt: Original-Kamerabild")
            elif key == ord('2'):
                current_mode = "2"
                print("Modus gewechselt: Virtuelles 3D-Overlay")
            elif key == ord('3'):
                current_mode = "3"
                print("Modus gewechselt: Halb-transparentes Overlay")
            elif key == 27 or key == ord('q'):  #ESC
                print("Fenster wird geschlossen...")
                break

        cv2.destroyWindow("Halb-transparentes Overlay")