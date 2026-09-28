import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision import PoseLandmarkerResult

# Make sure to download the model file and place it in the same directory
MODEL_PATH = "Tracking/FaceDetection/pose_landmarker_full.task"

class PoseDetector:

    def __init__(self, model_path=MODEL_PATH):
        base_options = python.BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.CPU,)
        self.options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.3,
        )
        self.result = None
        self.mp_image = None

    def detect_faces(self, image_path):
        self.mp_image = mp.Image.create_from_file(image_path)
        img_bgr = cv2.imread(image_path)

        with vision.PoseLandmarker.create_from_options(self.options) as landmarker:
            result = landmarker.detect(self.mp_image)

        if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
            self.result = result
            #self.show_landmarks_face(img_bgr) #for visualization
            return result
        else:
            print("Keine Pose/Gesicht gefunden. Starte iterative Suche...")
            return None #return self.search_head_pose_iterative(image_path, min_grid=2, max_grid=5, overlap=0.2)

    def convert_result(
        self,
        result: PoseLandmarkerResult,
        offset_x: float,
        offset_y: float,
        crop_w: float,
        crop_h: float,
        full_w: float,
        full_h: float,
    ) -> PoseLandmarkerResult:
        if not result or not result.pose_landmarks:
            return result

        converted_pose_landmarks = []

        for pose_landmarks in result.pose_landmarks:
            converted_single_pose = []
            for lm in pose_landmarks:
                crop_px_x = lm.x * crop_w
                crop_px_y = lm.y * crop_h

                global_norm_x = (crop_px_x + offset_x) / full_w
                global_norm_y = (crop_px_y + offset_y) / full_h

                new_lm = NormalizedLandmark(
                    x=global_norm_x,
                    y=global_norm_y,
                    z=lm.z,
                    visibility=getattr(lm, "visibility", None),
                    presence=getattr(lm, "presence", None),
                )
                converted_single_pose.append(new_lm)

            converted_pose_landmarks.append(converted_single_pose)

        return PoseLandmarkerResult(
            pose_landmarks=converted_pose_landmarks,
            pose_world_landmarks=result.pose_world_landmarks,
            segmentation_masks=result.segmentation_masks,
        )

    def get_face_rotation(self, z_threshold=0.02, nose_threshold=0.08):
        if not self.result or not self.result.pose_landmarks:
            return None

        landmarks = self.result.pose_landmarks[0]
        nose_z = landmarks[0].z
        left_ear_z = landmarks[7].z
        right_ear_z = landmarks[8].z

        z_diff = abs(right_ear_z - left_ear_z)

        print("\n")
        print("Nose z", nose_z)
        print("Left ear z:", left_ear_z)
        print("Right ear z:", right_ear_z)
        print("\n")

        if (left_ear_z - nose_z) > nose_threshold and (right_ear_z - nose_z) > nose_threshold:
            return "Nose up"

        if z_diff < z_threshold:
            return "Nose up"
        elif left_ear_z < right_ear_z:
            return "Left ear up"
        else:
            return "Right ear up"

    def show_detection_image(self, pitch, yaw, roll, rmat):
        img = self.mp_image.numpy_view()
        img_h, img_w, _ = img.shape

        nose_lm = self.result.pose_landmarks[0][0]
        origin_x = nose_lm.x * img_w
        origin_y = nose_lm.y * img_h

        axis_length = 80.0
        axes_3d = np.array(
            [
                [axis_length, 0, 0],   # X (Rot)
                [0, axis_length, 0],   # Y (Grün)
                [0, 0, axis_length],   # Z (Blau)
            ]
        )

        rmat_plot = rmat.copy()
        rmat_plot[1, :] *= -1  

        axes_2d = np.dot(axes_3d, rmat_plot.T)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)

        ax.plot([origin_x, origin_x + axes_2d[0, 0]], [origin_y, origin_y + axes_2d[0, 1]], color="red", linewidth=3, label="X (Pitch)")
        ax.plot([origin_x, origin_x + axes_2d[1, 0]], [origin_y, origin_y + axes_2d[1, 1]], color="green", linewidth=3, label="Y (Yaw)")
        ax.plot([origin_x, origin_x + axes_2d[2, 0]], [origin_y, origin_y + axes_2d[2, 1]], color="blue", linewidth=3, label="Z (Roll)")

        ax.scatter([origin_x], [origin_y], color="yellow", s=30, zorder=5)

        info_text = f"Pitch: {pitch:.1f}°\nYaw:   {yaw:.1f}°\nRoll:  {roll:.1f}°"
        ax.text(
            15, 35, info_text, color="white", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.6)
        )

        plt.title("Head Pose 3D-Koordinatensystem (Pose Landmarker)", fontsize=14)
        plt.axis("off")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    def get_roi(self, margin_top=1.5, margin_bottom=1.2, margin_sides=1.5):
        if not self.result or not self.result.pose_landmarks or self.mp_image is None:
            print("Keine Detektion vorhanden, um ein ROI zu erstellen.")
            return None, None

        img = self.mp_image.numpy_view()
        img_h, img_w, _ = img.shape
        landmarks = self.result.pose_landmarks[0]

        # Wichtige Kopf-Landmarks aus MediaPipe Pose:
        # 0: Nase, 1: Linkes Auge (innen), 2: Linkes Auge, 3: Linkes Auge (außen),
        # 4: Rechtes Auge (innen), 5: Rechtes Auge, 6: Rechtes Auge (außen),
        # 7: Linkes Ohr, 8: Rechtes Ohr
        head_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        head_pts_x = [landmarks[i].x * img_w for i in head_indices]
        head_pts_y = [landmarks[i].y * img_h for i in head_indices]

        raw_x_min, raw_x_max = min(head_pts_x), max(head_pts_x)
        raw_y_min, raw_y_max = min(head_pts_y), max(head_pts_y)

        head_w = raw_x_max - raw_x_min
        head_h = raw_y_max - raw_y_min

        head_w = max(head_w, 20.0)
        head_h = max(head_h, 20.0)

        face_max_with = max(head_w, head_h)

        x_min = int(raw_x_min - (face_max_with * margin_sides))
        x_max = int(raw_x_max + (face_max_with * margin_sides))
        y_min = int(raw_y_min - (face_max_with * margin_top))
        y_max = int(raw_y_max + (face_max_with * margin_bottom))

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w, x_max)
        y_max = min(img_h, y_max)

        crop_img = img[y_min:y_max, x_min:x_max]
        if crop_img.size == 0:
            print("Fehler: Erstellter Crop ist leer.")
            return None, None

        bbox = (x_min, y_min, x_max, y_max)
        return crop_img, bbox

    def show_landmarks(self, img_bgr, window_name="Mediapipe Pose Landmarker Test"):
        print("Showing detected landmarks...")
        result = self.result
        if not result or not result.pose_landmarks:
            print("Keine Landmarks zum Anzeigen vorhanden.")
            return

        annotated_img = img_bgr.copy()
        img_h, img_w, _ = annotated_img.shape

        for pose_landmarks in result.pose_landmarks:
            for lm in pose_landmarks:
                px_x = int(lm.x * img_w)
                px_y = int(lm.y * img_h)
                cv2.circle(annotated_img, (px_x, px_y), 2, (0, 255, 0), -1)

        cv2.imshow(window_name, annotated_img)
        print("\n[Landmarks-Anzeige] Drücke eine beliebige Taste zum Schließen des Fensters...")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    def show_landmarks_face(self, img_bgr, window_name="Mediapipe Pose Landmarker Test"):
        KEYPOINT_COLORS = {
            0: (0, 0, 255),      # 0: Nase -> Rot
            2: (255, 0, 0),      # 2: Linkes Auge -> Blau
            5: (0, 255, 255),    # 5: Rechtes Auge -> Gelb
            7: (255, 0, 255),    # 7: Linkes Ohr -> Magenta / Pink
            8: (0, 255, 0)       # 8: Rechtes Ohr -> Grün
        }

        KEYPOINT_NAMES = {
            0: "Nase",
            2: "L_Auge",
            5: "R_Auge",
            7: "L_Ohr",
            8: "R_Ohr"
        }
        print("Showing detected landmarks...")
        result = self.result
        if not result or not result.pose_landmarks:
            print("Keine Landmarks zum Anzeigen vorhanden.")
            return

        annotated_img = img_bgr.copy()
        img_h, img_w, _ = annotated_img.shape
        default_color = (200, 200, 200)  # Grau für alle restlichen Punkte

        for pose_landmarks in result.pose_landmarks:
            for idx, lm in enumerate(pose_landmarks):
                px_x = int(lm.x * img_w)
                px_y = int(lm.y * img_h)
                
                color = KEYPOINT_COLORS.get(idx, default_color)
                
                radius = 4 if idx in KEYPOINT_COLORS else 2
                if idx in KEYPOINT_NAMES:
                    cv2.circle(annotated_img, (px_x, px_y), radius, color, -1)
                    label = f"{KEYPOINT_NAMES[idx]}"
                    cv2.putText(
                        annotated_img, 
                        label, 
                        (int(px_x) + 7, int(px_y) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, 
                        color, 
                        1
                    )

        #----------- cropping image
        
        # img = annotated_img
        # img_h, img_w, _ = img.shape

        # crop_w = int(img_w * 0.3)
        # crop_h = int(img_h * 0.5)

        # x_start = (img_w - crop_w) // 2
        # x_end = x_start + crop_w
        # y_start = (img_h - crop_h) // 2
        # y_end = y_start + crop_h

        # annotated_img = img[y_start:y_end, x_start:x_end]

        #----------- cropping image

        cv2.imshow(window_name, annotated_img)
        print("\n[Landmarks-Anzeige] Drücke eine beliebige Taste zum Schließen des Fensters...")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)