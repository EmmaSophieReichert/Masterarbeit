import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision import FaceLandmarkerResult
import cv2

# Make sure to download the model file and place it in the same directory
MEDIAPIPE_MODEL_PATH = "Tracking/FaceDetection/face_landmarker.task"

class FaceDetector:

    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH)
        self.options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.3,
        )
        self.result = None
        self.mp_image = None
    
    def detect_faces(self, image_path):
        self.mp_image = mp.Image.create_from_file(image_path)
        with vision.FaceLandmarker.create_from_options(self.options) as landmarker:
            result = landmarker.detect(self.mp_image)

        if result.facial_transformation_matrixes and result.face_landmarks:
            self.result = result
            #self.show_landmarks(cv2.imread(image_path), window_name=f"Landmarks - full") #for visualization
            self.get_face_rotation()
            return result
        else:
            print("Kein Gesicht gefunden. Starte iterative Suche...")
            return self.search_head_pose_iterative(image_path, min_grid=2, max_grid=10, overlap=0.2)
        
    def convert_result(self, result: FaceLandmarkerResult, offset_x: float, offset_y: float, crop_w: float, crop_h: float, full_w: float, full_h: float) -> FaceLandmarkerResult:
        if not result or not result.face_landmarks:
            return result

        converted_face_landmarks = []

        for face_landmarks in result.face_landmarks:
            converted_single_face = []
            for lm in face_landmarks:
                crop_px_x = lm.x * crop_w
                crop_px_y = lm.y * crop_h

                global_norm_x = (crop_px_x + offset_x) / full_w
                global_norm_y = (crop_px_y + offset_y) / full_h

                new_lm = NormalizedLandmark(
                    x=global_norm_x,
                    y=global_norm_y,
                    z=lm.z,
                    visibility=getattr(lm, 'visibility', None),
                    presence=getattr(lm, 'presence', None)
                )
                converted_single_face.append(new_lm)

            converted_face_landmarks.append(converted_single_face)

        return FaceLandmarkerResult(
            face_landmarks=converted_face_landmarks,
            face_blendshapes=result.face_blendshapes,
            facial_transformation_matrixes=result.facial_transformation_matrixes
        )
    
    def search_head_pose_iterative(self, image_path, min_grid=2, max_grid=5, overlap=0.2):

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Fehler: Bild unter '{image_path}' konnte nicht geladen werden.")
            return None

        img_h, img_w, _ = img_bgr.shape

        with vision.FaceLandmarker.create_from_options(self.options) as landmarker:
            
            def check_tile(x_min, y_min, x_max, y_max, tile_label):
                x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
                x_max, y_max = min(img_w, int(x_max)), min(img_h, int(y_max))

                crop = img_bgr[y_min:y_max, x_min:x_max]
                if crop.size == 0:
                    return None

                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                
                result = landmarker.detect(mp_image)

                if result.facial_transformation_matrixes and len(result.facial_transformation_matrixes) > 0:
                    print(f"-> Gesicht gefunden in: {tile_label}")
                    self.result = result
                    self.show_landmarks(crop, window_name=f"Landmarks - {tile_label}")
                    global_result = self.convert_result(
                        result=result,
                        offset_x=x_min,
                        offset_y=y_min,
                        crop_w=x_max - x_min,
                        crop_h=y_max - y_min,
                        full_w=img_w,
                        full_h=img_h
                    )
                    self.result = global_result
                    self.show_landmarks(img_bgr, window_name=f"Landmarks - {tile_label}")
                    self.get_face_rotation()
                    return result
                return None

            for grid in range(min_grid, max_grid + 1):
                cols = grid
                rows = grid
                
                tile_w = img_w / cols
                tile_h = img_h / rows

                overlap_x = tile_w * overlap
                overlap_y = tile_h * overlap

                print(f"Prüfe Grid-Stufe {grid}x{grid} (Kachelgröße: {int(tile_w)}x{int(tile_h)} px)...")

                center_x = img_w / 2
                center_y = img_h / 2

                center_x_min = center_x - (tile_w / 2) - overlap_x
                center_y_min = center_y - (tile_h / 2) - overlap_y
                center_x_max = center_x + (tile_w / 2) + overlap_x
                center_y_max = center_y + (tile_h / 2) + overlap_y

                res = check_tile(center_x_min, center_y_min, center_x_max, center_y_max, f"Mitten-Kachel (Grid {grid}x{grid})")
                if res:
                    return res

                for r in range(rows):
                    for c in range(cols):
                        x_min = (c * tile_w) - overlap_x
                        y_min = (r * tile_h) - overlap_y
                        x_max = ((c + 1) * tile_w) + overlap_x
                        y_max = ((r + 1) * tile_h) + overlap_y

                        res = check_tile(x_min, y_min, x_max, y_max, f"Grid {grid}x{grid} [Zeile {r+1}, Spalte {c+1}]")
                        if res:
                            return res

        print("Kein Gesicht gefunden (bis Grid 5x5 durchsucht).")
        return None

    def show_detection_image(self, pitch, yaw, roll, rmat):
        img = self.mp_image.numpy_view()
        img_h, img_w, _ = img.shape

        nose_lm = self.result.face_landmarks[0][1]
        origin_x = nose_lm.x * img_w
        origin_y = nose_lm.y * img_h

        axis_length = 80.0

        axes_3d = np.array(
            [
                [axis_length, 0, 0],  # X-Achse (Rot)
                [0, axis_length, 0],  # Y-Achse (Grün)
                [0, 0, axis_length],  # Z-Achse (Blau / Blickrichtung)
            ]
        )

        rmat_plot = rmat.copy()
        rmat_plot[1, :] *= -1

        axes_2d = np.dot(axes_3d, rmat_plot.T)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)

        ax.plot(
            [origin_x, origin_x + axes_2d[0, 0]],
            [origin_y, origin_y + axes_2d[0, 1]],
            color="red",
            linewidth=3,
            label="X (Pitch)",
        )

        ax.plot(
            [origin_x, origin_x + axes_2d[1, 0]],
            [origin_y, origin_y + axes_2d[1, 1]],
            color="green",
            linewidth=3,
            label="Y (Yaw)",
        )

        ax.plot(
            [origin_x, origin_x + axes_2d[2, 0]],
            [origin_y, origin_y + axes_2d[2, 1]],
            color="blue",
            linewidth=3,
            label="Z (Roll)",
        )

        ax.scatter(
            [origin_x], [origin_y], color="yellow", s=30, zorder=5
        )

        info_text = f"Pitch: {pitch:.1f}°\nYaw:   {yaw:.1f}°\nRoll:  {roll:.1f}°"
        ax.text(
            15,
            35,
            info_text,
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="black", alpha=0.6
            ),
        )

        plt.title("Head Pose 3D-Koordinatensystem", fontsize=14)
        plt.axis("off")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    def get_roi(self):
        pass

    def get_face_rotation(self):
        if self.result:
            matrix = self.result.facial_transformation_matrixes[0]
            rmat = matrix[:3, :3]

            pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
            yaw = np.degrees(
                np.arctan2(-rmat[2, 0], np.sqrt(rmat[2, 1] ** 2 + rmat[2, 2] ** 2))
            )
            roll = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))

            print("--- MEDIAPIPE DIREKTBERECHNUNG ---")
            print(f"Pitch: {pitch:6.1f}°")
            print(f"Yaw:   {yaw:6.1f}°")
            print(f"Roll:  {roll:6.1f}°")

            self.show_detection_image(pitch, yaw, roll, rmat)

            return pitch, yaw, roll
        return None
    
    def show_landmarks(self, img_bgr, window_name="Face Landmarks"):
        print("Showing detected landmarks...")
        result = self.result
        if not result or not result.face_landmarks:
            print("Keine Landmarks zum Anzeigen vorhanden.")
            return

        annotated_img = img_bgr.copy()
        img_h, img_w, _ = annotated_img.shape

        for face_landmarks in result.face_landmarks:
            for lm in face_landmarks:
                px_x = int(lm.x * img_w)
                px_y = int(lm.y * img_h)

                cv2.circle(annotated_img, (px_x, px_y), 1, (0, 255, 0), -1)

        cv2.imshow(window_name, annotated_img)
        print("\n[Landmarks-Anzeige] Drücke eine beliebige Taste zum Schließen des Fensters...")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)