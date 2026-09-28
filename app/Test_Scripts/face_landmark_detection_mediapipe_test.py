"""
Script to test MediaPipe face and pose landmarker
"""

import cv2

from app.Test_Scripts.face_detector import FaceDetector

face_detector = FaceDetector()

# face_detector.detect_faces("CameraImages_Person1/test1.PNG")

#face_detector.detect_faces("CameraImages_Person1/rgb_05.png")

#face_detector.detect_faces("CameraImages_Person1/test.jpeg")

#face_detector.detect_faces("Images_Head_New/rgb_01.png")

from Tracking.FaceDetection.pose_detector import PoseDetector

pose_detector = PoseDetector()

for num_head in range(2, 10):
    for num_img in range(0, 3):
        img_path = f"CameraImages_Person{num_head}/rgb_{num_img:02d}.png"
        print(f"Processing {img_path}...")
        pose_detector.detect_faces(img_path)

        crop_head, bbox = pose_detector.get_roi(margin_top = 0.9, margin_bottom = 0.9, margin_sides = 1.2)

        if crop_head is not None:
            # ROI anzeigen (Achtung: RGB zu BGR für OpenCV konvertieren)
            crop_bgr = cv2.cvtColor(crop_head, cv2.COLOR_RGB2BGR)
            cv2.imshow("Kopf-ROI (Ganz)", crop_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# pose_detector.detect_faces("CameraImages_Person6/rgb_01.png")