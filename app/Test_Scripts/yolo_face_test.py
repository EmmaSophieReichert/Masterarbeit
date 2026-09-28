"""
Test YOLO face recognition
"""

import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo26l-pose.pt")
#model = YOLO("yoloe-26m-seg.pt")
#model = YOLO("yolov8m-face.pt")

#model.set_classes(["person"])

img = cv2.imread("CameraImages_Person2/rgb_01.png")
#img = cv2.imread("CameraImages_Person1/test4.png")
results = model.predict(img)#, conf=0.25)

results[0].show()

CONF_THRESHOLD = 0.0

KEYPOINT_COLORS = {
    0: (0, 0, 255),      # 0: Nase -> Rot
    1: (255, 0, 0),      # 1: Linkes Auge -> Blau
    2: (0, 255, 255),    # 2: Rechtes Auge -> Gelb
    3: (255, 0, 255),    # 3: Linkes Ohr -> Magenta / Pink
    4: (0, 255, 0)       # 4: Rechtes Ohr -> Grün
}

KEYPOINT_NAMES = {
    0: "Nase",
    1: "L_Auge",
    2: "R_Auge",
    3: "L_Ohr",
    4: "R_Ohr"
}

for r in results:
    if r.keypoints is not None and len(r.keypoints.data) > 0:
        coords = r.keypoints.xy.cpu().numpy()
        confs = r.keypoints.conf.cpu().numpy()

        for person_idx in range(len(coords)):
            face_coords = coords[person_idx][:5]
            face_confs = confs[person_idx][:5]

            for i, ((x, y), score) in enumerate(zip(face_coords, face_confs)):
                if score >= CONF_THRESHOLD:
                    color = KEYPOINT_COLORS.get(i, (255, 255, 255))
                    
                    cv2.circle(img, (int(x), int(y)), 5, color, -1)
                    
                    label = f"{KEYPOINT_NAMES[i]}"#: {score:.2f}"
                    cv2.putText(
                        img, 
                        label, 
                        (int(x) + 7, int(y) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, 
                        color, 
                        1
                    )

#----------- cropping image
img_h, img_w, _ = img.shape

crop_w = int(img_w * 0.3)
crop_h = int(img_h * 0.5)

x_start = (img_w - crop_w) // 2
x_end = x_start + crop_w
y_start = (img_h - crop_h) // 2
y_end = y_start + crop_h

img = img[y_start:y_end, x_start:x_end]

#----------- cropping image

cv2.namedWindow("Yolo26l Pose Test", cv2.WINDOW_NORMAL)
cv2.imshow("Yolo26l Pose Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()