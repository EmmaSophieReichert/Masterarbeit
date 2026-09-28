"""
Test YuNet face detection
"""

import cv2

detector = cv2.FaceDetectorYN.create(
    model="face_detection_yunet_2023mar.onnx",
    config="",
    input_size=(640, 480),
    score_threshold=0.6,
    nms_threshold=0.3
)

img = cv2.imread("CameraImages_Person1/test4.png")

if img is None:
    raise FileNotFoundError("Bild konnte nicht geladen werden!")

h, w, _ = img.shape
detector.setInputSize((w, h))

faces = detector.detect(img)[1]

if faces is not None:
    for face in faces:
        # 1. Bounding Box zeichnen
        bbox = face[0:4].astype(int)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
        
        # 2. Landmarks extrahieren und zeichnen (Indizes 4 bis 13)
        landmarks = face[4:14].astype(int).reshape((5, 2))
        
        # Farben für die einzelnen Punkte (Optional, z. B. BGR):
        # Augen: Blau/Cyan, Nase: Rot, Mundwinkel: Gelb
        colors = [
            (255, 0, 0),    # Rechtes Auge (Blau)
            (255, 255, 0),  # Linkes Auge (Cyan)
            (0, 0, 255),    # Nasenspitze (Rot)
            (0, 255, 255),  # Rechter Mundwinkel (Gelb)
            (0, 255, 255)   # Linker Mundwinkel (Gelb)
        ]
        
        for idx, (px, py) in enumerate(landmarks):
            cv2.circle(img, (px, py), 3, colors[idx], -1) # Radius 3, gefüllter Kreis

    cv2.imshow("YuNet Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else: 
    print("Kein Gesicht erkannt.")