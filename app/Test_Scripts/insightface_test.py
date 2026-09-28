import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(allowed_modules=['detection'])
app.prepare(ctx_id=0, det_size=(640, 640)) # Bei großen/hochauflösenden Bildern auf (1024, 1024) erhöhen

img = cv2.imread("CameraImages_Person1/test4.png")
faces = app.get(img)

for face in faces:
    bbox = face.bbox.astype(int)
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    landmarks = face.kps.astype(int)
    for px, py in landmarks:
        cv2.circle(img, (px, py), 3, (0, 0, 255), -1)

cv2.imshow("Insightface Test", img)
cv2.waitKey(0)