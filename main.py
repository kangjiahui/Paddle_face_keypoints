import paddlehub as hub
import numpy as np
import cv2

face_landmark = hub.Module(name="face_landmark_localization")
# Replace face detection module to speed up predictions but reduce performance
face_landmark.set_face_detector_module(hub.Module(name="ultra_light_fast_generic_face_detector_1mb_320"))

def priori_faces(path):
    face_dict = {}
    return face_dict


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    data = face_landmark.keypoint_detection(images=[frame])
    for points in data[0]['data']:
        # print(points)
        for x, y in points:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 2)
    # or
    # result = face_landmark.keypoint_detection(paths=['/PATH/TO/IMAGE'])
    cv2.imshow("Face_Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
