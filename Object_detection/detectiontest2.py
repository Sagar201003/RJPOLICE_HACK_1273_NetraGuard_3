import os
import random
import cv2
import cv
from ultralytics import YOLO
from tracker import Tracker
from scipy.optimize import linear_sum_assignment as linear_assignment


VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'People_walking_in_airport.mp4')
video_out_path = os.path.join('.', 'out.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
#H, W, _ = frame.shape
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model_path = os.path.join('.','yolov8n.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

threshold = 0.5

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, results.names[int(class_id)], (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
            # cv2.putText(frame,(int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    cap_out.write(frame)
    #cv2.imshow('OpenPose using OpenCV', frame)
    cv2.imshow('frame',frame)
    cv2.waitKey(250)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()
