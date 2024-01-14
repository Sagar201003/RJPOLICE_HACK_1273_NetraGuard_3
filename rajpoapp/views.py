from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
from ultralytics import YOLO
from rajpoapp.tracker import Tracker
import random
from .models import FilesUpload
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import load_model
import easyocr
import matplotlib.pyplot as plt
from glob import glob
from random import sample
from PIL import ImageFont, ImageDraw, Image
import pandas as pd
stop=False
res = True
reader = easyocr.Reader(['en'])

numbers = []

def vid_inp(request):
    if request.method == "POST":
        file = request.FILES["file"]
        document = FilesUpload.objects.create(file=file)
        document.save()
        return video_feed(request, document.file.path)
    else:
        return video_feed(request, 'rajpoapp/pik.mp4')


def video_feed(request, videopath):
    cap = cv2.VideoCapture(videopath)
    model = YOLO("yolov8n.pt")
    tracker = Tracker()
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]
    detection_threshold = 0.5


    def gen_frames():
        global res
        global stop
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (IMG_SIZE_L, IMG_SIZE_B))
            if res==True :
                    # Resnet classification
                frame_features = feature_extractor.predict(frame[None, ...])
                frame_features = np.repeat(frame_features[None, ...], MAX_SEQ_LENGTH, axis=1)
                frame_mask = np.ones(shape=(1, MAX_SEQ_LENGTH), dtype="bool")

                probabilities = sequence_model.predict([frame_features, frame_mask])[0]
                top_class_index = np.argmax(probabilities)
                top_class = class_labels[top_class_index]
                confidence = probabilities[top_class_index] * 100
                cv2.putText(frame, f"Class: {top_class} ({confidence:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if confidence >= 90 or res==False:
                # Continue with YOLO and DeepSORT tracking
                res=False
                #frame = cv2.resize(frame, (IMG_SIZE_L, IMG_SIZE_B))
                results = model(frame)

                for result in results:
                    detections = []
                    for r in result.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = r
                        x1 = int(x1)
                        x2 = int(x2)
                        y1 = int(y1)
                        y2 = int(y2)
                        class_id = int(class_id)
                        if score > detection_threshold:
                            detections.append([x1, y1, x2, y2, score])

                tracker.update(frame, detections)
                
                for track in tracker.tracks:
                    bbox = track.bbox
                    x1, y1, x2, y2 = bbox
                    track_id = track.track_id
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                
                if stop==False:
                    results = reader.readtext(frame)
                    for (bbox, text, prob) in results:
                            (top_left, top_right, bottom_right, bottom_left) = bbox
                            top_left = tuple(map(int, top_left))
                            bottom_right = tuple(map(int, bottom_right))
                            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                            cv2.putText(frame, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                   

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return StreamingHttpResponse(gen_frames(), content_type="multipart/x-mixed-replace;boundary=frame")


def index(request):
    return render(request, 'live_feed.html')


# Resnet code
IMG_SIZE_L = 480
IMG_SIZE_B = 640
MAX_SEQ_LENGTH = 100
NUM_FEATURES = 2048

# Load the trained Resnet model
sequence_model = keras.models.load_model('rajpoapp/saved_model')

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(640, 480, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((640, 480, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

# Dummy DataFrame for class labels
class_labels = ['Fencing', 'Punch', 'RockClimbingIndoor', 'RopeClimbing', 'WallPushUps']
label_processor = keras.layers.experimental.preprocessing.StringLookup(
    num_oov_indices=0, vocabulary=class_labels
)