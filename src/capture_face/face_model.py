# Model face landmarks

import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from collections import deque, Counter
from keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuration
SEQUENCE_LENGTH = 35
NUM_FEATURES = 1
CONFIDENCE_THRESHOLD =  0.6
SMOOTHING_WINDOW =  5

# Paths
DATA_PATH = "data"
MODEL_PATH = "models/bsl_multiclass_model.h5"

# Load Model
model = load_model(MODEL_PATH)

# MediaPipe Setup - Face
base_options = python.BaseOptions(model_asset_path="models/face_landmarker.task" )
options = vision.FaceLandmarkerOptions( base_options = base_options, num_faces = 1)
detector = vision.FaceLandmarker.create_from_options(options)

display_text = "..."
sequence = []
prediction_history = deque(maxlen = SMOOTHING_WINDOW)

class_names = sorted([
    folder for folder in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, folder))
])

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    frame_features = [0] * NUM_FEATURES
    result = detector.detect(mp_image)


    #if result.face_landmarks:
