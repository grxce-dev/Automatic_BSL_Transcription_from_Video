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
        image_format = mp.ImageFormat.SRGB,
        data = rgb
    )

    result = detector.detect(mp_image)

    frame_features = [0.0, 0.0]
    face_detected = False

    if result.face_landmarks:
        face = result.face_landmarks[0]

        nose = face[1]
        left_eye = face[468]
        right_eye = face[473]

        eye_centre_x = (left_eye.x + right_eye.x) / 2
        eye_centre_y = (left_eye.y + right_eye.y) / 2

        eye_dist = ((
            (left_eye.x - right_eye.x) **2 + 
            (left_eye.y - right_eye.y) **2
            ) **0.5)

        if eye_dist > 0:
            nose_rel_x = (nose.x - eye_centre_x) / eye_dist
            nose_rel_y = (nose.y - eye_centre_y) / eye_dist

        h, w, _ = frame.shape
        for idx in [1, 468, 473]:
            lm = face[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 197, 211), -1)
        
        # Only accumulate when a face is present (mirrors hand_model logic)
    if face_detected:
        sequence.append(frame_features)
    else:
        prediction_history.clear()
        sequence = []
        display_text = "..."
 
    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)
 
    # Predict when window is full
    if len(sequence) == SEQUENCE_LENGTH:
        input_data = np.array(sequence)                  # (10, 2)
        input_data = np.expand_dims(input_data, axis=0)  # (1, 10, 2)
 
        # Normalisation
        mean = np.mean(input_data, axis=(1, 2), keepdims=True)
        std  = np.std(input_data,  axis=(1, 2), keepdims=True) + 1e-8
        input_data = (input_data - mean) / std
 
        prediction = model.predict(input_data, verbose=0)
        predicted_index = np.argmax(prediction)
        confidence = prediction[0][predicted_index]
 
        if confidence > CONFIDENCE_THRESHOLD:
            prediction_history.append(predicted_index)
 
        if len(prediction_history) > 0:
            most_common = Counter(prediction_history).most_common(1)[0][0]
            display_text = class_names[most_common]
        else:
            display_text = "..."
 
    cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("BSL Face Model", frame)
 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 
cap.release()
cv2.destroyAllWindows()