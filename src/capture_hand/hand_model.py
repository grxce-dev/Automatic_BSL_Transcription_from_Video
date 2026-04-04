# Model hand landmarks
# ----------------------------
# Input: Hand Landmarks
# Output: Good, Bad, ...
# ----------------------------

import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from collections import deque, Counter
from keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# -------------------------------------------

# CONFIGURATION
SEQUENCE_LENGTH = 35
NUM_FEATURES = 126

CONFIDENCE_THRESHOLD =  0.6
SMOOTHING_WINDOW =  5

# PATHS
DATA_PATH = "data"
MODEL_PATH = "detection_model/bsl_multiclass_model.h5"

# LOAD MODEL
model = load_model(MODEL_PATH)

# MEDIAPIPE SETUP - HANDS
base_options = python.BaseOptions( model_asset_path="data/hand_landmarker.task")
options = vision.HandLandmarkerOptions( base_options = base_options, num_hands = 2)
detector = vision.HandLandmarker.create_from_options(options)

display_text = "..."
sequence = []
prediction_history = deque(maxlen = SMOOTHING_WINDOW)

class_names = sorted([
    folder for folder in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, folder))
])

# START WEBCAM
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

    # 1. DETECT HANDS
    if result.hand_landmarks and result.handedness:
        # 1.5. DRAW LANDMARKS
        for hand_landmarks in result.hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 2. CONVERT TO NORMALISED FEATURE VECTOR
        for i, hand in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].category_name
            offset = 0 if label == "Left" else NUM_FEATURES // 2

            wrist = hand[0]

            for j, landmark in enumerate(hand):
                base = offset + j * 3
                frame_features[base:base+3] = [
                    landmark.x - wrist.x,
                    landmark.y - wrist.y,
                    landmark.z - wrist.z
                ]

    # 3. STORE LAST 35 FRAMES
    if result.hand_landmarks:
        sequence.append(frame_features)
    
    if not result.hand_landmarks:
        prediction_history.clear()
        display_text = "..."

    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    # PREDICT WHEN WINDOW FULL. - 4. WHEN FULL:
    if len(sequence) == SEQUENCE_LENGTH:

        input_data = np.array(sequence)
        input_data = np.expand_dims(input_data, axis=0)  # Shape = (1, 35, 126)

        # Standardise (same as training) - 4.1. NORMALISE SEQUENCE
        mean = np.mean(input_data, axis=(1, 2), keepdims=True)
        std = np.std(input_data, axis=(1,2), keepdims=True) + 1e-8
        input_data = (input_data - mean) / std

        # MODEL PREDICTION - 4.2. PREDICT WITH MODEL
        prediction = model.predict(input_data, verbose=0)
        predicted_index = np.argmax(prediction)
        confidence = prediction[0][predicted_index]

        # 4.3. SMOOTH PREDICTIONS. -- ???  SEE IF STILL LAGGY??
        #if confidence > CONFIDENCE_THRESHOLD:
            #prediction_history.append(predicted_index)

        prediction_history.append(predicted_index)

        # 5. DISPLAY MORE STABLE PREDICTIONS
        if len(prediction_history) > 0:
            most_common = Counter(prediction_history).most_common(1)[0][0]
            display_text = f"{class_names[most_common]}"
        else:
            display_text = "..."


    cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("BSL Live Captioning", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()