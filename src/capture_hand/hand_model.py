# Model hand landmarks

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
NUM_FEATURES = 126
CONFIDENCE_THRESHOLD =  0.6
SMOOTHING_WINDOW =  5

# Paths
DATA_PATH = "data"
MODEL_PATH = "models/bsl_multiclass_model.h5"

# Load Model
model = load_model(MODEL_PATH)

# MediaPipe Setuup - Hands
base_options = python.BaseOptions( model_asset_path="models/hand_landmarker.task")
options = vision.HandLandmarkerOptions( base_options = base_options, num_hands = 2)
detector = vision.HandLandmarker.create_from_options(options)

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

    # Detect hands
    if result.hand_landmarks and result.handedness:
        # Normalised feature generation?
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

    # Store the last 35 frames (size of signs)
    if result.hand_landmarks:
        sequence.append(frame_features)
    
    # if no sign: clean and display '...'
    if not result.hand_landmarks:
        prediction_history.clear()
        sequence = []
        display_text = "..."

    # if sequence is longer than sequence length re-check and remove prev sign
    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    # Predict when window is full/sequence of frames reached
    if len(sequence) == SEQUENCE_LENGTH:

        input_data = np.array(sequence)
        input_data = np.expand_dims(input_data, axis=0)  # Shape = (1, 35, 126)

        # Normalisation
        mean = np.mean(input_data, axis=(1, 2), keepdims=True)
        std = np.std(input_data, axis=(1,2), keepdims=True) + 1e-8
        input_data = (input_data - mean) / std

        # Model Prediction
        prediction = model.predict(input_data, verbose=0)
        predicted_index = np.argmax(prediction)
        confidence = prediction[0][predicted_index]

        # Display is confidence is higher than the confidence threshold
        if confidence > CONFIDENCE_THRESHOLD:
            prediction_history.append(predicted_index)

        # Display stable predictions
        if len(prediction_history) > 0:
            most_common = Counter(prediction_history).most_common(1)[0][0]
            display_text = f"{class_names[most_common]}"
        else:
            display_text = "..."

    ## MAKE SURE THEY ABIDE BY THE WCAG STANDARDS:
    cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("BSL Live Captioning", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()