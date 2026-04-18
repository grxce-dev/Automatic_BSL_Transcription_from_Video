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
hand_sequence_length = 35
hand_num_features = 126

confidence_threshold =  0.6
smoothing_window =  5

# Load Model
hand_model = load_model("models/bsl_multiclass_model.h5")

# Load Hand classes 
hand_class_names = list(np.load("models/hand_class_names.npy", allow_pickle = True))
""" class_names = sorted([
    folder for folder in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, folder))
]) """

# MediaPipe Setup - Hand Detector
base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options = base_options, num_hands = 2)
detector = vision.HandLandmarker.create_from_options(options)

# State
hand_sequence = []
hand_prediction_history = deque(maxlen = smoothing_window)

display_text = "..."

# Helpers
def normalise(arr):
    mean = np.mean(input_data, axis = (1, 2), keepdims = True)
    std = np.std(input_data, axis = (1, 2), keepdims = True) + 1e-8
    input_data = (input_data - mean) / std

def predict_class(model, sequence, class_names, prediction_history, threshold):
    input_data = np.expand_dims(np.array(sequence), axis = 0)
    input_data = normalise(input_data)

    prediction = model.predict(input_data, verbose = 0)
    predicted_idx = np.argmax(prediction)
    confidence = prediction[0][predicted_idx]

    if confidence > threshold:
        prediction_history.append(predicted_idx)

    if prediction_history:
        best_idx = Counter(prediction_history).most_common(1)[0][0]
        return class_names[best_idx]
    return None

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb)

    result = detector.detect(mp_image)
    frame_features = [0] * hand_num_features
    hand_detected = False

    # Hand Pipeline
    if result.hand_landmarks and result.handedness:
        hand_detected = True
        for i, hand in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].category_name

            if label == "Left": 
                offset = 0
            else:
                hand_num_features // 2

            wrist = hand[0]

            for j, landmark in enumerate(hand):
                base = offset + j * 3
                frame_features[base:base+3] = [
                    landmark.x - wrist.x,
                    landmark.y - wrist.y,
                    landmark.z - wrist.z
                ]

    if hand_detected:
        hand_sequence.append(frame_features)
    else:
        hand_prediction_history.clear()
        hand_sequence = []

    if len(hand_sequence) > hand_sequence_length:
        hand_sequence.pop(0)

    # Predict when window is full/sequence of frames reached
    hand_label = None
    if len(hand_sequence) == hand_sequence_length:
        hand_label = predict_class(
            hand_model, hand_sequence,
            hand_class_names, hand_prediction_history,
            confidence_threshold
        )
        
    """     if len(hand_sequence) == sequence_length:
            input_data = np.array(hand_sequence)
            input_data = np.expand_dims(input_data, axis=0) """
        
    ## Display
    text_w, text_h, baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    cv2.rectangle(frame, (10, 10), (20 + text_w, 20 + text_h + baseline), (0, 0, 0), -1)
    cv2.putText(frame, display_text, (15, 15 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("BSL Live Captioning", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()