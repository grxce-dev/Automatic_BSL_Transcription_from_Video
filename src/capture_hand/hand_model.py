"""
hand_model.py
-----------------
Live hand gesture recognition using pre-trained LSTM model.

Loads a trained model and runs real time inference on webcam input,
displying the predicted BSL sign on screen.

Controls:
    q - quit
"""

import os
import cv2
import numpy as np
import mediapipe as mp

from config import *
from tensorflow import keras
from collections import deque, Counter
from keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

hand_model = load_model(hand_model_path)
hand_class_names = list(np.load(hand_class_path, allow_pickle = True))

# MediaPipe Setup - Face
base_options = python.BaseOptions(model_asset_path = hand_landmarker_path)
options = vision.HandLandmarkerOptions(base_options = base_options, num_hands = 2)
detector = vision.HandLandmarker.create_from_options(options)

# State
hand_sequence = []
hand_prediction_history = deque(maxlen = smoothing_window)
display_text = "..."

# Helpers
def normalize(input_data):
    """
    Normalize a sequence array along the time and feature axis

    Parameters:
    -----------
    input_data : np.ndarray, shape (batch, frames, features)
                Raw input sequences

    Returns:
    --------
    np.ndarray
        Zero mean, unit variance normalized array.
    """
    mean = np.mean(input_data, axis = (1, 2), keepdims = True)
    std = np.std(input_data, axis = (1, 2), keepdims = True) + 1e-8
    return (input_data - mean) / std
    

def predict_class(model, sequence, class_names, prediction_history, threshold):
    """
    Run inference on a landmark sequence and return the smoothed prediction.

    Predictions below the confidence threshold are discarded. The most
    frequent prediction across the smoothing window is returned.

    Parameters
    ----------
    model : keras.Model
        Trained LSTM classifier.
    sequence : list of list[float]
        Sliding window of normalised landmark frames.
    class_names : list[str]
        Ordered list of class labels matching model output indices.
    prediction_history : collections.deque
        Rolling buffer of recent high-confidence prediction indices.
    threshold : float
        Minimum confidence required to record a prediction.

    Returns
    -------
    str or None
        Most frequent predicted class name, or None if history is empty.
    """

    input_data = np.expand_dims(np.array(sequence), axis = 0)
    input_data = normalize(input_data)

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
                offset = hand_num_features // 2

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

    hand_label = None

    if len(hand_sequence) == hand_sequence_length:
        hand_label = predict_class(
            hand_model, hand_sequence,
            hand_class_names, hand_prediction_history,
            confidence_threshold
        )
        
        hand_class_names = list(np.load(hand_class_path, allow_pickle=True))

        if hand_label:
            display_text = hand_label
        
    ## Display
    (text_w, text_h), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    cv2.rectangle(frame, (10, 10), (20 + text_w, 20 + text_h + baseline), (0, 0, 0), -1)
    cv2.putText(frame, display_text, (15, 15 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("BSL Live Captioning", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()