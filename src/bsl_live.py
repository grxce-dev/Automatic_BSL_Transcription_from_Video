"""
bsl_live.py
-----------------
Live hand and face gesture recognition using pre-trained LSTM model.

Loads a trained model and runs real time inference on webcam input,
displying the predicted BSL sign on screen.

Controls:
    q - quit
"""

import os
import cv2
import numpy as np
import mediapipe as mp

from tensorflow import keras
from collections import deque, Counter
from keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Hand model
hand_sequence_length = 30
hand_num_features    = 126
hand_data_path       = "data/hand"
hand_model_path      = "models/detection_model/hand_model.h5"
hand_class_path      = "models/class_names/hand_class_names.npy"

# Face model  
face_sequence_length = 10
face_num_features    = 6
face_data_path       = "data/face"
face_model_path      = "models/detection_model/face_model.h5"
face_class_path      = "models/class_names/face_class_names.npy"

# Model
confidence_threshold = 0.6
smoothing_window     = 5

# MediaPipe
hand_landmarker_path = "models/mediaPipe/hand_landmarker.task"
face_landmarker_path = "models/mediaPipe/face_landmarker.task"


# Load model and class names 
hand_model = load_model(hand_model_path)
hand_class_names = list(np.load(hand_class_path, allow_pickle = True))

face_model = load_model(face_model_path)
face_class_names = list(np.load(face_class_path, allow_pickle = True))

# MediaPipe Setup - Hand detector
base_opts_hand = python.BaseOptions(model_asset_path = hand_landmarker_path)
hand_detector  = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(base_options=base_opts_hand, num_hands = 2)
)

# MediaPipe Setup - Face detector
base_opts_face = python.BaseOptions(model_asset_path = face_landmarker_path)
face_detector  = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(base_options=base_opts_face, num_faces = 1)
)

# State
hand_sequence = []
face_sequence = []
hand_prediction_history = deque(maxlen = smoothing_window)
face_prediction_history = deque(maxlen = smoothing_window)

last_face_prediction = "NEUTRAL"
display_text = "..."

# Helpers
def normalise(arr):
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
    mean = np.mean(arr, axis=(1, 2), keepdims = True)
    std  = np.std(arr,  axis=(1, 2), keepdims = True) + 1e-8
    return (arr - mean) / std


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
    input_data = normalise(input_data)

    prediction = model.predict(input_data, verbose=0)
    predicted_idx = np.argmax(prediction)
    confidence = prediction[0][predicted_idx]

    if confidence > threshold:
        prediction_history.append(predicted_idx)

    if prediction_history:
        best_idx = Counter(prediction_history).most_common(1)[0][0]
        return class_names[best_idx]
    return None


# Capture
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb)

    # Hand Pipeline
    hand_result = hand_detector.detect(mp_image)
    hand_features = [0.0] * hand_num_features
    hand_detected = False

    if hand_result.hand_landmarks and hand_result.handedness:
        hand_detected = True
        for i, hand in enumerate(hand_result.hand_landmarks):
            label = hand_result.handedness[i][0].category_name
            offset = 0 if label == 'Left' else hand_num_features // 2
            wrist = hand[0]

            for j, landmark in enumerate(hand):
                base = offset + j * 3
                hand_features[base:base+3] = [
                    landmark.x - wrist.x,
                    landmark.y - wrist.y,
                    landmark.z - wrist.z,
                ]

        # Draw hand landmarks
        for hand_landmarks in hand_result.hand_landmarks:
            for landmark in hand_landmarks:
                cx = int(landmark.x * frame_width)
                cy = int(landmark.y * frame_height)
                cv2.circle(frame, (cx, cy), 4, (255, 197, 211), -1)

    if hand_detected:
        hand_sequence.append(hand_features)
    else:
        hand_prediction_history.clear()
        hand_sequence = []

    if len(hand_sequence) > hand_sequence_length:
        hand_sequence.pop(0)

    # Face Pipeline
    face_result = face_detector.detect(mp_image)
    face_features = [0.0, 0.0] * 6
    face_detected = False

    if face_result.face_landmarks:
        face = face_result.face_landmarks[0]
        nose      = face[1]
        left_eye  = face[468]
        right_eye = face[473]
        mouth_l   = face[61]
        mouth_r   = face[291]

        eye_centre_x = (left_eye.x + right_eye.x) / 2
        eye_centre_y = (left_eye.y + right_eye.y) / 2
        eye_dist = (((left_eye.x - right_eye.x)**2 +
                    (left_eye.y - right_eye.y)**2) ** 0.5)

        if eye_dist > 0:
            face_features = [
                (nose.x    - eye_centre_x) / eye_dist,
                (nose.y    - eye_centre_y) / eye_dist,
                (mouth_l.x - eye_centre_x) / eye_dist,
                (mouth_l.y - eye_centre_y) / eye_dist,
                (mouth_r.x - eye_centre_x) / eye_dist,
                (mouth_r.y - eye_centre_y) / eye_dist,
            ]
            face_detected = True

        # Draw face landmarks
        h, w, _ = frame.shape
        for idx in [1, 468, 473, 61, 291]: # (nose, eye_l, eye_r, mouth_l, mouth_r)
            landmark = face[idx]
            centre_x, centre_y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (centre_x, centre_y), 5, (255, 197, 211), -1)

    if face_detected:
        face_sequence.append(face_features)
    else:
        face_prediction_history.clear()
        face_sequence = []

    if len(face_sequence) > face_sequence_length:
        face_sequence.pop(0)

    # Run Inference
    hand_label = None
    if len(hand_sequence) == hand_sequence_length:
        hand_label = predict_class(
            hand_model, hand_sequence,
            hand_class_names, hand_prediction_history,
            confidence_threshold
        )

    if len(face_sequence) == face_sequence_length:
        face_label = predict_class(
            face_model, face_sequence,
            face_class_names, face_prediction_history,
            confidence_threshold
        )
        if face_label:
            last_face_prediction = face_label

    # Decision Layer - Face + Hand
    # hand sign  :  { face state → modified output }

    modifier_map = {
        "NO"        : { "NOD": None,         "SHAKE": "NO",               "NEUTRAL": None },
        "YES"       : { "NOD": "YES",        "SHAKE": None,               "NEUTRAL": None },
        "UNDERSTAND": { "NOD": "UNDERSTAND", "SHAKE": "DON'T UNDERSTAND", "NEUTRAL": None }
    }
    
    if hand_label:
        modifiers = modifier_map.get(hand_label, {})
        result = modifiers.get(last_face_prediction, hand_label)
        display_text = result if result is not None else hand_label
    elif not hand_detected:
        display_text = "..."

    # Display
    (text_w, text_h), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    cv2.rectangle(frame, (10, 10), (20 + text_w, 20 + text_h + baseline), (0, 0, 0), -1)
    
    cv2.putText(frame, display_text, (15, 15 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, f"Face: {last_face_prediction}", (15, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("BSL Live Captioning", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()