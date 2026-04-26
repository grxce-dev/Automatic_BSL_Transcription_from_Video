"""
face_model.py
-----------------
Live face gesture recognition using pre-trained LSTM model.

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
from mediapipe.tasks import python
from keras.models import load_model
from collections import deque, Counter
from mediapipe.tasks.python import vision

# CONFIGURATION:
face_sequence_length = 10
face_num_features    = 6
confidence_threshold = 0.6
smoothing_window     = 5
face_model_path      = "models/detection_model/face_model.h5"
face_class_path      = "models/class_names/face_class_names.npy"
face_landmarker_path = "models/mediaPope/face_landmarker.task"

# Load Model and Face classes 
face_model = load_model(face_model_path)
face_class_names = list(np.load(face_class_path, allow_pickle = True))

# MediaPipe Setup - Face
base_options = python.BaseOptions(model_asset_path = face_landmarker_path)
options = vision.FaceLandmarkerOptions( base_options = base_options, num_faces = 1)
detector = vision.FaceLandmarker.create_from_options(options)

# State
face_sequence = []
face_prediction_history = deque(maxlen = smoothing_window)
last_face_prediction = "NEUTRAL"

# Helpers
def normalise(input_data):
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
    std  = np.std(input_data,  axis = (1, 2), keepdims = True) + 1e-8
    return (input_data - mean) / std


def predict_class(model, sequence, class_names, pred_history, threshold):
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
    input_data = np.expand_dims(np.array(sequence), axis=0)
    input_data = normalise(input_data)

    prediction = model.predict(input_data, verbose=0)
    predicted_idx = np.argmax(prediction)
    confidence = prediction[0][predicted_idx]

    if confidence > threshold:
        pred_history.append(predicted_idx)

    if pred_history:
        best_idx = Counter(pred_history).most_common(1)[0][0]
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

    face_result = detector.detect(mp_image)
    frame_features = [0.0, 0.0]
    face_detected = False

    # Detect Face
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
        for idx in [1, 468, 473]:
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
 
    # Predict when window is full/sequence of frames reached
    if len(face_sequence) == face_sequence_length:
        face_label = predict_class(
            face_model, face_sequence,
            face_class_names, face_prediction_history,
            confidence_threshold
        )
        if face_label:
            last_face_prediction = face_label
    
    ## Display
    (text_w, text_h), baseline = cv2.getTextSize(last_face_prediction, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    cv2.rectangle(frame, (10, 10), (20 + text_w, 20 + text_h + baseline), (0, 0, 0), -1)
    cv2.putText(frame, f"Face: {last_face_prediction}", (15, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imshow("BSL Live Captioning", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 
cap.release()
cv2.destroyAllWindows()