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

SEQUENCE_LENGTH = 35
NUM_FEATURES = 126
DATA_PATH = "data"
MODEL_PATH = "models/bsl_multiclass_model.h5"

CONFIDENCE_THRESHOLD =  0.3 # 0.80
SMOOTHING_WINDOW =  1 # 5

model = load_model(MODEL_PATH)

class_names = sorted([
    folder for folder in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, folder))
])

print("Loaded classes:", class_names)


base_options = python.BaseOptions(
    model_asset_path="data/hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

sequence = []
prediction_history = deque(maxlen=SMOOTHING_WINDOW)

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

    result = detector.detect(mp_image)
    frame_features = []

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for landmark in hand:
                frame_features.extend([landmark.x, landmark.y, landmark.z])

    # Pad to fixed size (126 features)
    while len(frame_features) < NUM_FEATURES:
        frame_features.append(0)

    sequence.append(frame_features)

    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    display_text = "..."


    # PREDICT WHEN WINDOW FULL

    if len(sequence) == SEQUENCE_LENGTH:

        input_data = np.array(sequence)
        input_data = np.expand_dims(input_data, axis=0)  # Shape = (1, 35, 126)

        # Standardise (same as training)
        mean = np.mean(input_data, axis=(1, 2), keepdims=True)
        std = np.std(input_data, axis=(1,2), keepdims=True) + 1e-8
        input_data = (input_data - mean) / std

        prediction = model.predict(input_data, verbose=0)
        predicted_index = np.argmax(prediction)
        confidence = prediction[0][predicted_index]

        if confidence > CONFIDENCE_THRESHOLD:
            prediction_history.append(predicted_index)

        if len(prediction_history) > 0:
            most_common = Counter(prediction_history).most_common(1)[0][0]
            display_text = f"{class_names[most_common]} ({confidence:.2f})"

    # DISPLAY RESULT
    cv2.putText(frame,
                display_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("BSL Live Captioning", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()