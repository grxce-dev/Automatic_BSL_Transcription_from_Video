# Responsible for: Webcam, MediaPipe, Extract landmarks, Save .npy sequences


import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

SEQUENCE_LENGTH = 30
sequence = []

# Save frames into .npy file in folder of choice
SAVE_FOLDER = "data/_good"
os.makedir(SAVE_FOLDER, exist_ok = True)

# Load model
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options = base_options,
    num_hands = 2
)

detector = vision.HandLandmarker.create_from_options(options)

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

    frame_features = []

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for landmark in hand:
                frame_features.extend([landmark.x, landmark.y, landmark.z])

    # If only one hand detected, pad with zeros for second hand
    while len(frame_features) < 126:
        frame_features.append(0)

    sequence.append(frame_features)

    if len(sequence) == SEQUENCE_LENGTH:
        print("Captured 30 frames!")
        print(frame_features)
        break

    print("Sequence length:", len(sequence))

    #print("Hands detected:", len(result.hand_landmarks))

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()