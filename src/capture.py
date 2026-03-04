# Responsible for: Webcam, MediaPipe, Extract landmarks, Save .npy sequences

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

SEQUENCE_LENGTH = 35
sequence = []

recording = False

# Save frames into .npy file in folder of choice
SAVE_FOLDER = "data/BAD/RH"
os.makedirs(SAVE_FOLDER, exist_ok = True)

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
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    # cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 's' to start recording
    if key == ord("s") and not recording:
        print("Recording started")
        recording = True
        sequence = []

    # Press 'q' to quit
    if key == ord("q"):
        break

    # If recording append frame (even if detection fails)
    if recording:

        frame_features = []

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                for landmark in hand:
                    frame_features.extend([landmark.x, landmark.y, landmark.z])

        # Pad to 126 features (2 hands max)
        while len(frame_features) < 126:
            frame_features.append(0)

        sequence.append(frame_features)

        print("Frames:", len(sequence))

        if len(sequence) == SEQUENCE_LENGTH:
            recording = False
            print("Captured 35 frames!")

            sequence_array = np.array(sequence)

            file_count = len(os.listdir(SAVE_FOLDER))
            filename = f"file_{file_count}.npy"
            filepath = os.path.join(SAVE_FOLDER, filename)

            np.save(filepath, sequence_array)

            # Testing - quality control
            print(f"Saved to {filepath}")
            print("Shape:", sequence_array.shape)
    
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()