# Capture landmarks
# ----------------------------
# Input: Webcam
# Output: 
# ----------------------------

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
        data = rgb
    )

    result = detector.detect(mp_image)
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

        if result.hand_landmarks and result.handedness:
            for i, hand in enumerate(result.hand_landmarks):
                label = result.handedness[i][0].category_name
                offset = 0 if label == "Left" else 63

                wrist = hand[0]

                for j, landmark in enumerate(hand):
                    base = offset + j * 3
                    frame_features[base:base] = [
                        landmark.x - wrist.x,    # Normalisation relative to wrist
                        landmark.y - wrist.y,
                        landmark.z - wrist.z
                    ]
                    
        # Pad to 126 features (2 hands max)
        while len(frame_features) < 126:
            frame_features.append(0)

        #if len(sequence) > 0:
        #    prev = sequence[-1]
        #    velocity = np.array(frame_features) - np.array(prev)
        #    frame_features = np.concatenate([frame_features, velocity]) 
        # NUM_FEATURES = 252

        sequence.append(frame_features)
        print("Frames:", len(sequence))

        if len(sequence) == SEQUENCE_LENGTH:
            recording = False
            sequence_array = np.array(sequence)

            file_count = len(os.listdir(SAVE_FOLDER))
            filename = f"file_{file_count}.npy"
            filepath = os.path.join(SAVE_FOLDER, filename)

            np.save(filepath, sequence_array)

            # Testing - quality control
            print(f"Saved to {filepath}")
            print("Shape:", sequence_array.shape)

    cv2.imshow("Recording", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()