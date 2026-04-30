"""
hand_capture.py
-------------------
Records hand landmark sequences for BSL gesture recognition training.

Captures 30 frame sequences of hand landmarks using MediaPipe,
normalized relative to the wrist joint. Sequences are saved as .npy
files under 'data/hand/<class_name>/.

Controls:
    s - start recording a sequence
    q - quit

"""

import os
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuration
sequence = []
recording = False
sequence_length = 30
hand_landmarker_path = "models/mediaPipe/hand_landmarker.task"

# IMPORTANT: Change this to the class you are recording before each session.
save_folder = "data/hand/"

# Load model
base_options_hands = python.BaseOptions(model_asset_path = hand_landmarker_path) 
options_hands = vision.HandLandmarkerOptions(
    base_options = base_options_hands,
    num_hands = 2
) 

hand_detector = vision.HandLandmarker.create_from_options(options_hands)

# Start video capture
cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format = mp.ImageFormat.SRGB,
        data = rgb
    )

    result = hand_detector.detect(mp_image)
    key = cv2.waitKey(1) & 0xFF

    for hand_landmarks in result.hand_landmarks:
        for landmark in hand_landmarks:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x,y), 5,(255, 197, 211), -1)

    if key == ord("s") and not recording:
        print("Recording started")
        recording = True
        sequence = []

    if key == ord("q"):
        break      

    # If recording append frame (even if detection fails)
    if recording:
        
        cv2.putText(frame, "Recording...", (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        frame_features = [0] * 126

        if result.hand_landmarks and result.handedness:

            for i, hand in enumerate(result.hand_landmarks):
                label = result.handedness[i][0].category_name
                offset = 0 if label == "Left" else 63

                wrist = hand[0]

                for j, landmark in enumerate(hand):
                    base = offset + j * 3
                    frame_features[base:base+3] = [
                        landmark.x - wrist.x,    
                        landmark.y - wrist.y,
                        landmark.z - wrist.z
                    ]

        sequence.append(frame_features)
        print("Frames:", len(sequence))

        # End recording and save
        if len(sequence) == sequence_length:
            recording = False
            sequence_array = np.array(sequence)

            file_count = len(os.listdir(save_folder))
            filename = f"file_{file_count}.npy"
            filepath = os.path.join(save_folder, filename)

            np.save(filepath, sequence_array)

    cv2.imshow("Recording", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()