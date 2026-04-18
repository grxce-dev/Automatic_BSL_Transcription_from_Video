# Capture face landmarks

import os
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# CONFIGURATION
SEQUENCE_LENGTH = 10
sequence = []
recording = False

# Save frames into .npy file in folder of choice
SAVE_FOLDER = "data/face"
os.makedirs(SAVE_FOLDER, exist_ok = True)

# Load model
base_options_face = python.BaseOptions(model_asset_path="models/face_landmarker.task" )
options_face = vision.FaceLandmarkerOptions(
    base_options = base_options_face,
    num_faces = 1
)

face_detector = vision.FaceLandmarker.create_from_options(options_face)

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

    result = face_detector.detect(mp_image)
    key = cv2.waitKey(1) & 0xFF

    # DRAW PINK DOTS <3
    for face_landmarks in result.face_landmarks:

        h, w, _ = frame.shape
        indices = [1, 468, 473] # Nose = 1, Left eye centre = 468, Right eye centre = 473
        
        for idx in indices:
            landmark = face_landmarks[idx]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx,cy), 5,(255, 197, 211), -1)

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

        frame_features = [0,0]

        if result.face_landmarks:
            
            face = result.face_landmarks[0]

            nose = face[1]
            left_eye = face[468]
            right_eye = face[473]

            eye_centre_x = (left_eye.x + right_eye.x) / 2
            eye_centre_y = (left_eye.y + right_eye.y) / 2

            eye_dist = ((
                (left_eye.x - right_eye.x) **2 + 
                (left_eye.y - right_eye.y) **2
                ) **0.5)

            if eye_dist > 0:
                nose_rel_x = (nose.x - eye_centre_x) / eye_dist
                nose_rel_y = (nose.y - eye_centre_y) / eye_dist

                frame_features = [nose_rel_x, nose_rel_y]

        sequence.append(frame_features)
        print("Frames:", len(sequence))

        # End recording and save
        if len(sequence) == SEQUENCE_LENGTH:
            recording = False
            sequence_array = np.array(sequence)

            file_count = len(os.listdir(SAVE_FOLDER))
            filename = f"file_{file_count}.npy"
            filepath = os.path.join(SAVE_FOLDER, filename)

            np.save(filepath, sequence_array)

    cv2.imshow("Recording", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()