"""
face_capture.py
-------------------
Records face (nose tip, irisis of both eyes) landmark sequences for BSL 
gesture recognition training.

Captures 15 frame sequences of face landmarks using MediaPipe.
Sequences are saved as .npy files under 'data/face/<class_name>/.

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

# CONFIGURATION
sequence    = []
recording   = False
face_sequence_length = 15
face_landmarker_path = "models/mediaPipe/face_landmarker.task"

# IMPORTANT: Change this to the class you are recording before each session.
save_folder = "data/face"

# Load model
base_options_face = python.BaseOptions(model_asset_path = face_landmarker_path )
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

    # Draw landmarks
    for face_landmarks in result.face_landmarks:
        h, w, _ = frame.shape
        indices = [1, 468, 473, 61, 291] # (nose, eye_l, eye_r, mouth_l, mouth_r)
        for index in indices:
            landmark = face_landmarks[index]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx,cy), 5,(255, 197, 211), -1)

    # Press 's' to start recording
    if key == ord("s") and not recording:
        print("Recording started")
        recording = True
        sequence = []

        cv2.putText(frame, "Recording...", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Press 'q' to quit
    if key == ord("q"):
        break

    if recording:
        face_features = [0.0] * 6
        
        if result.face_landmarks:
            
            face = result.face_landmarks[0]

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

        sequence.append(face_features)
        print("Frames:", len(sequence))

        # End recording and save
        if len(sequence) == face_sequence_length:
            recording = False
            sequence_array = np.array(sequence)

            cv2.putText(frame, "Recording Finished", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            file_count = len(os.listdir(save_folder))
            filename = f"file_{file_count}.npy"
            filepath = os.path.join(save_folder, filename)

            np.save(filepath, sequence_array)

    cv2.imshow("Recording", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()