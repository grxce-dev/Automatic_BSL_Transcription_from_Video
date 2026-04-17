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
        for landmark in face_landmarks:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x,y), 5,(255, 197, 211), -1)

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

        if result.detections:
            detection = result.detections[0]

            bounding_box = detection.location_data.relative_bounding_box
            face_x, face_y = bounding_box.xmin, bounding_box.ymin
            face_w, face_h = bounding_box.width, bounding_box.height

            nose_tip = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP)
            nose_x, nose_y = nose_tip.x, nose_tip.y

            nose_rel_x = (nose_x - face_x) / face_w
            nose_rel_y = (nose_y - face_y) / face_h

            frame_features = [nose_rel_x, nose_rel_y]

        else:
            frame_features = [0,0]

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