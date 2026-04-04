import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

SEQUENCE_LENGTH = 10
important_points = [70, 63, 105, 66, 107]  # eyebrow

sequence = []
recording = False

# Save frames into .npy file in folder of choice
SAVE_FOLDER = "data/FACE"
os.makedirs(SAVE_FOLDER, exist_ok = True)

# load model
base_options_face = python.BaseOptions(model_asset_path="models/mediapipe_model/face_landmarker.task" )
options_face = vision.FaceLandmarkerOptions(
    base_options = base_options_face,
    num_faces = 1
)

face_detector = vision.FaceLandmarker.create_from_options(options_face)

# Start video capture
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

    result = face_detector.detect(mp_image)
    key = cv2.waitKey(1) & 0xFF

    # Press 's' to start recording
    if key == ord("s") and not recording:
        print("Recording started")
        recording = True
        sequence = []

    # Press 'q' to quit
    if key == ord("q"):
        break

    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION )

    # If recording append frame (even if detection fails)
    if recording:
        frame_features = [0] * (len(important_points) * 2)
        if result.face_landmarks:
            face = result.face_landmarks[0]

            reference = face[1] # nose tip reference point (more stable)

            for idx, point in enumerate(important_points):
                landmark = face[point]
                base = idx * 2

                frame_features[base:base+2] = [
                    landmark.x - reference.x,
                    landmark.y - reference.y
                ]


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

            # Testing - quality control
            #print(f"Saved to {filepath}")
            #print("Shape:", sequence_array.shape)

    cv2.imshow("Recording", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()