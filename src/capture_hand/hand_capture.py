# Capture landmarks
# ----------------------------
# Input: Webcam
# Output: 
# ----------------------------

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

SEQUENCE_LENGTH = 35
sequence = []
recording = False

# Save frames into .npy file in folder of choice
SAVE_FOLDER = "data/HANDS/BAD/RH"
os.makedirs(SAVE_FOLDER, exist_ok = True)

# Load models
base_options_hands = python.BaseOptions(model_asset_path="models/mediapipe_model/hand_landmarker.task") 
options_hands = vision.HandLandmarkerOptions(
    base_options = base_options_hands,
    num_hands = 2
)

hand_detector = vision.HandLandmarker.create_from_options(options_hands)

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

    result = hand_detector.detect(mp_image)
    key = cv2.waitKey(1) & 0xFF

    # Draw hands :)
    for hand_landmarks in result.hand_landmarks:
        mp_drawing.draw_landmarks( frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Press 's' to start recording
    if key == ord("s") and not recording:
        print("Recording started")
        recording = True
        sequence = []
        
        cv2.putText(frame, "Recording...", (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Press 'q' to quit
    if key == ord("q"):
        break      

    # If recording append frame (even if detection fails)
    if recording:
        
        frame_features = [0] * 126 # incl. padding for missing hand

        if result.hand_landmarks and result.handedness:

            for i, hand in enumerate(result.hand_landmarks):
                label = result.handedness[i][0].category_name
                offset = 0 if label == "Left" else 63

                wrist = hand[0]

                for j, landmark in enumerate(hand):
                    base = offset + j * 3
                    frame_features[base:base+3] = [
                        landmark.x - wrist.x,    # Normalisation relative to wrist
                        landmark.y - wrist.y,
                        landmark.z - wrist.z
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