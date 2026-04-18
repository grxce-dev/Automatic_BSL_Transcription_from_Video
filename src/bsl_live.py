# bsl_live.py — Parallel hand + face inference for BSL live captioning
 
import os
import cv2
import numpy as np
import mediapipe as mp

from tensorflow import keras
from collections import deque, Counter
from keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
 
# ── CONFIGURATION ──────────────────────────────────────────────────────────────
hand_sequence_length = 35
hand_num_features = 126   # 2 hands × 21 landmarks × (x, y, z)
 
face_sequence_length  = 10
face_num_features = 2     # nose_rel_x, nose_rel_y
 
confidence_threshold  = 0.6
smoothing_window = 5
 
# ── LOAD MODELS ────────────────────────────────────────────────────────────────
hand_model = load_model("models/hand_model.h5")
face_model = load_model("models/face_model.h5")

# ── LOAD CLASS NAMES ───────────────────────────────────────────────────────────

# Hand classes come from data/ top-level folders
DATA_PATH = "data"
hand_class_names = sorted([
    f for f in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, f))
])

# Face classes saved separately during predict_face.py training
face_class_names = list(np.load("models/face_class_names.npy", allow_pickle=True))

print("Hand classes:", hand_class_names)
print("Face classes:", face_class_names)

# ── MEDIAPIPE DETECTORS ────────────────────────────────────────────────────────
base_opts_hand = python.BaseOptions(model_asset_path="models/hand_landmarker.task")
hand_detector  = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(base_options=base_opts_hand, num_hands=2)
)

base_opts_face = python.BaseOptions(model_asset_path="models/face_landmarker.task")
face_detector  = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(base_options=base_opts_face, num_faces=1)
)

# ── STATE ──────────────────────────────────────────────────────────────────────
hand_sequence = []
face_sequence = []
hand_pred_history = deque(maxlen = smoothing_window)
face_pred_history = deque(maxlen = smoothing_window)

last_face_prediction = "NEUTRAL"
display_text = "..."

# ── HELPERS ────────────────────────────────────────────────────────────────────
def normalize(arr):
    mean = np.mean(arr, axis=(1, 2), keepdims=True)
    std  = np.std(arr,  axis=(1, 2), keepdims=True) + 1e-8
    return (arr - mean) / std


def predict_class(model, sequence, class_names, pred_history, threshold):
    input_data = np.expand_dims(np.array(sequence), axis=0)
    input_data = normalize(input_data)

    prediction     = model.predict(input_data, verbose=0)
    predicted_idx  = np.argmax(prediction)
    confidence     = prediction[0][predicted_idx]

    if confidence > threshold:
        pred_history.append(predicted_idx)

    if pred_history:
        best_idx = Counter(pred_history).most_common(1)[0][0]
        return class_names[best_idx]
    return None


# ── CAPTURE ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # ── HAND PIPELINE ──────────────────────────────────────────────────────────
    hand_result   = hand_detector.detect(mp_image)
    hand_features = [0.0] * hand_num_features
    hand_detected = False

    if hand_result.hand_landmarks and hand_result.handedness:
        hand_detected = True
        for i, hand in enumerate(hand_result.hand_landmarks):
            label  = hand_result.handedness[i][0].category_name
            offset = 0 if label == "Left" else hand_num_features // 2
            wrist  = hand[0]
            for j, lm in enumerate(hand):
                base = offset + j * 3
                hand_features[base:base+3] = [
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z,
                ]

        # Draw pink dots on hands
        for hand_lms in hand_result.hand_landmarks:
            for lm in hand_lms:
                cx = int(lm.x * frame_width)
                cy = int(lm.y * frame_height)
                cv2.circle(frame, (cx, cy), 4, (255, 197, 211), -1)

    if hand_detected:
        hand_sequence.append(hand_features)
    else:
        hand_pred_history.clear()
        hand_sequence = []

    if len(hand_sequence) > hand_sequence_length:
        hand_sequence.pop(0)

    # ── FACE PIPELINE ──────────────────────────────────────────────────────────
    face_result   = face_detector.detect(mp_image)
    face_features = [0.0, 0.0]
    face_detected = False

    if face_result.face_landmarks:
        face      = face_result.face_landmarks[0]
        nose      = face[1]
        left_eye  = face[468]
        right_eye = face[473]

        eye_cx   = (left_eye.x + right_eye.x) / 2
        eye_cy   = (left_eye.y + right_eye.y) / 2
        eye_dist = ((left_eye.x - right_eye.x)**2 +
                    (left_eye.y - right_eye.y)**2) ** 0.5

        if eye_dist > 0:
            face_features = [
                (nose.x - eye_cx) / eye_dist,
                (nose.y - eye_cy) / eye_dist,
            ]
            face_detected = True

        # Draw pink dots on face
        h, w, _ = frame.shape
        for idx in [1, 468, 473]:
            lm = face[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 197, 211), -1)

    if face_detected:
        face_sequence.append(face_features)
    else:
        face_pred_history.clear()
        face_sequence = []

    if len(face_sequence) > face_sequence_length:
        face_sequence.pop(0)

    # ── RUN INFERENCE ──────────────────────────────────────────────────────────
    hand_label = None
    if len(hand_sequence) == hand_sequence_length:
        hand_label = predict_class(
            hand_model, hand_sequence,
            hand_class_names, hand_pred_history,
            confidence_threshold
        )

    if len(face_sequence) == face_sequence_length:
        face_label = predict_class(
            face_model, face_sequence,
            face_class_names, face_pred_history,
            confidence_threshold
        )
        if face_label:
            last_face_prediction = face_label

    # ── DECISION LAYER ─────────────────────────────────────────────────────────
    # hand sign  :  { face state → modified output }

    modifier_map = {
        "UNDERSTAND": { "NOD": "UNDERSTAND", "SHAKE": "DON'T UNDERSTAND", "NEUTRAL": None },
    }

    if hand_label:
        modifiers = modifier_map.get(hand_label, {})
        display_text = modifiers.get(last_face_prediction, hand_label)
    elif not hand_detected:
        display_text = "..."

    # ── DISPLAY ────────────────────────────────────────────────────────────────

    text_w, text_h, baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    cv2.rectangle(frame, (10, 10), (20 + text_w, 20 + text_h + baseline), (0, 0, 0), -1)
    cv2.putText(frame, display_text, (15, 15 + text_h), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
# TESTING
    cv2.putText(frame, f"Face: {last_face_prediction}", (15, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("BSL Live Captioning", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()