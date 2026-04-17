# main.py

import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from collections import deque, Counter
from keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# load BOTH models at the start
hand_model = load_model("models/hand_model.h5")
face_model = load_model("models/face_model.h5")

# setup BOTH mediapipe detectors
hand_detector = ...
face_detector = ...

# separate sequences for each
hand_sequence = []
face_sequence = []
last_face_prediction = "NEUTRAL"

while True:
    ret, frame = cap.read()
    
    # --- HAND DETECTION (same logic as hand_model.py) ---
    # detect, build frame_features, append to hand_sequence
    # if len(hand_sequence) == 35: predict, get hand_result
    
    # --- FACE DETECTION (same logic as face_model.py) ---  
    # detect, build frame_features, append to face_sequence
    # if len(face_sequence) == 10: predict, update last_face_prediction
    
    # --- DECISION LAYER ---
    # if hand_result == "UNDERSTAND":
    #     check last_face_prediction, output accordingly
    # else:
    #     display hand_result directly
    
    cv2.imshow(...)