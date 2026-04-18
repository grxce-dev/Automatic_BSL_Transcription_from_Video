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
hand_num_features    = 126   # 2 hands × 21 landmarks × (x, y, z)
 
face_sequence_length  = 10
face_num_features     = 2     # nose_rel_x, nose_rel_y
 
confidence_threshold  = 0.6
smoothing_window      = 5
 
# ── LOAD MODELS ────────────────────────────────────────────────────────────────
hand_model = load_model("models/hand_model.h5")
face_model = load_model("models/face_model.h5")
 
 