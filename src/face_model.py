# Model face landmarks
# ----------------------------
# Input: Facial Landmarks
# Output: Neutral, Questions
# ----------------------------

import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from collections import deque, Counter
from keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SEQUENCE_LENGTH = 35
NUM_FEATURES = 0
DATA_PATH = "data"
MODEL_PATH = "models/bsl_multiclass_model.h5"

CONFIDENCE_THRESHOLD =  0.6
SMOOTHING_WINDOW =  5

#model = load_model(MODEL_PATH)

model = Sequential([
    GRU(64, input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.4),
    Dense(len(class_names), activation="softmax")
])