# Model face landmarks

import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from collections import deque, Counter
from keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuration
SEQUENCE_LENGTH = 35
NUM_FEATURES = 0
CONFIDENCE_THRESHOLD =  0.6
SMOOTHING_WINDOW =  5

# Paths
DATA_PATH = "data"
MODEL_PATH = "models/bsl_multiclass_model.h5"

# Load Model
model = load_model(MODEL_PATH)

# MediaPipe Setup - Face
base_options = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face landmarker instance with the video mode:
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

#with FaceLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...