"""
Configuration constriants for BSL Live Transcription system.
Modify these values to match your trained models.

"""

# Hand model
hand_sequence_length = 30
hand_num_features    = 126
hand_data_path       = "data/hand"
hand_model_path      = "models/detection_model/hand_model.h5"
hand_class_path      = "models/class_names/hand_class_names.npy"

# Face model  
face_sequence_length = 10
face_num_features    = 3
face_data_path       = "data/face"
face_model_path      = "models/detection_model/face_model.h5"
face_class_path      = "models/class_names/face_class_names.npy"

# Model
confidence_threshold = 0.6
smoothing_window     = 5

# MediaPipe
hand_landmarker_path = "models/mediaPipe/hand_landmarker.task"
face_landmarker_path = "models/mediaPipe/face_landmarker.task"