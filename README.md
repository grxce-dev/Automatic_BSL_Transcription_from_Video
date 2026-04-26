# Automatic BSL Transcription from Video
Aberystwyth University Major Project — A proof-of-concept system for 
real-time British Sign Language transcription using hand and facial 
landmark recognition.

## Requirements
Python 3.13, TensorFlow, OpenCV, MediaPipe, scikit-learn

Install dependencies:
    pip install tensorflow opencv-python mediapipe scikit-learn matplotlib

## Project Structure
    data/           — recorded landmark sequences (.npy)
    models/         — trained models, class names, MediaPipe task files
    src/            — source code

## Usage
1. Record hand data:      python hand_capture.py
2. Train hand model:      python predict_hand.py
3. Record face data:      python face_capture.py
4. Train face model:      python predict_face.py
5. Run live captioning:   python bsl_live.py

## Controls
    s — start recording a sequence (capture scripts only)
    q — quit

## Notes
- Change save_folder in hand_capture.py and face_capture.py 
  before each recording session
- Models must be trained before running bsl_live.py