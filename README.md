# Automatic_BSL_Transcription_from_Video
Aberswywth University Major Project - A proof of concept project into the feasibility of real time British Sign Language transcription

1. Project Overview

Explain:
What problem you're solving
Why dynamic modelling (LSTM)
Why MediaPipe landmarks

2. Project Structure
Explain each folder briefly.

3. Installation
pip install -r requirements.txt

4. How to Collect Data
python src/capture.py

5. How to Train
python src/train.py

6. How to Predict
python src/predict.py

# Requirements:
numpy==1.26.4
opencv-python==4.11.0.86
opencv-contrib-python==4.11.0.86
mediapipe==0.10.32
tensorflow==2.20.0
pytest==9.0.2
sounddevice==0.5.5