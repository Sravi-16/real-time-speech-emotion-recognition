# Real-Time Speech Emotion Recognition for Call Centers

## Project Overview

This project implements a Real-Time Speech Emotion Recognition (SER) system designed for call center environments. The system detects human emotions from live speech input using a hybrid CNN-LSTM deep learning model. It captures audio through a microphone, processes the speech signal, extracts relevant acoustic features, and predicts the emotional state of the speaker in real time.

Understanding customer emotions during conversations helps organizations improve service quality, monitor customer satisfaction, and enhance agent performance. This project provides a practical AI-based solution for monitoring customer emotions during live calls.

## Objectives

* Detect emotions from live speech input in real time
* Apply deep learning techniques for emotion classification
* Provide visualization of emotional patterns through a dashboard
* Support monitoring of customer interactions in call center environments

## Features

* Real-time speech emotion detection using microphone input
* Hybrid CNN-LSTM deep learning model for emotion classification
* Extraction of MFCC, Chroma, and Mel Spectrogram features
* Dashboard visualization with emotion trends and distribution graphs
* Automatic storage of recorded calls for analysis

## Emotions Detected

The system detects the following eight emotional states:

* Neutral
* Calm
* Happy
* Sad
* Angry
* Fearful
* Surprised
* Disgust

## Technology Stack

Frontend:
HTML, CSS, JavaScript, Chart.js

Backend:
Python, Flask

Machine Learning:
TensorFlow, Keras, Librosa, NumPy, Scikit-learn

Tools:
Visual Studio Code

## Project Structure

src/ – Contains source code for the model, real-time processing, and dashboard
docs/ – Contains project documentation and presentation files
requirements.txt – Python dependencies required to run the project
architecture.png – System architecture diagram
setup_instructions.md – Instructions to run the project

## Applications

* Call center emotion monitoring
* Customer satisfaction analysis
* Agent performance evaluation
* Customer experience improvement

## Future Improvements

* Improve model accuracy using larger datasets
* Add advanced visualization graphs
* Support multilingual speech emotion recognition
* Deploy system on cloud platforms
