# Setup Instructions

## System Requirements

* Python 3.8 or above
* Microphone for real-time audio input
* Web browser (Chrome/Edge recommended)

## Step 1: Clone the Repository

Download or clone the GitHub repository to your local system.

Example:
git clone https://github.com/your-repository-link

## Step 2: Navigate to Project Folder

Open the project folder in Visual Studio Code or terminal.

Example:
cd Speech-Emotion-Recognition

## Step 3: Install Required Libraries

Install all required Python dependencies using the requirements file.

Command:
pip install -r requirements.txt

## Step 4: Train the Model (Optional)

If the model is not already available, run the training script to train the CNN-LSTM model.

Example:
python train_model.py

This step extracts features from the dataset and trains the deep learning model.

## Step 5: Run the Application

Start the Flask server to launch the real-time emotion recognition dashboard.

Command:
python app.py

## Step 6: Open the Dashboard

After running the server, open a web browser and go to:

http://localhost:5000

The dashboard will display the detected emotion, confidence score, system accuracy, emotion trend graph, and emotion distribution chart.

## Step 7: Start Real-Time Detection

Click the Start button on the dashboard to begin recording audio from the microphone and predicting emotions in real time.

## Output

The system will continuously detect emotions from speech and update the dashboard with live results.
