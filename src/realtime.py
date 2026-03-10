import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime
from collections import Counter
from scipy.signal import wiener

print("🔄 Loading CNN+LSTM realtime engine...")

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "emotion_model.h5")
LABEL_PATH = os.path.join(BASE_DIR, "model", "label_map.pkl")
PAST_DIR = os.path.join(BASE_DIR, "static", "past_calls")

os.makedirs(PAST_DIR, exist_ok=True)

# ================= LOAD MODEL =================
model = load_model(MODEL_PATH)
labels = joblib.load(LABEL_PATH)

print("✅ CNN+LSTM Model loaded")

# ================= LIVE STATE =================
current_emotion = "Listening..."
current_confidence = 0.0
current_accuracy = 0.0
call_time = 0
stop_flag = False

emotion_history = []
time_history = []
emotion_counts = Counter()

# ================= SETTINGS =================
SAMPLE_RATE = 16000
CHUNK = 2.0
SILENCE_TH = 0.01

# ================= RESET =================
def reset_live():
    global current_emotion, current_confidence, current_accuracy
    global call_time, stop_flag, emotion_history, time_history, emotion_counts

    current_emotion = "Listening..."
    current_confidence = 0.0
    current_accuracy = 0.0
    call_time = 0
    stop_flag = False
    emotion_history.clear()
    time_history.clear()
    emotion_counts.clear()

# ================= RECORD =================
def record_chunk():
    try:
        audio = sd.rec(int(SAMPLE_RATE * CHUNK),
                       samplerate=SAMPLE_RATE,
                       channels=1,
                       dtype="float32")
        sd.wait()
        return audio.flatten()
    except:
        return np.zeros(int(SAMPLE_RATE * CHUNK), dtype="float32")

# ================= FEATURE (Noise Robust) =================
def extract_features(audio):
    # Noise reduction
    audio = wiener(audio)

    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE)

    feat = np.hstack([
        np.mean(mfcc.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(mel.T, axis=0)
    ])

    return feat.reshape(1, -1, 1)

# ================= MAIN LOOP =================
def main_loop():
    global current_emotion, current_confidence, current_accuracy
    global call_time, stop_flag, emotion_history, time_history, emotion_counts

    reset_live()
    print("\n🎤 CNN+LSTM realtime started\n", flush=True)

    start = time.time()
    full_audio = []

    while not stop_flag:

        call_time = int(time.time() - start)

        audio = record_chunk()
        full_audio.extend(audio.tolist())

        volume = np.max(np.abs(audio))
        if volume < SILENCE_TH:
            continue

        feat = extract_features(audio)

        pred = model.predict(feat, verbose=0)[0]
        idx = np.argmax(pred)

        emotion = labels[idx]
        confidence = float(pred[idx]) * 100

        current_emotion = emotion
        current_confidence = round(confidence, 2)
        current_accuracy = round(0.8 * current_accuracy + 0.2 * confidence, 2)

        emotion_history.append(emotion)
        time_history.append(call_time)
        emotion_counts[emotion] += 1

        if len(emotion_history) > 40:
            emotion_history.pop(0)
            time_history.pop(0)

        print(f"{emotion} | {confidence:.2f}% | {call_time}s", flush=True)

    # ================= SAVE CALL =================
    try:
        if len(full_audio) > SAMPLE_RATE:
            filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"
            sf.write(os.path.join(PAST_DIR, filename),
                     np.array(full_audio, dtype="float32"),
                     SAMPLE_RATE)
            print("💾 Call saved:", filename)
    except Exception as e:
        print("Save error:", e)
