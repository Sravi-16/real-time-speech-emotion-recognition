import os
import numpy as np
import librosa
import joblib
import random
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = "dataset"

# ================= FEATURE EXTRACTION =================
def extract_features(file):
    audio, sr = librosa.load(file, sr=16000)

    # Noise augmentation
    if random.random() < 0.4:
        noise = np.random.randn(len(audio)) * 0.004
        audio = audio + noise

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)

    feat = np.vstack([mfcc, chroma, mel])
    feat = np.mean(feat.T, axis=0)

    return feat

# ================= LABEL EXTRACT (UNIVERSAL) =================
def get_label(file):
    name = os.path.basename(file).lower()

    # RAVDESS
    if "-" in name:
        try:
            code = int(name.split("-")[2])
            return {
                1:"neutral",2:"calm",3:"happy",4:"sad",
                5:"angry",6:"fearful",7:"disgust",8:"surprised"
            }.get(code, None)
        except:
            return None

    # TESS
    if "_" in name:
        return name.split("_")[-1].replace(".wav","")

    return None

# ================= LOAD DATA =================
X, y = [], []

for file in glob(os.path.join(DATASET_PATH, "**/*.wav"), recursive=True):
    label = get_label(file)
    if label is None:
        continue

    feat = extract_features(file)
    X.append(feat)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Loaded samples:", len(X))

# ================= ENCODE =================
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

os.makedirs("model", exist_ok=True)
joblib.dump(le.classes_, "model/label_map.pkl")

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_enc
)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ================= MODEL =================
model = Sequential([
    Conv1D(64, 3, activation="relu", input_shape=(X_train.shape[1],1)),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(128, 3, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(2),

    LSTM(128),

    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(y_cat.shape[1], activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ================= TRAIN =================
early = EarlyStopping(patience=12, restore_best_weights=True)
lr = ReduceLROnPlateau(patience=5, factor=0.3, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=80,
    batch_size=32,
    callbacks=[early, lr]
)

# ================= SAVE =================
model.save("model/emotion_model.h5")
print("Model saved!")

# ================= EVALUATE =================
pred = model.predict(X_test)
y_pred = np.argmax(pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_true, y_pred)

os.makedirs("static", exist_ok=True)

plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.savefig("static/confusion_matrix.png")

plt.figure(figsize=(7,4))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.legend()
plt.title("Performance")
plt.savefig("static/performance.png")
