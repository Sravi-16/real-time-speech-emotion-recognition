import hashlib
import os
from typing import Dict, List, Optional, Tuple

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf


TARGET_SR = 22050
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
MAX_SECONDS = 4.0
MAX_FRAMES = int(np.ceil((TARGET_SR * MAX_SECONDS) / HOP_LENGTH))


EMOTION_TO_ID = {
    "happy": 0,
    "angry": 1,
    "neutral": 2,
    "sad": 3,
    "frustrated": 4,
    "confused": 5,
}
ID_TO_EMOTION = {v: k for k, v in EMOTION_TO_ID.items()}


def file_sha1(path: str) -> str:
    hasher = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            block = f.read(1 << 16)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def load_audio(path: str, sr: int = TARGET_SR) -> np.ndarray:
    data, _ = librosa.load(path, sr=sr, mono=True)
    return data.astype(np.float32)


def remove_silence(audio: np.ndarray, top_db: int = 25) -> np.ndarray:
    if len(audio) == 0:
        return audio
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return audio
    non_silent = np.concatenate([audio[start:end] for start, end in intervals], axis=0)
    return non_silent if len(non_silent) > 0 else audio


def reduce_noise(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    if len(audio) < sr // 10:
        return audio
    try:
        return nr.reduce_noise(y=audio, sr=sr).astype(np.float32)
    except Exception:
        return audio


def normalize_volume(audio: np.ndarray, target_rms: float = 0.1, eps: float = 1e-8) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(audio)) + eps)
    gain = target_rms / (rms + eps)
    normalized = audio * gain
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def pad_or_truncate(audio: np.ndarray, sr: int = TARGET_SR, max_seconds: float = MAX_SECONDS) -> np.ndarray:
    max_len = int(sr * max_seconds)
    if len(audio) > max_len:
        return audio[:max_len]
    if len(audio) < max_len:
        return np.pad(audio, (0, max_len - len(audio)))
    return audio


def preprocess_audio(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    audio = remove_silence(audio)
    audio = reduce_noise(audio, sr=sr)
    audio = normalize_volume(audio)
    audio = pad_or_truncate(audio, sr=sr)
    return audio


def extract_features(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    feats = np.concatenate([mfcc, delta], axis=0)
    if feats.shape[1] < MAX_FRAMES:
        pad_cols = MAX_FRAMES - feats.shape[1]
        feats = np.pad(feats, ((0, 0), (0, pad_cols)))
    else:
        feats = feats[:, :MAX_FRAMES]
    return feats.astype(np.float32)


def preprocess_file_to_features(path: str, sr: int = TARGET_SR) -> np.ndarray:
    audio = load_audio(path, sr=sr)
    audio = preprocess_audio(audio, sr=sr)
    return extract_features(audio, sr=sr)


def write_wav(path: str, audio: np.ndarray, sr: int = TARGET_SR) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio, sr)


def summarize_array_stats(x: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }
