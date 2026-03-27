from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch

from backend.audio_utils import ID_TO_EMOTION, TARGET_SR, extract_features, preprocess_audio
from training.model import CNNLSTMEmotionModel, get_device


class EmotionPredictor:
    def __init__(
        self,
        model_path: str = "models/emotion_cnn_lstm.pt",
        smooth_window: int = 3,
        silence_rms_threshold: float = 0.01,
        ema_alpha: float = 0.65,
    ):
        self.device = get_device()
        ckpt = torch.load(model_path, map_location=self.device)
        self.emotion_to_id = ckpt["emotion_to_id"]
        self.id_to_emotion = {v: k for k, v in self.emotion_to_id.items()}
        self.model = CNNLSTMEmotionModel(n_classes=len(self.emotion_to_id)).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.prob_window: Deque[np.ndarray] = deque(maxlen=smooth_window)
        self.history: List[Tuple[float, str, float]] = []
        self.silence_rms_threshold = silence_rms_threshold
        self.ema_alpha = ema_alpha
        self._ema_probs: np.ndarray | None = None
        self._class_prior: np.ndarray | None = None
        self.prior_alpha = 0.03
        # Light post-training calibration (no retraining): promote neutral/happy, suppress confused drift.
        self.class_bias = {
            "happy": 1.25,
            "angry": 1.00,
            "neutral": 1.30,
            "sad": 1.00,
            "frustrated": 0.90,
            "confused": 0.70,
        }
        self._recent_labels: Deque[str] = deque(maxlen=6)

    def predict_chunk(self, audio_chunk: np.ndarray, timestamp: float) -> Dict[str, object]:
        if audio_chunk.ndim > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)
        audio_chunk = audio_chunk.astype(np.float32)

        rms = float(np.sqrt(np.mean(np.square(audio_chunk)) + 1e-10))
        if rms < self.silence_rms_threshold:
            return {
                "label": "no_speech",
                "confidence": 0.0,
                "rms": rms,
                "probs": {self.id_to_emotion[i]: 0.0 for i in range(len(self.id_to_emotion))},
            }

        audio_chunk = preprocess_audio(audio_chunk, sr=TARGET_SR)
        feats = extract_features(audio_chunk, sr=TARGET_SR)
        x = torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Adaptive debiasing: if one class dominates over time, down-weight it.
        if self._class_prior is None:
            self._class_prior = np.ones_like(probs) / len(probs)
        self._class_prior = (1.0 - self.prior_alpha) * self._class_prior + self.prior_alpha * probs
        adjusted = probs / np.power(self._class_prior + 1e-8, 0.35)
        probs = adjusted / (np.sum(adjusted) + 1e-8)

        # Apply class calibration weights.
        weight_vec = np.array(
            [self.class_bias[self.id_to_emotion[i]] for i in range(len(probs))],
            dtype=np.float32,
        )
        probs = probs * weight_vec
        probs = probs / (np.sum(probs) + 1e-8)

        if self._ema_probs is None:
            self._ema_probs = probs
        else:
            self._ema_probs = self.ema_alpha * probs + (1.0 - self.ema_alpha) * self._ema_probs

        self.prob_window.append(probs)
        window_mean = np.mean(np.stack(list(self.prob_window), axis=0), axis=0)
        smoothed = 0.35 * window_mean + 0.65 * self._ema_probs

        # Anti-dominance guard: if output is repeatedly "confused", downscale it temporarily.
        if len(self._recent_labels) >= 4 and list(self._recent_labels)[-4:].count("confused") >= 3:
            confused_id = next((i for i, name in self.id_to_emotion.items() if name == "confused"), None)
            if confused_id is not None:
                smoothed[confused_id] *= 0.75
                smoothed = smoothed / (np.sum(smoothed) + 1e-8)

        pred_id = int(np.argmax(smoothed))
        pred_label = self.id_to_emotion[pred_id]
        confidence = float(smoothed[pred_id] * 100.0)
        self._recent_labels.append(pred_label)
        self.history.append((timestamp, pred_label, confidence))

        return {
            "label": pred_label,
            "confidence": confidence,
            "rms": rms,
            "probs": {self.id_to_emotion[i]: float(smoothed[i] * 100.0) for i in range(len(smoothed))},
        }
