import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from backend.audio_utils import EMOTION_TO_ID, extract_features, file_sha1, preprocess_audio, preprocess_file_to_features


@dataclass
class SampleMeta:
    path: str
    dataset: str
    speaker: str
    label: str


RAVDESS_MAP = {
    1: "neutral",
    2: "neutral",      # calm -> neutral
    3: "happy",
    4: "sad",
    5: "angry",
    6: "frustrated",   # fearful -> frustrated
    7: "frustrated",   # disgust -> frustrated
    8: "confused",     # surprised -> confused
}

TESS_MAP = {
    "happy": "happy",
    "angry": "angry",
    "neutral": "neutral",
    "sad": "sad",
    "fear": "frustrated",
    "disgust": "frustrated",
    "pleasant_surprise": "confused",
    "surprise": "confused",
}

CREMAD_MAP = {
    "HAP": "happy",
    "ANG": "angry",
    "NEU": "neutral",
    "SAD": "sad",
    "FEA": "frustrated",
    "DIS": "frustrated",
}


def _parse_ravdess(path: str) -> Optional[SampleMeta]:
    name = os.path.basename(path).replace(".wav", "")
    parts = name.split("-")
    if len(parts) < 7:
        return None
    emotion_code = int(parts[2])
    speaker_id = parts[-1]
    if emotion_code not in RAVDESS_MAP:
        return None
    return SampleMeta(path=path, dataset="ravdess", speaker=f"ravdess_{speaker_id}", label=RAVDESS_MAP[emotion_code])


def _parse_tess(path: str) -> Optional[SampleMeta]:
    name = os.path.basename(path).replace(".wav", "")
    tokens = name.split("_")
    if len(tokens) < 3:
        return None
    emotion_token = tokens[-1].lower()
    speaker_token = tokens[0]
    if emotion_token not in TESS_MAP:
        return None
    return SampleMeta(path=path, dataset="tess", speaker=f"tess_{speaker_token}", label=TESS_MAP[emotion_token])


def _parse_cremad(path: str) -> Optional[SampleMeta]:
    name = os.path.basename(path).replace(".wav", "")
    tokens = name.split("_")
    if len(tokens) < 3:
        return None
    speaker_token = tokens[0]
    emotion_code = tokens[2].upper()
    if emotion_code not in CREMAD_MAP:
        return None
    return SampleMeta(path=path, dataset="cremad", speaker=f"cremad_{speaker_token}", label=CREMAD_MAP[emotion_code])


def collect_wav_files(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".wav"):
                paths.append(os.path.join(dirpath, fname))
    return sorted(paths)


def verify_audio(path: str) -> Tuple[bool, str]:
    try:
        info = sf.info(path)
        if info.frames <= 0:
            return False, "empty audio"
        data, _ = sf.read(path)
        if data.size == 0:
            return False, "no samples"
        return True, "ok"
    except Exception as e:
        return False, str(e)


def _resolve_dataset_dir(dataset_root: str, expected_name: str) -> Optional[str]:
    if not os.path.isdir(dataset_root):
        return None
    for entry in os.listdir(dataset_root):
        full = os.path.join(dataset_root, entry)
        if os.path.isdir(full) and entry.lower() == expected_name.lower():
            return full
    return None


def build_metadata(dataset_root: str) -> pd.DataFrame:
    all_rows: List[Dict[str, str]] = []
    checks: List[Dict[str, str]] = []

    dataset_dirs = ["ravdess", "tess", "cremad"]
    for ds in dataset_dirs:
        ds_dir = _resolve_dataset_dir(dataset_root, ds)
        if not ds_dir or not os.path.isdir(ds_dir):
            continue
        files = collect_wav_files(ds_dir)
        for fp in files:
            ok, message = verify_audio(fp)
            if not ok:
                checks.append({"path": fp, "status": "corrupt", "reason": message})
                continue
            parser = _parse_ravdess if ds == "ravdess" else _parse_tess if ds == "tess" else _parse_cremad
            parsed = parser(fp)
            if parsed is None:
                checks.append({"path": fp, "status": "skipped", "reason": "unmapped or invalid filename"})
                continue
            all_rows.append(
                {
                    "path": parsed.path,
                    "dataset": parsed.dataset,
                    "speaker": parsed.speaker,
                    "label": parsed.label,
                    "sha1": file_sha1(parsed.path),
                }
            )

    if checks:
        os.makedirs("training/results", exist_ok=True)
        pd.DataFrame(checks).to_csv("training/results/file_checks.csv", index=False)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["sha1"], keep="first").reset_index(drop=True)
    return df


def print_dataset_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No valid samples found.")
        return
    print("Total samples:", len(df))
    print("\nBy dataset:")
    print(df["dataset"].value_counts())
    print("\nBy label:")
    print(df["label"].value_counts())
    print("\nBy speaker:")
    print(df["speaker"].nunique())


def print_verification_summary(dataset_root: str, df: pd.DataFrame) -> None:
    ds_names = ["ravdess", "tess", "cremad"]
    print("Raw file counts:")
    for name in ds_names:
        ds_dir = _resolve_dataset_dir(dataset_root, name)
        count = len(collect_wav_files(ds_dir)) if ds_dir and os.path.isdir(ds_dir) else 0
        print(f"  {name}: {count}")

    checks_path = "training/results/file_checks.csv"
    if os.path.exists(checks_path):
        checks_df = pd.read_csv(checks_path)
        corrupt = int((checks_df["status"] == "corrupt").sum())
        skipped = int((checks_df["status"] == "skipped").sum())
    else:
        corrupt = 0
        skipped = 0

    print(f"Corrupted files: {corrupt}")
    print(f"Skipped/unmapped files: {skipped}")
    print(f"Valid unique files: {len(df)}")


def split_speaker_independent(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # First split off test set by speaker
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_train_val, idx_test = next(gss1.split(df, groups=df["speaker"]))
    train_val = df.iloc[idx_train_val].reset_index(drop=True)
    test_df = df.iloc[idx_test].reset_index(drop=True)

    # Split train_val into train and val by speaker
    adjusted_val = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=adjusted_val, random_state=seed)
    idx_train, idx_val = next(gss2.split(train_val, groups=train_val["speaker"]))
    train_df = train_val.iloc[idx_train].reset_index(drop=True)
    val_df = train_val.iloc[idx_val].reset_index(drop=True)
    return train_df, val_df, test_df


def augment_audio(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    aug = audio.copy()
    choice = random.choice(["noise", "pitch", "stretch", "none"])
    if choice == "noise":
        noise = np.random.normal(0, 0.003, size=aug.shape)
        aug = aug + noise
    elif choice == "pitch":
        aug = librosa.effects.pitch_shift(aug, sr=sr, n_steps=random.uniform(-1.5, 1.5))
    elif choice == "stretch":
        rate = random.uniform(0.9, 1.1)
        aug = librosa.effects.time_stretch(aug, rate=rate)
    return aug.astype(np.float32)


class EmotionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, training: bool = False):
        self.df = df.reset_index(drop=True)
        self.training = training

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["path"]
        y = EMOTION_TO_ID[row["label"]]

        if self.training:
            audio, sr = librosa.load(path, sr=22050, mono=True)
            audio = augment_audio(audio, sr=sr)
            audio = preprocess_audio(audio, sr=sr)
            feats = extract_features(audio, sr=sr)
        else:
            feats = preprocess_file_to_features(path)

        x = torch.tensor(feats, dtype=torch.float32)
        label = torch.tensor(y, dtype=torch.long)
        return x, label
