from typing import Tuple

import torch
import torch.nn as nn


class CNNLSTMEmotionModel(nn.Module):
    def __init__(self, n_classes: int = 6, n_mfcc_channels: int = 80, lstm_hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mfcc_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 80, frames)
        x = self.conv(x)  # (batch, 128, frames')
        x = x.transpose(1, 2)  # (batch, frames', 128)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        logits = self.fc(pooled)
        return logits


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_input_shape() -> Tuple[int, int]:
    # channels, frames
    return 80, 173
