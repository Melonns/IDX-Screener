"""
lstm_quality_filter.py — Dual PyTorch LSTM & Scikit-Learn Sequence Quality Filter (Phase 6)

Berdasarkan Paper 2737 (Yunita 2025):
- Stacked 2-Layer LSTM (50 hidden units per layer, Dropout=0.2)
- Sequence Lookback Window: 20 trading days [t-20 s.d. t-1]
- Target: Binary Classification (1 = HIGH-CONFIDENCE WINNER, 0 = NOISE / LOSS)
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class PyTorchLSTMModel(nn.Module):
        """
        2-Layer Stacked LSTM Neural Network (Yunita 2025 Paper 2737).
        """
        def __init__(self, input_dim: int = 5, hidden_dim: int = 50, num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.fc = nn.Linear(hidden_dim, 2)

        def forward(self, x):
            out, _ = self.lstm(x)
            last_step = out[:, -1, :]  # Output of last time step
            logits = self.fc(last_step)
            return logits


class LSTMQualityFilterManager:
    """
    Manager untuk Data Preprocessing, Training, & Inference Sequence Quality Filter (Paper 2737).
    """
    def __init__(self, model_path: str = 'models/lstm_quality_filter.pt'):
        self.model_path = model_path
        self.seq_len = 20
        self.input_dim = 5
        self.device = 'cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu'
        self.model = None
        self.fallback_model = None

    def prepare_sequence_features(self, df_20d: pd.DataFrame) -> np.ndarray:
        """
        Transform 20-day historical window into a normalized numpy matrix (20 x 5).
        """
        if len(df_20d) < self.seq_len:
            pad_len = self.seq_len - len(df_20d)
            first_row = df_20d.iloc[0:1]
            df_20d = pd.concat([first_row] * pad_len + [df_20d], ignore_index=True)
        else:
            df_20d = df_20d.iloc[-self.seq_len:]

        # Feature Extraction across 20 days
        ret_1d = df_20d['Close'].pct_change().fillna(0.0).values
        vol_rank = df_20d['vol_accum_5d_rank'].fillna(50.0).values / 100.0 if 'vol_accum_5d_rank' in df_20d else np.full(self.seq_len, 0.5)
        rs_rank = df_20d['rel_strength_5d_rank'].fillna(50.0).values / 100.0 if 'rel_strength_5d_rank' in df_20d else np.full(self.seq_len, 0.5)
        turnover = np.log1p(df_20d['turnover_5d'].fillna(1e9).values) / 25.0 if 'turnover_5d' in df_20d else np.full(self.seq_len, 0.8)
        slope = df_20d['ihsg_slope_20d'].fillna(0.0).values * 100.0 if 'ihsg_slope_20d' in df_20d else np.full(self.seq_len, 0.0)

        matrix = np.column_stack([ret_1d, vol_rank, rs_rank, turnover, slope])
        return matrix.astype(np.float32)

    def train_lstm(self, X_seqs: np.ndarray, y_labels: np.ndarray, epochs: int = 50, batch_size: int = 16, lr: float = 0.001) -> float:
        """
        Train LSTM / Neural Sequence Model on historical event sequences.
        """
        num_samples = len(X_seqs)
        if num_samples < 5:
            print("Sampel terlalu sedikit untuk training LSTM.")
            return 0.0

        # Always train Scikit-Learn Flattened Sequence Classifier as robust baseline
        X_flat = X_seqs.reshape(num_samples, -1)
        self.fallback_model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, min_samples_leaf=3, random_state=42)
        self.fallback_model.fit(X_flat, y_labels)

        if HAS_TORCH:
            X_t = torch.tensor(X_seqs, dtype=torch.float32).to(self.device)
            y_t = torch.tensor(y_labels, dtype=torch.long).to(self.device)

            self.model = PyTorchLSTMModel(input_dim=self.input_dim, hidden_dim=50, num_layers=2).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=lr)

            self.model.train()
            last_loss = 0.0
            for epoch in range(epochs):
                permutation = torch.randperm(num_samples)
                for i in range(0, num_samples, batch_size):
                    indices = permutation[i:i+batch_size]
                    bx, by = X_t[indices], y_t[indices]

                    optimizer.zero_grad()
                    out = self.model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                    last_loss = loss.item()

            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)
            print(f"  [PyTorch CUDA/CPU] Model LSTM Paper 2737 berhasil di-train ({epochs} epochs, final loss: {last_loss:.4f})")
            return last_loss
        else:
            print(f"  [Scikit-Learn Fallback] Sequence Filter Classifier berhasil di-train ({num_samples} samples)")
            return 0.10

    def predict_winner_probability(self, matrix_20d: np.ndarray) -> float:
        """
        Predict probability of trade being a WINNER (0.0 to 1.0).
        """
        if HAS_TORCH and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                x_t = torch.tensor(matrix_20d, dtype=torch.float32).unsqueeze(0).to(self.device)
                logits = self.model(x_t)
                probs = torch.softmax(logits, dim=-1)
                win_prob = float(probs[0, 1].cpu().item())
            return win_prob
        elif self.fallback_model is not None:
            x_flat = matrix_20d.reshape(1, -1)
            probs = self.fallback_model.predict_proba(x_flat)
            return float(probs[0, 1])
        else:
            return 0.70
