"""
models/lstm_model.py
====================
PyTorch 2-layer LSTM forecasting model.

Design decisions:
- Uses ALL 13 engineered features (not just Close) → multivariate input.
- 60-day look-back window: the model sees the past 60 trading days to
  predict the next step, rolled forward autoregressively for multi-step.
- Early stopping on validation loss prevents overfitting.
- Monte Carlo Dropout: at inference time, run N forward passes with
  dropout enabled → mean = point forecast, std = uncertainty → CI bands.
- Model saved as a .pt state_dict (PyTorch standard).
- Falls back to CPU automatically if CUDA is unavailable.

Architecture:
    Input:  (batch, 60, 13)   ← sequence_len × n_features
    LSTM:   2 layers, hidden=128, dropout=0.2
    Linear: 128 → 1           ← predicts next Close (scaled)
    Output: (batch, 1)

Usage:
    from models.lstm_model import LSTMModel
    model = LSTMModel("AAPL")
    model.fit(train_df, val_df)
    forecast = model.predict(30)
    result   = model.predict_with_ci(30)
    metrics  = model.evaluate(test_df, scaler)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.settings import LSTM_CONFIG, TARGET_COL, RANDOM_SEED
from models.base_model import BaseModel


# ── PyTorch module definition ─────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    """
    Internal PyTorch module: 2-layer LSTM + linear head.

    Not used directly — LSTMModel wraps this.
    """

    def __init__(
        self,
        n_features:  int,
        hidden_size: int,
        num_layers:  int,
        dropout:     float,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        # MC Dropout layer (separate so we can enable at inference time)
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        lstm_out, _ = self.lstm(x)
        # Take the output at the last time step
        last_out = lstm_out[:, -1, :]        # (batch, hidden_size)
        dropped  = self.dropout(last_out)
        return self.fc(dropped).squeeze(-1)  # (batch,)


# ── Helper: sequence builder ──────────────────────────────────────────────────

def _build_sequences(
    data:        np.ndarray,    # shape (T, n_features)
    target_idx:  int,           # column index of Close
    seq_len:     int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2-D array into (X, y) pairs for supervised learning.

    X[i] = data[i : i+seq_len]          shape (seq_len, n_features)
    y[i] = data[i+seq_len, target_idx]  scalar (next Close value)
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Main model class ──────────────────────────────────────────────────────────

class LSTMModel(BaseModel):
    """
    2-layer LSTM with early stopping and Monte Carlo Dropout CI.

    Attributes:
        _net:        The _LSTMNet PyTorch module.
        _device:     "cuda" or "cpu".
        _feature_cols: List of column names used during training.
        _target_idx: Index of TARGET_COL in feature_cols.
        _seq_len:    Look-back window length (from LSTM_CONFIG).
        _last_seq:   The last sequence seen during training — used as the
                     starting window for multi-step autoregressive forecasting.
    """

    MODEL_FILENAME    = "lstm_model.pt"
    META_FILENAME     = "lstm_meta.pkl"
    MC_DROPOUT_PASSES = 50    # number of stochastic forward passes for CI

    def __init__(self, ticker: str):
        super().__init__(ticker=ticker, model_name="lstm")
        self._net:          Optional[_LSTMNet]  = None
        self._device:       str                 = "cpu"
        self._feature_cols: Optional[list[str]] = None
        self._target_idx:   Optional[int]       = None
        self._seq_len:      int                 = LSTM_CONFIG["sequence_length"]
        self._last_seq:     Optional[np.ndarray] = None   # (seq_len, n_features)

    # ── Device selection ──────────────────────────────────────────────────────

    def _get_device(self) -> str:
        if LSTM_CONFIG["train_on_gpu"] and torch.cuda.is_available():
            self.logger.info("[%s | lstm] Using CUDA GPU", self.ticker)
            return "cuda"
        self.logger.info("[%s | lstm] Using CPU", self.ticker)
        return "cpu"

    # ── Abstract method implementations ──────────────────────────────────────

    def _fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        self._device       = self._get_device()
        self._feature_cols = list(train.columns)
        self._target_idx   = self._feature_cols.index(TARGET_COL)
        n_features         = len(self._feature_cols)

        # Build numpy arrays
        train_arr = train[self._feature_cols].values.astype(np.float32)
        val_arr   = val[self._feature_cols].values.astype(np.float32)

        # Check we have enough rows for at least one sequence
        if len(train_arr) <= self._seq_len:
            raise ValueError(
                f"Training data ({len(train_arr)} rows) is shorter than "
                f"sequence_length ({self._seq_len}). Reduce LSTM_CONFIG['sequence_length']."
            )

        X_train, y_train = _build_sequences(train_arr, self._target_idx, self._seq_len)
        X_val,   y_val   = _build_sequences(val_arr,   self._target_idx, self._seq_len)

        self.logger.info(
            "[%s | lstm] Sequences — train: %s, val: %s, features: %d",
            self.ticker, X_train.shape, X_val.shape, n_features,
        )

        # DataLoaders
        train_ds  = TensorDataset(
            torch.tensor(X_train), torch.tensor(y_train)
        )
        val_ds    = TensorDataset(
            torch.tensor(X_val), torch.tensor(y_val)
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=LSTM_CONFIG["batch_size"],
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=LSTM_CONFIG["batch_size"],
            shuffle=False,
        )

        # Build network
        self._net = _LSTMNet(
            n_features=n_features,
            hidden_size=LSTM_CONFIG["hidden_size"],
            num_layers=LSTM_CONFIG["num_layers"],
            dropout=LSTM_CONFIG["dropout"],
        ).to(self._device)

        optimiser = torch.optim.Adam(
            self._net.parameters(), lr=LSTM_CONFIG["learning_rate"]
        )
        criterion = nn.MSELoss()

        # Training loop with early stopping
        best_val_loss  = float("inf")
        patience_count = 0
        best_state     = None
        epochs         = LSTM_CONFIG["epochs"]
        patience       = LSTM_CONFIG["early_stopping_patience"]

        for epoch in range(1, epochs + 1):
            # ── Train ──
            self._net.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimiser.zero_grad()
                preds = self._net(X_batch)
                loss  = criterion(preds, y_batch)
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self._net.parameters(), LSTM_CONFIG["clip_grad_norm"]
                )
                optimiser.step()
                train_losses.append(loss.item())

            # ── Validate ──
            self._net.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self._device)
                    y_batch = y_batch.to(self._device)
                    preds   = self._net(X_batch)
                    val_losses.append(criterion(preds, y_batch).item())

            train_loss = np.mean(train_losses)
            val_loss   = np.mean(val_losses)

            if epoch % 10 == 0 or epoch == 1:
                self.logger.info(
                    "[%s | lstm] Epoch %3d/%d — train_loss=%.6f | val_loss=%.6f",
                    self.ticker, epoch, epochs, train_loss, val_loss,
                )

            # Early stopping check
            if val_loss < best_val_loss - 1e-6:
                best_val_loss  = val_loss
                patience_count = 0
                best_state     = {
                    k: v.cpu().clone()
                    for k, v in self._net.state_dict().items()
                }
            else:
                patience_count += 1
                if patience_count >= patience:
                    self.logger.info(
                        "[%s | lstm] Early stopping at epoch %d "
                        "(best val_loss=%.6f)",
                        self.ticker, epoch, best_val_loss,
                    )
                    break

        # Restore best weights
        if best_state is not None:
            self._net.load_state_dict(best_state)

        # Store the last sequence for autoregressive forecasting
        full_arr      = np.vstack([train_arr, val_arr])
        self._last_seq = full_arr[-self._seq_len:]   # (seq_len, n_features)

        self.logger.info(
            "[%s | lstm] Training complete — best val_loss=%.6f",
            self.ticker, best_val_loss,
        )

    def _autoregressive_forecast(self, n_steps: int) -> np.ndarray:
        """
        Multi-step forecast using autoregressive rolling-window prediction.

        At each step:
          1. Feed the current window → get ŷ (next Close, scaled).
          2. Build the next input row by copying the last row of the window
             and replacing the Close column with ŷ.
          3. Slide the window forward by one step.

        This keeps all other features constant (holding-last-value
        assumption for features we can't forecast independently).
        """
        self._net.eval()
        window = self._last_seq.copy()              # (seq_len, n_features)
        preds  = []

        with torch.no_grad():
            for _ in range(n_steps):
                x = torch.tensor(window[np.newaxis], dtype=torch.float32).to(
                    self._device
                )
                y_hat = self._net(x).item()
                preds.append(y_hat)

                # Slide window: drop oldest row, append new row
                new_row               = window[-1].copy()
                new_row[self._target_idx] = y_hat
                window = np.vstack([window[1:], new_row[np.newaxis]])

        return np.array(preds, dtype=float)

    def _predict(self, n_steps: int) -> np.ndarray:
        return self._autoregressive_forecast(n_steps)

    def _predict_with_ci(
        self,
        n_steps: int,
        confidence: float = 0.95,
    ) -> dict[str, np.ndarray]:
        """
        Monte Carlo Dropout: run MC_DROPOUT_PASSES stochastic forward passes.

        With dropout enabled during inference, each pass gives a slightly
        different forecast. The distribution across passes gives uncertainty.

        CI: mean ± z * std   where z corresponds to the confidence level.
        """
        from scipy.stats import norm as scipy_norm

        z = scipy_norm.ppf((1 + confidence) / 2)

        # Enable dropout (train mode for MC Dropout)
        self._net.train()
        all_preds = []

        for _ in range(self.MC_DROPOUT_PASSES):
            window = self._last_seq.copy()
            pass_preds = []

            with torch.no_grad():
                for _ in range(n_steps):
                    x = torch.tensor(
                        window[np.newaxis], dtype=torch.float32
                    ).to(self._device)
                    y_hat = self._net(x).item()
                    pass_preds.append(y_hat)

                    new_row = window[-1].copy()
                    new_row[self._target_idx] = y_hat
                    window = np.vstack([window[1:], new_row[np.newaxis]])

            all_preds.append(pass_preds)

        # Back to eval mode
        self._net.eval()

        all_preds = np.array(all_preds)          # (MC_passes, n_steps)
        mean_pred = all_preds.mean(axis=0)
        std_pred  = all_preds.std(axis=0)

        return {
            "forecast": mean_pred,
            "lower":    mean_pred - z * std_pred,
            "upper":    mean_pred + z * std_pred,
        }

    def save(self, path: Path) -> None:
        """Save model weights (.pt) and metadata (.pkl) separately."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save PyTorch weights
        torch.save(self._net.state_dict(), path / self.MODEL_FILENAME)

        # Save metadata needed to reconstruct the network
        meta = {
            "feature_cols": self._feature_cols,
            "target_idx":   self._target_idx,
            "seq_len":      self._seq_len,
            "last_seq":     self._last_seq,
            "n_features":   len(self._feature_cols),
            "ticker":       self.ticker,
        }
        with open(path / self.META_FILENAME, "wb") as f:
            pickle.dump(meta, f)

        self.logger.info("[%s | lstm] Model saved → %s", self.ticker, path)

    def load(self, path: Path) -> "LSTMModel":
        """Restore LSTM from weights + metadata files."""
        path = Path(path)

        with open(path / self.META_FILENAME, "rb") as f:
            meta = pickle.load(f)

        self._feature_cols = meta["feature_cols"]
        self._target_idx   = meta["target_idx"]
        self._seq_len      = meta["seq_len"]
        self._last_seq     = meta["last_seq"]

        self._device = self._get_device()
        self._net    = _LSTMNet(
            n_features=meta["n_features"],
            hidden_size=LSTM_CONFIG["hidden_size"],
            num_layers=LSTM_CONFIG["num_layers"],
            dropout=LSTM_CONFIG["dropout"],
        ).to(self._device)

        self._net.load_state_dict(
            torch.load(path / self.MODEL_FILENAME, map_location=self._device)
        )
        self._net.eval()
        self.is_fitted = True

        self.logger.info("[%s | lstm] Model loaded ← %s", self.ticker, path)
        return self

    def summary(self) -> str:
        if not self.is_fitted:
            return f"LSTMModel({self.ticker}) — not fitted"
        n_params = sum(p.numel() for p in self._net.parameters())
        return (
            f"LSTMModel({self.ticker}) — "
            f"features={len(self._feature_cols)}, "
            f"seq_len={self._seq_len}, "
            f"params={n_params:,}, "
            f"device={self._device}, "
            f"fit_time={self._fit_duration_seconds:.1f}s"
        )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    from data.preprocessing import preprocess_ticker

    np.random.seed(RANDOM_SEED)

    print("\n" + "═" * 60)
    print("  models/lstm_model.py — smoke test")
    print("═" * 60 + "\n")

    # Build synthetic data — enough rows for seq_len=60 + splits
    n = 600
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    raw = pd.DataFrame(
        {
            "Open": close * 0.99, "High": close * 1.02,
            "Low":  close * 0.97, "Close": close,
            "Volume": np.random.randint(500_000, 2_000_000, n).astype(float),
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )

    processed = preprocess_ticker(raw, "TEST")
    assert processed is not None

    print(f"Train={len(processed.train)} | Val={len(processed.val)} | Test={len(processed.test)}")
    print(f"Features ({len(processed.feature_cols)}): {processed.feature_cols}\n")

    # Use fewer epochs for the smoke test
    original_epochs = LSTM_CONFIG["epochs"]
    original_patience = LSTM_CONFIG["early_stopping_patience"]
    LSTM_CONFIG["epochs"] = 5
    LSTM_CONFIG["early_stopping_patience"] = 3

    print("1. Fitting LSTM (5 epochs for smoke test)…")
    model = LSTMModel("TEST")
    model.fit(processed.train, processed.val)
    print(f"   {model.summary()}")

    # Restore original config
    LSTM_CONFIG["epochs"] = original_epochs
    LSTM_CONFIG["early_stopping_patience"] = original_patience

    print("\n2. Point forecast (10 steps)…")
    fc = model.predict(10)
    print(f"   Forecast: {np.round(fc, 4)}")

    print("\n3. Forecast with 95% CI (10 steps, MC Dropout)…")
    result = model.predict_with_ci(10)
    for i in range(5):
        print(
            f"   step {i+1:02d}: "
            f"lower={result['lower'][i]:.4f}  "
            f"fc={result['forecast'][i]:.4f}  "
            f"upper={result['upper'][i]:.4f}"
        )

    print("\n4. Evaluating on test set…")
    metrics = model.evaluate(processed.test, processed.scaler)
    print(f"   RMSE={metrics['RMSE']:.4f} | MAPE={metrics['MAPE']:.2f}% | MAE={metrics['MAE']:.4f}")

    print("\n5. Save / Load round-trip…")
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(Path(tmpdir))
        model2 = LSTMModel("TEST")
        model2.load(Path(tmpdir))
        fc2 = model2.predict(5)
        assert np.allclose(model.predict(5), fc2, atol=1e-4), "Save/load mismatch!"
    print("   ✅ Save/load produces identical forecasts")

    print("\n✅ LSTMModel smoke test PASSED\n")
