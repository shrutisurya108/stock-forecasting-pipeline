"""
models/base_model.py
====================
Abstract base class that every forecasting model must implement.

Why this exists:
  The trainer (Phase 4) and predictor (Phase 5) loop over all three models
  using identical code. This is only possible because every model exposes the
  same interface. Without this contract, you'd need three separate training
  loops — messy and hard to extend.

Any new model added to this project must subclass BaseModel and implement
all abstract methods. Python will raise a TypeError at import time if a
subclass is missing any of them.

Shared interface:
    model.fit(train_df, val_df)        → trains the model
    model.predict(n_steps)             → point forecast array
    model.predict_with_ci(n_steps)     → dict with forecast + lower/upper CI
    model.evaluate(test_df, scaler)    → dict with RMSE, MAPE, MAE
    model.save(path)                   → persist model to disk
    model.load(path)                   → restore model from disk
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.logging_config import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all forecasting models.

    Subclasses must implement:
        _fit, _predict, _predict_with_ci, save, load

    Subclasses inherit:
        fit        (wraps _fit with timing + logging)
        predict    (wraps _predict with validation)
        evaluate   (computes RMSE/MAPE/MAE from _predict output)
    """

    def __init__(self, ticker: str, model_name: str):
        """
        Args:
            ticker:     Stock symbol this model is trained for (e.g. "AAPL").
            model_name: Human-readable name ("arima", "prophet", "lstm").
        """
        self.ticker     = ticker
        self.model_name = model_name
        self.is_fitted  = False
        self._fit_duration_seconds: Optional[float] = None
        self.logger     = get_logger(f"{model_name}.{ticker}")

    # ── Public methods (call these from outside) ──────────────────────────────

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "BaseModel":
        """
        Train the model. Wraps _fit with timing and logging.

        Args:
            train: Scaled training DataFrame (output of preprocessing).
            val:   Scaled validation DataFrame (used for early stopping etc.).

        Returns:
            self  (allows chaining: model.fit(train, val).predict(30))
        """
        self.logger.info(
            "[%s | %s] Starting fit — train=%d rows, val=%d rows",
            self.ticker, self.model_name, len(train), len(val),
        )
        t0 = time.time()
        self._fit(train, val)
        self._fit_duration_seconds = time.time() - t0
        self.is_fitted = True
        self.logger.info(
            "[%s | %s] Fit complete in %.1fs",
            self.ticker, self.model_name, self._fit_duration_seconds,
        )
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        """
        Generate a point forecast for n_steps ahead.

        Args:
            n_steps: Number of future time steps to forecast.

        Returns:
            1-D numpy array of shape (n_steps,) with predicted values.
        """
        self._assert_fitted()
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        return self._predict(n_steps)

    def predict_with_ci(
        self,
        n_steps: int,
        confidence: float = 0.95,
    ) -> dict[str, np.ndarray]:
        """
        Generate forecast with confidence interval bands.

        Args:
            n_steps:    Number of future steps to forecast.
            confidence: Confidence level (0–1). Default 0.95 → 95% CI.

        Returns:
            Dict with keys:
                "forecast" → np.ndarray (n_steps,)  point forecast
                "lower"    → np.ndarray (n_steps,)  lower CI bound
                "upper"    → np.ndarray (n_steps,)  upper CI bound
        """
        self._assert_fitted()
        result = self._predict_with_ci(n_steps, confidence)
        # Validate the returned dict has all required keys
        for key in ("forecast", "lower", "upper"):
            if key not in result:
                raise RuntimeError(
                    f"{self.model_name}._predict_with_ci must return key '{key}'"
                )
        return result

    def evaluate(
        self,
        test: pd.DataFrame,
        scaler,                     # sklearn MinMaxScaler
        target_col: str = "Close",
    ) -> dict[str, float]:
        """
        Evaluate the model on the test set.

        Generates a forecast for len(test) steps, inverse-transforms both
        the forecast and the true values back to real price space, then
        computes RMSE, MAPE, and MAE.

        Args:
            test:       Scaled test DataFrame.
            scaler:     The MinMaxScaler fitted on train (from ProcessedData).
            target_col: Column to evaluate against. Default "Close".

        Returns:
            Dict {"RMSE": float, "MAPE": float, "MAE": float}
        """
        self._assert_fitted()

        n_steps = len(test)
        forecast_scaled = self._predict(n_steps)

        # Build a dummy array to inverse-transform only the target column
        feature_cols = list(test.columns)
        target_idx   = feature_cols.index(target_col)

        # Use scaler's own dimension — not len(feature_cols) — because the
        # test DataFrame may contain raw OHLCV columns beyond what scaler saw.
        n_scaler_cols = len(scaler.scale_)

        def _inverse(values: np.ndarray) -> np.ndarray:
            """Inverse-transform a 1-D array using the fitted scaler."""
            dummy = np.zeros((len(values), n_scaler_cols))
            dummy[:, target_idx] = values
            return scaler.inverse_transform(dummy)[:, target_idx]

        y_pred = _inverse(np.array(forecast_scaled).ravel())
        y_true = _inverse(test[target_col].values)

        metrics = _compute_metrics(y_true, y_pred)
        self.logger.info(
            "[%s | %s] Evaluation — RMSE=%.4f | MAPE=%.4f%% | MAE=%.4f",
            self.ticker, self.model_name,
            metrics["RMSE"], metrics["MAPE"], metrics["MAE"],
        )
        return metrics

    # ── Abstract methods (subclasses must implement these) ────────────────────

    @abstractmethod
    def _fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        """Internal fit logic — called by the public fit() wrapper."""

    @abstractmethod
    def _predict(self, n_steps: int) -> np.ndarray:
        """Internal predict logic — return shape (n_steps,) array."""

    @abstractmethod
    def _predict_with_ci(
        self,
        n_steps: int,
        confidence: float,
    ) -> dict[str, np.ndarray]:
        """Internal CI predict — return dict with forecast/lower/upper arrays."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the fitted model to disk."""

    @abstractmethod
    def load(self, path: Path) -> "BaseModel":
        """Restore a fitted model from disk. Returns self."""

    # ── Private helpers ───────────────────────────────────────────────────────

    def _assert_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.model_name} for {self.ticker} has not been fitted. "
                "Call .fit(train, val) first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"{self.model_name.upper()}Model(ticker={self.ticker}, {status})"


# ── Shared metric utilities (used by all models + benchmarking) ───────────────

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute RMSE, MAPE, and MAE between true and predicted values.

    Args:
        y_true: Ground-truth values (real price scale).
        y_pred: Model forecast values (real price scale).

    Returns:
        Dict {"RMSE": float, "MAPE": float, "MAE": float}
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Guard against divide-by-zero in MAPE
    nonzero_mask = y_true != 0
    if not nonzero_mask.all():
        logger.warning(
            "y_true contains zeros — MAPE computed on non-zero subset only"
        )

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(
        np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask])
                       / y_true[nonzero_mask])) * 100
    ) if nonzero_mask.any() else float("nan")
    mae  = float(np.mean(np.abs(y_true - y_pred)))

    return {"RMSE": rmse, "MAPE": mape, "MAE": mae}


def naive_baseline_metrics(
    test: pd.DataFrame,
    scaler,
    target_col: str = "Close",
) -> dict[str, float]:
    """
    Compute metrics for the naive 'last known value' baseline forecast.

    The naive strategy predicts that tomorrow's price == today's price.
    This is the benchmark every model must beat.

    Args:
        test:       Scaled test DataFrame.
        scaler:     MinMaxScaler fitted on train.
        target_col: Column to evaluate against.

    Returns:
        Dict {"RMSE": float, "MAPE": float, "MAE": float}
    """
    feature_cols = list(test.columns)
    target_idx   = feature_cols.index(target_col)

    # Use scaler's own dimension (n_features it was fitted on),
    # NOT test.columns — test may have more columns than the scaler saw.
    n_scaler_cols = len(scaler.scale_)

    def _inverse(values: np.ndarray) -> np.ndarray:
        dummy = np.zeros((len(values), n_scaler_cols))
        dummy[:, target_idx] = values
        return scaler.inverse_transform(dummy)[:, target_idx]

    close_scaled = test[target_col].values
    y_true = _inverse(close_scaled)

    # Naive: predict t using value at t-1 (shift by 1)
    y_naive = np.roll(y_true, shift=1)
    y_naive[0] = y_true[0]   # first prediction = first true value

    return _compute_metrics(y_true, y_naive)
