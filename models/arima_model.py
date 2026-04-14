"""
models/arima_model.py
=====================
ARIMA forecasting model using pmdarima's auto_arima for automatic order
selection and statsmodels for the actual fitted model.

Design decisions:
- auto_arima searches for the best (p,d,q) using AIC — no manual tuning.
- Works on the Close price column only (univariate time series).
- Confidence intervals are computed analytically from the statsmodels
  ARIMA result object (no sampling needed → fast).
- The fitted model is serialised with pickle (standard for statsmodels).

Usage:
    from models.arima_model import ARIMAModel
    model = ARIMAModel("AAPL")
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
from scipy import stats

from config.settings import ARIMA_CONFIG, TARGET_COL
from models.base_model import BaseModel

# pmdarima is imported inside methods to allow the file to load even if
# the package is not installed (fails loudly only when ARIMA is used).


class ARIMAModel(BaseModel):
    """
    Auto-ARIMA model: finds optimal (p,d,q) order, fits, and forecasts.

    Attributes:
        _model:      Fitted pmdarima AutoARIMA object (contains the
                     statsmodels result internally).
        _train_vals: Numpy array of training Close values (needed for
                     predict — statsmodels ARIMA requires the history).
        _order:      Tuple (p, d, q) selected by auto_arima.
    """

    MODEL_FILENAME = "arima_model.pkl"

    def __init__(self, ticker: str):
        super().__init__(ticker=ticker, model_name="arima")
        self._model:      Optional[object] = None   # pmdarima AutoARIMA
        self._train_vals: Optional[np.ndarray] = None
        self._order:      Optional[tuple]      = None

    # ── Abstract method implementations ──────────────────────────────────────

    def _fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        """
        1. Extract scaled Close values from train.
        2. Run auto_arima to find optimal (p, d, q).
        3. Fit the selected model on train+val combined for best accuracy.
        """
        try:
            import pmdarima as pm
        except ImportError as exc:
            raise ImportError(
                "pmdarima is required for ARIMAModel. "
                "Install it with: pip install pmdarima"
            ) from exc

        # ARIMA is univariate — only the Close column
        train_vals = train[TARGET_COL].values.astype(float)
        val_vals   = val[TARGET_COL].values.astype(float)

        self.logger.info(
            "[%s | arima] Running auto_arima (max_p=%d, max_q=%d, criterion=%s)…",
            self.ticker,
            ARIMA_CONFIG["max_p"],
            ARIMA_CONFIG["max_q"],
            ARIMA_CONFIG["information_criterion"],
        )

        self._model = pm.auto_arima(
            train_vals,
            start_p=1,
            start_q=0,
            max_p=ARIMA_CONFIG["max_p"],
            max_q=ARIMA_CONFIG["max_q"],
            d=None,                              # let auto_arima choose d
            seasonal=ARIMA_CONFIG["seasonal"],
            information_criterion=ARIMA_CONFIG["information_criterion"],
            stepwise=True,                       # faster than exhaustive search
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )

        self._order = self._model.order
        self.logger.info(
            "[%s | arima] Selected order: (p=%d, d=%d, q=%d)",
            self.ticker, *self._order,
        )

        # Update the model with validation data (extends the training window)
        self._model.update(val_vals)

        # Store the full history for potential re-forecasting
        self._train_vals = np.concatenate([train_vals, val_vals])

    def _predict(self, n_steps: int) -> np.ndarray:
        """Generate a point forecast n_steps ahead (scaled space)."""
        forecast, _ = self._model.predict(
            n_periods=n_steps,
            return_conf_int=True,
            alpha=1 - 0.95,
        )
        return np.array(forecast, dtype=float)

    def _predict_with_ci(
        self,
        n_steps: int,
        confidence: float = 0.95,
    ) -> dict[str, np.ndarray]:
        """
        Return forecast + lower/upper CI bands.
        pmdarima returns confidence intervals directly.
        """
        alpha = 1.0 - confidence
        forecast, conf_int = self._model.predict(
            n_periods=n_steps,
            return_conf_int=True,
            alpha=alpha,
        )
        return {
            "forecast": np.array(forecast, dtype=float),
            "lower":    conf_int[:, 0].astype(float),
            "upper":    conf_int[:, 1].astype(float),
        }

    def save(self, path: Path) -> None:
        """Pickle the entire ARIMAModel to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file = path / self.MODEL_FILENAME
        with open(save_file, "wb") as f:
            pickle.dump(self, f)
        self.logger.info("[%s | arima] Model saved → %s", self.ticker, save_file)

    def load(self, path: Path) -> "ARIMAModel":
        """Load a pickled ARIMAModel from disk."""
        load_file = Path(path) / self.MODEL_FILENAME
        with open(load_file, "rb") as f:
            loaded = pickle.load(f)
        self.__dict__.update(loaded.__dict__)
        self.logger.info("[%s | arima] Model loaded ← %s", self.ticker, load_file)
        return self

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def order(self) -> Optional[tuple]:
        """Return the selected ARIMA (p, d, q) order tuple."""
        return self._order

    def summary(self) -> str:
        """Return a short text summary of the fitted model."""
        if not self.is_fitted:
            return f"ARIMAModel({self.ticker}) — not fitted"
        p, d, q = self._order
        return (
            f"ARIMAModel({self.ticker}) — "
            f"order=({p},{d},{q}), "
            f"fit_time={self._fit_duration_seconds:.1f}s"
        )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    from data.preprocessing import preprocess_ticker

    np.random.seed(42)

    print("\n" + "═" * 60)
    print("  models/arima_model.py — smoke test")
    print("═" * 60 + "\n")

    # Build a small synthetic dataset
    n = 400
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
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
    assert processed is not None, "Preprocessing failed"

    print("1. Fitting ARIMA…")
    model = ARIMAModel("TEST")
    model.fit(processed.train, processed.val)
    print(f"   {model.summary()}")

    print("\n2. Point forecast (10 steps)…")
    fc = model.predict(10)
    print(f"   Forecast: {np.round(fc, 4)}")

    print("\n3. Forecast with 95% CI (10 steps)…")
    result = model.predict_with_ci(10, confidence=0.95)
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
        model2 = ARIMAModel("TEST")
        model2.load(Path(tmpdir))
        fc2 = model2.predict(5)
        assert np.allclose(model.predict(5), fc2), "Save/load forecast mismatch!"
    print("   ✅ Save/load produces identical forecasts")

    print("\n✅ ARIMAModel smoke test PASSED\n")
