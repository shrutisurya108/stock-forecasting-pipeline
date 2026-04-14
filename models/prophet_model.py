"""
models/prophet_model.py
=======================
Facebook/Meta Prophet forecasting model.

Design decisions:
- Prophet requires a very specific DataFrame format: columns 'ds' (datetime)
  and 'y' (target value). This module handles that conversion transparently
  so the trainer never needs to know about it.
- Confidence intervals are Prophet's native yhat_lower / yhat_upper — no
  extra work required.
- The model is serialised with pickle (Prophet objects support this).
- Prophet suppresses its verbose Stan compiler output by default.

Usage:
    from models.prophet_model import ProphetModel
    model = ProphetModel("AAPL")
    model.fit(train_df, val_df)
    forecast = model.predict(30)
    result   = model.predict_with_ci(30)
    metrics  = model.evaluate(test_df, scaler)
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import PROPHET_CONFIG, TARGET_COL
from models.base_model import BaseModel

# Silence Prophet / cmdstanpy noise at module level
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*Stan.*")


class ProphetModel(BaseModel):
    """
    Prophet model wrapper with automatic ds/y formatting.

    Attributes:
        _model:        Fitted Prophet object.
        _last_date:    Last date in training history (needed to build
                       future DataFrame for prediction).
        _freq:         Pandas offset string for business-day frequency ("B").
    """

    MODEL_FILENAME = "prophet_model.pkl"
    FREQ           = "B"   # Business days — matches our stock data

    def __init__(self, ticker: str):
        super().__init__(ticker=ticker, model_name="prophet")
        self._model:     Optional[object]  = None
        self._last_date: Optional[pd.Timestamp] = None

    # ── Helper: DataFrame conversion ─────────────────────────────────────────

    @staticmethod
    def _to_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a standard preprocessed DataFrame to Prophet's ds/y format.

        Prophet requires:
            ds → datetime column
            y  → target value column
        """
        prophet_df = pd.DataFrame(
            {
                "ds": df.index.tz_localize(None),  # remove tz if present
                "y":  df[TARGET_COL].values,
            }
        )
        return prophet_df

    # ── Abstract method implementations ──────────────────────────────────────

    def _fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        """
        Combine train + val, convert to ds/y format, fit Prophet.
        """
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise ImportError(
                "prophet is required for ProphetModel. "
                "Install it with: pip install prophet"
            ) from exc

        # Combine train and val for maximum history
        combined = pd.concat([train, val]).sort_index()
        prophet_df = self._to_prophet_df(combined)
        self._last_date = combined.index.max()

        self.logger.info(
            "[%s | prophet] Fitting on %d rows (%s → %s)…",
            self.ticker,
            len(prophet_df),
            str(prophet_df["ds"].min().date()),
            str(prophet_df["ds"].max().date()),
        )

        self._model = Prophet(
            changepoint_prior_scale=PROPHET_CONFIG["changepoint_prior_scale"],
            seasonality_prior_scale=PROPHET_CONFIG["seasonality_prior_scale"],
            seasonality_mode=PROPHET_CONFIG["seasonality_mode"],
            yearly_seasonality=PROPHET_CONFIG["yearly_seasonality"],
            weekly_seasonality=PROPHET_CONFIG["weekly_seasonality"],
            daily_seasonality=PROPHET_CONFIG["daily_seasonality"],
            interval_width=PROPHET_CONFIG["interval_width"],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(prophet_df)

        self.logger.info("[%s | prophet] Fit complete", self.ticker)

    def _build_future_df(self, n_steps: int) -> pd.DataFrame:
        """
        Build Prophet's future DataFrame for n_steps business days
        starting the day after the last training date.
        """
        future_dates = pd.date_range(
            start=self._last_date + pd.offsets.BDay(1),
            periods=n_steps,
            freq=self.FREQ,
        )
        return pd.DataFrame({"ds": future_dates})

    def _predict(self, n_steps: int) -> np.ndarray:
        """Generate a point forecast (yhat) for n_steps business days ahead."""
        future     = self._build_future_df(n_steps)
        forecast   = self._model.predict(future)
        return forecast["yhat"].values.astype(float)

    def _predict_with_ci(
        self,
        n_steps: int,
        confidence: float = 0.95,
    ) -> dict[str, np.ndarray]:
        """
        Return forecast with Prophet's native confidence intervals.

        Note: Prophet's interval_width is set at fit time (from PROPHET_CONFIG).
        The confidence argument here is noted but Prophet always returns
        the CI it was configured with. For dynamic CI, refit with different
        interval_width — acceptable for this project.
        """
        if abs(confidence - PROPHET_CONFIG["interval_width"]) > 0.01:
            self.logger.warning(
                "[%s | prophet] Requested CI=%.2f but model fitted with "
                "interval_width=%.2f. Returning fitted CI.",
                self.ticker, confidence, PROPHET_CONFIG["interval_width"],
            )

        future   = self._build_future_df(n_steps)
        forecast = self._model.predict(future)

        return {
            "forecast": forecast["yhat"].values.astype(float),
            "lower":    forecast["yhat_lower"].values.astype(float),
            "upper":    forecast["yhat_upper"].values.astype(float),
        }

    def save(self, path: Path) -> None:
        """Pickle the ProphetModel to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file = path / self.MODEL_FILENAME
        with open(save_file, "wb") as f:
            pickle.dump(self, f)
        self.logger.info("[%s | prophet] Model saved → %s", self.ticker, save_file)

    def load(self, path: Path) -> "ProphetModel":
        """Load a pickled ProphetModel from disk."""
        load_file = Path(path) / self.MODEL_FILENAME
        with open(load_file, "rb") as f:
            loaded = pickle.load(f)
        self.__dict__.update(loaded.__dict__)
        self.logger.info("[%s | prophet] Model loaded ← %s", self.ticker, load_file)
        return self

    def summary(self) -> str:
        if not self.is_fitted:
            return f"ProphetModel({self.ticker}) — not fitted"
        return (
            f"ProphetModel({self.ticker}) — "
            f"last_date={self._last_date.date()}, "
            f"fit_time={self._fit_duration_seconds:.1f}s"
        )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    from data.preprocessing import preprocess_ticker

    np.random.seed(42)

    print("\n" + "═" * 60)
    print("  models/prophet_model.py — smoke test")
    print("═" * 60 + "\n")

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
    assert processed is not None

    print("1. Fitting Prophet…")
    model = ProphetModel("TEST")
    model.fit(processed.train, processed.val)
    print(f"   {model.summary()}")

    print("\n2. Point forecast (10 steps)…")
    fc = model.predict(10)
    print(f"   Forecast: {np.round(fc, 4)}")

    print("\n3. Forecast with CI (10 steps)…")
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
        model2 = ProphetModel("TEST")
        model2.load(Path(tmpdir))
        fc2 = model2.predict(5)
        assert np.allclose(model.predict(5), fc2), "Save/load forecast mismatch!"
    print("   ✅ Save/load produces identical forecasts")

    print("\n✅ ProphetModel smoke test PASSED\n")
