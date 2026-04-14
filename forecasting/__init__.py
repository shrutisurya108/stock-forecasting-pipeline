"""
forecasting/__init__.py
=======================
Clean exports for the forecasting layer.
"""
from forecasting.predictor import (
    generate_predictions,
    generate_all_predictions,
    save_prediction,
    save_all_predictions,
    load_prediction,
    load_all_forecasts,
    get_available_tickers,
    PredictionResult,
)

__all__ = [
    "generate_predictions",
    "generate_all_predictions",
    "save_prediction",
    "save_all_predictions",
    "load_prediction",
    "load_all_forecasts",
    "get_available_tickers",
    "PredictionResult",
]

