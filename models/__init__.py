"""
models/__init__.py
==================
Exports all model classes and the base so the trainer can import
everything from a single location:

    from models import ARIMAModel, ProphetModel, LSTMModel
"""

from models.base_model import BaseModel, naive_baseline_metrics, _compute_metrics
from models.arima_model import ARIMAModel
from models.prophet_model import ProphetModel
from models.lstm_model import LSTMModel

__all__ = [
    "BaseModel",
    "ARIMAModel",
    "ProphetModel",
    "LSTMModel",
    "naive_baseline_metrics",
    "_compute_metrics",
]
