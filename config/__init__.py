"""
config/__init__.py
==================
Makes `config` a Python package and re-exports the two most-used objects
so callers can write:

    from config import settings, get_logger

instead of the longer form.
"""

from config import settings                          # noqa: F401
from config.logging_config import get_logger         # noqa: F401

__all__ = ["settings", "get_logger"]
