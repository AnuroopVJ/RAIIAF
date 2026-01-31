"""RAIIAF public API.

Exposes the main file handler class for encoding/decoding RAIIAF artifacts.
"""

from .handlers.file_handler import raiiafFileHandler

__all__ = [
    "raiiafFileHandler",
]
