"""
Stub for the prediction logger. Full SQLite implementation in Step 28.

Currently a no-op that lets api/main.py import cleanly. Replaced wholesale
in Step 28 with real persistence.
"""

from __future__ import annotations


def log_prediction(request, response, latency_ms: float) -> None:
    """Stub: real implementation in Step 28."""
    return None
