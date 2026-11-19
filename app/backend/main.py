"""
FastAPI application entrypoint alias.
Re-exports app from app.backend.api.
"""

from app.backend.api import app

__all__ = ["app"]
