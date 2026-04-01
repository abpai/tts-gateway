"""Shim: re-exports create_app from routes.py for backward compatibility."""

from tts_gateway.routes import create_app  # noqa: F401
