#!/usr/bin/env python3
"""
Wrapper script that runs the groxaxo Qwen3-TTS server with the voice
registration router mounted.

This avoids modifying the upstream groxaxo server code. We import its
FastAPI app and mount our additional router before starting uvicorn.

Usage (replaces `python -m api.main`):
    python run_groxaxo.py --host 0.0.0.0 --port 8880

Environment variables (same as groxaxo server):
    TTS_BACKEND, TTS_MODEL_NAME, VOICE_LIBRARY_DIR, etc.
"""

import argparse
import os
import sys

# Ensure the groxaxo package is importable
# The groxaxo server is installed via pip, so api module should be available


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS server with voice registration")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8880")))
    parser.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "1")))
    args = parser.parse_args()

    # Import the groxaxo server's FastAPI app
    from api.main import app

    # Import and mount the voice registration router
    from voice_register_router import router as voice_register_router
    app.include_router(voice_register_router, prefix="/v1")

    # Log that voice registration is available
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Voice registration endpoint mounted at POST /v1/audio/voice-register")

    # Start the server
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
