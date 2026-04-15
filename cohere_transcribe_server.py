#!/usr/bin/env python3
"""
Cohere Transcribe HTTP Server

Lightweight FastAPI server that exposes Cohere Transcribe as an HTTP endpoint.
Uses scipy for high-quality 8kHz->16kHz resampling.
Designed to run inside the RunPod H200 container alongside the LLM and TTS servers.

API:
    POST /transcribe
        Body: raw ulaw 8kHz audio bytes (application/octet-stream)
        Query params:
            language (str, default 'en')
            sample_rate (int, default 8000)  # must be 8000 for ulaw SIP audio
        Response: {"text": "...", "duration_ms": 6}

    GET /health
        Response: {"status": "ok", "model": "..."}

Usage:
    python3 cohere_transcribe_server.py --host 0.0.0.0 --port 8881

Environment variables:
    COHERE_TRANSCRIBE_MODEL   HuggingFace model ID (default: CohereLabs/cohere-transcribe-03-2026)
    COHERE_TRANSCRIBE_DEVICE  torch device (default: cuda)
    HF_HOME                   HuggingFace cache dir
"""
import argparse
import audioop
import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import scipy.signal
import torch
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('cohere_transcribe_server')

MODEL_ID = os.getenv('COHERE_TRANSCRIBE_MODEL', 'CohereLabs/cohere-transcribe-03-2026')
DEVICE = os.getenv('COHERE_TRANSCRIBE_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
COHERE_SAMPLE_RATE = 16000

# Module-level singletons - loaded once, shared across all requests
_model = None
_processor = None


def load_model():
    global _model, _processor
    if _model is not None:
        return
    logger.info(f'Loading Cohere Transcribe model: {MODEL_ID} on {DEVICE}')
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
    _model.eval()
    logger.info('Cohere Transcribe model loaded.')


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title='Cohere Transcribe Server', lifespan=lifespan)


def resample_2x(audio: np.ndarray) -> np.ndarray:
    """Upsample 8kHz -> 16kHz using polyphase resampling (scipy)."""
    return scipy.signal.resample_poly(audio, up=2, down=1).astype(np.float32)


def transcribe_ulaw(ulaw_bytes: bytes, language: str = 'en') -> str:
    """Convert ulaw 8kHz bytes to text via Cohere Transcribe."""
    # ulaw -> PCM int16 -> float32
    pcm_bytes = audioop.ulaw2lin(ulaw_bytes, 2)
    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0

    # Resample 8kHz -> 16kHz
    audio_16k = resample_2x(audio_float)

    # Use model.transcribe() - the trust_remote_code API (works with transformers 4.57.3)
    # compile=True: torch.compile encoder for faster throughput (one-time warmup on first call)
    texts = _model.transcribe(
        processor=_processor,
        audio_arrays=[audio_16k],
        sample_rates=[COHERE_SAMPLE_RATE],
        language=language,
        compile=True,
    )
    text = texts[0] if texts else ''

    return text.strip()


@app.post('/transcribe')
async def transcribe(request: Request, language: str = 'en', sample_rate: int = 8000):
    """
    Transcribe ulaw 8kHz audio bytes.

    Body: raw ulaw bytes (application/octet-stream)
    Returns: {"text": "...", "duration_ms": N}
    """
    if _model is None:
        raise HTTPException(status_code=503, detail='Model not loaded yet')

    ulaw_bytes = await request.body()
    if not ulaw_bytes:
        raise HTTPException(status_code=400, detail='Empty audio body')

    t0 = time.time()
    try:
        text = transcribe_ulaw(ulaw_bytes, language=language)
    except Exception as e:
        logger.error(f'Transcription error: {e}')
        raise HTTPException(status_code=500, detail=str(e))

    duration_ms = int((time.time() - t0) * 1000)
    logger.info(f'Transcribed {len(ulaw_bytes)} bytes in {duration_ms}ms -> "{text}"')
    return JSONResponse({'text': text, 'duration_ms': duration_ms})


@app.get('/health')
async def health():
    return {'status': 'ok', 'model': MODEL_ID, 'device': DEVICE, 'loaded': _model is not None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8881)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')
