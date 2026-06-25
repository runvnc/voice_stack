#!/usr/bin/env python3
"""Batched Cohere Transcribe HTTP Server using nano-cohere-transcribe.

Drop-in replacement for cohere_transcribe_server.py with:
- 1.5-3.6x faster inference (CUDA graph decoder, KV cache)
- Request batching for concurrent STT (configurable window + max batch size)
- Minimal dependencies (no transformers needed)

API (unchanged):
    POST /transcribe
        Body: raw ulaw 8kHz audio bytes (application/octet-stream)
        Query params: language (str, default 'en')
        Response: {"text": "...", "duration_ms": N}

    GET /health
        Response: {"status": "ok", "model": "...", ...}

Usage:
    python3 nano_cohere_transcribe_server.py --host 0.0.0.0 --port 8881

Environment variables:
    COHERE_TRANSCRIBE_MODEL   HuggingFace model ID (default: CohereLabs/cohere-transcribe-03-2026)
    COHERE_TRANSCRIBE_DEVICE  torch device (default: cuda)
    BATCH_WINDOW_MS           Max time to wait collecting batch (default: 30)
    BATCH_MAX_SIZE            Max requests per batch (default: 8)
    HF_HOME                   HuggingFace cache dir
"""
import argparse
import asyncio
import audioop
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

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
logger = logging.getLogger('nano_cohere_transcribe_server')

MODEL_ID = os.getenv('COHERE_TRANSCRIBE_MODEL', 'CohereLabs/cohere-transcribe-03-2026')
DEVICE = os.getenv('COHERE_TRANSCRIBE_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
BATCH_WINDOW_MS = int(os.getenv('BATCH_WINDOW_MS', '5'))
BATCH_MAX_SIZE = int(os.getenv('BATCH_MAX_SIZE', '8'))

_model = None
_batch_queue: Optional[asyncio.Queue] = None
_request_count = 0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    global _model
    if _model is not None:
        return

    logger.info(f'Loading nano-cohere-transcribe model: {MODEL_ID} on {DEVICE}')
    logger.info(f'torch version: {torch.__version__}')

    if torch.cuda.is_available():
        logger.info(f'CUDA device: {torch.cuda.get_device_name(0)}')
        logger.info(f'CUDA memory total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    from nano_cohere_transcribe import from_pretrained

    _model = from_pretrained(MODEL_ID, device=DEVICE)

    if torch.cuda.is_available():
        logger.info(f'CUDA memory after load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB')

    # Warmup
    logger.info('Running warmup...')
    dummy = torch.zeros(16000, dtype=torch.float32)  # 1 second of silence
    _model.transcribe(dummy, language='en')
    _model.transcribe(dummy, language='en')  # second warmup for CUDA graphs
    logger.info('Warmup complete.')


# ---------------------------------------------------------------------------
# Audio preprocessing (same as original server)
# ---------------------------------------------------------------------------

def resample_2x(audio: np.ndarray) -> np.ndarray:
    """Upsample 8kHz -> 16kHz using polyphase resampling."""
    return scipy.signal.resample_poly(audio, up=2, down=1).astype(np.float32)


def ulaw_to_float16k(ulaw_bytes: bytes) -> torch.Tensor:
    """Convert ulaw 8kHz bytes to float32 16kHz torch tensor."""
    pcm_bytes = audioop.ulaw2lin(ulaw_bytes, 2)
    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0
    audio_16k = resample_2x(audio_float)
    return torch.from_numpy(audio_16k)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

@dataclass
class TranscribeRequest:
    waveform: torch.Tensor
    language: str
    future: asyncio.Future
    submit_time: float


async def batch_worker():
    """Background worker that collects and batch-processes transcription requests."""
    global _batch_queue
    logger.info(f'Batch worker started (window={BATCH_WINDOW_MS}ms, max_size={BATCH_MAX_SIZE})')

    while True:
        batch: list[TranscribeRequest] = []

        # Wait for first request
        try:
            first = await _batch_queue.get()
            batch.append(first)
        except Exception:
            continue

        # Collect more requests within the time window
        deadline = time.monotonic() + BATCH_WINDOW_MS / 1000.0
        while len(batch) < BATCH_MAX_SIZE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = await asyncio.wait_for(_batch_queue.get(), timeout=remaining)
                batch.append(req)
            except asyncio.TimeoutError:
                break
            except Exception:
                break

        # Process batch
        t0 = time.perf_counter()
        try:
            if len(batch) == 1:
                # Single request - no batching overhead
                text = _model.transcribe(
                    batch[0].waveform,
                    language=batch[0].language,
                )
                batch[0].future.set_result(text)
            else:
                # Batch transcription
                waveforms = [r.waveform for r in batch]
                # All requests in a batch use the same language (simplification)
                language = batch[0].language
                texts = _model.transcribe_batch(
                    waveforms,
                    language=language,
                    batch_size=len(waveforms),
                )
                for req, text in zip(batch, texts):
                    req.future.set_result(text)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            total_audio_s = sum(len(r.waveform) / 16000.0 for r in batch)
            avg_wait_ms = sum(t0 - r.submit_time for r in batch) / len(batch) * 1000
            logger.info(
                f'Batch processed: size={len(batch)}, '
                f'inference={elapsed_ms:.0f}ms, '
                f'total_audio={total_audio_s:.1f}s, '
                f'avg_wait={avg_wait_ms:.0f}ms'
            )

        except Exception as e:
            logger.exception(f'Batch processing error: {e}')
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _batch_queue
    load_model()
    _batch_queue = asyncio.Queue()
    worker_task = asyncio.create_task(batch_worker())
    yield
    worker_task.cancel()


app = FastAPI(title='Nano Cohere Transcribe Server', lifespan=lifespan)


@app.post('/transcribe')
async def transcribe(request: Request, language: str = 'en'):
    global _request_count
    if _model is None:
        raise HTTPException(status_code=503, detail='Model not loaded yet')

    ulaw_bytes = await request.body()
    if not ulaw_bytes:
        raise HTTPException(status_code=400, detail='Empty audio body')

    _request_count += 1
    req_id = _request_count
    audio_duration_s = len(ulaw_bytes) / 8000.0
    logger.info(f'[req#{req_id}] {len(ulaw_bytes)} bytes = {audio_duration_s:.2f}s audio')

    t0 = time.time()

    # Preprocess audio
    t_pre = time.perf_counter()
    waveform = ulaw_to_float16k(ulaw_bytes)
    pre_ms = (time.perf_counter() - t_pre) * 1000
    logger.info(f'[req#{req_id}] preprocess: {pre_ms:.1f}ms, samples: {len(waveform)}')

    # Submit to batch queue
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    await _batch_queue.put(TranscribeRequest(
        waveform=waveform,
        language=language,
        future=future,
        submit_time=time.perf_counter(),
    ))

    # Wait for result
    try:
        text = await asyncio.wait_for(future, timeout=30.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail='Transcription timeout')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    duration_ms = int((time.time() - t0) * 1000)
    rtfx = audio_duration_s / (duration_ms / 1000) if duration_ms > 0 else 0
    logger.info(
        f'[req#{req_id}] result: "{text}" | '
        f'RTFx={rtfx:.0f}x | audio={audio_duration_s:.2f}s | total={duration_ms}ms'
    )
    return JSONResponse({'text': text, 'duration_ms': duration_ms})


@app.get('/health')
async def health():
    return {
        'status': 'ok',
        'model': MODEL_ID,
        'device': DEVICE,
        'loaded': _model is not None,
        'backend': 'nano-cohere-transcribe',
        'batch_window_ms': BATCH_WINDOW_MS,
        'batch_max_size': BATCH_MAX_SIZE,
        'requests_served': _request_count,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8881)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')
