#!/usr/bin/env python3
"""
Cohere Transcribe HTTP Server
...

Lightweight FastAPI server that exposes Cohere Transcribe as an HTTP endpoint.
Uses the native transformers>=5.4.0 API (CohereAsrForConditionalGeneration +
model.generate()) for lowest latency - bypasses the trust_remote_code
model.transcribe() wrapper overhead.

Runs in /opt/cohere-venv (separate venv with transformers>=5.4.0) so it does
not conflict with the pinned transformers==4.57.3 required by qwen-tts.

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
    /opt/cohere-venv/bin/python3 cohere_transcribe_server.py --host 0.0.0.0 --port 8881

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

_request_count = 0

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('cohere_transcribe_server')

MODEL_ID = os.getenv('COHERE_TRANSCRIBE_MODEL', 'CohereLabs/cohere-transcribe-03-2026')
DEVICE = os.getenv('COHERE_TRANSCRIBE_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
COHERE_SAMPLE_RATE = 16000
# Bucket sizes for encoder input padding (mel frames at ~100 fps for 16kHz audio).
# Each bucket gets its own CUDA graph after first use.
# Input is padded to the smallest bucket >= actual frame count.
# 3 buckets warmed up at startup; longer audio falls back to eager (no CUDA graph).
_default_buckets = '150,300,500,1000'
ENCODER_BUCKETS = sorted(int(x) for x in os.getenv('COHERE_ENCODER_BUCKETS', _default_buckets).split(','))
COHERE_PROFILE = os.getenv('COHERE_PROFILE', '').lower() in ('1', 'true', 'yes')

# Module-level singletons - loaded once, shared across all requests
_model = None
_processor = None


def load_model():
    global _model, _processor
    if _model is not None:
        return
    logger.info(f'Loading Cohere Transcribe model: {MODEL_ID} on {DEVICE}')
    logger.info(f'torch version: {torch.__version__}')

    import transformers
    logger.info(f'transformers version: {transformers.__version__}')

    if torch.cuda.is_available():
        logger.info(f'CUDA device: {torch.cuda.get_device_name(0)}')
        logger.info(f'CUDA memory total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
        logger.info(f'CUDA memory allocated before load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB')

    # Native transformers>=5.4.0 API - no trust_remote_code needed
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    # bfloat16: same dynamic range as float32, ~2x faster on H200 (memory-bandwidth bound)
    _model = CohereAsrForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    _model.eval()

    # Compile encoder with reduce-overhead (CUDA graphs).
    # Inputs are padded to the nearest bucket so the encoder always sees a fixed shape.
    # CUDA graphs capture the kernel sequence once; subsequent calls replay without
    # per-kernel launch overhead (~5us/kernel * ~20 kernels * 48 layers = ~5ms saved).
    logger.info(f'Compiling encoder with torch.compile(mode=reduce-overhead) '
                f'(buckets: {ENCODER_BUCKETS})...')
    _model.model.encoder = torch.compile(_model.model.encoder, mode='reduce-overhead')

    # Compile each decoder layer with dynamic=True.
    # dynamic=True handles variable-length generation without recompilation.
    # Per-layer compilation reduces Python + kernel launch overhead per decode step.
    decoder_layers = _model.model.decoder.layers
    logger.info(f'Compiling {len(decoder_layers)} decoder layers with torch.compile(dynamic=True)...')
    for layer in decoder_layers:
        layer.forward = torch.compile(layer.forward, dynamic=True)

    # Compile the mel filterbank if accessible (reduces feature extraction overhead)
    try:
        filterbank = _processor.feature_extractor.filterbank
        filterbank.forward = torch.compile(filterbank.forward)
        logger.info('Compiled mel filterbank')
    except AttributeError:
        logger.info('No filterbank found on processor.feature_extractor (skipping)')

    if torch.cuda.is_available():
        logger.info(f'CUDA memory allocated after load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB')
        logger.info(f'Model dtype: {next(_model.parameters()).dtype}')
        logger.info(f'Model device: {next(_model.parameters()).device}')

    logger.info('Cohere Transcribe model loaded. Running warmup...')
    _warmup()
    logger.info('Warmup complete.')

    # Log dynamo compilation stats
    try:
        from torch._dynamo.utils import counters
        logger.info(f'dynamo counters after warmup: {dict(counters)}')
    except Exception as e:
        logger.info(f'dynamo counters unavailable: {e}')


def _warmup():
    """Pre-warm all encoder buckets so CUDA graphs are compiled before real calls."""
    try:
        if COHERE_PROFILE:
            logger.info('COHERE_PROFILE=1: running torch.profiler on warmup call...')
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=False,
            ) as prof:
                _transcribe_array(np.zeros(16000, dtype=np.float32), language='en')
            table = prof.key_averages().table(sort_by='cuda_time_total', row_limit=25)
            logger.info(f'CUDA profile (warmup):\n{table}')
        else:
            # Warm up each bucket by running a dummy inference at that frame count.
            # Each bucket triggers CUDA graph compilation once here, not during live calls.
            for bucket in ENCODER_BUCKETS:
                # (bucket - 1) * 160 samples produces exactly bucket mel frames
                # (processor adds 1 frame: frames = samples/160 + 1)
                n_samples = (bucket - 1) * 160
                dummy = np.zeros(n_samples, dtype=np.float32)
                logger.info(f'Warming up bucket {bucket} frames ({n_samples} samples)...')
                _transcribe_array(dummy, language='en')
            logger.info(f'Warmup complete for {len(ENCODER_BUCKETS)} buckets: {ENCODER_BUCKETS}')
    except Exception as e:
        logger.warning(f'Warmup inference failed (non-fatal): {e}')


def _transcribe_array(audio_16k: np.ndarray, language: str = 'en') -> str:
    """Core transcription: float32 16kHz numpy array -> text string."""
    return _transcribe_array_impl(audio_16k, language)


def _pad_to_fixed_frames(input_features: torch.Tensor):
    """[DEPRECATED - use _pad_audio_to_bucket instead]"""
    return input_features, None


def _pad_audio_to_bucket(audio_16k: np.ndarray) -> tuple:
    """Pad raw audio to the smallest bucket that fits, before processor.

    Padding audio (not mel spectrogram) means the processor produces a fixed-size
    output with no attention_mask needed -> consistent CUDA graph shapes.

    Returns (padded_audio, bucket_frames) where bucket_frames is the target bucket.
    frames = samples/160 + 1, so target_samples = (bucket - 1) * 160.
    """
    # Estimate frame count: frames = len(audio)/160 + 1
    estimated_frames = len(audio_16k) // 160 + 1
    bucket = next((b for b in ENCODER_BUCKETS if b >= estimated_frames), None)
    if bucket is None:
        logger.warning(f'Audio ~{estimated_frames} frames exceeds all buckets (max {ENCODER_BUCKETS[-1]}), no CUDA graph')
        return audio_16k, None
    target_samples = (bucket - 1) * 160
    if len(audio_16k) >= target_samples:
        return audio_16k, bucket
    padded = np.pad(audio_16k, (0, target_samples - len(audio_16k)))
    return padded, bucket


def _transcribe_array_impl(audio_16k: np.ndarray, language: str = 'en') -> str:
    t_proc = time.perf_counter()
    # Pad raw audio to bucket size before processor -> fixed mel frame count -> no attention_mask
    audio_padded, bucket = _pad_audio_to_bucket(audio_16k)
    inputs = _processor(
        audio_padded,
        sampling_rate=COHERE_SAMPLE_RATE,
        return_tensors='pt',
        language=language,
    )
    proc_ms = (time.perf_counter() - t_proc) * 1000

    input_features = inputs['input_features'].to(DEVICE, dtype=_model.dtype)
    T_orig = input_features.shape[1]  # should equal bucket

    # Run encoder separately to profile it
    t_enc = time.perf_counter()
    with torch.inference_mode():
        # Required when using CUDA graphs to prevent output tensor overwrite between calls
        torch.compiler.cudagraph_mark_step_begin()
        encoder_outputs = _model.model.encoder(
            input_features,
            attention_mask=None,
        )
    enc_ms = (time.perf_counter() - t_enc) * 1000

    # Run decoder (generate) with pre-computed encoder outputs
    inputs_for_gen = inputs.to(DEVICE, dtype=_model.dtype)
    t_dec = time.perf_counter()
    with torch.inference_mode():
        outputs = _model.generate(
            **inputs_for_gen,
            encoder_outputs=encoder_outputs,
            max_new_tokens=64,
        )
    dec_ms = (time.perf_counter() - t_dec) * 1000

    n_tokens = outputs.shape[-1] if hasattr(outputs, 'shape') else '?'
    logger.info(
        f'profile: processor={proc_ms:.1f}ms encoder={enc_ms:.1f}ms '
        f'decoder={dec_ms:.1f}ms tokens={n_tokens} '
        f'orig_frames={len(audio_16k)//160+1} bucket={bucket} actual_frames={T_orig}'
    )

    text = _processor.decode(outputs, skip_special_tokens=True)
    if isinstance(text, list):
        text = text[0]
    return text.strip()


def resample_2x(audio: np.ndarray) -> np.ndarray:
    """Upsample 8kHz -> 16kHz using polyphase resampling (scipy)."""
    return scipy.signal.resample_poly(audio, up=2, down=1).astype(np.float32)


def transcribe_ulaw(ulaw_bytes: bytes, language: str = 'en') -> str:
    """Convert ulaw 8kHz bytes to text via Cohere Transcribe."""
    global _request_count
    _request_count += 1
    req_id = _request_count
    audio_duration_s = len(ulaw_bytes) / 8000.0
    logger.info(f'[req#{req_id}] transcribe_ulaw: {len(ulaw_bytes)} bytes = {audio_duration_s:.2f}s audio')

    # ulaw -> PCM int16 -> float32
    t_decode = time.perf_counter()
    pcm_bytes = audioop.ulaw2lin(ulaw_bytes, 2)
    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0
    logger.info(f'[req#{req_id}] ulaw decode: {(time.perf_counter()-t_decode)*1000:.1f}ms')

    # Resample 8kHz -> 16kHz
    t_resample = time.perf_counter()
    audio_16k = resample_2x(audio_float)
    logger.info(f'[req#{req_id}] resample: {(time.perf_counter()-t_resample)*1000:.1f}ms, output shape: {audio_16k.shape}')

    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated(0) / 1e6
        logger.info(f'[req#{req_id}] GPU mem before transcribe: {mem_before:.0f} MB')

    t_transcribe = time.perf_counter()
    text = _transcribe_array(audio_16k, language=language)
    transcribe_ms = (time.perf_counter() - t_transcribe) * 1000
    logger.info(f'[req#{req_id}] model.generate: {transcribe_ms:.1f}ms')

    if torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated(0) / 1e6
        logger.info(f'[req#{req_id}] GPU mem after transcribe: {mem_after:.0f} MB')

    rtfx = audio_duration_s / (transcribe_ms / 1000) if transcribe_ms > 0 else 0
    logger.info(f'[req#{req_id}] result: "{text}" | RTFx={rtfx:.0f}x | audio={audio_duration_s:.2f}s | transcribe={transcribe_ms:.0f}ms')

    return text


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title='Cohere Transcribe Server', lifespan=lifespan)


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
