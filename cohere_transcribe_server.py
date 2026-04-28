#!/usr/bin/env python3
"""
Cohere Transcribe HTTP Server

Lightweight FastAPI server that exposes Cohere Transcribe as an HTTP endpoint.
Uses the native transformers>=5.4.0 API (CohereAsrForConditionalGeneration)
for lowest latency.

Custom greedy decode loop (COHERE_CUSTOM_DECODE=1, default) bypasses HF generate()
Python overhead for ~2x faster decoder. Falls back to generate() if disabled.

API:
    POST /transcribe
        Body: raw ulaw 8kHz audio bytes (application/octet-stream)
        Query params: language (str, default 'en')
        Response: {"text": "...", "duration_ms": N}

    GET /health
        Response: {"status": "ok", "model": "..."}

Usage:
    /opt/cohere-venv/bin/python3 cohere_transcribe_server.py --host 0.0.0.0 --port 8881

Environment variables:
    COHERE_TRANSCRIBE_MODEL   HuggingFace model ID (default: CohereLabs/cohere-transcribe-03-2026)
    COHERE_TRANSCRIBE_DEVICE  torch device (default: cuda)
    COHERE_CUSTOM_DECODE      Use custom greedy decode (default: 1)
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
_default_buckets = '150,300,500,1000'
ENCODER_BUCKETS = sorted(int(x) for x in os.getenv('COHERE_ENCODER_BUCKETS', _default_buckets).split(','))
COHERE_PROFILE = os.getenv('COHERE_PROFILE', '').lower() in ('1', 'true', 'yes')
CUSTOM_DECODE = os.getenv('COHERE_CUSTOM_DECODE', '1').lower() in ('1', 'true', 'yes')

# Max new tokens for decoder. 128 is safe for any phone utterance (EOS stops it early).
# The trust_remote_code version uses 256. Lower = faster worst-case but risks truncation.
MAX_NEW_TOKENS = int(os.getenv('COHERE_MAX_NEW_TOKENS', '128'))

# Module-level singletons
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

    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = CohereAsrForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    _model.eval()

    # Compile encoder with CUDA graphs
    logger.info(f'Compiling encoder with torch.compile(mode=reduce-overhead) '
                f'(buckets: {ENCODER_BUCKETS})...')
    _model.model.encoder = torch.compile(_model.model.encoder, mode='reduce-overhead')

    # Compile decoder layers individually
    decoder_layers = _model.model.decoder.layers
    logger.info(f'Compiling {len(decoder_layers)} decoder layers with torch.compile(dynamic=True)...')
    for layer in decoder_layers:
        layer.forward = torch.compile(layer.forward, dynamic=True)

    # Compile mel filterbank if accessible
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

    logger.info(f'Custom greedy decode: {"ENABLED" if CUSTOM_DECODE else "DISABLED (using HF generate)"}')
    logger.info('Running warmup...')
    _warmup()
    logger.info('Warmup complete.')

    try:
        from torch._dynamo.utils import counters
        logger.info(f'dynamo counters after warmup: {dict(counters)}')
    except Exception as e:
        logger.info(f'dynamo counters unavailable: {e}')


def _warmup():
    """Pre-warm all encoder buckets."""
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
            for bucket in ENCODER_BUCKETS:
                n_samples = (bucket - 1) * 160
                dummy = np.zeros(n_samples, dtype=np.float32)
                logger.info(f'Warming up bucket {bucket} frames ({n_samples} samples)...')
                _transcribe_array(dummy, language='en')
            logger.info(f'Warmup complete for {len(ENCODER_BUCKETS)} buckets: {ENCODER_BUCKETS}')
    except Exception as e:
        logger.warning(f'Warmup inference failed (non-fatal): {e}')


# ---------------------------------------------------------------------------
# Custom greedy decode - bypasses HF generate() Python overhead
# ---------------------------------------------------------------------------

def _greedy_decode(encoder_outputs, processor_inputs, max_new_tokens=64):
    """Custom greedy decode loop that bypasses HF generate() overhead.

    For ASR with ~18-20 tokens, HF generate() adds ~1-1.5ms of Python overhead
    per token (LogitsProcessor, StoppingCriteria, output management, etc.).
    This tight loop eliminates that overhead.

    Args:
        encoder_outputs: Pre-computed encoder outputs (BaseModelOutput)
        processor_inputs: Dict from processor, moved to device. May contain
                         'input_ids' or 'decoder_input_ids' for the prompt.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        torch.Tensor of token IDs (1, seq_len) including prompt.
    """
    model = _model
    config = model.config

    eos_token_id = getattr(config, 'eos_token_id', None)
    pad_token_id = getattr(config, 'pad_token_id', None)

    # Get decoder prompt tokens from processor output
    if 'decoder_input_ids' in processor_inputs:
        decoder_input_ids = processor_inputs['decoder_input_ids']
    elif 'input_ids' in processor_inputs:
        decoder_input_ids = processor_inputs['input_ids']
    else:
        # Fallback: use decoder_start_token_id
        start_id = getattr(config, 'decoder_start_token_id', None)
        if start_id is None:
            start_id = getattr(config, 'bos_token_id', 0)
        decoder_input_ids = torch.tensor([[start_id]], device=DEVICE, dtype=torch.long)

    # Ensure on device
    decoder_input_ids = decoder_input_ids.to(DEVICE)
    batch_size = decoder_input_ids.shape[0]

    # Collect all generated token IDs (including prompt)
    all_token_ids = decoder_input_ids.clone()

    # Prefill: run forward with all prompt tokens to build KV cache
    past_key_values = None
    with torch.inference_mode():
        prefill_out = model(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = prefill_out.past_key_values
        # Get the last token's logits for the first generated token
        next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        all_token_ids = torch.cat([all_token_ids, next_token], dim=-1)

        # Check EOS after prefill
        if eos_token_id is not None and next_token.item() == eos_token_id:
            return all_token_ids

        # Autoregressive decode loop
        for step in range(max_new_tokens - 1):
            out = model(
                decoder_input_ids=next_token,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            all_token_ids = torch.cat([all_token_ids, next_token], dim=-1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    return all_token_ids


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------

def _pad_audio_to_bucket(audio_16k: np.ndarray) -> tuple:
    """Pad raw audio to the smallest bucket that fits."""
    estimated_frames = len(audio_16k) // 160 + 1
    bucket = next((b for b in ENCODER_BUCKETS if b >= estimated_frames), None)
    if bucket is None:
        logger.warning(f'Audio ~{estimated_frames} frames exceeds all buckets (max {ENCODER_BUCKETS[-1]})')
        return audio_16k, None
    target_samples = (bucket - 1) * 160
    if len(audio_16k) >= target_samples:
        return audio_16k, bucket
    padded = np.pad(audio_16k, (0, target_samples - len(audio_16k)))
    return padded, bucket


def _transcribe_array(audio_16k: np.ndarray, language: str = 'en') -> str:
    """Core transcription: float32 16kHz numpy array -> text string."""
    t_proc = time.perf_counter()
    audio_padded, bucket = _pad_audio_to_bucket(audio_16k)
    inputs = _processor(
        audio_padded,
        sampling_rate=COHERE_SAMPLE_RATE,
        return_tensors='pt',
        language=language,
        punctuation=True,
    )
    proc_ms = (time.perf_counter() - t_proc) * 1000

    input_features = inputs['input_features'].to(DEVICE, dtype=_model.dtype)
    T_orig = input_features.shape[1]

    # Run encoder
    t_enc = time.perf_counter()
    with torch.inference_mode():
        torch.compiler.cudagraph_mark_step_begin()
        encoder_outputs = _model.model.encoder(
            input_features,
            attention_mask=None,
        )
    enc_ms = (time.perf_counter() - t_enc) * 1000

    # Run decoder
    inputs_on_device = inputs.to(DEVICE, dtype=_model.dtype)
    t_dec = time.perf_counter()

    if CUSTOM_DECODE:
        # Custom greedy decode - bypasses HF generate() overhead
        with torch.inference_mode():
            outputs = _greedy_decode(
                encoder_outputs=encoder_outputs,
                processor_inputs=inputs_on_device,
                max_new_tokens=MAX_NEW_TOKENS,
            )
    else:
        # HF generate() fallback
        with torch.inference_mode():
            outputs = _model.generate(
                **inputs_on_device,
                encoder_outputs=encoder_outputs,
                max_new_tokens=MAX_NEW_TOKENS,
            )

    dec_ms = (time.perf_counter() - t_dec) * 1000

    n_tokens = outputs.shape[-1] if hasattr(outputs, 'shape') else '?'
    logger.info(
        f'profile: processor={proc_ms:.1f}ms encoder={enc_ms:.1f}ms '
        f'decoder={dec_ms:.1f}ms tokens={n_tokens} '
        f'decode_mode={"custom" if CUSTOM_DECODE else "generate"} '
        f'orig_frames={len(audio_16k)//160+1} bucket={bucket} actual_frames={T_orig}'
    )

    text = _processor.decode(outputs[0] if outputs.dim() > 1 else outputs, skip_special_tokens=True)
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

    t_decode = time.perf_counter()
    pcm_bytes = audioop.ulaw2lin(ulaw_bytes, 2)
    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0
    logger.info(f'[req#{req_id}] ulaw decode: {(time.perf_counter()-t_decode)*1000:.1f}ms')

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
    if _model is None:
        raise HTTPException(status_code=503, detail='Model not loaded yet')

    ulaw_bytes = await request.body()
    if not ulaw_bytes:
        raise HTTPException(status_code=400, detail='Empty audio body')

    t0 = time.time()
    try:
        text = transcribe_ulaw(ulaw_bytes, language=language)
    except Exception as e:
        logger.error(f'Transcription error: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    duration_ms = int((time.time() - t0) * 1000)
    logger.info(f'Transcribed {len(ulaw_bytes)} bytes in {duration_ms}ms -> "{text}"')
    return JSONResponse({'text': text, 'duration_ms': duration_ms})


@app.get('/health')
async def health():
    return {'status': 'ok', 'model': MODEL_ID, 'device': DEVICE, 'loaded': _model is not None,
            'custom_decode': CUSTOM_DECODE}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8881)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')
