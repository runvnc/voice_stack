# coding=utf-8
"""
Voice registration router for the groxaxo Qwen3-TTS server.

Adds POST /v1/audio/voice-register endpoint that downloads reference audio
from a URL and creates a voice library profile, making it immediately
available as voice="clone:ProfileName" in /v1/audio/speech requests.

Auto-transcribes reference audio with Whisper for ICL mode (best quality).
This router is mounted by run_groxaxo.py wrapper - no modification to the
upstream groxaxo server code is needed.
"""

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Voice library directory (must match groxaxo server's VOICE_LIBRARY_DIR)
VOICE_LIBRARY_DIR = Path(
    os.environ.get("VOICE_LIBRARY_DIR", "./voice_library")
).resolve()

router = APIRouter(tags=["Voice Registration"])


def _transcribe_with_whisper(audio_np: np.ndarray, sr: int, language: str = None) -> str:
    """
    Transcribe audio using OpenAI Whisper.

    Returns the transcribed text, or empty string on failure.
    """
    try:
        import whisper
        import tempfile
        import io as _io

        logger.info(f"Transcribing audio with Whisper ({len(audio_np)/sr:.1f}s)...")

        # Whisper expects a file path, so write to a temp WAV
        model_name = os.environ.get("WHISPER_MODEL", "base")
        model = whisper.load_model(model_name)

        # Write to temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        # Save as 16kHz WAV (Whisper's expected format)
        import scipy.signal
        if sr != 16000:
            audio_16k = scipy.signal.resample_poly(audio_np, 16000, sr).astype(np.float32)
        else:
            audio_16k = audio_np.astype(np.float32)

        # Normalize
        max_val = np.max(np.abs(audio_16k))
        if max_val > 0:
            audio_16k = audio_16k / max_val

        sf.write(tmp_path, audio_16k, 16000)

        # Transcribe
        whisper_opts = {}
        if language and language != "Auto":
            whisper_opts["language"] = language

        result = model.transcribe(tmp_path, **whisper_opts)
        text = result.get("text", "").strip()

        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        logger.info(f"Whisper transcription: '{text[:100]}...'" if len(text) > 100 else f"Whisper transcription: '{text}'")
        return text

    except ImportError:
        logger.warning("Whisper not installed, skipping auto-transcription")
        return ""
    except Exception as e:
        logger.warning(f"Whisper transcription failed: {e}")
        return ""


class VoiceRegisterRequest(BaseModel):
    """Request schema for voice registration endpoint."""
    name: str = Field(
        ...,
        description="Profile name for the voice. Will be available as clone:Name.",
        max_length=64,
    )
    ref_audio_url: str = Field(
        ...,
        description="URL to download the reference audio file (WAV, MP3, etc.).",
    )
    ref_text: Optional[str] = Field(
        default=None,
        description="Transcript of the reference audio. If not provided, auto-transcribed with Whisper. Recommended for best quality.",
        max_length=4096,
    )
    language: Optional[str] = Field(
        default="Auto",
        description="Language of the voice. Default: Auto.",
    )
    x_vector_only_mode: bool = Field(
        default=False,
        description="If True, use x-vector only mode (no ref_text needed, lower quality). If False, use ICL mode (recommended, auto-transcribes with Whisper if no ref_text provided).",
    )


class VoiceRegisterResponse(BaseModel):
    """Response schema for voice registration endpoint."""
    status: str = Field(..., description="Registration status")
    name: str = Field(..., description="Profile name")
    voice_id: str = Field(..., description="Voice ID to use: clone:Name")
    message: str = Field(..., description="Human-readable status message")
    ref_text: Optional[str] = Field(None, description="Transcript used (auto-transcribed or provided)")


@router.post("/audio/voice-register", response_model=VoiceRegisterResponse)
async def register_voice(request: VoiceRegisterRequest):
    """
    Register a voice from a reference audio URL.

    Downloads the audio, validates it, and creates a voice library profile.
    The voice is immediately available as `voice="clone:Name"` in
    `/v1/audio/speech` requests.

    If ref_text is not provided, the audio is auto-transcribed with Whisper
    for ICL mode (best voice cloning quality). Set x_vector_only_mode=True
    to skip transcription (lower quality but faster).

    If a profile with the same name already exists, it is updated (re-registered).
    """
    profile_name = request.name.strip()
    if not profile_name:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_name",
                "message": "Profile name cannot be empty",
                "type": "invalid_request_error",
            },
        )

    # Validate name doesn't contain path traversal or weird chars
    if not all(c.isalnum() or c in ('_', '-', ' ') for c in profile_name):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_name",
                "message": f"Profile name '{profile_name}' contains invalid characters. Use alphanumeric, underscore, hyphen, or space.",
                "type": "invalid_request_error",
            },
        )

    # Download the reference audio
    logger.info(f"Downloading reference audio from: {request.ref_audio_url}")
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
        ) as client:
            response = await client.get(request.ref_audio_url, follow_redirects=True)
            response.raise_for_status()
            audio_bytes = response.content
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "download_failed",
                "message": f"Failed to download audio: HTTP {e.response.status_code}",
                "type": "invalid_request_error",
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "download_failed",
                "message": f"Failed to download audio: {e}",
                "type": "invalid_request_error",
            },
        )

    if len(audio_bytes) < 1000:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "audio_too_small",
                "message": f"Downloaded audio is only {len(audio_bytes)} bytes - likely invalid",
                "type": "invalid_request_error",
            },
        )

    # Validate the audio is readable
    import io
    try:
        audio_np, sr = sf.read(io.BytesIO(audio_bytes))
        if len(audio_np.shape) > 1:
            audio_np = audio_np.mean(axis=1)
        duration = len(audio_np) / sr
        logger.info(f"Reference audio: {duration:.1f}s at {sr}Hz, {len(audio_bytes)} bytes")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "audio_invalid",
                "message": f"Downloaded file is not valid audio: {e}",
                "type": "invalid_request_error",
            },
        )

    # Determine ref_text: use provided, or auto-transcribe with Whisper
    ref_text = request.ref_text
    x_vector_only_mode = request.x_vector_only_mode

    if not x_vector_only_mode and not ref_text:
        # Auto-transcribe with Whisper for ICL mode
        logger.info("No ref_text provided, auto-transcribing with Whisper for ICL mode...")
        ref_text = _transcribe_with_whisper(audio_np, sr, request.language)
        if ref_text:
            logger.info(f"Auto-transcribed ref_text: '{ref_text[:80]}'")
        else:
            # Whisper failed or not installed - fall back to x-vector mode
            logger.warning("Whisper transcription unavailable, falling back to x-vector mode")
            x_vector_only_mode = True

    # Create the voice library profile
    profiles_dir = VOICE_LIBRARY_DIR / "profiles"
    profile_dir = profiles_dir / profile_name.replace(" ", "_")
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Determine file extension from URL or content
    ext = ".wav"
    url_lower = request.ref_audio_url.lower()
    for candidate in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
        if url_lower.endswith(candidate):
            ext = candidate
            break

    ref_filename = f"reference{ext}"
    ref_path = profile_dir / ref_filename

    # Write the audio file
    ref_path.write_bytes(audio_bytes)

    # Write meta.json
    meta = {
        "profile_id": str(uuid.uuid4()),
        "name": profile_name,
        "ref_audio_filename": ref_filename,
        "ref_text": ref_text or "",
        "language": request.language or "Auto",
        "x_vector_only_mode": x_vector_only_mode,
        "source_url": request.ref_audio_url,
        "duration_seconds": round(duration, 2),
        "sample_rate": sr,
        "auto_transcribed": ref_text is not None and request.ref_text is None,
    }
    meta_path = profile_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Clear the in-process ref audio cache so the server picks up the new profile
    try:
        from api.routers.openai_compatible import _ref_audio_cache
        cache_key = profile_name.lower()
        if cache_key in _ref_audio_cache:
            del _ref_audio_cache[cache_key]
            logger.info(f"Cleared ref audio cache for '{profile_name}'")
    except Exception:
        pass  # Cache may not exist yet, that's fine

    voice_id = f"clone:{profile_name}"
    mode_str = "x-vector" if x_vector_only_mode else "ICL"
    transcript_str = f", ref_text='{ref_text[:50]}...'" if ref_text and len(ref_text) > 50 else f", ref_text='{ref_text}'" if ref_text else ""
    logger.info(f"Voice profile registered: {voice_id} ({duration:.1f}s, {mode_str} mode{transcript_str})")

    return VoiceRegisterResponse(
        status="registered",
        name=profile_name,
        voice_id=voice_id,
        message=f"Voice '{profile_name}' registered successfully ({duration:.1f}s, {mode_str} mode{transcript_str}). Use voice='{voice_id}' in speech requests.",
        ref_text=ref_text,
    )
