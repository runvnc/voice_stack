# vLLM-Omni TTS Container Handoff

**Date:** 2026-04-07  
**Session summary:** Added groxaxo OpenAI-FastAPI server as default TTS backend (lowest latency). Updated plugin with openai_impl. Refactored container startup to support 4 runtime-selectable TTS backends.

---

## Current State (Working)

- **Container:** `vllm_qwen35_35b_a3b` at `/files/upd6/mr_verification_dashboard/containers/vllm_qwen35_35b_a3b/`
- **vllm-omni version:** `0.18.0` (PyPI, released 2026-03-28, latest stable)
- **vllm base image:** `vllm/vllm-openai:v0.18.0`
- **LLM:** `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` on port 8000
- **TTS (default):** groxaxo OpenAI-FastAPI on port 8880, `/v1/audio/speech`
- **Voice Registration:** `POST /v1/audio/voice-register` (auto-register voices from URLs)
- **Plugin:** `mr_qwen3tts` at `/xfiles/plugins_ah/mr_qwen3tts`

---

## TTS Backend Selection

The container supports four TTS backends, selected via the `TTS_BACKEND` environment variable at container start (set in RunPod pod config):

| `TTS_BACKEND` | Server | Port | Notes |
|---|---|---|---|
| `qwen3tts_openai` (default) | groxaxo OpenAI-FastAPI | 8880 | Lowest latency (~97ms TTFA), torch.compile + CUDA graphs |
| `qwen3tts` | vllm-omni | 8091 | Integrated with vLLM, ~205ms TTFA |
| `qwen3tts_custom` | Custom WebSocket server | 8765 | Voice cloning with auto-transcription |
| `cosyvoice3` | vllm-omni | 8091 | CosyVoice3, multilingual, higher quality |

Optionally override the model with `TTS_MODEL=<hf_model_id>`.

**How it works:** `start.sh` (container entrypoint) reads `TTS_BACKEND`, assembles `/etc/supervisord.conf` from `supervisord_base.conf` + a dynamic TTS program block, then execs supervisord. No rebuild needed to switch backends.

---

## File Locations

| File | Purpose |
|---|---|
| `Dockerfile` | Container build - installs vllm-omni 0.18.0, groxaxo server, copies all configs |
| `start.sh` | Entrypoint - selects TTS backend, assembles supervisord.conf, starts supervisord |
| `supervisord_base.conf` | LLM-only supervisord config (TTS section added dynamically by start.sh) |
| `qwen3_tts_optimized.yaml` | Qwen3-TTS stage config for vllm-omni (memory-tuned for H200 + LLM cohabitation) |
| `cosyvoice3_optimized.yaml` | CosyVoice3 stage config for vllm-omni (memory-tuned for H200 + LLM cohabitation) |
| `qwen3_tts_groxaxo.yaml` | Config for groxaxo server (copied to /root/qwen3-tts/config.yaml in container) |
| `voice_register_router.py` | FastAPI router adding /v1/audio/voice-register endpoint |
| `run_groxaxo.py` | Wrapper that runs groxaxo server with voice register router mounted |
| `/xfiles/plugins_ah/mr_qwen3tts/` | MindRoot TTS plugin |

**Note:** `supervisord.conf` is no longer a static file in this directory - it is generated at container start by `start.sh`.

---

## Plugin Backend Selection

The plugin supports three backends via `MR_QWEN3TTS_BACKEND`:

| Backend | Default | Connects to |
|---|---|---|
| `openai` (default) | groxaxo OpenAI-FastAPI | `http://localhost:8880/v1/audio/speech` |
| `websocket` | Custom WebSocket server | `ws://localhost:8765` |
| `vllm` | vllm-omni HTTP API | `http://localhost:8091/v1/audio/speech` |

Plugin structure:
```
src/mr_qwen3tts/
  mod.py              # Backend switcher (reads MR_QWEN3TTS_BACKEND)
  audio_pacer.py      # Shared AudioPacer for SIP pacing
  openai_impl/        # groxaxo OpenAI-FastAPI (default)
    mod.py            # stream_tts, speak, on_interrupt
  vllm_impl/          # vllm-omni HTTP API
    mod.py, audio_pacer.py
  websocket_impl/     # Custom WebSocket server
    mod.py, realtime_stream.py, audio_pacer.py, README.md
```

---

## Voice Auto-Registration

When the plugin receives a `voice_id` that is a URL (e.g. `https://example.com/alice_voice.wav`),
it automatically:

1. Calls `POST /v1/audio/voice-register` on the groxaxo server with the URL
2. Server downloads the audio, validates it, creates a voice library profile
3. Plugin caches the URL -> `clone:Name` mapping locally
4. Subsequent requests with the same URL skip registration and use `clone:Name` directly

The server also caches the decoded audio array and speaker embeddings in-process,
so repeat requests are fast (no re-processing of the reference audio).

Agent persona config example:
```json
{"voice_id": "https://example.com/agent_voices/alice.wav"}
```

First utterance has ~1-2s overhead for registration. All subsequent utterances use cached profile.

---

## GPU Memory Layout (H200, 143.7 GiB total)

### qwen3tts_openai backend (groxaxo)

| Component | ~GiB |
|---|---|
| LLM (Qwen3.5-35B-A3B-AWQ) | ~122 GiB |
| Qwen3-TTS 0.6B (bfloat16) | ~4 GiB (1.2 weights + 2-3 KV/CUDA graphs) |
| Qwen3-TTS 1.7B (bfloat16) | ~8 GiB (3.4 weights + 4-5 KV/CUDA graphs) |
| Free buffer | ~5-21 GiB |

The groxaxo server manages its own GPU memory via PyTorch/transformers (not vllm), so no `gpu_memory_utilization` setting needed.

### qwen3tts / cosyvoice3 backends (vllm-omni)

| Process | gpu_memory_utilization | ~GiB |
|---|---|---|
| LLM (Qwen3.5-35B-A3B-AWQ) | 0.85 | ~122 GiB |
| TTS Stage 0 (Talker AR) | 0.08 (qwen3tts) / 0.12 (cosyvoice3) | ~11-17 GiB |
| TTS Stage 1 (Code2Wav/Flow) | 0.04 | ~5.6 GiB |
| Free buffer | ~5 GiB (qwen3tts) / ~-1 GiB (cosyvoice3, tight!) |

---

## Key Technical Notes

### groxaxo OpenAI-FastAPI (qwen3tts_openai backend)
- Optimized backend: `TTS_BACKEND=optimized` env var in supervisord command
- torch.compile + CUDA graphs for production throughput
- First 2-3 requests slow (~10-30s) during warmup. Set `TTS_WARMUP_ON_START=true` to warm at container start.
- Config at `/root/qwen3-tts/config.yaml` - defines available models (0.6B-Base, 0.6B-CustomVoice, 1.7B-Base, 1.7B-CustomVoice)
- Voice Library: save voice profiles on server, use `voice="clone:Name"` - cached embeddings save ~0.7s per request
- Voice cloning: pass `ref_audio` URL for Base model type
- Auto-registration: plugin detects URL voice_ids, calls `/v1/audio/voice-register` on first use
- `run_groxaxo.py` wrapper mounts `voice_register_router.py` onto the groxaxo server without modifying upstream code
- PCM streaming: `response_format=pcm` + `stream=true` -> raw 16-bit signed PCM at 24kHz
- Plugin resamples 24kHz PCM -> 8kHz ulaw on the fly
- `TTS_MAX_CONCURRENT=1` default - may need to increase for multi-agent scenarios

### Qwen3-TTS (qwen3tts backend, vllm-omni)
- `enforce_eager: false` on both stages (CUDA graphs active)
- `initial_codec_chunk_frames=2` in plugin payload = ~10x TTFA reduction (205ms avg)
- `async_chunk: true` for streaming
- Voice cloning via `task_type=Base`, `ref_audio`, `ref_text`

### CosyVoice3 (cosyvoice3 backend, vllm-omni)
- `enforce_eager: true` on Stage 1 - **required**, not a bug. Dynamic conv shapes in flow matching are incompatible with CUDA graphs.
- `dtype: float32` required on both stages
- `initial_codec_chunk_frames` does NOT apply - CosyVoice3 uses flow matching, not codec chunking.
- Better quality than Qwen3-TTS-0.6B for voice cloning; multilingual (9 languages, 18+ Chinese dialects)

---

## Environment Variables

### Container (RunPod pod config)

| Var | Default | Notes |
|---|---|---|
| `TTS_BACKEND` | `qwen3tts_openai` | Select TTS backend at container start |
| `TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Override model for selected backend |

### Plugin (mr_qwen3tts)

| Var | Default | Notes |
|---|---|---|
| `MR_QWEN3TTS_BACKEND` | `openai` | Plugin backend: openai, websocket, vllm |
| `MR_QWEN3TTS_OPENAI_URL` | `http://localhost:8880` | groxaxo server URL |
| `MR_QWEN3TTS_VOICE` | `Vivian` | Default voice name (or `clone:Name` for Voice Library) |
| `MR_QWEN3TTS_MODEL` | `qwen3-tts` | Model name for API requests |
| `MR_QWEN3TTS_WS_URL` | `ws://localhost:8765` | WebSocket server URL (websocket backend) |
| `MR_QWEN3TTS_REF_AUDIO` | `` | Reference audio for voice cloning |
| `MR_QWEN3TTS_REF_TEXT` | `` | Transcript of ref audio |
| `MR_QWEN3TTS_LANGUAGE` | `Auto` | Language setting |

---

## Latency Comparison

| Backend | TTFA | RTF | Notes |
|--------|------|-----|-------|
| qwen3tts_openai (groxaxo) | ~97ms | 0.65-0.70 | Optimized backend, torch.compile + CUDA graphs |
| qwen3tts (vllm-omni) | ~205ms | ~0.83 | initial_codec_chunk_frames=2 trick |
| qwen3tts_custom (WebSocket) | ~100ms+ | varies | Voice cache helps on repeat requests |
| cosyvoice3 (vllm-omni) | higher | varies | Flow matching, no codec chunking trick |

---

## Next Steps / TODO

1. **Deploy and test on H200** - build container with `TTS_BACKEND=qwen3tts_openai`, verify TTFA and RTF.
2. **Benchmark** - measure actual TTFA/RTF vs vllm-omni backend.
3. **Voice Library setup** - register agent persona voices as profiles on groxaxo server at container start.
4. **Warmup strategy** - test `TTS_WARMUP_ON_START=true` or add warmup to start.sh.
5. **Concurrency** - test `TTS_MAX_CONCURRENT` > 1 for multi-agent scenarios.
6. **FlashAttention2** - monitor for issues on 50xx GPUs (H200 should be fine).
7. **Monitor** `tts.err` log for OOM or errors during LLM + TTS cohabitation.
