#!/bin/bash
# =============================================================================
# Container entrypoint - assembles supervisord.conf based on TTS_BACKEND
# env var and starts supervisord as PID 1.
#
# Environment variables:
#   TTS_BACKEND   qwen3tts_openai (default) | qwen3tts | cosyvoice3 | qwen3tts_custom
#   TTS_MODEL     Override the default model for the selected backend.
#                 Default qwen3tts_openai: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
#                 Default qwen3tts:        Qwen/Qwen3-TTS-12Hz-0.6B-Base
#                 Default cosyvoice3: FunAudioLLM/Fun-CosyVoice3-0.5B-2512
#   LLM_MODEL     Override the default LLM model.
#                 Default: Intel/Qwen3.6-27B-int4-AutoRound
#
# Usage (RunPod pod environment variables):
#   TTS_BACKEND=qwen3tts_openai      -> runs groxaxo OpenAI-FastAPI server (default, port 8880, lowest latency)
#   TTS_BACKEND=cosyvoice3            -> runs CosyVoice3
#   TTS_BACKEND=qwen3tts              -> runs Qwen3-TTS via vllm-omni
#   TTS_BACKEND=qwen3tts_custom       -> runs custom Qwen3-TTS WebSocket server (port 8765)
# =============================================================================
set -e

mkdir -p /workspace/logs

# LLM model selection - override via RunPod env var LLM_MODEL
export LLM_MODEL=${LLM_MODEL:-Intel/Qwen3.6-27B-int4-AutoRound}
echo "[start.sh] LLM model: ${LLM_MODEL}"

TTS_BACKEND=${TTS_BACKEND:-qwen3tts_openai}

if [ "$TTS_BACKEND" = "cosyvoice3" ]; then
    TTS_MODEL=${TTS_MODEL:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}
    TTS_STAGE_CONFIG=/etc/cosyvoice3_optimized.yaml
    TTS_PROGRAM_NAME=cosyvoice3-tts
    echo "[start.sh] TTS backend: CosyVoice3 (${TTS_MODEL})"
elif [ "$TTS_BACKEND" = "qwen3tts_openai" ]; then
    TTS_MODEL=${TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}
    TTS_PROGRAM_NAME=qwen3tts-openai
    echo "[start.sh] TTS backend: groxaxo OpenAI-FastAPI (${TTS_MODEL}) on port 8880"
elif [ "$TTS_BACKEND" = "qwen3tts" ]; then
    TTS_MODEL=${TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}
    TTS_STAGE_CONFIG=/etc/qwen3_tts_optimized.yaml
    TTS_PROGRAM_NAME=qwen3-tts
    echo "[start.sh] TTS backend: Qwen3-TTS via vllm-omni (${TTS_MODEL})"
elif [ "$TTS_BACKEND" = "qwen3tts_custom" ]; then
    TTS_MODEL=${TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}
    TTS_PROGRAM_NAME=qwen3tts-custom
    echo "[start.sh] TTS backend: Custom Qwen3-TTS WebSocket server (${TTS_MODEL}) on port 8765"
else
    echo "[start.sh] ERROR: Unknown TTS_BACKEND='${TTS_BACKEND}'. Must be 'qwen3tts_openai', 'qwen3tts', 'cosyvoice3', or 'qwen3tts_custom'."
    exit 1
fi

# Assemble supervisord.conf = base (LLM section) + dynamic TTS section
cp /etc/supervisord_base.conf /etc/supervisord.conf

if [ "$TTS_BACKEND" = "qwen3tts" ] || [ "$TTS_BACKEND" = "cosyvoice3" ]; then
cat >> /etc/supervisord.conf << EOF

; =============================================================================
; TTS: ${TTS_PROGRAM_NAME} (vllm-omni) - port 8091
; Backend selected at container start via TTS_BACKEND=${TTS_BACKEND}
; Model: ${TTS_MODEL}
; Stage config: ${TTS_STAGE_CONFIG}
; Streaming: POST /v1/audio/speech with stream=true, response_format=pcm
;   -> raw 16-bit signed PCM at 24kHz, chunked HTTP transfer.
; Voice cloning: pass ref_audio (URL or path), ref_text
; =============================================================================
[program:${TTS_PROGRAM_NAME}]
command=/bin/bash -c 'VLLM_LOGGING_LEVEL=INFO vllm-omni serve ${TTS_MODEL} --stage-configs-path ${TTS_STAGE_CONFIG} --omni --host 0.0.0.0 --port 8091 --trust-remote-code'
autostart=true
autorestart=true
startretries=3
stopwaitsecs=30
stdout_logfile=/workspace/logs/tts.log
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=5
stderr_logfile=/workspace/logs/tts.err
stderr_logfile_maxbytes=100MB
stderr_logfile_backups=5
environment=HF_HOME="/workspace/huggingface",PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
EOF

elif [ "$TTS_BACKEND" = "qwen3tts_custom" ]; then
cat >> /etc/supervisord.conf << EOF

; =============================================================================
; TTS: qwen3tts-custom (WebSocket server) - port 8765
; Backend: custom Qwen3-TTS WebSocket server from https://github.com/runvnc/qwen3tts
; Model: ${TTS_MODEL}
; Protocol: WebSocket ws://host:8765
;   init: {type: init, ref_audio_base64, ref_text}
;   generate: {type: generate_stream, text}
;   returns: binary ulaw 8kHz chunks + {type: audio_end}
; Plugin env: MR_QWEN3TTS_BACKEND=websocket, MR_QWEN3TTS_WS_URL=ws://host:8765
; =============================================================================
[program:qwen3tts-custom]
command=python3 /app/qwen3tts_server/server.py --model ${TTS_MODEL} --port 8765 --host 0.0.0.0
autostart=true
autorestart=true
startretries=3
stopwaitsecs=30
stdout_logfile=/workspace/logs/tts.log
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=5
stderr_logfile=/workspace/logs/tts.err
stderr_logfile_maxbytes=100MB
stderr_logfile_backups=5
environment=HF_HOME="/workspace/huggingface",QWEN3_TTS_MODEL="${TTS_MODEL}",QWEN3_TTS_DEVICE="cuda:0",PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
EOF

elif [ "$TTS_BACKEND" = "qwen3tts_openai" ]; then
cat >> /etc/supervisord.conf << EOF

; =============================================================================
; TTS: qwen3tts-openai (groxaxo OpenAI-FastAPI) - port 8880
; Backend: groxaxo/Qwen3-TTS-Openai-Fastapi with optimized backend
; Model: ${TTS_MODEL}
; Config: /root/qwen3-tts/config.yaml (copied in Dockerfile)
; API: POST /v1/audio/speech with stream=true, response_format=pcm
;   -> raw 16-bit signed PCM at 24kHz, chunked HTTP transfer.
; Voice cloning: voice="clone:Name" (Voice Library) or ref_audio URL
; Plugin env: MR_QWEN3TTS_BACKEND=openai, MR_QWEN3TTS_OPENAI_URL=http://host:8880
; Note: First 2-3 requests slow (~10-30s) during torch.compile warmup.
;       Set TTS_WARMUP_ON_START=true to warm up at container start.
; =============================================================================
[program:qwen3tts-openai]
command=/bin/bash -c 'TTS_BACKEND=optimized TTS_MODEL_NAME=${TTS_MODEL} python3 /app/run_groxaxo.py --host 0.0.0.0 --port 8880'
autostart=true
autorestart=true
startretries=3
stopwaitsecs=30
stdout_logfile=/workspace/logs/tts.log
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=5
stderr_logfile=/workspace/logs/tts.err
stderr_logfile_maxbytes=100MB
stderr_logfile_backups=5
environment=HF_HOME="/workspace/huggingface",CUDA_VISIBLE_DEVICES="0",PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",TTS_MAX_CONCURRENT="4"
EOF

fi

echo "[start.sh] supervisord.conf assembled. Starting supervisord..."

# =============================================================================
# Cohere Transcribe server (always started, port 8881)
# Used by mr_sip STT_PROVIDER=silero_cohere for remote GPU transcription.
# Silero VAD runs on the client (Hetzner VPS), audio POSTed here after VAD.
# =============================================================================
cat >> /etc/supervisord.conf << 'EOF'

[program:cohere-transcribe]
command=/opt/cohere-venv/bin/python3 /app/cohere_transcribe_server.py --host 0.0.0.0 --port 8881
autostart=true
autorestart=true
startretries=3
stopwaitsecs=30
stdout_logfile=/workspace/logs/cohere_transcribe.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=3
stderr_logfile=/workspace/logs/cohere_transcribe.err
stderr_logfile_maxbytes=50MB
stderr_logfile_backups=3
environment=HF_HOME="/workspace/huggingface",CUDA_VISIBLE_DEVICES="0"
EOF

# =============================================================================
# Mindroot voice agent platform (always started, port 8010)
# All backend services on localhost - no network round trip for STT/TTS/LLM.
# Credentials (JWT_SECRET_KEY, ADMIN_USER, ADMIN_PASS) set via RunPod env vars.
# =============================================================================
cat >> /etc/supervisord.conf << 'EOF'

[program:mindroot]
command=/app/.venv/bin/python -m mindroot.server --port 8010
directory=/app
autostart=true
autorestart=true
startretries=3
stopwaitsecs=30
stdout_logfile=/workspace/logs/mindroot.log
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=5
stderr_logfile=/workspace/logs/mindroot.err
stderr_logfile_maxbytes=100MB
stderr_logfile_backups=5
EOF

exec supervisord -c /etc/supervisord.conf
