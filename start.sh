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
#   TTS_BACKEND=cosyvoice3            -> runs CosyVoice3
#   TTS_BACKEND=kyutai                -> runs Kyutai TTS 1.6B (moshi) incremental streaming server (port 8765)
#   TTS_BACKEND=kyutai_batched        -> runs Kyutai TTS 1.6B batched server (WS port 8765, TCP port 8766)
# =============================================================================
set -e

mkdir -p /workspace/logs

# Make CUDA/cuDNN shared libraries from pip-installed NVIDIA wheels visible to
# subprocesses launched by supervisord. This matters especially for the isolated
# Kyutai venv: torch may depend on libcudnn.so.9 from nvidia-cudnn-cu12, but the
# dynamic linker does not always search Python site-packages/nvidia/*/lib at
# runtime. Without this, Kyutai can fail with:
#   ImportError: libcudnn.so.9: cannot open shared object file
collect_python_nvidia_libs() {
	local py="$1"
	if [ ! -x "$py" ]; then
		return 0
	fi
	"$py" - <<'PY' 2>/dev/null || true
import glob
import site

paths = []
for base in site.getsitepackages():
    for path in glob.glob(base + "/nvidia/*/lib"):
        if path not in paths:
            paths.append(path)
print(":".join(paths))
PY
}

BASE_NVIDIA_LD_LIBRARY_PATH="$(collect_python_nvidia_libs python3)"
KYUTAI_NVIDIA_LD_LIBRARY_PATH="$(collect_python_nvidia_libs /opt/kyutai-venv/bin/python)"

# Export only the base image's NVIDIA wheel libs globally. The Kyutai venv can
# have a different torch/CUDA wheel stack, so its lib dirs are injected only into
# Kyutai supervisord programs below to avoid perturbing vLLM's dynamic linker.
if [ -n "${BASE_NVIDIA_LD_LIBRARY_PATH}" ]; then
	export LD_LIBRARY_PATH="${BASE_NVIDIA_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "[start.sh] base NVIDIA LD_LIBRARY_PATH: ${BASE_NVIDIA_LD_LIBRARY_PATH:-}"
echo "[start.sh] kyutai NVIDIA LD_LIBRARY_PATH: ${KYUTAI_NVIDIA_LD_LIBRARY_PATH:-}"
echo "[start.sh] process LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"

# vLLM sizing defaults. Deploy tooling may override these for smaller GPUs such
# as H100 80GB so Kyutai + STT have enough VRAM to start alongside vLLM.
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-32768}
export VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-16192}
export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-64}
echo "[start.sh] vLLM sizing: gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION}, max_model_len=${VLLM_MAX_MODEL_LEN}, max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS}, max_num_seqs=${VLLM_MAX_NUM_SEQS}"

# LLM model selection - override via RunPod env var LLM_MODEL
export LLM_MODEL=${LLM_MODEL:-Intel/Qwen3.6-27B-int4-AutoRound}
echo "[start.sh] LLM model: ${LLM_MODEL}"

TTS_BACKEND=${TTS_BACKEND:-kyutai_batched}

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
elif [ "$TTS_BACKEND" = "kyutai" ]; then
	TTS_MODEL=${TTS_MODEL:-kyutai/tts-1.6b-en_fr}
	TTS_PROGRAM_NAME=kyutai-tts
	echo "[start.sh] TTS backend: Kyutai TTS 1.6B (moshi) incremental streaming on port 8765"
elif [ "$TTS_BACKEND" = "kyutai_batched" ]; then
	TTS_MODEL=${TTS_MODEL:-kyutai/tts-1.6b-en_fr}
	TTS_PROGRAM_NAME=kyutai-tts-batched
	echo "[start.sh] TTS backend: Kyutai TTS 1.6B BATCHED (WS:8765, TCP:8766, batch_size=${MR_KYUTAI_BATCH_SIZE:-8})"
else
	echo "[start.sh] ERROR: Unknown TTS_BACKEND='${TTS_BACKEND}'. Must be 'qwen3tts_openai', 'qwen3tts', 'cosyvoice3', 'qwen3tts_custom', 'kyutai', or 'kyutai_batched'."
	exit 1
fi

# Override KYUTAI_REMOTE for batched backend (WebSocket instead of TCP)
if [ "$TTS_BACKEND" = "kyutai_batched" ]; then
	export KYUTAI_REMOTE=ws://localhost:8765
	echo "[start.sh] KYUTAI_REMOTE overridden to ws://localhost:8765 for batched backend"
fi

# Assemble supervisord.conf = base (LLM section) + dynamic TTS section
cp /etc/supervisord_base.conf /etc/supervisord.conf

if [ "$TTS_BACKEND" = "qwen3tts" ] || [ "$TTS_BACKEND" = "cosyvoice3" ]; then
	cat >>/etc/supervisord.conf <<EOF

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
	cat >>/etc/supervisord.conf <<EOF

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
	cat >>/etc/supervisord.conf <<EOF

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

elif [ "$TTS_BACKEND" = "kyutai" ]; then
	cat >>/etc/supervisord.conf <<EOF

; =============================================================================
; TTS: kyutai-tts (Kyutai TTS 1.6B moshi) - port 8765
; Backend: mr_kyutai.remote_server (framed TCP protocol)
; Model: ${TTS_MODEL}
; Protocol: TCP framed (1 byte type + 4-byte length + payload)
;   Client sends JSON frames: {op:start, voice:...}, {op:text, text:...}, {op:finish}
;   Server returns: 'A' frames (ulaw 8kHz audio), 'E' (end), 'X' (error)
; Plugin env: KYUTAI_REMOTE=tcp://localhost:8765, MR_KYUTAI_REALTIME_STREAM=1
; =============================================================================
[program:kyutai-tts]
command=/opt/kyutai-venv/bin/python -m mr_kyutai.remote_server --host 0.0.0.0 --port 8765
autostart=true
autorestart=true
startretries=3
stopwaitsecs=30
stdout_logfile=/workspace/logs/kyutai_tts.log
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=5
stderr_logfile=/workspace/logs/kyutai_tts.err
stderr_logfile_maxbytes=100MB
stderr_logfile_backups=5
environment=PYTHONPATH="/app/local/plugins/mr_kyutai/mr_kyutai/src",LD_LIBRARY_PATH="${KYUTAI_NVIDIA_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
EOF
fi

if [ "$TTS_BACKEND" = "kyutai_batched" ]; then
	cat >>/etc/supervisord.conf <<EOF

; =============================================================================
; TTS: kyutai-tts-batched (Kyutai TTS 1.6B moshi) - WS port 8765, TCP port 8766
; Backend: mr_kyutai.batched_tts_server wrapping TTSService for concurrent sessions
; Model: ${TTS_MODEL}
; Protocol: WebSocket (msgpack, matches moshi-server) + legacy TCP (framed J/A/E/X)
;   WS client sends: {type:Text, text:word}, {type:Eos}
;   WS server returns: {type:Audio, pcm:[f32...]}, {type:Ready}
;   TCP client sends JSON frames: {op:start}, {op:text}, {op:finish}
;   TCP server returns: 'A' ulaw8k, 'E' end, 'X' error
; Plugin env: KYUTAI_REMOTE=ws://localhost:8765, MR_KYUTAI_REALTIME_STREAM=1
; =============================================================================
[program:kyutai-tts-batched]
command=/opt/kyutai-venv/bin/python -m mr_kyutai.batched_tts_server --ws-port 8765 --tcp-port 8766 --batch-size ${MR_KYUTAI_BATCH_SIZE:-8}
autostart=true
autorestart=true
startretries=3
stopwaitsecs=30
stdout_logfile=/workspace/logs/kyutai_tts.log
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=5
stderr_logfile=/workspace/logs/kyutai_tts.err
stderr_logfile_maxbytes=100MB
stderr_logfile_backups=5
environment=PYTHONPATH="/app/local/plugins/mr_kyutai/mr_kyutai/src",MR_KYUTAI_HF_REPO="${TTS_MODEL}",LD_LIBRARY_PATH="${KYUTAI_NVIDIA_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
EOF
fi

echo "[start.sh] supervisord.conf assembled. Starting supervisord..."

# =============================================================================
# Cohere Transcribe server (always started, port 8881)
# Used by mr_sip STT_PROVIDER=silero_cohere for remote GPU transcription.
# Silero VAD runs on the client (Hetzner VPS), audio POSTed here after VAD.
# STT_SERVER=nano (default) uses nano-cohere-transcribe (1.5-3.6x faster, batched)
# STT_SERVER=legacy uses original cohere_transcribe_server.py (transformers)
# =============================================================================
STT_SERVER=${STT_SERVER:-nano}
if [ "$STT_SERVER" = "nano" ]; then
	STT_COMMAND="/opt/cohere-venv/bin/python3 /app/nano_cohere_transcribe_server.py --host 0.0.0.0 --port 8881"
	echo "[start.sh] STT server: nano-cohere-transcribe (batched, port 8881)"
else
	STT_COMMAND="/opt/cohere-venv/bin/python3 /app/cohere_transcribe_server.py --host 0.0.0.0 --port 8881"
	echo "[start.sh] STT server: legacy cohere_transcribe_server (transformers, port 8881)"
fi
cat >>/etc/supervisord.conf <<EOF

[program:cohere-transcribe]
command=${STT_COMMAND}
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
cat >>/etc/supervisord.conf <<'EOF'

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

if [ "${STT_PROVIDER:-smart_turn_v3}" = "smart_turn_v3" ] && \
   [ "${SMART_TURN_DEVICE:-cuda}" = "cuda" ] && \
   [ "${SMART_TURN_CUDA_PREFLIGHT:-1}" = "1" ]; then
	echo "[start.sh] Running Smart Turn CUDA preflight..."
	if ! /app/.venv/bin/python /app/check_smart_turn_cuda.py; then
		echo "[start.sh] ERROR: Smart Turn CUDA preflight failed."
		echo "[start.sh] Refusing to start with silent CPU fallback because SMART_TURN_DEVICE=cuda."
		echo "[start.sh] Set SMART_TURN_CUDA_PREFLIGHT=0 only for debugging, not production."
		exit 1
	fi
	echo "[start.sh] Smart Turn CUDA preflight passed."
fi

exec supervisord -c /etc/supervisord.conf
