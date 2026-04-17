# =============================================================================
# Stage 1: Base vLLM image
# vllm/vllm-openai:v0.18.0 supports Qwen3.5-35B-A3B (Gated DeltaNet MoE)
# and is the latest release as of 2026-03-28.
# =============================================================================
FROM vllm/vllm-openai:v0.18.0 AS base

# Clear the base image's ENTRYPOINT so start.sh can be PID 1 via supervisord
ENTRYPOINT []

# RunPod persistent storage paths
ENV HF_HOME=/workspace/huggingface
ENV VLLM_HOME=/workspace/vllm

# Optimization for NVIDIA H200 (Hopper) - FlashInfer is fastest on Hopper
ENV VLLM_ATTENTION_BACKEND=FLASHINFER

# Binding to 0.0.0.0 is required for RunPod proxy to reach the container
ENV VLLM_HOST=0.0.0.0
ENV VLLM_PORT=8000

# H200 / Hopper optimizations
# Enable TF32 for faster matrix math on Hopper (no precision loss for inference)
ENV NVIDIA_TF32_OVERRIDE=1
# PyTorch TF32
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
# CUDA memory allocator optimization
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Enable torch.compile caching
ENV TORCHINDUCTOR_CACHE_DIR=/workspace/torchinductor_cache

# =============================================================================
# Stage 2: Install vllm-omni and system dependencies
#
# vllm-omni 0.18.0 (PyPI release, 2026-03-28) supports:
#   - Qwen3-TTS (all variants)
#   - CosyVoice3 / Fun-CosyVoice3-0.5B-2512 (added in PR #498)
# Both TTS backends are available in the same package.
# =============================================================================
FROM base AS with-omni

# Configure apt to retry on network failures (Ubuntu mirrors can be flaky)
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/99retries && \
    echo 'Acquire::http::Timeout "120";' >> /etc/apt/apt.conf.d/99retries && \
    echo 'Acquire::https::Timeout "120";' >> /etc/apt/apt.conf.d/99retries && \
    sed -i 's|http://archive.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list

# ffmpeg required by librosa (used internally by vllm-omni for audio)
# git required for any pip installs from source if needed
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg sox libsox-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install vllm-omni from PyPI (v0.18.0 is the latest stable release)
# Both Qwen3-TTS and CosyVoice3 are supported in this release.
RUN pip install --no-cache-dir vllm-omni==0.18.0

# Install supervisord for process management
RUN pip install --no-cache-dir supervisor

# =============================================================================
# Stage 3a: Install custom Qwen3-TTS WebSocket server
# Cloned from https://github.com/runvnc/qwen3tts
# Runs on port 8765 (WebSocket), selected via TTS_BACKEND=qwen3tts_custom
# =============================================================================
RUN pip install --no-cache-dir \
    websockets>=12.0 \
    soundfile \
    librosa \
    openai-whisper \
    "qwen-tts @ git+https://github.com/dffdeeq/Qwen3-TTS-streaming.git"

# Install flash-attn for the custom TTS server (vllm image has CUDA/torch already)
RUN git clone https://github.com/runvnc/qwen3tts.git /app/qwen3tts_server

# =============================================================================
# Stage 3b: Install groxaxo Qwen3-TTS OpenAI-FastAPI server
# https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi
# Runs on port 8880 (HTTP), selected via TTS_BACKEND=qwen3tts_openai
# =============================================================================
RUN pip install --no-cache-dir \
    "qwen-tts[api] @ git+https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi.git"
# scipy for Whisper resampling in voice registration, openai-whisper for auto-transcription
RUN pip install --no-cache-dir scipy openai-whisper

# =============================================================================
# Install flash-attn from pre-built wheel (no compilation - avoids ~90min build)
# Detects PyTorch version at build time and downloads the matching wheel.
# Official wheels only go up to torch 2.9. For torch 2.10, use community wheel
# from lesj0610 (reproducible build, confirmed working on cu12.9).
# FA3 also installed for future model support.
# =============================================================================
RUN pip install --no-cache-dir \
    "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
    || echo '[flash-attn] WARNING: FA2 install failed, will use SDPA fallback'
RUN pip install --no-cache-dir flash_attn_3 \
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu129_torch2100 \
    || echo '[flash-attn3] WARNING: FA3 install failed'

# =============================================================================
# Stage 3c: Install local VAD + ASR dependencies
# Used by mr_sip STT_PROVIDER=silero_cohere (replaces Deepgram Flux)
#
# silero-vad: Silero VAD v6 (MIT license, ~260K params, 8kHz native)
# transformers: for Cohere Transcribe (CohereLabs/cohere-transcribe-03-2026)
# accelerate: required by device_map='auto' in from_pretrained
# =============================================================================
RUN pip install --no-cache-dir \
    silero-vad \
    accelerate
# Pin transformers to 4.57.3 - required by qwen-tts (groxaxo TTS backend)
RUN pip install --no-cache-dir "transformers==4.57.3"

# =============================================================================
# Stage 3d: Separate venv for Cohere Transcribe with native transformers>=5.4.0
# Uses --system-site-packages to inherit torch/cuda/scipy from base image.
# Only transformers (and its direct deps) are overridden in this venv.
# =============================================================================
RUN python3 -m venv /opt/cohere-venv --system-site-packages && \
    /opt/cohere-venv/bin/pip install --no-cache-dir "transformers>=5.4.0" sentencepiece protobuf

# =============================================================================
# Stage 4: Final image - copy configs and set up runtime
# =============================================================================
FROM with-omni AS final

COPY cohere_transcribe_server.py /app/cohere_transcribe_server.py

# Copy supervisord base config (LLM section only; TTS section added at runtime)
COPY supervisord_base.conf /etc/supervisord_base.conf

# Copy TTS stage configs for both backends
# Selected at container start via TTS_BACKEND env var in start.sh
COPY qwen3_tts_optimized.yaml /etc/qwen3_tts_optimized.yaml
COPY cosyvoice3_optimized.yaml /etc/cosyvoice3_optimized.yaml

# Copy groxaxo config for optimized backend
COPY qwen3_tts_groxaxo.yaml /root/qwen3-tts/config.yaml

# Copy voice registration router and wrapper script
# These mount an extra endpoint onto the groxaxo server without modifying upstream code
COPY voice_register_router.py /app/voice_register_router.py
COPY run_groxaxo.py /app/run_groxaxo.py

# Copy entrypoint script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Expose LLM API port and TTS API port
EXPOSE 8000
EXPOSE 8091
EXPOSE 8765
EXPOSE 8880
EXPOSE 8881

# start.sh reads TTS_BACKEND env var, assembles supervisord.conf, starts supervisord.
# Set TTS_BACKEND=qwen3tts_openai (default) for groxaxo server with lowest latency.
# Set TTS_BACKEND=cosyvoice3 in RunPod pod env to use CosyVoice3.
# Set TTS_BACKEND=qwen3tts for vllm-omni backend.
CMD ["/start.sh"]

# =============================================================================
# Stage 5: Mindroot voice agent platform
# Installed via pip; plugins pre-installed at build time from GitHub.
# Runs on port 8010. All backend services on localhost (no network round trip).
#
# Plugins installed:
#   runvnc/mr_sip       - SIP/voice call handling (v2 + silero_cohere STT)
#   runvnc/ah_openrouter - OpenRouter LLM backend
#   runvnc/mr_qwen3tts  - Qwen3-TTS plugin (openai backend -> localhost:8880)
# =============================================================================
FROM final AS with-mindroot

# Create Mindroot runtime directory structure
RUN mkdir -p /app/imgs /app/data/chat /app/models /app/static/personas \
    /app/personas/local /app/personas/shared /app/data/sessions

WORKDIR /app

# Create isolated venv and install Mindroot from PyPI
RUN python3 -m venv /app/.venv && \
    /app/.venv/bin/pip install --no-cache-dir mindroot

# Pre-install plugins at build time
RUN /app/.venv/bin/mindroot plugin install \
    runvnc/mr_sip \
    runvnc/ah_openrouter \
    runvnc/mr_qwen3tts \
    runvnc/mr_any_llm

# Mindroot env vars - all backend services on localhost (eliminates network round trip)
ENV ANY_LLM_SERVER_URL=http://localhost:8000/v1
ENV MR_QWEN3TTS_BACKEND=openai
ENV MR_QWEN3TTS_OPENAI_URL=http://localhost:8880
ENV STT_PROVIDER=silero_cohere
ENV COHERE_TRANSCRIBE_URL=http://localhost:8881
ENV SILERO_MIN_SILENCE_MS=400
ENV ANY_LLM_EXTRA_PARAMS='{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}'
# Credentials - override at runtime via RunPod env vars
ENV JWT_SECRET_KEY=change_me_at_runtime
ENV ADMIN_USER=admin
ENV ADMIN_PASS=change_me_at_runtime

EXPOSE 8010
