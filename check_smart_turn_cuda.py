#!/usr/bin/env python3
"""
Runtime preflight for Smart Turn v3 ONNX Runtime CUDA.

Purpose:
  - Fail fast if CUDAExecutionProvider is not actually active.
  - Avoid silent CPU fallback in production.
  - Warm up the Smart Turn ONNX session and print rough latency.

This script intentionally tests the ONNX model input directly:
  input_features: [1, 80, 800] float32
so it validates ONNX Runtime CUDA independent of audio preprocessing.
"""
import os
import statistics
import time

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download


def main():
    print("[smart-turn-cuda-check] ORT version:", ort.__version__, flush=True)
    print("[smart-turn-cuda-check] available providers:", ort.get_available_providers(), flush=True)
    print("[smart-turn-cuda-check] ORT device:", ort.get_device(), flush=True)
    print("[smart-turn-cuda-check] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)

    try:
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
            print("[smart-turn-cuda-check] ort.preload_dlls() OK", flush=True)
        else:
            print("[smart-turn-cuda-check] ort.preload_dlls() not available in this ORT version", flush=True)
    except Exception as e:
        print("[smart-turn-cuda-check] WARNING: ort.preload_dlls() failed:", repr(e), flush=True)

    try:
        import torch
        print("[smart-turn-cuda-check] torch:", torch.__version__, flush=True)
        print("[smart-turn-cuda-check] torch.version.cuda:", getattr(torch.version, "cuda", None), flush=True)
        print("[smart-turn-cuda-check] torch.cuda.is_available:", torch.cuda.is_available(), flush=True)
        if torch.cuda.is_available():
            print("[smart-turn-cuda-check] torch device 0:", torch.cuda.get_device_name(0), flush=True)
    except Exception as e:
        print("[smart-turn-cuda-check] WARNING: torch import/check failed:", repr(e), flush=True)

    model_path = os.environ.get("SMART_TURN_MODEL_PATH")
    if not model_path:
        filename = os.environ.get("SMART_TURN_MODEL_FILENAME", "smart-turn-v3.2-gpu.onnx")
        print("[smart-turn-cuda-check] downloading/loading HF model:", filename, flush=True)
        model_path = hf_hub_download("pipecat-ai/smart-turn-v3", filename=filename)
    print("[smart-turn-cuda-check] model_path:", model_path, flush=True)

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise SystemExit("[smart-turn-cuda-check] ERROR: CUDAExecutionProvider not in available providers")

    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = [
        ("CUDAExecutionProvider", {
            "device_id": 0,
            "cudnn_conv_algo_search": "DEFAULT",
            "do_copy_in_default_stream": "1",
            "use_tf32": "1",
        }),
    ]

    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    actual = sess.get_providers()
    print("[smart-turn-cuda-check] session providers:", actual, flush=True)

    if "CUDAExecutionProvider" not in actual:
        raise SystemExit(f"[smart-turn-cuda-check] ERROR: CUDA requested but session providers are {actual}")

    # Smart Turn ONNX input is Whisper log-mel features: [batch, 80, 800].
    x = np.random.randn(1, 80, 800).astype(np.float32)

    print("[smart-turn-cuda-check] warmup...", flush=True)
    for _ in range(10):
        sess.run(None, {"input_features": x})

    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        sess.run(None, {"input_features": x})
        times.append((time.perf_counter() - t0) * 1000.0)

    print(
        "[smart-turn-cuda-check] latency_ms "
        f"min={min(times):.3f} "
        f"p50={statistics.median(times):.3f} "
        f"p95={statistics.quantiles(times, n=20)[18]:.3f} "
        f"max={max(times):.3f}",
        flush=True,
    )
    print("[smart-turn-cuda-check] OK: CUDA Smart Turn preflight passed", flush=True)


if __name__ == "__main__":
    main()
