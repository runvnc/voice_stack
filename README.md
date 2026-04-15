# Multi-Model vLLM H200 Deployment CLI

This repository provides a flexible, production-ready **vLLM** deployment system designed for **NVIDIA H200** GPUs on **RunPod**. It is optimized for **AWQ 4-bit** models (like Qwen 3.5, Llama 3.1, and QwQ) requiring low latency and high context windows.

## 🚀 Features
- **Fully Parameterized**: Switch models, Docker images, and context lengths via CLI flags.
- **H200 Optimized**: Automatic configuration for `FlashInfer`, `FP8 KV Cache`, and `Marlin` kernels.
- **Smart Templates**: Creates unique, slugified RunPod templates for every Model/Image combination.
- **Private Registry Support**: Integrated support for private Docker images via `RUNPOD_REGISTRY_AUTH_ID`.
- **Lifecycle Management**: Simple commands to launch or terminate pods to control costs.

## 📋 Prerequisites

1. **RunPod API Key**: Found in your RunPod Settings.
2. **Hugging Face Token**: Required for accessing gated or large model repositories.
3. **Docker Image**: Use the provided Dockerfile. Ensure it is pushed to a registry.
4. **Python Dependencies**:
   pip install runpod

## ⚙️ Environment Setup

Export these variables in your terminal:

# Required
export RUNPOD_API_KEY="your_runpod_api_key"
export DOCKER_USER="your_docker_username"
export HF_TOKEN="your_hf_read_token"

# Optional (Required for private Docker images)
export RUNPOD_REGISTRY_AUTH_ID="cr-xxxxxxxx"

## 🛠️ Usage

### 1. Launch Default (Qwen 3.5 35B-A3B on H200)
python deploy_vllm.py

### 2. Launch a Different Model (e.g., Llama 3.1 70B)
python deploy_vllm.py --model "casperhansen/llama-3.1-70b-instruct-awq" --len 65536

### 3. Use a Custom Docker Image
python deploy_vllm.py --image "youruser/custom-vllm:v1.0" --model "Qwen/Qwen2.5-72B-Instruct-AWQ"

### 4. Cleanup & Termination
# Terminate all pods started by this script
python deploy_vllm.py --terminate

# Terminate a specific Pod ID
python deploy_vllm.py --terminate [POD_ID]

## 🔧 CLI Arguments Reference


| Flag | Default | Description |
| :--- | :--- | :--- |
| --model | cyankiwi/Qwen3.5... | HuggingFace AWQ Model Repository |
| --image | $DOCKER_USER/... | Docker image to deploy |
| --gpu | NVIDIA H200 | RunPod GPU Type ID |
| --count | 1 | Number of GPUs to request |
| --len | 32768 | Max Model Context Length |
| --terminate | N/A | Stop and remove matching pods |

## 🔍 Architecture Details
- **Persistence**: Models are cached in `/workspace/huggingface` using a 120GB network volume.
- **Efficiency**: Uses `--kv-cache-dtype fp8` to maximize the H200's 141GB VRAM for high concurrency.
- **Stability**: Uses `--enforce-eager` to ensure compatibility with Mixture-of-Experts (MoE) architectures like Qwen 3.5.

## 🔗 Connection Example

Once live, the script provides a proxy URL. Use it as the `base_url` in any OpenAI-compatible client:

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("RUNPOD_API_KEY"),
    base_url="https://[YOUR_POD_ID]-8000.proxy.runpod.net/v1"
)

