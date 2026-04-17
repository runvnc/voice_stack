import os
import time
import argparse
import secrets
import sys
import re
import json
import requests
import urllib3
import runpod
from pathlib import Path

def check_env_vars():
    required = ["RUNPOD_API_KEY", "DOCKER_USER", "HF_TOKEN"]
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

def check_sip_env_vars():
    """Check SIP env vars required for deploy (passed into container)."""
    required = ["SIP_USER", "SIP_PASSWORD", "SIP_GATEWAY"]
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        print(f"Error: Missing SIP environment variables required for deploy: {', '.join(missing)}")
        print(f"  Set these before deploying: SIP_USER, SIP_PASSWORD, SIP_GATEWAY")
        sys.exit(1)

# Configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
DOCKER_USER = os.getenv("DOCKER_USER")
HF_TOKEN = os.getenv("HF_TOKEN")
REGISTRY_AUTH_ID = os.getenv("RUNPOD_REGISTRY_AUTH_ID")
SIP_USER = os.getenv("SIP_USER")
SIP_PASSWORD = os.getenv("SIP_PASSWORD")
SIP_GATEWAY = os.getenv("SIP_GATEWAY")
POD_BASE_NAME = "vLLM-Inference"
STATE_FILE = Path(__file__).parent / "pod_state.json"

if RUNPOD_API_KEY:
    runpod.api_key = RUNPOD_API_KEY

def slugify(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')

def load_pod_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_pod_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def query_templates():
    """Query templates via REST API."""
    response = requests.get(
        "https://rest.runpod.io/v1/templates",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    )
    response.raise_for_status()
    return response.json()

def delete_template(template_id):
    """Delete a template by ID via REST API."""
    response = requests.delete(
        f"https://rest.runpod.io/v1/templates/{template_id}",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    )
    if response.status_code in (200, 204):
        print(f"Deleted template: {template_id}")
    else:
        print(f"Warning: delete returned {response.status_code}: {response.text}")

def delete_templates_by_name(name=None):
    """Delete templates matching name, or all vLLM templates if no name given."""
    templates = query_templates()
    found = False
    for t in templates:
        if name:
            if t['name'] == name:
                delete_template(t['id'])
                found = True
        else:
            if t['name'].startswith('vLLM-'):
                print(f"Deleting template: {t['name']} ({t['id']})")
                delete_template(t['id'])
                found = True
    if not found:
        print("No matching templates found.")

def get_saved_pod_id():
    state = load_pod_state()
    return state.get('pod_id')

def save_pod_id(pod_id):
    state = load_pod_state()
    state['pod_id'] = pod_id
    save_pod_state(state)
    print(f"Saved pod ID: {pod_id}")

def get_or_create_template(model_repo, max_len, image_name, tts_backend="qwen3tts_openai"):
    """
    Creates a unique template per model/image combination.
    """
    model_slug = slugify(model_repo.split('/')[-1])
    image_slug = slugify(image_name.split('/')[-1])
    tts_slug = slugify(tts_backend)
    template_name = f"vLLM-{model_slug}-{image_slug}-{tts_slug}"

    templates = query_templates()
    for t in templates:
        if t['name'] == template_name:
            print(f"Using existing template: {template_name}")
            return t['id']

    print(f"Creating new template: {template_name}...")

    # NOTE: docker_start_cmd overrides the container CMD.
    # We leave it empty so supervisord (set in the Dockerfile CMD) runs instead.
    # vllm is managed by supervisord inside the container.
    template_args = {
        "name": template_name,
        "image_name": image_name,
        "container_disk_in_gb": 50,
        "volume_in_gb": 120,
        "volume_mount_path": "/workspace",
        "ports": "8000/http,8091/http,8765/tcp,8880/http,8881/http,8010/http",
        "env": {
            "HF_HOME": "/workspace/huggingface",
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
            "HF_TOKEN": HF_TOKEN,
            "TTS_BACKEND": tts_backend,
            "STT_PROVIDER": "silero_cohere",
            "SILERO_MIN_SILENCE_MS": "400",
            "ANY_LLM_SERVER_URL": "http://localhost:8000/v1",
            "SIP_USER": SIP_USER,
            "SIP_PASSWORD": SIP_PASSWORD,
            "SIP_GATEWAY": SIP_GATEWAY,
        }
    }

    if REGISTRY_AUTH_ID:
        template_args["container_registry_auth_id"] = REGISTRY_AUTH_ID

    # Auto-generate Mindroot credentials
    template_args["env"]["JWT_SECRET_KEY"] = secrets.token_hex(32)
    template_args["env"]["ADMIN_USER"] = "admin_" + secrets.token_hex(4)
    template_args["env"]["ADMIN_PASS"] = secrets.token_urlsafe(16)

    # Save credentials locally so we can display them later
    # (RunPod pod API does not return env vars)
    state = load_pod_state()
    state['mindroot_creds'] = {'ADMIN_USER': template_args["env"]["ADMIN_USER"],
                               'ADMIN_PASS': template_args["env"]["ADMIN_PASS"]}
    save_pod_state(state)

    new_template = runpod.create_template(**template_args)
    return new_template['id']

def get_ws_url_from_pod(pod):
    """Extract the WebSocket URL for port 8765 from pod runtime TCP port mappings."""
    try:
        ports = pod.get('runtime', {}).get('ports', [])
        for p in ports:
            if p.get('privatePort') == 8765 and p.get('isIpPublic'):
                ip = p.get('ip')
                pub_port = p.get('publicPort')
                if ip and pub_port:
                    return f"ws://{ip}:{pub_port}"
    except Exception:
        pass
    return None

def print_tts_info(pod_id, tts_backend, pod=None):
    """Print TTS endpoint info based on backend type."""
    if tts_backend == 'qwen3tts_openai':
        print(f"   TTS (groxaxo OpenAI-FastAPI): https://{pod_id}-8880.proxy.runpod.net/v1/audio/speech")
        print(f"   Voice Registration: https://{pod_id}-8880.proxy.runpod.net/v1/audio/voice-register")
        print(f"   Plugin env:")
        print(f"     MR_QWEN3TTS_BACKEND=openai")
        print(f"     MR_QWEN3TTS_OPENAI_URL=https://{pod_id}-8880.proxy.runpod.net")
        print(f"     ANY_LLM_SERVER_URL=https://{pod_id}-8000.proxy.runpod.net/v1")
    elif tts_backend == 'qwen3tts_custom':
        ws_url = None
        if pod:
            ws_url = get_ws_url_from_pod(pod)
        if ws_url:
            print(f"   TTS (WebSocket TCP): {ws_url}")
        else:
            print(f"   TTS (WebSocket TCP): check RunPod Connect menu -> Direct TCP Ports for port 8765")
        print(f"   Plugin env:")
        print(f"     MR_QWEN3TTS_BACKEND=websocket")
        if ws_url:
            print(f"     MR_QWEN3TTS_WS_URL={ws_url}")
        else:
            print(f"     MR_QWEN3TTS_WS_URL=ws://<PUBLIC_IP>:<TCP_PORT>  (from Connect menu)")
    elif tts_backend == 'qwen3tts':
        print(f"   TTS (vllm-omni): https://{pod_id}-8091.proxy.runpod.net/v1/audio/speech")
        print(f"   Plugin env:")
        print(f"     MR_QWEN3TTS_BACKEND=vllm")
        print(f"     MR_QWEN3TTS_API_URL=https://{pod_id}-8091.proxy.runpod.net")
    elif tts_backend == 'cosyvoice3':
        print(f"   TTS (CosyVoice3 via vllm-omni): https://{pod_id}-8091.proxy.runpod.net/v1/audio/speech")
        print(f"   Plugin env:")
        print(f"     MR_QWEN3TTS_BACKEND=vllm")
        print(f"     MR_QWEN3TTS_API_URL=https://{pod_id}-8091.proxy.runpod.net")
        print(f"     MR_QWEN3TTS_MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    else:
        print(f"   TTS backend: {tts_backend}")

def print_stt_info(pod_id):
    """Print STT endpoint info."""
    print(f"   STT (Cohere Transcribe): https://{pod_id}-8881.proxy.runpod.net/transcribe")
    print(f"   STT health: https://{pod_id}-8881.proxy.runpod.net/health")
    print(f"   Plugin env:")
    print(f"     STT_PROVIDER=silero_cohere")
    print(f"     COHERE_TRANSCRIBE_URL=https://{pod_id}-8881.proxy.runpod.net")
    print(f"     SILERO_MIN_SILENCE_MS=400")

def print_mindroot_info(pod_id, env=None):
    """Print Mindroot endpoint and credentials."""
    if env is None:
        state = load_pod_state()
        env = state.get('mindroot_creds') or {}
    print(f"   Mindroot: https://{pod_id}-8010.proxy.runpod.net")
    print(f"   Admin UI: https://{pod_id}-8010.proxy.runpod.net/admin")
    print(f"   Credentials:")
    print(f"     ADMIN_USER={env.get('ADMIN_USER', '(see pod_state.json)')}")
    print(f"     ADMIN_PASS={env.get('ADMIN_PASS', '(see pod_state.json)')}")


def get_pod_status(pod_id):
    try:
        return runpod.get_pod(pod_id)
    except:
        return None

def terminate_pods(target_id=None):
    pods = runpod.get_pods()
    found = False
    for p in pods:
        if target_id == p['id'] or (not target_id and p['name'].startswith(POD_BASE_NAME)):
            print(f"Terminating Pod: {p['name']} ({p['id']})")
            runpod.terminate_pod(p['id'])
            found = True
    if not found: print("No matching pods found.")

def stop_pod(pod_id=None):
    """Stop a pod by ID or use saved pod ID."""
    if not pod_id:
        pod_id = get_saved_pod_id()

    if not pod_id:
        print("No pod ID saved. Use --deploy first to create a pod.")
        return

    pod = get_pod_status(pod_id)
    if not pod:
        print(f"Pod {pod_id} not found or already removed.")
        return

    print(f"Stopping pod: {pod['name']} ({pod_id})")
    runpod.stop_pod(pod_id)
    print(f"Pod stop requested.")

def start_pod(pod_id=None, tts_backend=None):
    """Start a stopped pod by ID or use saved pod ID."""
    if not pod_id:
        pod_id = get_saved_pod_id()

    if not pod_id:
        print("No pod ID saved. Use --deploy first to create a pod.")
        return

    pod = get_pod_status(pod_id)
    if not pod:
        print(f"Pod {pod_id} not found.")
        return

    if pod.get('runtime') and pod.get('address'):
        print(f"Pod already running: https://{pod_id}-8000.proxy.runpod.net/v1")
        return

    # Determine TTS backend from pod env if not specified
    if not tts_backend:
        tts_backend = pod.get('env', {}).get('TTS_BACKEND', 'qwen3tts_openai')

    print(f"Starting pod: {pod['name']} ({pod_id})")
    runpod.start_pod(pod_id)

    print(f"Waiting for initialization...")
    while True:
        p = runpod.get_pod(pod_id)
        status = p.get('desiredStatus') if p else None
        if p and status == 'RUNNING':
            url = f"https://{pod_id}-8000.proxy.runpod.net/health"
            try:
                r = requests.get(url, timeout=5)
                is_live = r.status_code == 200
            except Exception:
                is_live = False
            if is_live:
                print(f"\nLIVE: https://{pod_id}-8000.proxy.runpod.net/v1")
                print_tts_info(pod_id, tts_backend, p)
                print_stt_info(pod_id)
                print_mindroot_info(pod_id)
                break
        sys.stdout.write("."); sys.stdout.flush()
        time.sleep(15)

def deploy_pod(args, tid):
    """Deploy a new pod and save its ID."""
    print(f"Requesting {args.count}x {args.gpu}...")
    print(f"Image: {args.image}")
    print(f"Model: {args.model}")

    pod = runpod.create_pod(
        name=f"{POD_BASE_NAME}-{args.gpu.replace(' ', '-')}",
        template_id=tid,
        gpu_type_id=args.gpu,
        gpu_count=args.count,
        cloud_type="SECURE"
    )

    pod_id = pod['id']
    save_pod_id(pod_id)

    print(f"Waiting for initialization...")
    while True:
        p = runpod.get_pod(pod_id)
        status = p.get('desiredStatus') if p else None
        if p and status == 'RUNNING':
            url = f"https://{pod_id}-8000.proxy.runpod.net/health"
            try:
                r = requests.get(url, timeout=5)
                is_live = r.status_code == 200
            except Exception:
                is_live = False
            if is_live:
                print(f"\nLIVE: https://{pod_id}-8000.proxy.runpod.net/v1")
                print_tts_info(pod_id, args.tts_backend, p)
                print_stt_info(pod_id)
                print_mindroot_info(pod_id)
                break
        sys.stdout.write("."); sys.stdout.flush()
        time.sleep(15)

    return pod_id

def status_pod(pod_id=None):
    """Show status of a pod."""
    if not pod_id:
        pod_id = get_saved_pod_id()

    if not pod_id:
        print("No pod ID saved. Use --deploy first to create a pod.")
        return

    pod = get_pod_status(pod_id)
    if not pod:
        print(f"Pod {pod_id} not found.")
        return

    status = pod.get('status', 'unknown')
    name = pod.get('name', 'unknown')

    tts_backend = pod.get('env', {}).get('TTS_BACKEND', 'qwen3tts_openai')

    print(f"Pod Status:")
    print(f"   Name: {name}")
    print(f"   ID: {pod_id}")
    print(f"   Status: {status}")
    print(f"   TTS Backend: {tts_backend}")
    print(f"   LLM endpoint:  https://{pod_id}-8000.proxy.runpod.net/v1")
    print_tts_info(pod_id, tts_backend, pod)
    print_stt_info(pod_id)
    print_mindroot_info(pod_id, pod.get('env', {}))

    if pod.get('runtime') and pod.get('address'):
        print(f"Running:")
        print(f"   LLM: https://{pod_id}-8000.proxy.runpod.net/v1")
    if status == 'INACTIVE':
        print(f"Stopped. Use --start to restart.")

def main():
    parser = argparse.ArgumentParser(description="Multi-Model RunPod vLLM CLI")

    # Model & Image Config
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B-FP8",
                        help="HuggingFace model repo")
    parser.add_argument("--image", type=str, default=f"{DOCKER_USER}/qwen_vllm:latest",
                        help="Custom Docker image name (default uses DOCKER_USER env var)")

    # Hardware Config
    parser.add_argument("--gpu", type=str, default="NVIDIA H200", help="GPU Type")
    parser.add_argument("--count", type=int, default=1, help="GPU Count")
    parser.add_argument("--len", type=int, default=32768, help="Max Context Length")

    # Management Commands
    parser.add_argument("--deploy", action="store_true", help="Deploy a new pod (or restart stopped one)")
    parser.add_argument("--start", action="store_true", help="Start a stopped pod")
    parser.add_argument("--stop", action="store_true", help="Stop a running pod")
    parser.add_argument("--status", action="store_true", help="Show pod status")
    parser.add_argument("--terminate", nargs='?', const=True, help="Terminate pods")
    parser.add_argument("--delete-template", nargs='?', const=True,
                        help="Delete RunPod template(s). Pass a name to delete specific one, or no arg to delete all vLLM- templates.")
    parser.add_argument("--list-gpus", action="store_true", help="List available GPU types on RunPod")
    parser.add_argument("--pod-id", type=str, help="Specify pod ID for start/stop/status")

    parser.add_argument("--tts-backend", type=str, default="qwen3tts_openai",
                        choices=["qwen3tts_openai", "qwen3tts", "cosyvoice3", "qwen3tts_custom"],
                        help="TTS backend to use (default: qwen3tts_openai)")

    args = parser.parse_args()
    check_env_vars()

    # Handle delete-template
    if args.delete_template is not None:
        name = args.delete_template if isinstance(args.delete_template, str) else None
        delete_templates_by_name(name)
        return

    # Handle list-gpus
    if args.list_gpus:
        gpus = runpod.get_gpus()
        print(f"{'GPU ID':<30} {'Display Name':<25} {'Memory':<10}")
        print("-" * 65)
        for g in gpus:
            gpu_id = g.get('id', '')
            display = g.get('displayName', '')
            mem = g.get('memoryInGb', '?')
            print(f"{gpu_id:<30} {display:<25} {mem}GB")
        return

    # Handle terminate separately (doesn't need template)
    if args.terminate:
        terminate_pods(args.terminate if isinstance(args.terminate, str) else None)
        return

    # Handle status command
    if args.status:
        status_pod(args.pod_id)
        return

    # Handle stop command
    if args.stop:
        stop_pod(args.pod_id)
        return

    # Handle start command (reuse stopped pod)
    if args.start:
        start_pod(args.pod_id, args.tts_backend)
        return

    # Handle deploy command (reuse stopped pod or create new)
    if args.deploy or not any([args.start, args.stop, args.status]):
        saved_id = get_saved_pod_id()

        if saved_id:
            pod = get_pod_status(saved_id)
            if pod:
                if pod.get('runtime') and pod.get('address'):
                    print(f"Pod already running: https://{saved_id}-8000.proxy.runpod.net/v1")
                    return
                elif pod.get('status') == 'INACTIVE':
                    print(f"Found stopped pod: {saved_id}")
                    print(f"Starting existing pod instead of deploying new...")
                    start_pod(saved_id, args.tts_backend)
                    return
            else:
                print(f"Saved pod {saved_id} no longer exists, creating new...")

        check_sip_env_vars()
        # No saved pod or it doesn't exist - deploy new
        tid = get_or_create_template(args.model, args.len, args.image, args.tts_backend)
        deploy_pod(args, tid)
        return

if __name__ == "__main__":
    main()
