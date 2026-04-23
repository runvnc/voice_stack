#!/usr/bin/env python3
"""
Unified Multi-Cloud Deployment Script for vLLM Voice Agent
Supports: RunPod, Nebius AI Cloud
"""

# Fix: venvs don't include user site-packages; inject them so nebius is found
import site, sys
try:
    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.append(user_site)
except Exception:
    pass

import os
import time
import json
import re
import shlex
import secrets
import argparse
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, List, Any

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

STATE_FILE = Path(__file__).parent / "state.json"
POD_STATE_FILE = Path(__file__).parent / "pod_state.json"


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    if POD_STATE_FILE.exists():
        with open(POD_STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    with open(POD_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def check_common_env():
    required = ["DOCKER_USER", "HF_TOKEN"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"Error: Missing required env vars: {', '.join(missing)}")
        sys.exit(1)


def check_sip_env():
    required = ["SIP_USER", "SIP_PASSWORD", "SIP_GATEWAY"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"Error: Missing SIP env vars: {', '.join(missing)}")
        sys.exit(1)


def get_ssh_public_keys() -> List[str]:
    keys: List[str] = []

    env_key = os.getenv("NEBIUS_SSH_PUBLIC_KEY") or os.getenv("SSH_PUBLIC_KEY")
    if env_key and env_key.strip():
        keys.append(env_key.strip())

    default_pubkey = Path.home() / ".ssh" / "id_ed25519.pub"
    fallback_pubkey = Path.home() / ".ssh" / "id_rsa.pub"
    for path in (default_pubkey, fallback_pubkey):
        try:
            if path.exists():
                key = path.read_text().strip()
                if key and key not in keys:
                    keys.append(key)
        except Exception:
            pass

    return keys


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class CloudProvider(ABC):
    @abstractmethod
    def deploy(self, name: str, image: str, env: Dict[str, str],
               ports: Dict[int, int], hardware_config: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def start(self, resource_id: str) -> None:
        pass

    @abstractmethod
    def stop(self, resource_id: str) -> None:
        pass

    @abstractmethod
    def terminate(self, resource_id: str) -> None:
        pass

    @abstractmethod
    def status(self, resource_id: str) -> Optional[Any]:
        pass

    @abstractmethod
    def list_hardware(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_urls(self, resource: Any) -> Dict[str, str]:
        pass

    @abstractmethod
    def wait_for_ready(self, resource_id: str, timeout: int = 600) -> bool:
        pass

    def print_info(self, resource: Any):
        urls = self.get_urls(resource)
        print("\n--- Service URLs ---")
        for name, url in urls.items():
            print(f"  {name}: {url}")
        state = load_state()
        creds = state.get("mindroot_creds", {})
        if creds:
            print("\n--- Mindroot Credentials ---")
            print(f"  ADMIN_USER={creds.get('ADMIN_USER', 'N/A')}")
            print(f"  ADMIN_PASS={creds.get('ADMIN_PASS', 'N/A')}")


# ---------------------------------------------------------------------------
# RunPod Provider
# ---------------------------------------------------------------------------

class RunPodProvider(CloudProvider):
    def __init__(self):
        import runpod as _runpod
        self.api_key = os.getenv("RUNPOD_API_KEY")
        if not self.api_key:
            print("Error: RUNPOD_API_KEY env var required")
            sys.exit(1)
        _runpod.api_key = self.api_key
        self.runpod = _runpod
        self.registry_auth_id = os.getenv("RUNPOD_REGISTRY_AUTH_ID")

    def _query_templates(self):
        import requests
        resp = requests.get(
            "https://rest.runpod.io/v1/templates",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _get_or_create_template(self, model_repo: str, image_name: str,
                                 tts_backend: str, env: Dict[str, str]) -> str:
        model_slug = slugify(model_repo.split("/")[-1])
        image_slug = slugify(image_name.split("/")[-1])
        tts_slug = slugify(tts_backend)
        template_name = f"vLLM-{model_slug}-{image_slug}-{tts_slug}"
        templates = self._query_templates()
        for t in templates:
            if t.get("name") == template_name:
                print(f"Using existing template: {template_name}")
                return t["id"]
        print(f"Creating new template: {template_name}...")
        template_args = {
            "name": template_name,
            "image_name": image_name,
            "container_disk_in_gb": 50,
            "volume_in_gb": 120,
            "volume_mount_path": "/workspace",
            "ports": "8000/http,8091/http,8765/tcp,8880/http,8881/http,8010/http",
            "env": env,
        }
        if self.registry_auth_id:
            template_args["container_registry_auth_id"] = self.registry_auth_id
        new_template = self.runpod.create_template(**template_args)
        return new_template["id"]

    def deploy(self, name, image, env, ports, hardware_config) -> str:
        gpu = hardware_config.get("gpu", "NVIDIA H200")
        count = hardware_config.get("count", 1)
        model = hardware_config.get("model", "Qwen/Qwen3.5-27B-FP8")
        tts_backend = env.get("TTS_BACKEND", "qwen3tts_openai")
        tid = self._get_or_create_template(model, image, tts_backend, env)
        print(f"Requesting {count}x {gpu}...")
        pod = self.runpod.create_pod(
            name=f"vLLM-Inference-{gpu.replace(' ', '-')}",
            template_id=tid,
            gpu_type_id=gpu,
            gpu_count=count,
            cloud_type="SECURE",
        )
        return pod["id"]

    def start(self, resource_id: str):
        self.runpod.start_pod(resource_id)

    def stop(self, resource_id: str):
        self.runpod.stop_pod(resource_id)

    def terminate(self, resource_id: str):
        self.runpod.terminate_pod(resource_id)

    def status(self, resource_id: str):
        try:
            return self.runpod.get_pod(resource_id)
        except Exception:
            return None

    def list_hardware(self):
        gpus = self.runpod.get_gpus()
        return [
            {"id": g.get("id", ""), "name": g.get("displayName", ""),
             "memory_gb": g.get("memoryInGb", "?")}
            for g in gpus
        ]

    def get_urls(self, resource) -> Dict[str, str]:
        pod_id = resource.get("id", "unknown")
        return {
            "LLM": f"https://{pod_id}-8000.proxy.runpod.net/v1",
            "TTS": f"https://{pod_id}-8880.proxy.runpod.net/v1/audio/speech",
            "STT": f"https://{pod_id}-8881.proxy.runpod.net/transcribe",
            "Mindroot": f"https://{pod_id}-8010.proxy.runpod.net",
            "SIP/RTP": "(RunPod proxy does not support UDP - use Nebius for SIP)",
        }

    def wait_for_ready(self, resource_id: str, timeout: int = 600) -> bool:
        print("Waiting for initialization...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            p = self.status(resource_id)
            status = p.get("desiredStatus") if p else None
            if p and status == "RUNNING":
                url = f"https://{resource_id}-8000.proxy.runpod.net/health"
                try:
                    import requests
                    r = requests.get(url, timeout=5)
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(15)
        return False


# ---------------------------------------------------------------------------
# Nebius Provider
# ---------------------------------------------------------------------------

class NebiusProvider(CloudProvider):
    def __init__(self):
        try:
            from nebius.sdk import SDK
            from nebius.api.nebius.compute.v1 import (
                InstanceServiceClient, PlatformServiceClient,
                DiskSpec, SourceImageFamily, PublicIPAddress, IPAddress,
            )
            from nebius.api.nebius.vpc.v1 import (
                SecurityGroupServiceClient, SecurityRuleServiceClient,
            )
            from nebius.api.nebius.vpc.v1alpha1 import SubnetServiceClient
            self.SDK = SDK
            self.InstanceServiceClient = InstanceServiceClient
            self.PlatformServiceClient = PlatformServiceClient
            self.SecurityGroupServiceClient = SecurityGroupServiceClient
            self.SecurityRuleServiceClient = SecurityRuleServiceClient
            self.SubnetServiceClient = SubnetServiceClient
            self.DiskSpec = DiskSpec
            self.SourceImageFamily = SourceImageFamily
            self.PublicIPAddress = PublicIPAddress
            self.IPAddress = IPAddress
        except ImportError:
            print(f"Error: nebius SDK not found in {sys.executable}")
            print(f"Run: {sys.executable} -m pip install nebius")
            sys.exit(1)

        # Auth
        creds_file = os.getenv("NEBIUS_CREDENTIALS_FILE")
        token = os.getenv("NEBIUS_ACCESS_TOKEN") or os.getenv("NEBIUS_IAM_TOKEN")
        if creds_file:
            from nebius.base.service_account.credentials_file import Reader
            self.sdk = self.SDK(credentials=Reader(filename=creds_file))
        elif token:
            from nebius.aio.token.static import Bearer
            self.sdk = self.SDK(credentials=Bearer(token))
        else:
            self.sdk = self.SDK()

        # Auto-discover project_id from CLI config if not set
        self.project_id = os.getenv("NEBIUS_PROJECT_ID")
        if not self.project_id:
            try:
                result = subprocess.run(
                    ["nebius", "config", "get", "parent-id"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    self.project_id = result.stdout.strip()
                    print(f"Auto-discovered project_id: {self.project_id}")
            except Exception:
                pass
        if not self.project_id:
            print("Error: NEBIUS_PROJECT_ID env var required (or configure nebius CLI)")
            sys.exit(1)

        # Auto-discover subnet_id if not set
        self.subnet_id = os.getenv("NEBIUS_SUBNET_ID")
        if not self.subnet_id:
            try:
                subnet_svc = self.SubnetServiceClient(self.sdk)
                from nebius.api.nebius.vpc.v1alpha1 import ListSubnetsRequest
                resp = subnet_svc.list(ListSubnetsRequest(parent_id=self.project_id)).wait()
                if resp.items:
                    self.subnet_id = resp.items[0].metadata.id
                    print(f"Auto-discovered subnet_id: {self.subnet_id}")
            except Exception as e:
                print(f"Warning: could not auto-discover subnet: {e}")
        if not self.subnet_id:
            print("Error: NEBIUS_SUBNET_ID env var required")
            sys.exit(1)

        # Auto-discover network_id from subnet
        self.network_id = os.getenv("NEBIUS_NETWORK_ID")
        if not self.network_id:
            try:
                subnet_svc = self.SubnetServiceClient(self.sdk)
                from nebius.api.nebius.vpc.v1alpha1 import GetSubnetRequest
                resp = subnet_svc.get(GetSubnetRequest(id=self.subnet_id)).wait()
                self.network_id = resp.spec.network_id
                print(f"Auto-discovered network_id: {self.network_id}")
            except Exception as e:
                print(f"Warning: could not auto-discover network: {e}")

        self.image_family = os.getenv("NEBIUS_IMAGE_FAMILY", "ubuntu24.04-cuda13.0")
        self._services: Dict[str, Any] = {}

    def _svc(self, name: str, cls):
        if name not in self._services:
            self._services[name] = cls(self.sdk)
        return self._services[name]

    def _instance_svc(self):
        return self._svc("instance", self.InstanceServiceClient)

    def _platform_svc(self):
        return self._svc("platform", self.PlatformServiceClient)

    def _sg_svc(self):
        return self._svc("sg", self.SecurityGroupServiceClient)

    def _sr_svc(self):
        return self._svc("sr", self.SecurityRuleServiceClient)

    def _build_cloud_init(self, image: str, env: Dict[str, str], ports: Dict[int, int]) -> str:
        ssh_keys = get_ssh_public_keys()
        env_file_lines = []
        for k, v in env.items():
            env_file_lines.append(f"{k}={shlex.quote(str(v))}")

        env_file_content = "\n".join(env_file_lines)
        env_file_block = "\n".join('      ' + line for line in env_file_content.splitlines())

        # SSH keys: we emit them in TWO places as a belt+suspenders workaround for
        # a known cloud-init 24.04 issue where `users.ssh_authorized_keys` sometimes
        # writes 0 bytes (canonical/cloud-init#6175):
        #   1. Top-level `ssh_authorized_keys` -> applied to the image's default user
        #      (`ubuntu` on Nebius ubuntu24.04-* images) via cc_ssh module.
        #   2. `users:` block with `- default` preserved and an explicit `ubuntu`
        #      entry with sudo/shell configured.
        top_level_ssh_block = ""
        users_block = ""
        if ssh_keys:
            top_level_ssh_lines = "\n".join('  - ' + key for key in ssh_keys)
            top_level_ssh_block = f"ssh_authorized_keys:\n{top_level_ssh_lines}\n"

            users_ssh_lines = "\n".join('      - ' + key for key in ssh_keys)
            users_block = (
                "users:\n"
                "  - default\n"
                "  - name: ubuntu\n"
                "    sudo: ALL=(ALL) NOPASSWD:ALL\n"
                "    shell: /bin/bash\n"
                "    groups: [sudo, adm, docker]\n"
                "    ssh_authorized_keys:\n"
                f"{users_ssh_lines}\n"
            )

        return f"""#cloud-config
{top_level_ssh_block}{users_block}write_files:
  - path: /opt/vllm-agent.env
    permissions: '0600'
    owner: root:root
    content: |
{env_file_block}
  - path: /etc/apt/apt.conf.d/99force-ipv4
    permissions: '0644'
    owner: root:root
    content: |
      Acquire::ForceIPv4 "true";
  - path: /usr/local/bin/vllm-agent-bootstrap.sh
    permissions: '0755'
    owner: root:root
    content: |
      #!/bin/bash
      # vLLM voice-agent bootstrap. Idempotent; safe to re-run.
      set -u
      LOGFILE=/var/log/vllm-agent-bootstrap.log
      exec > >(tee -a "$LOGFILE") 2>&1
      echo "=== vLLM bootstrap started at $(date -u +%FT%TZ) ==="

      retry() {{
        local n=1 max=5 delay=15
        while true; do
          "$@" && return 0
          if [ $n -ge $max ]; then
            echo "Command failed after $n attempts: $*" >&2
            return 1
          fi
          echo "Attempt $n failed for: $* ; retrying in ${{delay}}s" >&2
          n=$((n+1))
          sleep "$delay"
        done
      }}

      # Force IPv4 everywhere: apt already configured via 99force-ipv4.
      # Use curl -4 for all external fetches.
      retry apt-get update
      retry apt-get install -y ca-certificates curl gnupg
      install -m 0755 -d /etc/apt/keyrings
      retry bash -c "curl -4 -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg"
      chmod a+r /etc/apt/keyrings/docker.gpg
      echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list
      retry apt-get update
      retry apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

      # NVIDIA container toolkit (may already be present on ubuntu24.04-cuda13.0 image)
      if ! command -v nvidia-ctk >/dev/null 2>&1; then
        retry bash -c "curl -4 -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        retry bash -c "curl -4 -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list"
        retry apt-get update
        retry apt-get install -y nvidia-container-toolkit
      fi
      nvidia-ctk runtime configure --runtime=docker || true
      systemctl restart docker

      # Pull and run the app container.
      retry docker pull {image}
      # Remove any stale container from a prior attempt.
      docker rm -f vllm-agent 2>/dev/null || true
      docker run -d --name vllm-agent --gpus all --network host --restart unless-stopped \\
        --env-file /opt/vllm-agent.env \\
        -v /mnt/data:/workspace \\
        {image}
      echo "=== Container started at $(date -u +%FT%TZ) ==="
      echo "Container started" > /var/log/container-start.log
runcmd:
  - bash /usr/local/bin/vllm-agent-bootstrap.sh
"""

    def _ensure_security_group(self, name: str) -> str:
        from nebius.api.nebius.vpc.v1 import (
            ListSecurityGroupsRequest, CreateSecurityGroupRequest,
            CreateSecurityRuleRequest, ListSecurityRulesRequest,
            DeleteSecurityRuleRequest,
        )
        from nebius.api.nebius.common.v1 import ResourceMetadata
        from nebius.api.nebius.vpc.v1.security_group_pb2 import SecurityGroupSpec
        from nebius.api.nebius.vpc.v1.security_rule_pb2 import SecurityRuleSpec, RuleIngress, RuleEgress

        sg_name = f"sg-{slugify(name)}-{secrets.token_hex(4)}"
        if not self.network_id:
            print("Error: NEBIUS_NETWORK_ID required to create security group")
            sys.exit(1)

        print(f"Creating security group: {sg_name}...")
        sg_spec = SecurityGroupSpec()
        sg_spec.network_id = self.network_id
        op = self._sg_svc().create(
            CreateSecurityGroupRequest(
                metadata=ResourceMetadata(name=sg_name, parent_id=self.project_id),
                spec=sg_spec,
            )
        ).wait()
        sg_id = op.resource_id

        desired_rules = [
            ("tcp-api-1", 2, [8000, 8091, 8765, 8880], "ingress"),
            ("tcp-api-2", 2, [8881, 8010, 22], "ingress"),
            ("udp-all-in", 3, [], "ingress"),
            ("tcp-web-out", 2, [80, 443], "egress"),
            ("udp-all-out", 3, [], "egress"),
        ]

        for rule_name, protocol, ports, direction in desired_rules:
            spec = SecurityRuleSpec()
            spec.access = 1  # ALLOW
            spec.protocol = protocol
            spec.priority = 100
            spec.type = 1  # STATEFUL
            if direction == "ingress":
                ing = RuleIngress()
                ing.source_cidrs.append("0.0.0.0/0")
                for p in ports:
                    ing.destination_ports.append(p)
                spec.ingress.CopyFrom(ing)
            else:
                eg = RuleEgress()
                eg.destination_cidrs.append("0.0.0.0/0")
                for p in ports:
                    eg.destination_ports.append(p)
                spec.egress.CopyFrom(eg)
            self._sr_svc().create(
                CreateSecurityRuleRequest(
                    metadata=ResourceMetadata(name=rule_name, parent_id=sg_id),
                    spec=spec,
                )
            ).wait()

        return sg_id

    def deploy(self, name, image, env, ports, hardware_config) -> str:
        from nebius.api.nebius.compute.v1 import (
            CreateInstanceRequest, InstanceSpec, ResourcesSpec,
            AttachedDiskSpec, ManagedDisk, NetworkInterfaceSpec,
            SecurityGroup,
        )
        from nebius.api.nebius.common.v1 import ResourceMetadata

        platform = hardware_config.get("platform", "gpu-h200-sxm")
        preset = hardware_config.get("preset", "1gpu-16vcpu-200gb")
        disk_gb = hardware_config.get("disk_gb", 200)

        print(f"Deploying on Nebius: {platform}/{preset}")
        print(f"Image: {image}")
        deploy_id = secrets.token_hex(4)

        cloud_init = self._build_cloud_init(image, env, ports)
        sg_id = self._ensure_security_group(name)

        disk_spec = self.DiskSpec(
            type=self.DiskSpec.DiskType.NETWORK_SSD,
            size_gibibytes=disk_gb,
            source_image_family=self.SourceImageFamily(image_family=self.image_family),
        )

        pub_ip = self.PublicIPAddress()
        pub_ip.static = True
        ip_addr = self.IPAddress()

        req = CreateInstanceRequest(
            metadata=ResourceMetadata(name=name, parent_id=self.project_id),
            spec=InstanceSpec(
                resources=ResourcesSpec(platform=platform, preset=preset),
                boot_disk=AttachedDiskSpec(
                    attach_mode="READ_WRITE",
                    managed_disk=ManagedDisk(
                        name=f"{name}-boot-{deploy_id}",
                        spec=disk_spec,
                    ),
                ),
                network_interfaces=[
                    NetworkInterfaceSpec(
                        name="eth0",
                        subnet_id=self.subnet_id,
                        ip_address=ip_addr,
                        public_ip_address=pub_ip,
                        security_groups=[SecurityGroup(id=sg_id)],
                    )
                ],
                cloud_init_user_data=cloud_init,
            ),
        )

        op = self._instance_svc().create(req).wait()
        instance_id = op.resource_id
        print(f"Instance created: {instance_id}")

        # Persist sg_id so we can clean it up on terminate
        state = load_state()
        state["nebius_sg_id"] = sg_id
        save_state(state)

        return instance_id

    def start(self, resource_id: str):
        from nebius.api.nebius.compute.v1 import StartInstanceRequest
        self._instance_svc().start(StartInstanceRequest(id=resource_id)).wait()
        print(f"Instance {resource_id} started")

    def stop(self, resource_id: str):
        from nebius.api.nebius.compute.v1 import StopInstanceRequest
        self._instance_svc().stop(StopInstanceRequest(id=resource_id)).wait()
        print(f"Instance {resource_id} stopped")

    def terminate(self, resource_id: str):
        from nebius.api.nebius.compute.v1 import DeleteInstanceRequest
        from nebius.api.nebius.vpc.v1 import (
            DeleteSecurityGroupRequest,
            ListSecurityRulesRequest,
            DeleteSecurityRuleRequest,
        )

        # Delete the instance first
        self._instance_svc().delete(DeleteInstanceRequest(id=resource_id)).wait()
        print(f"Instance {resource_id} deleted")

        # Clean up the security group and its rules
        state = load_state()
        sg_id = state.get("nebius_sg_id")
        if sg_id:
            print(f"Cleaning up security group {sg_id}...")
            try:
                # Delete all rules in the security group first
                rules_resp = self._sr_svc().list(
                    ListSecurityRulesRequest(parent_id=sg_id)
                ).wait()
                for rule in rules_resp.items:
                    rule_id = rule.metadata.id
                    self._sr_svc().delete(
                        DeleteSecurityRuleRequest(id=rule_id)
                    ).wait()
                    print(f"  Deleted rule: {rule_id}")
                # Now delete the security group itself
                self._sg_svc().delete(
                    DeleteSecurityGroupRequest(id=sg_id)
                ).wait()
                print(f"Security group {sg_id} deleted")
            except Exception as e:
                print(f"Warning: failed to clean up security group {sg_id}: {e}")

    def status(self, resource_id: str):
        from nebius.api.nebius.compute.v1 import GetInstanceRequest
        try:
            return self._instance_svc().get(GetInstanceRequest(id=resource_id)).wait()
        except Exception:
            return None

    def list_hardware(self):
        from nebius.api.nebius.compute.v1 import ListPlatformsRequest
        resp = self._platform_svc().list(ListPlatformsRequest(parent_id=self.project_id)).wait()
        results = []
        for plat in resp.items:
            p_name = plat.metadata.name
            for preset in plat.spec.presets:
                results.append({
                    "platform": p_name,
                    "preset": preset.name,
                    "vcpu": preset.resources.vcpu_count,
                    "memory_gb": preset.resources.memory_gibibytes,
                    "gpu_count": preset.resources.gpu_count,
                })
        return results

    def get_urls(self, resource) -> Dict[str, str]:
        try:
            public_ip = resource.status.network_interfaces[0].public_ip_address.address
            if "/" in public_ip:
                public_ip = public_ip.split("/")[0]
        except (AttributeError, IndexError):
            public_ip = "(no public IP yet)"
        return {
            "LLM": f"http://{public_ip}:8000/v1",
            "TTS": f"http://{public_ip}:8880/v1/audio/speech",
            "STT": f"http://{public_ip}:8881/transcribe",
            "Mindroot": f"http://{public_ip}:8010",
            "SIP": f"sip:USER@{public_ip}:5060",
            "RTP": f"udp://{public_ip}:10000-20000",
        }

    def wait_for_ready(self, resource_id: str, timeout: int = 600) -> bool:
        print("Waiting for Nebius VM initialization...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            inst = self.status(resource_id)
            if inst is None:
                time.sleep(10)
                continue
            state_name = inst.status.state.name if hasattr(inst.status.state, "name") else str(inst.status.state)
            if state_name == "RUNNING":
                time.sleep(30)
                urls = self.get_urls(inst)
                llm_url = urls.get("LLM", "") + "/health"
                if llm_url.startswith("http"):
                    try:
                        import requests
                        r = requests.get(llm_url, timeout=5)
                        if r.status_code == 200:
                            return True
                    except Exception:
                        pass
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(15)
        return False


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

PROVIDER_MAP = {
    "runpod": RunPodProvider,
    "nebius": NebiusProvider,
}


def main():
    parser = argparse.ArgumentParser(description="Unified vLLM Voice Agent Deployment")
    parser.add_argument("--provider", type=str, choices=list(PROVIDER_MAP.keys()),
                        default="nebius", help="Cloud provider to use")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B-FP8",
                        help="HuggingFace model repo")
    parser.add_argument("--image", type=str, default=None,
                        help="Docker image (default: $DOCKER_USER/qwen_vllm:latest)")
    parser.add_argument("--tts-backend", type=str, default="qwen3tts_openai",
                        choices=["qwen3tts_openai", "qwen3tts", "cosyvoice3", "qwen3tts_custom"],
                        help="TTS backend")
    parser.add_argument("--gpu", type=str, default="NVIDIA H200", help="RunPod GPU type")
    parser.add_argument("--platform", type=str, default="gpu-h200-sxm", help="Nebius platform")
    parser.add_argument("--preset", type=str, default="1gpu-16vcpu-200gb", help="Nebius preset")
    parser.add_argument("--disk-gb", type=int, default=200, help="Nebius boot disk size in GB")
    parser.add_argument("--deploy", action="store_true", help="Deploy a new instance")
    parser.add_argument("--start", action="store_true", help="Start a stopped instance")
    parser.add_argument("--stop", action="store_true", help="Stop a running instance")
    parser.add_argument("--status", action="store_true", help="Show instance status")
    parser.add_argument("--terminate", action="store_true", help="Terminate/delete an instance")
    parser.add_argument("--list-hardware", action="store_true", help="List available hardware")
    parser.add_argument("--resource-id", type=str, help="Specific resource ID (uses saved state if omitted)")

    args = parser.parse_args()
    check_common_env()

    docker_user = os.getenv("DOCKER_USER")
    image = args.image or f"{docker_user}/qwen_vllm:latest"

    env = {
        "HF_HOME": "/workspace/huggingface",
        "VLLM_ATTENTION_BACKEND": "FLASHINFER",
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "TTS_BACKEND": args.tts_backend,
        "STT_PROVIDER": "silero_cohere",
        "SILERO_MIN_SILENCE_MS": "400",
        "ANY_LLM_SERVER_URL": "http://localhost:8000/v1",
        "SIP_USER": os.getenv("SIP_USER", ""),
        "SIP_PASSWORD": os.getenv("SIP_PASSWORD", ""),
        "SIP_GATEWAY": os.getenv("SIP_GATEWAY", ""),
    }
    env["JWT_SECRET_KEY"] = secrets.token_hex(32)
    env["ADMIN_USER"] = "admin_" + secrets.token_hex(4)
    env["ADMIN_PASS"] = secrets.token_urlsafe(16)

    state = load_state()
    state["mindroot_creds"] = {
        "ADMIN_USER": env["ADMIN_USER"],
        "ADMIN_PASS": env["ADMIN_PASS"],
    }
    save_state(state)

    ports = {8000: 8000, 8091: 8091, 8765: 8765, 8880: 8880,
             8881: 8881, 8010: 8010, 5060: 5060}

    if args.provider == "runpod":
        hardware = {"gpu": args.gpu, "count": 1, "model": args.model}
    else:
        hardware = {"platform": args.platform, "preset": args.preset,
                    "disk_gb": args.disk_gb, "model": args.model}

    ProviderClass = PROVIDER_MAP[args.provider]
    provider = ProviderClass()

    if args.list_hardware:
        hw_list = provider.list_hardware()
        if args.provider == "runpod":
            print(f"{'GPU ID':<30} {'Name':<25} {'Memory':<10}")
            print("-" * 65)
            for g in hw_list:
                print(f"{g['id']:<30} {g['name']:<25} {g['memory_gb']}GB")
        else:
            print(f"{'Platform':<20} {'Preset':<20} {'vCPU':<6} {'Mem':<8} {'GPUs':<6}")
            print("-" * 65)
            for h in hw_list:
                print(f"{h['platform']:<20} {h['preset']:<20} "
                      f"{h['vcpu']:<6} {h['memory_gb']}GB   {h['gpu_count']}")
        return

    if args.status:
        rid = args.resource_id or state.get("resource_id")
        if not rid:
            print("No resource ID saved. Use --deploy first.")
            return
        resource = provider.status(rid)
        if resource is None:
            print(f"Resource {rid} not found.")
            return
        print(f"Resource ID: {rid}")
        if args.provider == "runpod":
            print(f"Status: {resource.get('status', 'unknown')}")
            print(f"Name: {resource.get('name', 'N/A')}")
        else:
            st = resource.status
            print(f"Status: {st.state.name if hasattr(st.state, 'name') else st.state}")
            try:
                ip = st.network_interfaces[0].public_ip_address.address
                print(f"Public IP: {ip}")
            except (AttributeError, IndexError):
                print("Public IP: (not assigned)")
        provider.print_info(resource)
        return

    if args.terminate:
        rid = args.resource_id or state.get("resource_id")
        if not rid:
            print("No resource ID saved.")
            return
        provider.terminate(rid)
        state.pop("resource_id", None)
        state.pop("nebius_sg_id", None)
        save_state(state)
        print("Resource terminated and state cleared.")
        return

    if args.stop:
        rid = args.resource_id or state.get("resource_id")
        if not rid:
            print("No resource ID saved.")
            return
        provider.stop(rid)
        return

    if args.start:
        rid = args.resource_id or state.get("resource_id")
        if not rid:
            print("No resource ID saved.")
            return
        provider.start(rid)
        if provider.wait_for_ready(rid):
            provider.print_info(provider.status(rid))
        return

    # Default / --deploy
    saved_rid = state.get("resource_id")
    if saved_rid:
        existing = provider.status(saved_rid)
        if existing is not None:
            if args.provider == "runpod":
                if existing.get("runtime") and existing.get("address"):
                    print(f"Already running: {saved_rid}")
                    provider.print_info(existing)
                    return
                elif existing.get("status") == "INACTIVE":
                    print(f"Starting stopped instance: {saved_rid}")
                    provider.start(saved_rid)
                    if provider.wait_for_ready(saved_rid):
                        provider.print_info(provider.status(saved_rid))
                    return
            else:
                st_name = existing.status.state.name if hasattr(existing.status.state, "name") else str(existing.status.state)
                if st_name == "RUNNING":
                    print(f"Already running: {saved_rid}")
                    provider.print_info(existing)
                    return
                elif st_name == "STOPPED":
                    print(f"Starting stopped instance: {saved_rid}")
                    provider.start(saved_rid)
                    if provider.wait_for_ready(saved_rid):
                        provider.print_info(provider.status(saved_rid))
                    return

    check_sip_env()
    if args.provider == "nebius" and not get_ssh_public_keys():
        print("Warning: no SSH public key found for Nebius deploy. Set NEBIUS_SSH_PUBLIC_KEY or create ~/.ssh/id_ed25519.pub")

    name = f"vllm-voice-{args.provider}"
    rid = provider.deploy(name, image, env, ports, hardware)
    # Reload state since deploy() may have saved nebius_sg_id
    state = load_state()
    state["resource_id"] = rid
    state["provider"] = args.provider
    save_state(state)

    # Print known info immediately (don't wait for health check)
    print(f"\n--- Deploy Info ---")
    print(f"  Resource ID: {rid}")
    print(f"  Provider: {args.provider}")
    print(f"  ADMIN_USER: {env['ADMIN_USER']}")
    print(f"  ADMIN_PASS: {env['ADMIN_PASS']}")

    if not provider.wait_for_ready(rid, timeout=1200):
        print("WARNING: Timed out waiting for ready. Check status manually.")

    # Always print info (URLs + creds) after deploy, even on timeout
    resource = provider.status(rid)
    if resource:
        provider.print_info(resource)
    else:
        print(f"Resource ID: {rid}")
        print(f"Provider: {args.provider}")
        print(f"Run: python3 deploy_unified.py --provider {args.provider} --status")

if __name__ == "__main__":
    main()
