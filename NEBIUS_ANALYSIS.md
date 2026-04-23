# Nebius Deployment Analysis for vLLM Voice Agent Container

## Executive Summary

After thorough investigation of the Nebius Python SDK, API protobuf definitions, and official documentation, here are the findings:

**Answer to your question**: Nebius does NOT have a direct "deploy container" API suitable for long-running voice agents. You deploy a **VM** and then run the container on it. The Serverless AI Endpoint API (`nebius.ai.v1`) does accept container specs, but it's designed for HTTP request/response workloads and is inappropriate for a multi-service voice agent that takes several minutes to load models and needs persistent UDP sockets for SIP/RTP.

**UDP Support**: YES. Nebius VMs support UDP ingress and egress via:
1. Security groups with UDP protocol rules (`--protocol udp`)
2. Stateful firewall that allows return traffic automatically
3. Public IP addresses (dynamic or static) with 1:1 NAT

This makes Nebius viable for SIP voice agents, unlike RunPod which only exposes HTTP/TCP proxy ports.

---

## Nebius API Landscape

### 1. Serverless AI (`nebius.ai.v1`) - NOT SUITABLE

The `EndpointServiceClient` and `JobServiceClient` accept:
- `image`: Docker image URL
- `environment_variables`: Key-value env vars
- `ports`: With `container_port`, `host_port`, `protocol` (HTTP/TCP/UDP)
- `public_ip`: Boolean
- `platform` + `preset`: Hardware selection

**Why this is wrong for our workload:**
- Designed for inference APIs, not persistent SIP agents
- Unknown idle timeout behavior
- Container restart on any spec change = minutes of model reload
- No direct SSH/debug access
- The `public_endpoints` may only support HTTP (needs verification)

### 2. Compute VMs (`nebius.compute.v1`) - CORRECT CHOICE

The `InstanceServiceClient` provides raw VM control:
- `boot_disk`: OS disk (use container-optimized image or standard Ubuntu)
- `network_interfaces`: Assign subnet, public IP, security groups
- `cloud_init_user_data`: Bootstrap script to install Docker and run container
- `resources.platform` + `resources.preset`: GPU hardware
- Full lifecycle: create, start, stop, delete

**Advantages for voice agent:**
- Full control over container lifecycle
- Persistent VM across restarts
- Direct UDP port exposure via security groups
- SSH access for debugging
- Custom cloud-init for complex bootstrap

### 3. Security Groups (`nebius.vpc.v1`) - UDP FIREWALL

Security groups are **stateful** firewalls assigned to VM network interfaces:

```
RuleProtocol: ANY, TCP, UDP, ICMP
RuleDirection: INGRESS, EGRESS
RuleAccessAction: ALLOW, DENY
RuleType: STATEFUL, STATELESS
```

**For SIP/RTP, you need:**
- Ingress UDP on SIP signaling port (usually 5060)
- Ingress UDP on RTP media port range (usually 10000-20000)
- Egress UDP allow-all (for outbound calls, STUN, DNS)
- Ingress TCP on HTTP ports (8000, 8880, 8881, 8010, 8091)

**Key insight**: Security groups are stateful, so outbound RTP will work if you allow the initial ingress SIP invite (return traffic is automatically permitted).

---

## UDP Port Exposure: RunPod vs Nebius

| Feature | RunPod | Nebius |
|---------|--------|--------|
| Port exposure | HTTP proxy only (`{pod}-{port}.proxy.runpod.net`) | Direct public IP + security group rules |
| UDP support | NO (TCP/HTTP only) | YES (full UDP via security groups) |
| SIP signaling | Not possible directly | Possible with UDP ingress rule |
| RTP media | Not possible | Possible with UDP port range ingress |
| WebSocket | Yes (TCP) | Yes (TCP via security group) |
| HTTPS proxy | Automatic | Manual (need LB or direct IP) |

**This is a major advantage of Nebius for voice agents.**

---

## Container vs VM: The Correct Architecture

```
RunPod:
  Template -> Pod (container runs directly, HTTP proxy only)

Nebius:
  VM Instance (Ubuntu/container-optimized OS)
    -> cloud-init installs Docker
    -> Docker pulls and runs our image
    -> Security groups open UDP/TCP ports
    -> Public IP with 1:1 NAT
```

**You do NOT deploy "container + VM" separately.** You deploy a VM that runs a container. This is the same model as GCP's Container-Optimized OS or AWS ECS on EC2.

### Bootstrap Strategy for Nebius

The VM's `cloud_init_user_data` should:
1. Install Docker
2. Install NVIDIA Container Toolkit (for GPU passthrough)
3. Authenticate to private registry if needed
4. `docker run` our image with:
   - `--gpus all`
   - `-p` for all required ports (TCP and UDP)
   - All env vars
   - Volume mounts for model cache

### Port Mapping on Nebius

Unlike RunPod's automatic HTTP proxy, on Nebius you must explicitly:
1. Map container ports to host ports in Docker run command
2. Create security group rules for those host ports
3. Access via `public_ip:host_port`

For SIP, the container should use `--network host` or explicit `-p` mappings.

---

## Nebius Hardware Selection

Nebius uses a **platform + preset** model:

1. **Platform** = GPU type family (e.g., `gpu-h100-sxm`, `gpu-h200`)
2. **Preset** = specific configuration within platform (e.g., `8gpu-80gb`, `1gpu-80gb`)

**API to list hardware:**
- `PlatformServiceClient.list()` -> returns platforms with nested presets
- Each preset has: `name`, `vcpu_count`, `memory_gibibytes`, `gpu_count`, `gpu_memory_gigabytes`

**For our workload (Qwen3.5-27B-FP8 + TTS + STT):**
- Need ~80GB+ VRAM for FP8 model
- H100 80GB or H200 would work
- Start with single GPU preset

---

## Networking Requirements for Voice Agent on Nebius

### Required Security Group Rules

| Direction | Protocol | Ports | Source | Purpose |
|-----------|----------|-------|--------|---------|
| Ingress | TCP | 8000 | 0.0.0.0/0 | vLLM API |
| Ingress | TCP | 8880 | 0.0.0.0/0 | TTS API |
| Ingress | TCP | 8881 | 0.0.0.0/0 | STT API |
| Ingress | TCP | 8010 | 0.0.0.0/0 | Mindroot UI |
| Ingress | TCP | 8091 | 0.0.0.0/0 | vllm-omni TTS |
| Ingress | TCP | 8765 | 0.0.0.0/0 | WebSocket TTS |
| Ingress | UDP | 5060 | 0.0.0.0/0 | SIP signaling |
| Ingress | UDP | 10000-20000 | 0.0.0.0/0 | RTP media |
| Egress | ANY | ALL | 0.0.0.0/0 | Outbound (STUN, DNS, model download) |

### Public IP Strategy

- Use **static public IP** (allocation) for SIP, since dynamic IPs change on stop >1hr
- Or accept dynamic IP and update SIP registration when it changes
- SIP registrar must point to the public IP

---

## Design Recommendations

### 1. Unified Deployment Script Architecture

Create `deploy_unified.py` with:

```
CloudProvider (ABC)
  - deploy(...)
  - start(resource_id)
  - stop(resource_id)
  - terminate(resource_id)
  - status(resource_id)
  - list_hardware()
  - get_urls(resource)

RunPodProvider(CloudProvider)
  - Uses runpod SDK + REST API
  - Templates for env/ports
  - Proxy URLs for endpoints

NebiusProvider(CloudProvider)
  - Uses nebius SDK
  - Creates VM with cloud-init
  - Manages security groups
  - Direct public IP access
```

### 2. Nebius-Specific Implementation Plan

**Phase 1: VM Deployment**
- Use `InstanceServiceClient.create()` with:
  - Container-optimized boot image or Ubuntu
  - `cloud_init_user_data` for Docker bootstrap
  - Public IP enabled
  - Subnet ID from user config
  - Platform + preset for H100/H200

**Phase 2: Security Group Setup**
- Create security group in the VM's network
- Add rules for all required TCP and UDP ports
- Attach security group to VM's network interface

**Phase 3: Container Bootstrap**
- Cloud-init runs on first boot
- Installs Docker and NVIDIA runtime
- Runs `docker run` with all ports and env vars
- Container starts loading models

**Phase 4: Health Checking**
- Poll vLLM health endpoint on public IP:8000
- Similar to RunPod approach

### 3. Why Not Use Nebius "Containers over VMs" Web Feature?

The web console has a "Containers over VMs" feature, but:
- It's a simplified UI wrapper
- Limited port configuration
- No UDP port exposure in UI
- No security group integration in the simple flow
- Better to use the full VM API for control

---

## SDK Installation

```bash
pip install nebius
```

The SDK uses gRPC under the hood and requires asyncio for async operations.

---

## Authentication

Nebius supports multiple auth methods:
1. `NEBIUS_IAM_TOKEN` env var (short-lived, for CLI use)
2. Service account with private key file (recommended for automation)
3. CLI config file (if Nebius CLI is configured)

For deployment scripts, service account auth is most robust:

```python
from nebius.sdk import SDK
from nebius.base.service_account.credentials_file import Reader

sdk = SDK(credentials_file_name="/path/to/credentials.json")
```

---

## Open Questions / TODO

1. **Does Nebius Endpoint (Serverless AI) support UDP in `public_endpoints`?** - The protobuf has `UDP` as a protocol enum, but whether it actually exposes UDP to the internet needs testing. Given the serverless nature, raw VMs are still preferred.

2. **What's the exact `cloud_init_user_data` format Nebius expects?** - Likely standard cloud-init YAML. Need to verify on first deployment.

3. **Do we need to pre-create a subnet, or is there a default?** - Need to check if `subnet_id` can be omitted or if there's a default subnet per project.

4. **Container-optimized OS image ID** - Need to find the image family name for container-optimized VMs, or use Ubuntu + cloud-init Docker install.

5. **GPU driver availability** - Does Nebius pre-install NVIDIA drivers, or do we need to install them in cloud-init?

6. **Security group default behavior** - If no security groups are assigned, does the default security group allow all traffic or deny all?

---

## Migration Path from RunPod to Nebius

| Step | RunPod | Nebius |
|------|--------|--------|
| Auth | `RUNPOD_API_KEY` | `NEBIUS_IAM_TOKEN` or service account |
| Hardware | `gpu_type_id` (e.g., "NVIDIA H200") | `platform` + `preset` (e.g., `gpu-h100-sxm` + `1gpu-80gb`) |
| Image | Template references image | `cloud_init_user_data` pulls image |
| Ports | Template `ports` string | Security group rules + Docker `-p` |
| Env vars | Template `env` dict | Cloud-init sets env vars for Docker |
| Public access | `{pod_id}-{port}.proxy.runpod.net` | `public_ip:port` |
| UDP | Not supported | Full support via security groups |
| Stop/Start | Native pod operations | VM stop/start |
| Volume | Network volume at `/workspace` | Attach disk or use local VM storage |

---

## Conclusion

Nebius is **viable and attractive** for this voice agent deployment because:
1. **UDP support** enables native SIP/RTP without STUN/TURN workarounds
2. **Raw VM + container** gives full control matching our current architecture
3. **Security groups** provide proper firewall control
4. **gRPC SDK** is well-documented and supports async operations

The deployment complexity is higher than RunPod (need cloud-init, security groups, no automatic HTTPS proxy), but the networking flexibility is worth it for voice workloads.

**Recommended approach**: Build the unified deployment script with both RunPod and Nebius providers, using raw VMs for Nebius.
