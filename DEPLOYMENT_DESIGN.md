# Unified Deployment Design: RunPod + Nebius

## Executive Summary

This document describes the refactoring of `deploy_qwen.py` from a RunPod-only deployment tool into a **provider-agnostic multi-cloud deployment framework** supporting both **RunPod** and **Nebius AI Cloud**.

## Why Nebius?

Nebius AI Cloud is a GPU cloud provider offering competitive pricing and, critically for this project, **full UDP ingress/egress support** through security groups with explicit `ALLOW` rules for UDP traffic. This is essential for the SIP voice agent framework, which uses RTP media streams over UDP.

## Nebius Compute Model: Container vs VM

Nebius offers multiple compute abstractions:

### 1. AI Endpoints / Jobs (Serverless AI, `nebius.ai.v1`)
- Managed containers with automatic scaling
- Docker image, env vars, port mappings specified directly in API
- **Not suitable for our use case** because:
  - Designed for inference services with fast cold-start
  - No explicit security group attachment (networking is abstracted)
  - UDP port exposure is not documented/verified
  - Long startup times (minutes for model loading) may hit timeout limits
  - No persistent volume mounting for model caching

### 2. Compute VMs with Containers (`compute.v1` + cloud-init)
- Full VM with Docker container deployed via cloud-init
- Complete control over networking, security groups, storage
- **This is the correct approach** for our workload because:
  - Verified UDP support via security group rules (`protocol: UDP`, ingress/egress)
  - Persistent disk for HuggingFace model cache (`/workspace`)
  - No startup timeout - VM stays running
  - Full Docker capability including multi-stage builds
  - Can attach security groups with explicit UDP ALLOW rules for SIP/RTP

### 3. Raw Compute Instances
- Same as #2 but without the container convenience layer
- We use #2 (containers over VMs) as it matches our existing Docker-based workflow

## Nebius Networking for SIP/RTP

### Security Groups
Nebius uses **stateful security groups** (firewall rules at the VM network interface level). Key capabilities:

- **Protocols**: `TCP`, `UDP`, `ICMP`, `ANY` (confirmed in protobuf `RuleProtocol` enum)
- **Directions**: `INGRESS` (incoming) and `EGRESS` (outgoing)
- **Actions**: `ALLOW` or `DENY`
- **Stateful**: Return traffic is automatically allowed for stateful rules
- **Port ranges**: Multiple destination ports can be specified per rule

### Required Rules for SIP Voice Agent
For a SIP endpoint on Nebius, the following security group rules are needed:

| Direction | Protocol | Ports | Source/Dest | Action |
|-----------|----------|-------|-------------|--------|
| Ingress | TCP | 5060 | 0.0.0.0/0 | ALLOW |
| Ingress | UDP | 5060 | 0.0.0.0/0 | ALLOW |
| Ingress | UDP | 10000-20000 | 0.0.0.0/0 | ALLOW |
| Egress | UDP | 5060 | 0.0.0.0/0 | ALLOW |
| Egress | UDP | 10000-20000 | 0.0.0.0/0 | ALLOW |
| Egress | TCP | 443 | 0.0.0.0/0 | ALLOW |

Note: The default security group denies all traffic. If no security group is assigned, the default applies. Our deployment script will create and attach a custom security group with the necessary rules.

### Public IP Addresses
- VMs can have **dynamic** or **static** public IPs
- Dynamic IPs are returned to the pool if VM is stopped > 1 hour
- Static IPs persist across stops but return on deletion
- One public IP per VM maximum
- SIP/RTP traffic goes directly to the public IP (no proxy layer like RunPod)

## Architecture Design

### Provider Abstraction Layer

```
CloudProvider (abstract base)
  ├── deploy(name, image, env, ports, hardware_config)
  ├── start(resource_id)
  ├── stop(resource_id)
  ├── terminate(resource_id)
  ├── status(resource_id)
  ├── list_hardware()
  └── get_urls(resource) -> dict

RunPodProvider (CloudProvider)
  - Uses runpod Python SDK + REST API
  - Template-based deployment
  - Proxy URLs: https://{pod_id}-{port}.proxy.runpod.net

NebiusProvider (CloudProvider)
  - Uses nebius Python SDK (gRPC)
  - VM + cloud-init deployment
  - Direct IP URLs: http://{public_ip}:{port}
  - Security group management for UDP
```

### Unified CLI

```bash
# Deploy on RunPod
python deploy.py --provider runpod --deploy

# Deploy on Nebius
python deploy.py --provider nebius --deploy

# List hardware options
python deploy.py --provider nebius --list-hardware

# Status / stop / start / terminate (provider-agnostic commands)
python deploy.py --provider nebius --status
python deploy.py --provider nebius --stop
python deploy.py --provider nebius --start
python deploy.py --provider nebius --terminate
```

### Environment Variables

**RunPod** (unchanged):
- `RUNPOD_API_KEY`
- `DOCKER_USER`
- `HF_TOKEN`
- `SIP_USER`, `SIP_PASSWORD`, `SIP_GATEWAY`

**Nebius** (new):
- `NEBIUS_IAM_TOKEN` or `NEBIUS_CREDENTIALS_FILE`
- `NEBIUS_PROJECT_ID` (parent resource for all deployments)
- `NEBIUS_SUBNET_ID` (required for VM networking)
- `DOCKER_USER` (same)
- `HF_TOKEN` (same)
- `SIP_USER`, `SIP_PASSWORD`, `SIP_GATEWAY` (same)

### State Management

- `pod_state.json` -> `state.json` (renamed to be provider-agnostic)
- Each provider stores its own resource ID format
- Both providers store: resource_id, mindroot_creds, provider_name

### Endpoint URL Generation

**RunPod**:
```
LLM:   https://{pod_id}-8000.proxy.runpod.net/v1
TTS:   https://{pod_id}-8880.proxy.runpod.net/v1/audio/speech
STT:   https://{pod_id}-8881.proxy.runpod.net/transcribe
Mindroot: https://{pod_id}-8010.proxy.runpod.net
SIP:   Not directly accessible (proxy doesn't support UDP)
```

**Nebius**:
```
LLM:   http://{public_ip}:8000/v1
TTS:   http://{public_ip}:8880/v1/audio/speech
STT:   http://{public_ip}:8881/transcribe
Mindroot: http://{public_ip}:8010
SIP:   sip:{sip_user}@{public_ip}:5060  (direct UDP access!)
RTP:   {public_ip}:10000-20000 UDP
```

The critical difference: **Nebius gives us direct public IP access with UDP**, making it viable for SIP endpoint deployment where RunPod's HTTP proxy layer cannot handle UDP RTP media.

## Implementation Plan

### Phase 1: Unified Script (`deploy.py`)
- [x] Abstract `CloudProvider` base class
- [x] `RunPodProvider` implementation (refactored from `deploy_qwen.py`)
- [x] `NebiusProvider` implementation using `nebius` SDK
- [x] Unified argument parser with `--provider` flag
- [x] Provider-agnostic state file

### Phase 2: Nebius Security Group Automation
- [x] Auto-create security group with SIP/RTP rules
- [x] Attach security group to VM network interface
- [x] Support for static vs dynamic IP selection

### Phase 3: Testing
- [ ] Deploy on Nebius and verify all services start
- [ ] Verify UDP ingress on port range 10000-20000
- [ ] Test SIP registration and RTP media flow
- [ ] Benchmark latency vs RunPod

## Nebius SDK Usage Notes

### SDK Installation
```bash
pip install nebius
```

### Authentication
```python
from nebius.sdk import SDK
sdk = SDK()  # Reads NEBIUS_IAM_TOKEN env var
```

### Key APIs Used
- `nebius.api.nebius.compute.v1.InstanceServiceClient` - VM CRUD
- `nebius.api.nebius.compute.v1.PlatformServiceClient` - List GPU platforms/presets
- `nebius.api.nebius.vpc.v1.SecurityGroupServiceClient` - Firewall rules
- `nebius.api.nebius.vpc.v1.SecurityRuleServiceClient` - Individual rules
- `nebius.api.nebius.ai.v1.EndpointServiceClient` - (for reference, not used)

### Operations Pattern
Most mutating APIs return `Operation` objects:
```python
operation = await service.create(request)
await operation.wait()  # Polls until completion
print(f"Created: {operation.resource_id}")
```

### Hardware Selection
Nebius uses **Platform + Preset** instead of GPU type:
1. Call `PlatformServiceClient.list()` to get platforms (e.g., "gpu-h100-sxm")
2. Each platform has `.spec.presets[]` with resource details
3. Select preset by name (e.g., "8gpu-80gb")

## Migration from RunPod to Nebius

### For Existing Users
1. Set up Nebius credentials (`NEBIUS_IAM_TOKEN`)
2. Get project ID and subnet ID from Nebius console
3. Run `python deploy.py --provider nebius --list-hardware` to see GPU options
4. Deploy: `python deploy.py --provider nebius --deploy`
5. Update SIP gateway config to point to Nebius public IP instead of RunPod proxy

### Docker Image Compatibility
The existing Dockerfile is fully compatible with Nebius Compute VMs. No changes needed to the container image itself. The VM runs standard Docker and pulls the same image.

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Nebius security groups don't support UDP | **False** - protobuf confirms UDP protocol enum. Docs show `--protocol udp` CLI flag. |
| Nebius Endpoint API (serverless) has timeouts | **Avoided** - We use Compute VMs, not Serverless Endpoints. |
| Different URL format breaks client configs | `get_urls()` method generates correct URLs per provider. Clients read from state file. |
| Nebius disk performance slower than RunPod | Both use network-attached SSD. Comparable. Can tune disk type. |
| No RunPod-style proxy for HTTPS | Nebius gives direct IP. Can add reverse proxy (nginx/traefik) in container if needed. |

## Conclusion

Nebius Compute VMs with custom security groups provide a viable alternative to RunPod for our voice agent deployment. The key advantage is **direct UDP access** for SIP/RTP media, which RunPod's HTTP proxy architecture cannot support. The refactored `deploy.py` script provides a unified interface across both providers, minimizing operational complexity.
