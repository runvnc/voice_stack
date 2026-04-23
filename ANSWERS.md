# Answers to Nebius Open Questions - VERIFIED via API Testing

## Test Results Summary

All questions were answered by running actual API calls against Nebius using the SDK.

### 1. SourceImageFamily scope (CORRECTED)

**Finding**: `SourceImageFamily` works with **empty parent_id** (not project_id).

```python
DiskSpec(
    type=DiskSpec.DiskType.NETWORK_SSD,
    size_gibibytes=50,
    source_image_family=SourceImageFamily(image_family="ubuntu24.04-cuda13.0"),
    # parent_id omitted!
)
```

Using `parent_id=PROJECT_ID` fails with "Image family not found". Using `parent_id=tenant_id` fails with permission denied. Empty parent_id works and auto-resolves to the correct scope.

**Also**: Minimum disk size for this image is ~40 GiB (42949672960 bytes).

### 2. PublicIPAddress object creation

**Finding**: Must create `PublicIPAddress()` object and set fields via attribute, not dict.

```python
pub_ip = PublicIPAddress()
pub_ip.static = True  # NOT {"static": True}
```

### 3. NetworkInterfaceSpec required fields

**Finding**: Requires `name`, `ip_address`, and `public_ip_address`.

```python
NetworkInterfaceSpec(
    name="eth0",                    # REQUIRED
    subnet_id=SUBNET_ID,
    ip_address=IPAddress(),         # REQUIRED (empty = auto-assign)
    public_ip_address=pub_ip,       # REQUIRED
)
```

### 4. Boot disk creation timing

**Finding**: When creating a disk separately then attaching to VM, the disk must be READY before VM creation. VM creation fails with "disk is not ready".

**Solution**: Use `managed_disk` within `AttachedDiskSpec` — disk is created atomically with VM.

```python
AttachedDiskSpec(
    attach_mode="READ_WRITE",
    managed_disk=ManagedDisk(
        name="boot-disk",
        spec=DiskSpec(...),
    ),
)
```

### 5. Public IP format

**Finding**: Address includes `/32` CIDR suffix.

```
Private: '10.96.0.20/32'
Public:  '204.12.168.7/32'
```

Use `.split("/")[0]` to get clean IP.

### 6. Security rule direction detection

**Finding**: Use `getattr(r.spec, 'ingress', None)` / `getattr(r.spec, 'egress', None)`.

`HasField()` does not work on SDK wrapper objects.

### 7. Security group deletion

**Finding**: Must delete all rules first, then delete the security group. Deleting SG with rules fails with `FAILED_PRECONDITION`.

### 8. Image family name

**Confirmed**: `ubuntu24.04-cuda13.0` is the correct family name (with drivers + CUDA 13.0 pre-installed).

### 9. Auto-discovery

**Verified**:
- `project_id`: from `nebius config get parent-id` CLI
- `subnet_id`: from `SubnetServiceClient.list()` first item
- `network_id`: from `SubnetServiceClient.get(subnet_id).spec.network_id`

### 10. Platform/preset format

**Confirmed**: H200 platform is `gpu-h200-sxm`, preset is `1gpu-16vcpu-200gb`.

```
Platform: gpu-h200-sxm
  Preset: 1gpu-16vcpu-200gb
    vCPU: 16
    Memory: 200 GiB
    GPUs: 1
```

## Updated deploy_unified.py

The deployment script has been updated with all verified API patterns:
- Auto-discovery of project_id, subnet_id, network_id
- Proper `SourceImageFamily(image_family=...)` with empty parent_id
- `managed_disk` for atomic disk+VM creation
- `PublicIPAddress()` object with `.static = True`
- `NetworkInterfaceSpec` with `name`, `ip_address`, `public_ip_address`
- Security group with TCP + UDP ingress rules
- Public IP stripping `/32` suffix
- Rule deletion before SG deletion on cleanup
