#!/usr/bin/env python3
"""Clean up orphaned Nebius security groups from failed/old deploys.

Usage:
    python3 cleanup_security_groups.py              # dry run (list what would be deleted)
    python3 cleanup_security_groups.py --delete      # actually delete
    python3 cleanup_security_groups.py --delete --keep-current  # delete all except the one in state.json
"""
import site, sys
try:
    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.append(user_site)
except Exception:
    pass

import os
import json
import argparse
import subprocess
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


def get_sdk():
    from nebius.sdk import SDK
    creds_file = os.getenv("NEBIUS_CREDENTIALS_FILE")
    token = os.getenv("NEBIUS_ACCESS_TOKEN") or os.getenv("NEBIUS_IAM_TOKEN")
    if creds_file:
        from nebius.base.service_account.credentials_file import Reader
        return SDK(credentials=Reader(filename=creds_file))
    elif token:
        from nebius.aio.token.static import Bearer
        return SDK(credentials=Bearer(token))
    else:
        return SDK()


def get_project_id():
    pid = os.getenv("NEBIUS_PROJECT_ID")
    if not pid:
        try:
            result = subprocess.run(
                ["nebius", "config", "get", "parent-id"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                pid = result.stdout.strip()
        except Exception:
            pass
    if not pid:
        print("Error: NEBIUS_PROJECT_ID required")
        sys.exit(1)
    return pid


def main():
    parser = argparse.ArgumentParser(description="Clean up orphaned Nebius security groups")
    parser.add_argument("--delete", action="store_true", help="Actually delete (default is dry run)")
    parser.add_argument("--keep-current", action="store_true",
                        help="Keep the SG referenced in state.json")
    parser.add_argument("--keep-default", action="store_true", default=True,
                        help="Keep the default security group (default: True)")
    args = parser.parse_args()

    sdk = get_sdk()
    project_id = get_project_id()

    from nebius.api.nebius.vpc.v1 import (
        SecurityGroupServiceClient, SecurityRuleServiceClient,
        ListSecurityGroupsRequest, DeleteSecurityGroupRequest,
        ListSecurityRulesRequest, DeleteSecurityRuleRequest,
    )

    sg_svc = SecurityGroupServiceClient(sdk)
    sr_svc = SecurityRuleServiceClient(sdk)

    # Load state to find current SG
    state_file = Path(__file__).parent / "state.json"
    current_sg_id = None
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
            current_sg_id = state.get("nebius_sg_id")

    # List all security groups
    resp = sg_svc.list(ListSecurityGroupsRequest(parent_id=project_id)).wait()

    print(f"Found {len(resp.items)} security groups:\n")

    to_delete = []
    for sg in resp.items:
        sg_id = sg.metadata.id
        sg_name = sg.metadata.name
        is_default = getattr(sg.status, 'default', False)
        is_current = (sg_id == current_sg_id)

        skip_reason = None
        if is_default and args.keep_default:
            skip_reason = "DEFAULT"
        elif is_current and args.keep_current:
            skip_reason = "CURRENT"

        status = f"  SKIP ({skip_reason})" if skip_reason else "  DELETE"
        print(f"  {sg_name} ({sg_id}){status}")

        if not skip_reason:
            to_delete.append((sg_id, sg_name))

    print(f"\n{len(to_delete)} security groups to delete.")

    if not args.delete:
        print("\nDry run. Use --delete to actually delete.")
        return

    for sg_id, sg_name in to_delete:
        print(f"\nDeleting {sg_name} ({sg_id})...")

        # Delete all rules first
        try:
            rules_resp = sr_svc.list(ListSecurityRulesRequest(parent_id=sg_id)).wait()
            for rule in rules_resp.items:
                rule_id = rule.metadata.id
                rule_name = rule.metadata.name
                print(f"  Deleting rule: {rule_name} ({rule_id})")
                sr_svc.delete(DeleteSecurityRuleRequest(id=rule_id)).wait()
        except Exception as e:
            print(f"  Warning: error listing/deleting rules: {e}")

        # Delete the security group
        try:
            sg_svc.delete(DeleteSecurityGroupRequest(id=sg_id)).wait()
            print(f"  Deleted {sg_name}")
        except Exception as e:
            print(f"  Error deleting {sg_name}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
