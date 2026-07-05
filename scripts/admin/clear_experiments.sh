#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

exec python3 "${SCRIPT_DIR}/experiment_admin.py" clear \
  --workspace-root "${WORKSPACE_ROOT}" "$@"
