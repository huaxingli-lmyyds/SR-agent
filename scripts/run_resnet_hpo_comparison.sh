#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${WORKSPACE_ROOT}"
exec python3 "${SCRIPT_DIR}/run_comparison_experiments.py" \
  --model-family resnet \
  --config-path recipes/voxceleb/hparams/train_resnet.yaml \
  --suite smoke \
  "$@"