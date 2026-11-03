#!/bin/bash
# Resolve resources from this checkout, regardless of the current directory.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"
if [[ ! -x .venv/bin/egohand-live ]]; then
  printf '%s\n' 'Run ./setup.sh first.' >&2; exit 1
fi
needs_calibration=true
needs_output=true
for arg in "$@"; do
  case "$arg" in
    --calibration|--calibration=*|--estimate-intrinsics) needs_calibration=false ;;
  esac
  case "$arg" in
    --output|--output=*) needs_output=false ;;
  esac
done
command_args=(\
  --source assets/egopost \
  --checkpoint weights/egohand/egohand_s060000.ckpt \
  --assets weights/soma \
  --detector weights/detector/hand_detector.onnx
)
if "$needs_calibration"; then command_args+=(--estimate-intrinsics); fi
if "$needs_output"; then command_args+=(--output "runs/demo-$(date +%Y%m%d-%H%M%S)-$$"); fi
exec .venv/bin/egohand-live "${command_args[@]}" "$@"
