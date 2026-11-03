#!/bin/bash
# Create a project-local environment and fetch pinned inference resources.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"
case "${1:-}" in
  --help|-h) printf '%s\n' 'Usage: ./setup.sh [--skip-assets]' 'Requires macOS on Apple Silicon and uv.' 'Default: install dependencies, fetch upstream source and download/verify model assets.' 'Use --skip-assets to install software without the model download.'; exit 0 ;;
  ''|--skip-assets) ;;
  *) printf '%s\n' 'Unknown option. Run ./setup.sh --help.' >&2; exit 2 ;;
esac
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  printf '%s\n' 'This setup targets macOS on Apple Silicon.' >&2; exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1
fi
if [[ -x .venv/bin/python ]]; then
  .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use a Python 3.12 project environment"'
else
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python -r requirements-macos-tested.txt
uv pip install --python .venv/bin/python --no-deps -r requirements-dinov3.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python scripts/fetch_upstream.py assets/egopost
if [[ "${1:-}" != --skip-assets ]]; then
  .venv/bin/python scripts/fetch_assets.py weights
  .venv/bin/python scripts/verify_assets.py weights --output work/assets-integrity.json
fi
.venv/bin/python scripts/doctor.py
printf '%s\n' 'Software setup complete. See README.md for camera permissions and model asset requirements.'
