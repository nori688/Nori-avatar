#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec env \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  __NV_PRIME_RENDER_OFFLOAD=1 \
  "./.venv-avatar/bin/python" main.py
