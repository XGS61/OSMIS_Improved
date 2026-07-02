#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

ENV_NAME="${ENV_NAME:-osmis_5090}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.11 -y
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install --upgrade \
  torch torchvision --index-url https://download.pytorch.org/whl/cu128
conda run -n "${ENV_NAME}" python -m pip install -r requirements-modern.txt

echo
echo "Environment ready. Run:"
echo "conda activate ${ENV_NAME}"
echo "bash train_improved_5090.sh"
