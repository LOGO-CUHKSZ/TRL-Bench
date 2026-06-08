#!/bin/bash
# TaBERT Environment Setup Script
#
# This script creates a conda environment with all dependencies needed
# for running TaBERT column embedding generation.
#
# Usage:
#   bash models/tabert/setup_env.sh [env_name]
#
# Arguments:
#   env_name: Name of the conda environment (default: tabert)
#
# After setup, activate with:
#   conda activate tabert
#   python models/tabert/generate_column_embeddings.py --help

set -e

ENV_NAME="${1:-tabert}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_VERSION="${CUDA_VERSION:-cu121}"

echo "============================================================"
echo "TaBERT Environment Setup"
echo "============================================================"
echo "Environment name: ${ENV_NAME}"
echo "CUDA version: ${CUDA_VERSION}"
echo "============================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment '${ENV_NAME}'..."
conda create -n "${ENV_NAME}" python=3.10 -y

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# Install PyTorch 2.5.x (required for torch_scatter compatibility)
echo ""
echo "Installing PyTorch 2.5.1..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"

# Install torch_scatter
echo ""
echo "Installing torch_scatter..."
pip install torch_scatter -f "https://data.pyg.org/whl/torch-2.5.1+${CUDA_VERSION}.html"

# Install remaining requirements
echo ""
echo "Installing remaining dependencies..."
pip install transformers>=4.30.0 fairseq==0.10.2 h5py>=3.0.0
pip install numpy>=1.24.0 pandas>=2.0.0 tqdm>=4.60.0
pip install msgpack>=1.0.0 ujson>=5.0.0
pip install PyYAML>=6.0 regex>=2022.0.0 sacrebleu>=2.0.0
pip install hydra-core>=1.3.0 omegaconf>=2.3.0
pip install redis>=4.0.0 pyzmq>=25.0.0

# Patch fairseq for NumPy 2.x compatibility
echo ""
echo "Patching fairseq for NumPy compatibility..."
python "${SCRIPT_DIR}/patch_fairseq.py"

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
from torch_scatter import scatter_max
print('torch_scatter: OK')
import fairseq
print('fairseq: OK')
import transformers
print(f'transformers: {transformers.__version__}')
"

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "To use the environment:"
echo "  conda activate ${ENV_NAME}"
echo "  python models/tabert/generate_column_embeddings.py \\"
echo "      --input=datasets/adult \\"
echo "      --checkpoint=checkpoints/tabert/tabert_large_k3/model.bin \\"
echo "      --output=embeddings.pkl \\"
echo "      --device=cuda"
echo ""
echo "============================================================"
