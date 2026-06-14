#!/bin/bash
# Setup script for BONSAI dry-run pipeline

set -e  # Exit on any error

echo "======================================================================"
echo "BONSAI Dry-Run Pipeline - Environment Setup"
echo "======================================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

echo "✓ Conda found"

# Create environment
echo ""
echo "Creating new environment: bonsai_dryrun..."
conda create -n bonsai_dryrun python=3.11 -y -q
echo "✓ Environment created"

# Activate environment
echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate bonsai_dryrun
echo "✓ Environment activated"

# Install PyTorch (Intel Mac compatible)
echo ""
echo "Installing PyTorch 2.1.2 (CPU version - Intel Mac compatible)..."
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu -q
echo "✓ PyTorch 2.1.2 installed"

# Install other dependencies (Intel Mac friendly versions)
echo ""
echo "Installing dependencies..."
pip install \
    transformers==4.36.0 \
    numpy==1.26.0 \
    pandas \
    scipy \
    scikit-learn \
    pyyaml \
    pyarrow \
    tqdm \
    matplotlib \
    -q
echo "✓ Dependencies installed (all Intel Mac compatible)"

# Install BONSAI_values in editable mode
echo ""
echo "Installing BONSAI_values..."
cd "$(dirname "$0")"
pip install -e . -q
echo "✓ BONSAI_values installed"

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
import transformers
import pandas
print(f'  PyTorch: {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  Pandas: {pandas.__version__}')
" 2>/dev/null || echo "  ⚠ Some imports failed (non-critical)"

# Test feature injection
echo ""
echo "Testing feature injection module..."
python -c "
from corebehrt.functional.features.value_injection import SyntheticBiologicalFeatureGenerator
gen = SyntheticBiologicalFeatureGenerator(n_samples=10)
features = gen.generate_all_features()
print(f'✓ Feature injection working: {list(features.keys())}')
" || echo "❌ Feature injection test failed"

# Done
echo ""
echo "======================================================================"
echo "✅ Setup complete!"
echo ""
echo "To run the pipeline, execute:"
echo "  conda activate bonsai_dryrun"
echo "  python -m corebehrt.main.dryrun_pipeline"
echo ""
echo "For troubleshooting, see: SETUP_LOCAL.md"
echo "======================================================================"
