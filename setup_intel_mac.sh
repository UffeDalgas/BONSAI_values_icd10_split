#!/bin/bash
# Intel Mac setup - simplified and reliable

echo "========================================================================"
echo "BONSAI Dry-Run Pipeline - Intel Mac Setup"
echo "========================================================================"

# Setup conda properly
source /opt/anaconda3/etc/profile.d/conda.sh

# Remove old environment if exists
echo "Cleaning up old environment (if any)..."
conda remove -n bonsai_dryrun -y --all 2>/dev/null || true
sleep 2

# Create new environment
echo "Creating bonsai_dryrun environment..."
conda create -n bonsai_dryrun python=3.11 -y

# Activate and install
echo "Activating environment..."
conda activate bonsai_dryrun

echo ""
echo "Installing PyTorch 2.1.2 (Intel Mac CPU)..."
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

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
    matplotlib

echo ""
echo "Installing BONSAI_values..."
pip install -e .

echo ""
echo "Verifying installation..."
python -c "
import torch
import transformers
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ Transformers {transformers.__version__}')
"

python -c "
from corebehrt.functional.features.value_injection import SyntheticBiologicalFeatureGenerator
gen = SyntheticBiologicalFeatureGenerator(n_samples=10)
features = gen.generate_all_features()
print(f'✓ Feature injection: {list(features.keys())}')
"

echo ""
echo "========================================================================"
echo "✅ Setup complete!"
echo ""
echo "To activate the environment and run the pipeline:"
echo "  conda activate bonsai_dryrun"
echo "  python -m corebehrt.main.dryrun_pipeline"
echo ""
echo "========================================================================"
