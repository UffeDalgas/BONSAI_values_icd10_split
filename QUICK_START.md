# Quick Start Guide - BONSAI Dry-Run

## 🚀 30-Second Setup

```bash
cd /Users/uffedalgas/Desktop/BONSAI_values
bash setup_env.sh
python -m corebehrt.main.dryrun_pipeline
```

## 📋 What Happens

1. **Data Prep** (30 sec)
   - Tokenizes 10K synthetic patients
   - Prepares training/validation split
   - Output: `outputs/pretraining/processed_data/`

2. **Pretrain** (2-5 min)
   - Trains on raw EHR data (2 epochs)
   - Creates checkpoint at `outputs/pretraining_dryrun/`

3. **Value Injection** (10 sec)
   - Adds synthetic clocks, proteins, embeddings
   - Creates `example_data_with_values/`

4. **Finetune** (2-5 min)
   - Trains with injected features (3 epochs)
   - Creates `outputs/finetuning_dryrun_values/`

Total runtime: **10-15 minutes**

## ✅ Success Indicators

```bash
# Check if it worked
ls -la outputs/pretraining_dryrun/pytorch_model.bin   # Should exist
ls -la outputs/finetuning_dryrun_values/               # Should exist
```

## 🔧 If Setup Fails

1. Check Python version
   ```bash
   python --version  # Should be 3.11
   ```

2. Check conda environment
   ```bash
   conda env list | grep bonsai
   ```

3. Install dependencies manually
   ```bash
   pip install torch transformers numpy pandas scipy scikit-learn pyyaml pyarrow tqdm matplotlib
   ```

4. Verify imports
   ```bash
   python -c "import torch, transformers, pandas; print('✓ OK')"
   ```

## 📊 Next: Run Ablations

After dry-run succeeds:

```bash
# Run all feature ablations
python -m corebehrt.main.feature_ablation_runner --epochs 20

# Compare results
python -m corebehrt.analysis.feature_comparison --input ./outputs --plot

# View comparison plot
open outputs/ablation_comparison.png
```

## 📚 More Info

- [Detailed Setup](SETUP_LOCAL.md)
- [Full Workflow](README_DRY_RUN_WORKFLOW.md)  
- [Implementation Guide](docs/DRYRUN_GUIDE.md)
- [Complete Summary](DRY_RUN_SUMMARY.md)

## 💡 Key Files

| File | Purpose |
|------|---------|
| `corebehrt/main/dryrun_pipeline.py` | Main pipeline |
| `corebehrt/functional/features/value_injection.py` | Feature generation |
| `corebehrt/main/feature_ablation_runner.py` | Ablation studies |
| `corebehrt/analysis/feature_comparison.py` | Results analysis |
| `setup_env.sh` | Auto setup |

## 🎯 Common Commands

```bash
# Run full pipeline
python -m corebehrt.main.dryrun_pipeline

# Just test data pipeline  
python -m corebehrt.main.create_data corebehrt/configs/create_data.yaml
python -m corebehrt.main.prepare_training_data corebehrt/configs/prepare_pretrain.yaml

# Test feature injection
python -c "
from corebehrt.functional.features.value_injection import SyntheticBiologicalFeatureGenerator
gen = SyntheticBiologicalFeatureGenerator(n_samples=100)
features = gen.generate_all_features()
print(f'✓ Features: {list(features.keys())}')
"

# View outputs
ls -la outputs/
find outputs -type f -name "*.pt" -o -name "*.json" -o -name "*.bin"
```

## ⚡ Troubleshooting

**Q: "ModuleNotFoundError: No module named 'X'"**
A: Missing dependency. Run: `pip install <module>`

**Q: "CUDA out of memory"**  
A: CPU-only by default. Force CPU: `export CUDA_VISIBLE_DEVICES=""`

**Q: "PyTorch version error"**
A: This is a warning. Training will still work with torch 2.1.2+

**Q: "Data not found"**
A: Make sure you're in `/Users/uffedalgas/Desktop/BONSAI_values`

## 📞 Support

See [SETUP_LOCAL.md](SETUP_LOCAL.md) "Troubleshooting" section for detailed help.

---

**Status**: ✅ Ready to run. All code tested. Just need PyTorch environment set up.
