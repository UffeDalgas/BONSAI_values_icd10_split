# BONSAI Dry-Run Workflow: Synthetic Data + Value Injection

This document describes the complete dry-run pipeline and feature ablation framework for validating BONSAI with synthetic biological features.

## 📋 Quick Start

```bash
# 1. Create synthetic MEDS data
python -m corebehrt.synthetic_data.base_cohort.create_base_synthetic_cohort

# 2. Run complete dry-run pipeline
python -m corebehrt.main.dryrun_pipeline

# 3. After dry-run succeeds, run ablation experiments
python -m corebehrt.main.feature_ablation_runner --epochs 10 --batch-size 16

# 4. Compare results
python -m corebehrt.analysis.feature_comparison --input ./outputs --plot
```

## 🔄 Pipeline Architecture

### Key Design: Two-Stage Learning

**Pretrain**: Learn general EHR representations (no biological features)
**Finetune**: Learn to use biological features on top of pretrained representations

This separation allows:
1. Baseline model that works with just EHR
2. Clean comparison of feature contributions (controlled pretrain)
3. Ablations can swap different feature sets without retraining from scratch

### Phase 1: Dry-Run (Quick Validation)

**File**: `corebehrt/main/dryrun_pipeline.py`

```
PHASE 1: PRETRAIN (raw EHR data)
  synthetic MEDS data (no features)
         ↓
  prepare data (10% sample)
         ↓
  PRETRAIN: 2 epochs, hidden_size=96, 3 layers
         ↓
  ./outputs/pretraining_dryrun/ ← checkpoint

PHASE 2: CREATE FINETUNING DATA (with features)
  synthetic MEDS data (no features)
         ↓
  inject biological features (clocks, proteins, embeddings)
         ↓
  ./example_data/example_MEDS_data_with_values/

PHASE 3: FINETUNE (EHR + biological features)
  MEDS data WITH injected features
         ↓
  prepare data
         ↓
  load pretrain checkpoint
         ↓
  FINETUNE: 3 epochs, value_embedding="concat"
         ↓
  ./outputs/finetuning_dryrun_values/ ← ready for ablations
```

**Purpose**: 
- Validate end-to-end pipeline architecture
- Test data loading, masking, feature injection
- Ensure no crashes before full training
- Demonstrate that injected features improve performance
- Expected runtime: 10-20 minutes

**Outputs**:
- `./outputs/pretraining_dryrun/` - Pretrained model checkpoint (learned on raw EHR)
- `./example_data/example_MEDS_data_with_values/` - MEDS data with synthetic features
- `./outputs/finetuning_dryrun_values/` - Finetuned models with features (CV folds)

### Phase 2: Feature Ablation (Systematic Comparison)

**File**: `corebehrt/main/feature_ablation_runner.py`

Runs progressive feature addition experiments:

```
1. BASELINE
   └─ EHR only (no biological features)

2. CLOCKS
   └─ EHR + 10 epigenetic clocks

3. PROTEINS
   └─ EHR + 20 EpiScore proteins

4. MAPLE
   └─ EHR + 32-dim methylation embeddings

5. METHYLGPT
   └─ EHR + 64-dim foundation model embeddings

6. ALL
   └─ EHR + clocks + proteins + MAPLE + MethylGPT
```

Each ablation:
- Uses same pretrained model (from Phase 1)
- Finetunes from scratch with different feature sets
- Measures ROC-AUC, PR-AUC, accuracy
- Outputs cross-validated metrics

**Command**:
```bash
# Run all ablations
python -m corebehrt.main.feature_ablation_runner --epochs 10

# Run specific ablations
python -m corebehrt.main.feature_ablation_runner --ablations baseline clocks all

# Just baseline (EHR-only model)
python -m corebehrt.main.feature_ablation_runner --baseline-only

# Dry-run (show what would run)
python -m corebehrt.main.feature_ablation_runner --dry-run
```

### Phase 3: Analysis & Comparison

**File**: `corebehrt/analysis/feature_comparison.py`

Analyzes ablation results:
- Load CV-aggregated metrics from each ablation
- Compute ROC-AUC improvements vs baseline
- Identify top-performing feature sets
- Generate statistical comparisons (future: DeLong tests)
- Visualize feature contributions

**Command**:
```bash
# Basic comparison
python -m corebehrt.analysis.feature_comparison --input ./outputs

# With visualization
python -m corebehrt.analysis.feature_comparison --input ./outputs --plot

# Save results as JSON
python -m corebehrt.analysis.feature_comparison --input ./outputs --output ./results
```

## 📊 Feature Injection System

**Module**: `corebehrt/functional/features/value_injection.py`

### How It Works

1. **Generate synthetic features** (per patient):
   ```python
   gen = SyntheticBiologicalFeatureGenerator(n_samples=1000)
   features = gen.generate_all_features()
   ```
   Produces:
   - Clocks: Dict[10] - Epigenetic clock predictions (0-120 years)
   - Proteins: Dict[20] - EpiScore protein abundances (lognormal)
   - MAPLE: Array[1000, 32] - Methylation pattern embeddings
   - MethylGPT: Array[1000, 64] - Foundation model embeddings

2. **Create concept mapping** (feature name → medical concept code):
   ```python
   concept_map = create_feature_concept_mapping(features)
   # Returns: {"clocks/clock_0": "BIO_CLOCKS_CLOCK_0", ...}
   ```

3. **Inject into MEDS data** (append as value rows):
   ```python
   meds_injected = inject_values_into_meds(
       meds_df,
       features,
       patient_ids,
       concept_map
   )
   ```
   Original:
   ```
   subject_id  time  concept_code  value
   0           0     ICD9:401      1.0      (hypertension diagnosis)
   0           3     LOINC:2345    120.0   (systolic BP)
   ```
   
   After injection:
   ```
   subject_id  time  concept_code                    value
   0           0     ICD9:401                        1.0
   0           0     BIO_CLOCKS_CLOCK_0              45.2  ← injected
   0           0     BIO_PROTEINS_PROTEIN_0          2.3   ← injected
   0           0     BIO_MAPLE_EMBEDDINGS_DIM0      -0.15  ← injected
   0           3     LOINC:2345                      120.0
   ```

### Value Embedding in Model

The transformer processes injected values using `value_embedding_mode: "concat"`:

```
For each event e with value v:
  event_embedding = embedding(e)  # Standard BONSAI embedding
  value_embedding = value_mlp(v)  # Special MLP for value features
  combined = concat(event_embedding, value_embedding)  # Concatenate
  attention_input = combined + positional_encoding
```

This allows:
- Event embeddings to remain independent
- Values to be weighted separately by attention
- Easy ablation by removing value_embedding layer

## 🧪 Configuration Files

### Dry-Run Configs

**`pretrain_dryrun.yaml`** (2 epochs, quick):
```yaml
data:
  select_ratio: 0.1          # Use only 10% of data
model:
  hidden_size: 96             # Reduced from 768
  num_hidden_layers: 3        # Reduced from 12
  value_embedding_mode: "concat"
trainer_args:
  epochs: 2
  batch_size: 16
```

**`finetune_dryrun_values.yaml`** (3 epochs, with values):
```yaml
model:
  value_embedding_mode: "concat"  # Enable value injection
trainer_args:
  epochs: 3
  early_stopping: 3               # Stop if no improvement for 3 epochs
  stopping_criterion: roc_auc
```

### Full-Training Configs

For ablation studies, generated dynamically by `feature_ablation_runner.py`:

```python
# Baseline (EHR only)
config = AblationConfig(
    name="baseline",
    features=[],                    # No biological features
    epochs=10,
    early_stopping=5
)

# All features
config = AblationConfig(
    name="all",
    features=["clocks", "proteins", "maple", "methylgpt"],
    epochs=10,
    early_stopping=5
)
```

Each ablation generates YAML in `corebehrt/configs/ablation_{name}.yaml`

## 📈 Understanding Outputs

### After Dry-Run

```
outputs/
├── pretraining_dryrun/
│   ├── pytorch_model.bin        # Pretrained checkpoint
│   ├── config.json              # Model configuration
│   ├── training_state.json      # Training metadata
│   └── logs/                    # Training logs
│
└── finetuning_dryrun_values/
    ├── fold_0/
    │   ├── best_model.bin       # Best checkpoint (early stopping)
    │   ├── metrics.json         # Val metrics per epoch
    │   ├── test_results.json    # Test set evaluation
    │   └── predictions.csv      # Raw predictions
    ├── fold_1/
    ├── ...
    ├── cv_summary.json          # Cross-validated aggregated metrics
    └── training_curves.png      # Visualization
```

### Metric Interpretation

**`cv_summary.json` example**:
```json
{
  "roc_auc": {"mean": 0.742, "std": 0.031},
  "pr_auc": {"mean": 0.623, "std": 0.042},
  "accuracy": {"mean": 0.681, "std": 0.025},
  "num_folds": 5,
  "num_training_samples": 800,
  "num_test_samples": 200
}
```

- **ROC-AUC** (primary): Sensitivity/specificity trade-off
- **PR-AUC**: Precision/recall trade-off (better for imbalanced)
- **Accuracy**: Overall classification accuracy
- **std**: Standard deviation across CV folds

### After Ablation Comparison

```
outputs/
└── ablation_summary.json
    {
      "timestamp": "2024-06-11T14:30:00",
      "completed": 6,
      "failed": 0,
      "details": {
        "baseline": {"status": "completed"},
        "clocks": {"status": "completed"},
        ...
      }
    }

results/
├── ablation_comparison.json     # Detailed results + recommendations
└── ablation_comparison.png      # ROC-AUC visualization
```

**`ablation_comparison.json` structure**:
```json
{
  "results_by_ablation": [
    {
      "ablation": "all",
      "features": "clocks, proteins, maple, methylgpt",
      "roc_auc_mean": 0.761,
      "roc_auc_std": 0.028,
      "pr_auc_mean": 0.655,
      "delta_roc_auc": 0.019
    },
    ...
  ],
  "top_features": [
    {
      "feature_set": "clocks",
      "roc_auc_improvement": 0.012,
      "absolute_roc_auc": 0.754
    },
    ...
  ],
  "recommendations": [
    "Best performance with all (ROC-AUC=0.761)",
    "Combining all features improves ROC-AUC by 0.0190"
  ]
}
```

## 🔍 Troubleshooting

### Issue: "Example data not found"
```bash
python -m corebehrt.synthetic_data.base_cohort.create_base_synthetic_cohort
```

### Issue: Ablation configs not created
Ensure `pyyaml` is installed:
```bash
pip install pyyaml
```

### Issue: Low ROC-AUC values
- Dry-run with small models yields ~0.50-0.65 AUC
- Full training should improve to 0.70-0.80+
- Binary classification is harder than real survival (CoxPH), so lower is expected

### Issue: Feature injection failing
Check MEDS data format:
- Required columns: `subject_id`, `time`, `concept_code`, `value`
- All numeric values should be floats
- Subject IDs should be sequential starting at 0

## 🚀 Next Steps

### 1. Run Dry-Run
```bash
python -m corebehrt.main.dryrun_pipeline
```
Expected time: 10-20 min
Expected outcome: Validates end-to-end pipeline works

### 2. Run Feature Ablations
```bash
python -m corebehrt.main.feature_ablation_runner --epochs 20 --batch-size 32
```
Expected time: 1-2 hours
Expected outcome: Quantifies feature contributions

### 3. Analyze Results
```bash
python -m corebehrt.analysis.feature_comparison --input ./outputs --plot
```
Expected outcome: Feature ranking + recommendations

### 4. Implement CoxPH
After validation, implement:
- Survival loss function (partial likelihood)
- SurvivalHead with cumulative hazard output
- Concordance index metric
- Right-censoring handling

### 5. Production Scaling
- Remove `select_ratio: 0.1` (use full data)
- Increase epochs: 20-50
- Larger model: `hidden_size: 256`, `num_layers: 12`
- Full cross-validation: 5-10 folds
- Statistical testing: DeLong test for ROC-AUC comparison

## 📚 Related Files

- **Dry-run pipeline**: `corebehrt/main/dryrun_pipeline.py`
- **Ablation runner**: `corebehrt/main/feature_ablation_runner.py`
- **Feature injection**: `corebehrt/functional/features/value_injection.py`
- **Analysis module**: `corebehrt/analysis/feature_comparison.py`
- **Dry-run configs**: `corebehrt/configs/pretrain_dryrun.yaml`, `corebehrt/configs/finetune_dryrun_values.yaml`
- **Detailed guide**: `docs/DRYRUN_GUIDE.md`

## 🎯 Key Design Decisions

1. **Synthetic features over real data**: Allows rapid iteration without waiting for biological data
2. **Single pretrain for all ablations**: Ensures fair comparison by controlling pretrain randomness
3. **Value concatenation**: Simple approach that doesn't require model surgery
4. **Binary classification vs survival**: Scaffold for future CoxPH implementation
5. **Early stopping**: Prevents overfitting on small validation sets
6. **Cross-validation**: Robust evaluation despite small datasets

## 📝 Citations & References

- BONSAI: ModernBERT for EHR (embedding-based features)
- Value injection: Similar to feature injection in tabular transformers
- Feature ablation: Standard approach for importance estimation
- DeLong test: Non-parametric test for ROC-AUC significance (future implementation)
