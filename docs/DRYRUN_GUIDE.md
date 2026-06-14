# BONSAI Dry-Run Pipeline with Synthetic Value Injection

This guide explains how to run the BONSAI dry-run pipeline, which validates learning with biological features while keeping pretraining independent.

## Overview: Two-Stage Learning

The dry-run pipeline uses a two-stage approach:

**Stage 1 (Pretrain)**: Learn from raw EHR data only
- Builds general clinical representations
- No biological features
- Checkpoint reused for all finetuning

**Stage 2 (Finetune)**: Learn to use biological features
- Starts from pretrained checkpoint
- Adds: epigenetic clocks, protein scores, embeddings
- Tests feature contribution systematically

This design ensures:
1. Fair comparison (same pretrain for all ablations)
2. Clean feature isolation (can measure value-only contribution)
3. Realistic workflow (pretrain is expensive, finetune is quick)

## Pipeline Steps

1. **Prepare synthetic MEDS data** - Example EHR records
2. **Prepare pretrain data** - Convert to tokenized format
3. **Pretrain model** - 2 epochs on 10% of data (raw EHR only)
4. **Inject biological features** - Create separate MEDS copy with features
5. **Prepare finetune data** - Tokenize the injected MEDS data
6. **Create outcomes** - Generate binary mortality labels
7. **Finetune with injected features** - 3 epochs with early stopping
8. **Evaluate** - Compare baseline vs. feature-enhanced models

## Running the Pipeline

### Prerequisites

```bash
# Install BONSAI_values dependencies
cd BONSAI_values
pip install -e .

# Ensure you have example data
python -m corebehrt.synthetic_data.base_cohort.create_base_synthetic_cohort
```

### Run Complete Dry-Run

```bash
cd BONSAI_values
python -m corebehrt.main.dryrun_pipeline
```

This will:
- Check for example MEDS data in `./example_data/example_MEDS_data`
- Generate synthetic biological features for all patients
- Run pretraining (2 epochs, 10% data sample)
- Create binary mortality outcomes
- Run finetuning with value injection (3 epochs)
- Save outputs to:
  - `./outputs/pretraining_dryrun` - Pretrained model
  - `./outputs/finetuning_dryrun_values` - Finetuned models with values

## What's Being Tested

### Pretraining Phase (Raw EHR Only)

**Configuration**: `corebehrt/configs/pretrain_dryrun.yaml`

- **Input**: Synthetic MEDS data with EHR events (diagnoses, labs, medications)
- **Data sample**: 10% of synthetic patients
- **Epochs**: 2 (very fast)
- **Model**: Smaller architecture (hidden_size=96, 3 layers)
- **Features included**: None (raw EHR only)
- **Purpose**: 
  - Learn general clinical representations from EHR
  - Validate MEDS data loading, masking, training loop
  - Create reusable checkpoint for all finetuning ablations

**Output**: Single pretrained checkpoint at `./outputs/pretraining_dryrun`

**Why separate from finetuning?**
- Pretrain is computationally expensive and needs a large, diverse dataset
- Value injection is optional—some applications may not use it
- Checkpoint is frozen during value-based finetuning (only top layers unfreeze)
- Enables fair ablation: all finetuning runs use identical pretrained representations

### Value Injection Module

**Location**: `corebehrt/functional/features/value_injection.py`

Generates for each patient:
- **10 epigenetic clocks**: Base age ± realistic noise (0-120 years)
- **20 EpiScore proteins**: Lognormal-distributed values
- **32-dim MAPLE embeddings**: Normalized random vectors
- **64-dim MethylGPT embeddings**: Normalized random vectors

Injects as MEDS concept rows with:
- Concept codes: `BIO_CLOCKS_CLOCK_0`, `BIO_PROTEINS_PROTEIN_0`, etc.
- Time: 0 (added at sequence start)
- Values: Pre-computed feature values

### Finetuning Phase (EHR + Biological Features)

**Configuration**: `corebehrt/configs/finetune_dryrun_values.yaml`

- **Input**: Synthetic MEDS data with EHR events + injected biological features
- **Data source**: `./example_data/example_MEDS_data_with_values/` (created in value injection step)
- **Pretrained model**: Loaded from `./outputs/pretraining_dryrun`
- **Features injected**: 
  - 10 epigenetic clocks (biological age predictions)
  - 20 EpiScore proteins (disease-associated proteins)
  - 32-dim MAPLE embeddings (methylation patterns)
  - 64-dim MethylGPT embeddings (foundation model)
- **Value embedding**: Enabled with `value_embedding_mode: "concat"`
- **Freezing**: Top layers only unfreeze (lower layers from pretrain frozen)
- **Epochs**: 3 with early stopping (3 epochs patience)
- **Stopping criterion**: ROC-AUC improvement on validation set
- **Output**: Finetuned model at `./outputs/finetuning_dryrun_values`

**Why inject features only at finetune time?**
- Biological features are task-specific (mortality prediction)
- Pretraining benefits from general EHR patterns, not prediction-specific proxies
- Finetuning on task-relevant features (clocks, proteins) improves specialized performance
- Ablations can test: EHR only vs. +clocks vs. +proteins vs. all

## Configuration Files

### Quick-Test Configurations

These were designed for rapid validation on small data:

#### `pretrain_dryrun.yaml`
- `select_ratio: 0.1` - Use only 10% of prepared data
- `epochs: 2` - Just 2 training epochs
- `batch_size: 16` - Small batch for memory efficiency
- `hidden_size: 96` - Reduced from 768 for speed
- `num_hidden_layers: 3` - Reduced from 12 layers

#### `finetune_dryrun_values.yaml`
- `epochs: 3` - 3 epochs with early stopping
- `batch_size: 16` - Small batch size
- `early_stopping: 3` - Stop if 3 epochs without improvement
- `stopping_criterion: roc_auc` - Use ROC-AUC for early stopping
- `value_embedding_mode: "concat"` - Inject values as concatenated features

## Feature Ablation Structure

After dry-run validation, the pipeline is designed to support ablation studies:

### Baseline Configuration
- **Feature set**: EHR only (no biological features)
- **Config**: `finetune_baseline.yaml`
- **Purpose**: Control model performance with existing EHR data

### Progressive Feature Addition

Each adds biological features incrementally:

1. **+ Clocks**: `finetune_ablation_clocks.yaml`
   - Epigenetic clock predictions
   - Expected improvement: Captures biological age information

2. **+ EpiScores**: `finetune_ablation_episcore.yaml`
   - Protein abundance predictions
   - Expected improvement: Adds disease-specific protein signals

3. **+ MAPLE**: `finetune_ablation_maple.yaml`
   - Methylation pattern embeddings
   - Expected improvement: Captures methylation pattern diversity

4. **+ MethylGPT**: `finetune_ablation_methylgpt.yaml`
   - Foundation model embeddings
   - Expected improvement: Leverages pretrained representations

5. **+ All Combined**: `finetune_ablation_all.yaml`
   - All four feature types
   - Purpose: Measure synergistic effects

### Experimental Design Pattern

Each ablation config:
```yaml
paths:
  prepared_data: ./outputs/finetuning/processed_data_with_values/
  pretrain_model: ./outputs/pretraining_dryrun  # Reuse same pretrain
  model: ./outputs/finetuning_ablation_{name}    # Different output per ablation

trainer_args:
  epochs: 10                    # Full training (not dry-run)
  early_stopping: 5             # Real early stopping
  stopping_criterion: roc_auc
  
metrics:
  roc_auc: {...}               # Primary metric
  delong_roc_auc: {...}        # Statistical significance testing
```

## Expected Output Structure

After dry-run completion:

```
outputs/
├── pretraining_dryrun/
│   ├── pytorch_model.bin      # Pretrained checkpoint
│   ├── config.json            # Model config
│   └── training_state.json    # Training metadata
│
└── finetuning_dryrun_values/
    ├── fold_0/
    │   ├── best_model.bin     # Best checkpoint (early stopping)
    │   ├── metrics.json       # Val/test metrics
    │   └── training_curves.png
    ├── fold_1/
    ├── ...
    └── cv_summary.json        # Cross-validation aggregated metrics
```

## Key Concepts

### Value Embedding Mode: "concat"

When injecting biological features, BONSAI concatenates them:
```
Original sequence: [event_embed_1, event_embed_2, event_embed_3, ...]
With values:       [event_embed_1 + value_embed_1, 
                    event_embed_2 + value_embed_2, ...]
```

This allows the transformer to:
- Learn value-specific attention patterns
- Compare with event-only attention patterns
- Measure feature importance via ablation

### Stopping Criterion: roc_auc

For mortality prediction (binary classification):
- **Metric**: Area under ROC curve
- **Stopping**: Early stopping stops training if ROC-AUC doesn't improve for N epochs
- **Rationale**: ROC-AUC captures both sensitivity and specificity

For future survival analysis (CoxPH):
- Will use **concordance index** instead
- Stopping criterion will adjust accordingly

## Troubleshooting

### Error: "Example data not found"
```
Please run: python -m corebehrt.synthetic_data.base_cohort.create_base_synthetic_cohort
```

### Error: "Pretrain preparation failed"
- Check that `./outputs/pretraining/processed_data/` exists
- Ensure prepare_training_data step completed successfully
- Look at logs in `./outputs/logs/`

### Error: "Value injection failed"
- Verify biological features were generated correctly
- Check patient ID alignment between MEDS data and feature arrays
- Ensure parquet files are readable

### Running Specific Steps Only

To debug specific steps:

```bash
# Just prepare data
python -m corebehrt.main.prepare_training_data corebehrt/configs/prepare_pretrain.yaml

# Just pretrain
python -m corebehrt.main.pretrain corebehrt/configs/pretrain_dryrun.yaml

# Just finetune (assumes pretrain complete)
python -m corebehrt.main.finetune_cv corebehrt/configs/finetune_dryrun_values.yaml
```

## Next Steps After Validation

1. **Extend to full training**: Update configs with `select_ratio: 1.0` and `epochs: 20+`
2. **Implement CoxPH loss**: Add survival analysis loss function
3. **Run ablation studies**: Use provided ablation configs for feature comparison
4. **Statistical testing**: Use DeLong tests to compare model performance
5. **Production pipeline**: Scale up with full datasets

## Related Documentation

- [BONSAI Overview](../README.md)
- [Value Injection Module](../corebehrt/functional/features/value_injection.py)
- [Pretrain Config Details](../corebehrt/configs/pretrain_dryrun.yaml)
- [Finetune Config Details](../corebehrt/configs/finetune_dryrun_values.yaml)
