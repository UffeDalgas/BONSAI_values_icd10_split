# Value Injection Finetuning - Complete Workflow

After baseline finetuning works, run finetuning with biological features for conditions 0-7.

## Quick Start (3 commands)

```bash
# Step 1: Align value injection files to your MEDS subject IDs
bash align_value_injection_to_meds.sh \
  --value-injection-path /cluster/outputs/bonsai_value_injection \
  --output-path /cluster/outputs \
  --meds-path /cluster/data/meds_for_bonsai

# Step 2: Create finetuning configs for all conditions
bash create_condition_finetuning_configs.sh \
  --output-path /cluster/outputs \
  --value-injection-aligned-path /cluster/outputs/value_injection_aligned

# Step 3: Finetune each condition (run in parallel or sequentially)
for condition in 0 1 2 3 4 5 6 7; do
  sbatch finetune_condition.sh \
    --condition $condition \
    --output-path /cluster/outputs \
    --pretrain-model /cluster/outputs/pretraining_dryrun
done
```

---

## Detailed Workflow

### Step 1: Align Value Injection Files

Your epimap output has sample IDs like `SYNTH_000000`, but BONSAI needs subject IDs from your MEDS data.

```bash
bash align_value_injection_to_meds.sh \
  --value-injection-path /cluster/outputs/bonsai_value_injection \
  --output-path /cluster/outputs \
  --meds-path /cluster/data/meds_for_bonsai
```

**What it does:**
- Loads condition_0.csv, condition_1.csv, ..., condition_7.csv
- Maps epimap sample IDs → MEDS subject IDs
- Saves aligned files to `outputs/value_injection_aligned/`

**Output:**
```
outputs/value_injection_aligned/
├── condition_0_baseline_aligned.csv
├── condition_1_metadata_aligned.csv
├── condition_2_grimage_v2_aligned.csv
├── condition_3_systemsage_aligned.csv
├── condition_4_deep_embeddings_aligned.csv
├── condition_5_maple_predictions_aligned.csv
├── condition_6_dmi_aligned.csv
└── condition_7_episcores_aligned.csv
```

### Step 2: Create Finetuning Configs

Generate `finetune_config.yaml` for each condition specifying:
- Which value injection file to use
- Training hyperparameters (batch size, epochs, LR)
- Model output path

```bash
bash create_condition_finetuning_configs.sh \
  --output-path /cluster/outputs \
  --value-injection-aligned-path /cluster/outputs/value_injection_aligned \
  --conditions 0 1 2 3 4 5 6 7
```

**What it does:**
- Creates a config for each condition
- Specifies the aligned CSV file as input
- Sets injection mode to `concat` (can change to FiLM, discrete, comb)

**Output:**
```
outputs/finetune_models/
├── condition_0_baseline/finetune_config.yaml
├── condition_1_metadata/finetune_config.yaml
├── condition_2_grimage_v2/finetune_config.yaml
├── ... (one per condition)
└── condition_7_episcores/finetune_config.yaml
```

### Step 3: Run Finetuning

Run for each condition. Can do sequentially or in parallel:

**Sequential (one at a time):**
```bash
for condition in {0..7}; do
  bash finetune_condition.sh \
    --condition $condition \
    --output-path /cluster/outputs \
    --pretrain-model /cluster/outputs/pretraining_dryrun
  echo "✓ Condition $condition done"
done
```

**Parallel (submit all at once via SLURM):**
```bash
for condition in {0..7}; do
  sbatch finetune_condition.sh \
    --condition $condition \
    --output-path /cluster/outputs \
    --pretrain-model /cluster/outputs/pretraining_dryrun
done

# Monitor
squeue -u $USER
```

**For a single condition:**
```bash
bash finetune_condition.sh \
  --condition 3 \
  --output-path /cluster/outputs \
  --pretrain-model /cluster/outputs/pretraining_dryrun
```

---

## Conditions Explained

| ID | Name | Features | Description |
|----|------|----------|-------------|
| 0 | baseline | 0 | EHR-only (control) |
| 1 | metadata | 4 | Age, sex, BMI, disease status |
| 2 | grimage_v2 | 2 | Epigenetic aging (altumage, dunedinpace) |
| 3 | systemsage | 11 | Organ-specific aging (11 systems) |
| 4 | deep_embeddings | 96 | MAPLE embeddings (32-dim × 3 modalities) |
| 5 | maple_predictions | 2 | CVD + T2D risk predictions |
| 6 | dmi | 1 | DNA methylation instability |
| 7 | episcores | 103 | Protein biomarkers (Gadd et al. 2022) |

---

## Value Injection Modes

In `create_condition_finetuning_configs.sh`, change `"mode"`:

```yaml
biological_features:
  - condition: 3
    path: /path/to/condition_3_systemsage_aligned.csv
    mode: concat  # Options: concat, FiLM, discrete, comb
```

- **concat** (default): Concatenate features after embeddings
- **FiLM**: Feature-wise linear modulation
- **discrete**: Treat as discrete variables
- **comb**: Combine multiple injection methods

---

## Checking Results

After each condition finishes:

```bash
# View metrics for condition 3
cat /cluster/outputs/finetune_models/condition_3_systemsage/avg_metrics.csv

# Compare across conditions
for cond in {0..7}; do
  echo "Condition $cond:"
  tail -1 /cluster/outputs/finetune_models/condition_${cond}_*/avg_metrics.csv 2>/dev/null || echo "  Not ready"
done
```

---

## Expected Outputs

After all conditions complete:

```
outputs/
├── finetune_models/
│   ├── ehr_only/
│   │   ├── fold_1/metrics.csv
│   │   ├── fold_2/metrics.csv
│   │   └── avg_metrics.csv  ← Baseline ROC-AUC
│   │
│   ├── condition_0_baseline/
│   │   └── avg_metrics.csv  ← Same as baseline
│   │
│   ├── condition_1_metadata/
│   │   └── avg_metrics.csv  ← Improvement from metadata
│   │
│   ├── condition_2_grimage_v2/
│   │   └── avg_metrics.csv  ← Improvement from aging
│   │
│   ├── condition_3_systemsage/
│   │   └── avg_metrics.csv  ← Improvement from organ systems
│   │
│   ├── condition_4_deep_embeddings/
│   │   └── avg_metrics.csv  ← Improvement from MAPLE embeddings
│   │
│   ├── condition_5_maple_predictions/
│   │   └── avg_metrics.csv  ← Improvement from disease risk
│   │
│   ├── condition_6_dmi/
│   │   └── avg_metrics.csv  ← Improvement from methylation
│   │
│   └── condition_7_episcores/
│       └── avg_metrics.csv  ← Improvement from proteins
```

---

## Analysis & Comparison

Compare C-index across conditions:

```python
import pandas as pd
from pathlib import Path

results = {}
for cond_id in range(8):
    # Find condition directory
    cond_dir = list(Path('/cluster/outputs/finetune_models').glob(f'condition_{cond_id}_*'))
    if cond_dir:
        metrics_file = cond_dir[0] / 'avg_metrics.csv'
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            results[f'Condition {cond_id}'] = df.iloc[0]

results_df = pd.DataFrame(results).T
print(results_df[['roc_auc', 'pr_auc', 'accuracy']])
```

---

## Troubleshooting

### Config not found
Make sure you ran `create_condition_finetuning_configs.sh` first.

### Value injection file not found
Check that alignment step created files in `outputs/value_injection_aligned/`.

### Shape mismatch in value injection
Make sure the aligned CSV has same number of rows as your training data.

### Out of memory
Reduce `batch_size` in the finetune_config.yaml (default: 8 → try 4).

---

## Next Steps

After all conditions complete:

1. **Compare results** across conditions to see which features help most
2. **Combine conditions** (e.g., condition_3 + condition_6 + condition_7) for ensemble
3. **Optimize injection mode** (try FiLM or comb instead of concat)
4. **Fine-tune hyperparameters** (learning rate, epochs, batch size)

The condition with highest ROC-AUC is your best model!
