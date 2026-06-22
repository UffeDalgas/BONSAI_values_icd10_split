# Quick Start: Baseline Finetuning in One Command

Everything you need is in **`baseline_finetuning_complete.sh`**

## Prerequisites

You should already have:
```
/cluster/outputs/
├── pretraining_dryrun/              ← Pretrained model checkpoint
├── features/                         ← Tokenized features (from create_data)
├── tokenized/
│   ├── pids_tuning.pt               ← Required: tuning cohort IDs
│   ├── pids_train.pt
│   └── pids_held_out.pt

/cluster/data/
└── meds_for_bonsai/                 ← Your MEDS parquet files
    ├── train/*.parquet
    ├── tuning/*.parquet
    └── held_out/*.parquet
```

If you don't have `features/` and `tokenized/`, run `create_data` first:
```bash
python -m corebehrt.main.create_data --config corebehrt/configs/create_data.yaml
```

## Single Command Run

```bash
bash baseline_finetuning_complete.sh \
  --meds-path /cluster/data/meds_for_bonsai \
  --output-path /cluster/outputs \
  --pretrain-model /cluster/outputs/pretraining_dryrun
```

That's it! The script will:
1. ✅ Scan MEDS for DOD codes → mortality labels
2. ✅ Create CV folds from tuning cohort
3. ✅ Prepare patient sequences for finetuning
4. ✅ Train baseline model (EHR-only, no biological features)
5. ✅ Evaluate on validation folds
6. ✅ Report results

## Submit to SLURM

```bash
sbatch baseline_finetuning_complete.sh \
  --meds-path /cluster/data/meds_for_bonsai \
  --output-path /cluster/outputs \
  --pretrain-model /cluster/outputs/pretraining_dryrun
```

Default SLURM settings:
- Partition: `gpu`
- GPUs: 1
- Memory: 32 GB
- Time: 04:00:00 (4 hours)
- CPUs: 4

Override if needed:
```bash
sbatch --partition=gpu_v100 \
       --time=08:00:00 \
       baseline_finetuning_complete.sh \
  --meds-path /cluster/data/meds_for_bonsai \
  --output-path /cluster/outputs \
  --pretrain-model /cluster/outputs/pretraining_dryrun
```

## Monitor Execution

```bash
# Watch logs in real-time
tail -f logs/baseline_finetune_*.log

# Check SLURM job status
squeue -u $USER

# Cancel if needed
scancel JOB_ID
```

## Skip Already-Completed Steps

If the script fails mid-way and you fix it, resume without re-running completed steps:

```bash
# Skip outcome creation (if MORTALITY.csv exists)
bash baseline_finetuning_complete.sh \
  --meds-path /cluster/data/meds_for_bonsai \
  --output-path /cluster/outputs \
  --pretrain-model /cluster/outputs/pretraining_dryrun \
  --skip-outcomes

# Skip both outcomes and cohort selection
bash baseline_finetuning_complete.sh \
  ... \
  --skip-outcomes \
  --skip-cohort

# Skip everything except finetuning (just train)
bash baseline_finetuning_complete.sh \
  ... \
  --skip-outcomes \
  --skip-cohort \
  --skip-prepare

# Evaluate only (model already trained)
bash baseline_finetuning_complete.sh \
  ... \
  --skip-outcomes \
  --skip-cohort \
  --skip-prepare \
  --skip-finetune
```

## What Gets Created

```
/cluster/outputs/
├── outcomes/
│   └── MORTALITY.csv                 # Mortality labels from DOD codes
│
├── cohort/finetune/
│   ├── folds.pt                      # CV fold assignments
│   ├── pids.pt                       # Subject IDs in cohort
│   ├── index_dates.csv               # Index dates per subject
│   └── ...
│
├── finetuning/processed_data_no_values/
│   ├── folds.pt                      # Folds for training
│   ├── patients.pt                   # Prepared sequences
│   ├── data_train.pt                 # Training split
│   └── data_val.pt                   # Validation split
│
├── finetune_models/ehr_only/
│   ├── fold_1/
│   │   ├── checkpoints/
│   │   │   ├── epoch_0.pt            # Best checkpoint
│   │   │   └── ...
│   │   ├── metrics.csv               # Per-epoch metrics
│   │   ├── train_pids.pt
│   │   └── val_pids.pt
│   │
│   ├── fold_2/
│   │   ├── checkpoints/
│   │   └── metrics.csv
│   │
│   └── avg_metrics.csv               # Average across folds
│
└── logs/
    └── baseline_finetune_*.log       # Execution log
```

## Check Results

After completion:

```bash
# View final metrics
cat /cluster/outputs/finetune_models/ehr_only/avg_metrics.csv

# View fold-specific metrics
cat /cluster/outputs/finetune_models/ehr_only/fold_1/metrics.csv
cat /cluster/outputs/finetune_models/ehr_only/fold_2/metrics.csv

# Load trained model checkpoint
python -c "
import torch
model_path = '/cluster/outputs/finetune_models/ehr_only/fold_1/checkpoints/epoch_0.pt'
checkpoint = torch.load(model_path)
print('Model loaded, keys:', list(checkpoint.keys()))
"
```

## Expected Output Format

```
[1/5] Create Outcomes (scan MEDS for DOD codes)
      ✓ Create Outcomes completed

[2/5] Select Cohort (create CV folds)
      ✓ Select Cohort completed

[3/5] Prepare Finetuning Data
      ✓ Prepare Finetuning Data completed

[4/5] Run Baseline Finetuning (EHR-only, no features)
      ✓ Run Baseline Finetuning completed

[5/5] Evaluate and Report Results
Final Metrics (Averaged across folds):
─────────────────────────────────────────────
val_loss,roc_auc,pr_auc,accuracy
0.456,0.823,0.891,0.785
─────────────────────────────────────────────

✓ BASELINE FINETUNING COMPLETE
```

## Troubleshooting

### Out of Memory
Edit the script and reduce batch size:
```bash
'trainer_args': {
    'batch_size': 4,           # ← Reduce from 8
    'effective_batch_size': 4,
    ...
}
```

### GPU Not Available
```bash
# Run on CPU instead
CUDA_VISIBLE_DEVICES="" bash baseline_finetuning_complete.sh ...
```

### Python Module Not Found
```bash
# Ensure you're in the right environment
source /path/to/venv/bin/activate
python -c "import corebehrt; print('OK')"

# Then run the script
bash baseline_finetuning_complete.sh ...
```

### MEDS Path Error
Check structure:
```bash
ls /cluster/data/meds_for_bonsai/train/*.parquet | head -3
ls /cluster/data/meds_for_bonsai/tuning/*.parquet | head -3
ls /cluster/data/meds_for_bonsai/held_out/*.parquet | head -3
```

## Next: Value Injection Finetuning

Once baseline completes successfully, create conditions 0-7 for value injection:

```bash
# Copy value injection files to cluster
scp -r /local/path/bonsai_value_injection/ \
  user@cluster:/cluster/outputs/

# Generate configs for each condition
python create_value_injection_finetuning_configs.py \
  --output-path /cluster/outputs \
  --conditions 0 1 2 3 4 5 6 7
```

Then run finetuning for each condition (repeat script for each).

## That's It!

One script, one command, end-to-end baseline finetuning with evaluation. 🚀
