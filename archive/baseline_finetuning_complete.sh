#!/bin/bash
################################################################################
# BONSAI Baseline Finetuning: Complete Pipeline in One Script
#
# This script runs the entire baseline finetuning pipeline:
#   1. Create outcomes (scan MEDS for DOD codes)
#   2. Select cohort (create folds)
#   3. Prepare finetuning data
#   4. Run baseline finetuning
#   5. Evaluate and report results
#
# USAGE:
#   bash baseline_finetuning_complete.sh \
#     --meds-path /cluster/data/meds_for_bonsai \
#     --output-path /cluster/outputs \
#     --pretrain-model /cluster/outputs/pretraining_dryrun
#
# REQUIRED:
#   --meds-path             Path to MEDS data (contains train/, tuning/, held_out/)
#   --output-path           Where to save results
#   --pretrain-model        Path to pretrained BERT checkpoint
#
# OPTIONAL:
#   --skip-outcomes         Skip outcome creation (if MORTALITY.csv exists)
#   --skip-cohort           Skip cohort selection (if folds exist)
#   --skip-prepare          Skip data preparation (if prepared data exists)
#   --skip-finetune         Skip finetuning (just evaluate if checkpoints exist)
#   --job-name              SLURM job name (default: baseline-finetune)
#   --partition             SLURM partition (default: gpu)
#   --gpus                  Number of GPUs (default: 1)
#   --time                  Wall time in HH:MM:SS (default: 04:00:00)
#   --mem                   Memory in GB (default: 32)
#   --cpus                  CPUs per task (default: 4)
#
# EXAMPLE - LOCAL RUN:
#   bash baseline_finetuning_complete.sh \
#     --meds-path /local/data/meds \
#     --output-path /local/outputs \
#     --pretrain-model /local/outputs/pretraining_dryrun
#
# EXAMPLE - SLURM SUBMISSION:
#   sbatch baseline_finetuning_complete.sh \
#     --meds-path /cluster/data/meds \
#     --output-path /cluster/outputs \
#     --pretrain-model /cluster/outputs/pretraining_dryrun \
#     --time 08:00:00
#
################################################################################

#SBATCH --job-name=baseline-finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/baseline_finetune_%j.log

set -e

# ============================================================================
# COLORS FOR OUTPUT
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# PARSE COMMAND-LINE ARGUMENTS
# ============================================================================
MEDS_PATH=""
OUTPUT_PATH=""
PRETRAIN_MODEL=""
SKIP_OUTCOMES=false
SKIP_COHORT=false
SKIP_PREPARE=false
SKIP_FINETUNE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --meds-path)
            MEDS_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --pretrain-model)
            PRETRAIN_MODEL="$2"
            shift 2
            ;;
        --skip-outcomes)
            SKIP_OUTCOMES=true
            shift
            ;;
        --skip-cohort)
            SKIP_COHORT=true
            shift
            ;;
        --skip-prepare)
            SKIP_PREPARE=true
            shift
            ;;
        --skip-finetune)
            SKIP_FINETUNE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# VALIDATE ARGUMENTS
# ============================================================================
if [ -z "$MEDS_PATH" ] || [ -z "$OUTPUT_PATH" ] || [ -z "$PRETRAIN_MODEL" ]; then
    echo -e "${RED}ERROR: Missing required arguments${NC}"
    echo "Usage: bash baseline_finetuning_complete.sh \\"
    echo "  --meds-path /path/to/meds \\"
    echo "  --output-path /path/to/outputs \\"
    echo "  --pretrain-model /path/to/pretrained/model"
    exit 1
fi

if [ ! -d "$MEDS_PATH" ]; then
    echo -e "${RED}ERROR: MEDS path does not exist: $MEDS_PATH${NC}"
    exit 1
fi

if [ ! -d "$PRETRAIN_MODEL" ]; then
    echo -e "${RED}ERROR: Pretrain model path does not exist: $PRETRAIN_MODEL${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_PATH/logs"
mkdir -p "$OUTPUT_PATH/outcomes"
mkdir -p "$OUTPUT_PATH/cohort/finetune"
mkdir -p "$OUTPUT_PATH/finetuning"

# ============================================================================
# PRINT HEADER
# ============================================================================
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          BONSAI BASELINE FINETUNING - COMPLETE PIPELINE        ║${NC}"
echo -e "${BLUE}║                  Mortality Prediction (EHR-only)               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  MEDS data:        $MEDS_PATH"
echo "  Output path:      $OUTPUT_PATH"
echo "  Pretrain model:   $PRETRAIN_MODEL"
echo "  Skip outcomes:    $SKIP_OUTCOMES"
echo "  Skip cohort:      $SKIP_COHORT"
echo "  Skip prepare:     $SKIP_PREPARE"
echo "  Skip finetune:    $SKIP_FINETUNE"
echo ""

# Helper function to run steps
run_step() {
    local step_name="$1"
    local step_num="$2"
    local total_steps="$3"
    shift 3
    local cmd=("$@")

    echo -e "${YELLOW}[${step_num}/${total_steps}] $step_name${NC}"
    echo -e "${BLUE}Command: ${cmd[@]}${NC}"

    if "${cmd[@]}"; then
        echo -e "${GREEN}✓ $step_name completed${NC}"
    else
        echo -e "${RED}✗ $step_name FAILED${NC}"
        exit 1
    fi
    echo ""
}

# ============================================================================
# STEP 1: CREATE OUTCOMES (Scan MEDS for DOD codes)
# ============================================================================
if [ "$SKIP_OUTCOMES" = false ]; then
    MORTALITY_CSV="$OUTPUT_PATH/outcomes/MORTALITY.csv"
    if [ -f "$MORTALITY_CSV" ]; then
        echo -e "${YELLOW}[1/5] Create Outcomes${NC}"
        echo -e "${YELLOW}      MORTALITY.csv already exists, skipping...${NC}"
        echo ""
    else
        run_step "Create Outcomes (scan MEDS for DOD codes)" 1 5 \
            python -c "
import logging
import sys
from pathlib import Path
import yaml
import tempfile
from corebehrt.modules.setup.config import load_config
from corebehrt.main.create_outcomes import main_data

# Create temporary config
cfg = {
    'logging': {'level': 'INFO', 'path': '$OUTPUT_PATH/logs'},
    'paths': {
        'data': '$MEDS_PATH',
        'splits': ['train', 'tuning', 'held_out'],
        'outcomes': '$OUTPUT_PATH/outcomes',
        'features': '$OUTPUT_PATH/features'
    },
    'outcomes': {
        'MORTALITY': {
            'type': ['code'],
            'match': [['DOD']],
            'match_how': 'exact',
            'case_sensitive': True
        }
    }
}

# Write temp config
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    yaml.dump(cfg, f)
    temp_config = f.name

try:
    main_data(temp_config)
finally:
    Path(temp_config).unlink()
"
    fi
else
    echo -e "${YELLOW}[1/5] Create Outcomes${NC}"
    echo -e "${YELLOW}      Skipped (--skip-outcomes)${NC}"
    echo ""
fi

# ============================================================================
# STEP 2: SELECT COHORT (Create folds from tuning split)
# ============================================================================
if [ "$SKIP_COHORT" = false ]; then
    FOLDS_FILE="$OUTPUT_PATH/cohort/finetune/folds.pt"
    if [ -f "$FOLDS_FILE" ]; then
        echo -e "${YELLOW}[2/5] Select Cohort${NC}"
        echo -e "${YELLOW}      folds.pt already exists, skipping...${NC}"
        echo ""
    else
        run_step "Select Cohort (create CV folds)" 2 5 \
            python -c "
import logging
import sys
from pathlib import Path
import yaml
import tempfile
from corebehrt.main.select_cohort import main_select_cohort

# Create temporary config
cfg = {
    'logging': {'level': 'INFO', 'path': '$OUTPUT_PATH/logs'},
    'paths': {
        'features': '$OUTPUT_PATH/features',
        'tokenized': '$OUTPUT_PATH/tokenized',
        'initial_pids': 'pids_tuning.pt',
        'outcomes': '$OUTPUT_PATH/outcomes',
        'outcome': 'MORTALITY.csv',
        'cohort': '$OUTPUT_PATH/cohort/finetune'
    },
    'selection': {
        'exclude_prior_outcomes': False,
        'exposed_only': False,
        'age': {'min_years': 18, 'max_years': 120}
    },
    'index_date': {'mode': 'relative', 'relative': {'n_hours_from_exposure': -24}},
    'cv_folds': 2,
    'val_ratio': 0.1,
    'test_ratio': 0.1
}

# Write temp config
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    yaml.dump(cfg, f)
    temp_config = f.name

try:
    main_select_cohort(temp_config)
finally:
    Path(temp_config).unlink()
"
    fi
else
    echo -e "${YELLOW}[2/5] Select Cohort${NC}"
    echo -e "${YELLOW}      Skipped (--skip-cohort)${NC}"
    echo ""
fi

# ============================================================================
# STEP 3: PREPARE FINETUNING DATA
# ============================================================================
if [ "$SKIP_PREPARE" = false ]; then
    PREPARED_FOLDS="$OUTPUT_PATH/finetuning/processed_data_no_values/folds.pt"
    if [ -f "$PREPARED_FOLDS" ]; then
        echo -e "${YELLOW}[3/5] Prepare Finetuning Data${NC}"
        echo -e "${YELLOW}      prepared data already exists, skipping...${NC}"
        echo ""
    else
        run_step "Prepare Finetuning Data" 3 5 \
            python -c "
import logging
import sys
from pathlib import Path
import yaml
import tempfile
from corebehrt.main.prepare_training_data import main_prepare_data

# Create temporary config
cfg = {
    'logging': {'level': 'INFO', 'path': '$OUTPUT_PATH/logs'},
    'paths': {
        'features': '$OUTPUT_PATH/features',
        'tokenized': '$OUTPUT_PATH/tokenized',
        'cohort': '$OUTPUT_PATH/cohort/finetune',
        'outcomes': '$OUTPUT_PATH/outcomes',
        'outcome': 'MORTALITY.csv',
        'prepared_data': '$OUTPUT_PATH/finetuning/processed_data_no_values'
    },
    'data': {
        'type': 'finetune',
        'truncation_len': 30,
        'min_len': 2
    },
    'outcome': {
        'n_hours_censoring': -10,
        'n_hours_start_follow_up': 1,
        'n_hours_end_follow_up': None
    },
    'concept_pattern_hours_delay': {'^D': 72}
}

# Write temp config
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    yaml.dump(cfg, f)
    temp_config = f.name

try:
    main_prepare_data(temp_config)
finally:
    Path(temp_config).unlink()
"
    fi
else
    echo -e "${YELLOW}[3/5] Prepare Finetuning Data${NC}"
    echo -e "${YELLOW}      Skipped (--skip-prepare)${NC}"
    echo ""
fi

# ============================================================================
# STEP 4: RUN BASELINE FINETUNING
# ============================================================================
if [ "$SKIP_FINETUNE" = false ]; then
    CHECKPOINT_DIR="$OUTPUT_PATH/finetune_models/ehr_only/fold_1/checkpoints"
    if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR)" ]; then
        echo -e "${YELLOW}[4/5] Run Baseline Finetuning${NC}"
        echo -e "${YELLOW}      Checkpoints already exist, skipping...${NC}"
        echo ""
    else
        run_step "Run Baseline Finetuning (EHR-only, no features)" 4 5 \
            python -c "
import logging
import sys
from pathlib import Path
import yaml
import tempfile
from corebehrt.main.finetune_cv import main_finetune

# Create temporary config
cfg = {
    'biological_features': [],
    'evaluate': False,
    'logging': {'level': 'INFO', 'path': '$OUTPUT_PATH/logs'},
    'metrics': {
        'accuracy': {
            '_target_': 'corebehrt.modules.monitoring.metrics.Accuracy',
            'threshold': 0.5
        },
        'pr_auc': {
            '_target_': 'corebehrt.modules.monitoring.metrics.PR_AUC'
        },
        'roc_auc': {
            '_target_': 'corebehrt.modules.monitoring.metrics.ROC_AUC'
        }
    },
    'model': {'cls': 'default', 'value_embedding_mode': 'concat'},
    'optimizer': {'eps': 1.0e-06, 'lr': 0.0005},
    'paths': {
        'model': '$OUTPUT_PATH/finetune_models/ehr_only',
        'prepared_data': '$OUTPUT_PATH/finetuning/processed_data_no_values',
        'pretrain_model': '$PRETRAIN_MODEL'
    },
    'scheduler': {
        '_target_': 'transformers.get_linear_schedule_with_warmup',
        'num_training_steps': 20,
        'num_warmup_steps': 5
    },
    'trainer_args': {
        'batch_size': 8,
        'checkpoint_frequency': 1,
        'early_stopping': 2,
        'effective_batch_size': 8,
        'epochs': 2,
        'gradient_clip': {'clip_value': 1.0},
        'info': True,
        'n_layers_to_freeze': 1,
        'shuffle': True,
        'stopping_criterion': 'roc_auc',
        'val_batch_size': 8
    }
}

# Write temp config
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    yaml.dump(cfg, f)
    temp_config = f.name

try:
    main_finetune(temp_config)
finally:
    Path(temp_config).unlink()
"
    fi
else
    echo -e "${YELLOW}[4/5] Run Baseline Finetuning${NC}"
    echo -e "${YELLOW}      Skipped (--skip-finetune)${NC}"
    echo ""
fi

# ============================================================================
# STEP 5: EVALUATE AND REPORT RESULTS
# ============================================================================
echo -e "${YELLOW}[5/5] Evaluate and Report Results${NC}"

RESULTS_DIR="$OUTPUT_PATH/finetune_models/ehr_only"

echo -e "${GREEN}Results Directory:${NC} $RESULTS_DIR"
echo ""

# Check if metrics exist
if [ -f "$RESULTS_DIR/avg_metrics.csv" ]; then
    echo -e "${GREEN}Final Metrics (Averaged across folds):${NC}"
    echo "─────────────────────────────────────────────"
    cat "$RESULTS_DIR/avg_metrics.csv"
    echo "─────────────────────────────────────────────"
    echo ""
else
    echo -e "${YELLOW}Average metrics file not found yet${NC}"
    echo ""
fi

# List folds with metrics
if [ -d "$RESULTS_DIR" ]; then
    echo -e "${GREEN}Per-Fold Metrics:${NC}"
    for fold_dir in "$RESULTS_DIR"/fold_*; do
        if [ -d "$fold_dir" ]; then
            fold_name=$(basename "$fold_dir")
            if [ -f "$fold_dir/metrics.csv" ]; then
                echo ""
                echo -e "${BLUE}$fold_name:${NC}"
                head -5 "$fold_dir/metrics.csv" | tail -1
            fi
        fi
    done
    echo ""
fi

# Summary statistics
echo -e "${GREEN}File Structure:${NC}"
echo "  Outcomes:       $OUTPUT_PATH/outcomes/MORTALITY.csv"
echo "  Folds:          $OUTPUT_PATH/cohort/finetune/folds.pt"
echo "  Prepared data:  $OUTPUT_PATH/finetuning/processed_data_no_values/"
echo "  Model/metrics:  $RESULTS_DIR/"
echo ""

echo -e "${GREEN}Total file sizes:${NC}"
du -sh "$OUTPUT_PATH"/* 2>/dev/null | tail -10
echo ""

# ============================================================================
# COMPLETION
# ============================================================================
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}                                                                ${BLUE}║${NC}"
if [ -f "$RESULTS_DIR/avg_metrics.csv" ]; then
    echo -e "${BLUE}║${NC}  ${GREEN}✓ BASELINE FINETUNING COMPLETE${NC}                           ${BLUE}║${NC}"
else
    echo -e "${BLUE}║${NC}  ${YELLOW}⚠ FINETUNING COMPLETE (awaiting metric aggregation)${NC}  ${BLUE}║${NC}"
fi
echo -e "${BLUE}║${NC}                                                                ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Check metrics: cat $RESULTS_DIR/avg_metrics.csv"
echo "  2. Review per-fold results: ls $RESULTS_DIR/fold_*/metrics.csv"
echo "  3. Load trained model: $RESULTS_DIR/fold_1/checkpoints/"
echo ""
