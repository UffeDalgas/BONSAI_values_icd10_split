#!/bin/bash
#SBATCH --job-name=bonsai-baseline-finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/baseline_finetune_%j.log

# ============================================================================
# BONSAI Baseline Finetuning Job Script
# ============================================================================
#
# Usage:
#   sbatch submit_baseline_finetuning.sh \
#     --meds-path /cluster/path/meds_for_bonsai \
#     --output-path /cluster/path/outputs
#
# Or run directly:
#   bash submit_baseline_finetuning.sh \
#     --meds-path /path/to/meds \
#     --output-path /path/to/outputs
#
# ============================================================================

# Exit on error
set -e

# Default paths (override with arguments)
MEDS_PATH=""
OUTPUT_PATH="./outputs"
FEATURES_PATH="./outputs/features"
TOKENIZED_PATH="./outputs/tokenized"
PRETRAIN_MODEL="./outputs/pretraining_dryrun"
SKIP_OUTCOMES=false
SKIP_COHORT=false

# Parse arguments
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
        --features-path)
            FEATURES_PATH="$2"
            shift 2
            ;;
        --tokenized-path)
            TOKENIZED_PATH="$2"
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MEDS_PATH" ]; then
    echo "Error: --meds-path is required"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_PATH/logs"

# Load environment (adjust as needed for your cluster)
# module load cuda/12.0
# module load gcc/11.0
# module load python/3.11
source /path/to/venv/bin/activate  # Adjust this path

# Print job info
echo "========================================================================"
echo "BONSAI Baseline Finetuning"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "MEDS path: $MEDS_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Features path: $FEATURES_PATH"
echo "Tokenized path: $TOKENIZED_PATH"
echo "Pretrain model: $PRETRAIN_MODEL"
echo "Skip outcomes: $SKIP_OUTCOMES"
echo "Skip cohort: $SKIP_COHORT"
echo "========================================================================"

# Run pipeline
PYTHON_ARGS=(
    "--meds-path" "$MEDS_PATH"
    "--output-path" "$OUTPUT_PATH"
    "--features-path" "$FEATURES_PATH"
    "--tokenized-path" "$TOKENIZED_PATH"
    "--pretrain-model" "$PRETRAIN_MODEL"
)

if [ "$SKIP_OUTCOMES" = true ]; then
    PYTHON_ARGS+=("--skip-outcomes")
fi

if [ "$SKIP_COHORT" = true ]; then
    PYTHON_ARGS+=("--skip-cohort")
fi

python run_baseline_finetuning.py "${PYTHON_ARGS[@]}"

echo ""
echo "========================================================================"
echo "✅ Finetuning Complete"
echo "========================================================================"
echo "Results: $OUTPUT_PATH/finetune_models/ehr_only"
