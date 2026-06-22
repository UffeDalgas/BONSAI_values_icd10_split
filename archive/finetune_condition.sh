#!/bin/bash
################################################################################
# Finetune with Value Injection for a Specific Condition
#
# Runs finetuning on a single condition (0-7) with biological features injected.
#
# USAGE:
#   bash finetune_condition.sh \
#     --condition 3 \
#     --output-path /cluster/outputs \
#     --pretrain-model /cluster/outputs/pretraining_dryrun
#
# CONDITIONS:
#   0: baseline (no features)
#   1: metadata (age, sex, labs)
#   2: grimage_v2 (aging)
#   3: systemsage (11 organ systems)
#   4: deep_embeddings (MAPLE, 96-dim)
#   5: maple_predictions (CVD, T2D risk)
#   6: dmi (methylation instability)
#   7: episcores (103 proteins)
#
################################################################################

#SBATCH --job-name=bonsai-finetune-condition
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/finetune_condition_%j.log

set -e

CONDITION=""
OUTPUT_PATH=""
PRETRAIN_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --condition)
            CONDITION="$2"
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CONDITION" ] || [ -z "$OUTPUT_PATH" ] || [ -z "$PRETRAIN_MODEL" ]; then
    echo "Usage: bash finetune_condition.sh \\"
    echo "  --condition <0-7> \\"
    echo "  --output-path /path/to/outputs \\"
    echo "  --pretrain-model /path/to/pretrained"
    exit 1
fi

# Map condition ID to name
case $CONDITION in
    0) COND_NAME="baseline" ;;
    1) COND_NAME="metadata" ;;
    2) COND_NAME="grimage_v2" ;;
    3) COND_NAME="systemsage" ;;
    4) COND_NAME="deep_embeddings" ;;
    5) COND_NAME="maple_predictions" ;;
    6) COND_NAME="dmi" ;;
    7) COND_NAME="episcores" ;;
    *) echo "Invalid condition: $CONDITION"; exit 1 ;;
esac

CONFIG_PATH="$OUTPUT_PATH/finetune_models/condition_${CONDITION}_${COND_NAME}/finetune_config.yaml"
MODEL_OUTPUT="$OUTPUT_PATH/finetune_models/condition_${CONDITION}_${COND_NAME}"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "✗ Config not found: $CONFIG_PATH"
    echo "  Run: bash create_condition_finetuning_configs.sh --output-path $OUTPUT_PATH"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          BONSAI CONDITION FINETUNING - Condition $CONDITION              ║"
echo "║                    ($COND_NAME)"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Condition ID:     $CONDITION"
echo "  Condition name:   $COND_NAME"
echo "  Config file:      $CONFIG_PATH"
echo "  Output path:      $MODEL_OUTPUT"
echo "  Pretrain model:   $PRETRAIN_MODEL"
echo ""

# Run finetuning
python -m corebehrt.main.finetune_cv --config "$CONFIG_PATH"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          ✓ CONDITION $CONDITION FINETUNING COMPLETE                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results:"
echo "  Checkpoints: $MODEL_OUTPUT/fold_*/checkpoints/"
echo "  Metrics:     $MODEL_OUTPUT/fold_*/metrics.csv"
echo "  Average:     $MODEL_OUTPUT/avg_metrics.csv"
echo ""
