#!/bin/bash
################################################################################
# Create Finetuning Configs for Each Value Injection Condition
#
# Generates finetune_config.yaml for each condition (0-7) that specifies:
# - Which biological features to inject
# - Model output path
# - All training hyperparameters
#
# USAGE:
#   bash create_condition_finetuning_configs.sh \
#     --output-path /cluster/outputs \
#     --value-injection-aligned-path /cluster/outputs/value_injection_aligned \
#     --conditions 0 1 2 3 4 5 6 7
#
################################################################################

set -e

OUTPUT_PATH=""
VALUE_INJECTION_ALIGNED_PATH=""
CONDITIONS=(0 1 2 3 4 5 6 7)

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --value-injection-aligned-path)
            VALUE_INJECTION_ALIGNED_PATH="$2"
            shift 2
            ;;
        --conditions)
            shift
            CONDITIONS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                CONDITIONS+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$OUTPUT_PATH" ]; then
    echo "Usage: bash create_condition_finetuning_configs.sh \\"
    echo "  --output-path /path/to/outputs \\"
    echo "  --value-injection-aligned-path /path/to/aligned_files \\"
    echo "  --conditions 0 1 2 3 4 5 6 7"
    exit 1
fi

mkdir -p "$OUTPUT_PATH/finetune_models"

echo "Creating finetuning configs for conditions: ${CONDITIONS[@]}"
echo ""

python3 << 'PYTHON_SCRIPT'
import json
import yaml
from pathlib import Path

output_path = Path("$OUTPUT_PATH")
vi_aligned_path = Path("$VALUE_INJECTION_ALIGNED_PATH")
conditions = [int(c) for c in "$CONDITIONS".split()]

# Condition metadata
condition_info = {
    0: ("baseline", "EHR-only (no biological features)"),
    1: ("metadata", "Patient metadata (age, sex, labs)"),
    2: ("grimage_v2", "Epigenetic aging markers"),
    3: ("systemsage", "System-specific aging (11 organ systems)"),
    4: ("deep_embeddings", "Deep embeddings (MAPLE, 96-dim)"),
    5: ("maple_predictions", "MAPLE disease predictions"),
    6: ("dmi", "DNA Methylation Instability"),
    7: ("episcores", "Epigenetic protein biomarkers (103)"),
}

# Load baseline config as template
baseline_config_path = output_path / "finetune_models/ehr_only/finetune_config.yaml"
if not baseline_config_path.exists():
    print(f"✗ Baseline config not found: {baseline_config_path}")
    exit(1)

with open(baseline_config_path) as f:
    baseline_cfg = yaml.safe_load(f)

# Create config for each condition
for cond_id in conditions:
    cond_name, cond_desc = condition_info[cond_id]

    # Copy baseline config
    cfg = baseline_cfg.copy()

    # Update for this condition
    if cond_id == 0:
        # Baseline: no features
        cfg["biological_features"] = []
    else:
        # Other conditions: specify value injection file
        vi_file = vi_aligned_path / f"condition_{cond_id}_{cond_name}_aligned.csv"
        cfg["biological_features"] = [
            {
                "condition": cond_id,
                "path": str(vi_file),
                "mode": "concat"  # How to inject: concat, FiLM, discrete, comb
            }
        ]

    # Update output path
    cfg["paths"]["model"] = str(
        output_path / f"finetune_models/condition_{cond_id}_{cond_name}"
    )

    # Create output directory
    cond_output_dir = output_path / f"finetune_models/condition_{cond_id}_{cond_name}"
    cond_output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = cond_output_dir / "finetune_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"✓ Condition {cond_id}: {cond_name}")
    print(f"  Description: {cond_desc}")
    print(f"  Config: {config_path}")
    print()

PYTHON_SCRIPT

echo "✓ All condition configs created"
echo "  Output: $OUTPUT_PATH/finetune_models/condition_*/finetune_config.yaml"
echo ""
echo "Next step: Run finetuning for each condition"
echo "  bash finetune_condition.sh --condition 0 --output-path $OUTPUT_PATH"
