#!/bin/bash
################################################################################
# Align Value Injection Files to MEDS Subject IDs
#
# Maps epimap sample IDs to MEDS subject IDs and creates aligned CSV files
# for each condition (0-7) ready for BONSAI finetuning.
#
# USAGE:
#   bash align_value_injection_to_meds.sh \
#     --value-injection-path /cluster/outputs/bonsai_value_injection \
#     --output-path /cluster/outputs \
#     --meds-path /cluster/data/meds_for_bonsai
#
################################################################################

set -e

VALUE_INJECTION_PATH=""
OUTPUT_PATH=""
MEDS_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --value-injection-path)
            VALUE_INJECTION_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --meds-path)
            MEDS_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$VALUE_INJECTION_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Usage: bash align_value_injection_to_meds.sh \\"
    echo "  --value-injection-path /path/to/value_injection \\"
    echo "  --output-path /path/to/outputs \\"
    echo "  --meds-path /path/to/meds (optional)"
    exit 1
fi

mkdir -p "$OUTPUT_PATH/value_injection_aligned"

echo "Aligning value injection files to MEDS subject IDs..."
echo ""

python3 << 'PYTHON_SCRIPT'
import pandas as pd
import numpy as np
from pathlib import Path

value_injection_path = Path("$VALUE_INJECTION_PATH")
output_path = Path("$OUTPUT_PATH")
meds_path = Path("$MEDS_PATH") if "$MEDS_PATH" else None

# Get list of MEDS subject IDs
meds_subject_ids = set()
if meds_path and meds_path.exists():
    for split in ["train", "tuning", "held_out"]:
        split_path = meds_path / split
        if split_path.exists():
            parquet_files = list(split_path.glob("*.parquet"))
            # Assume parquet filenames are subject IDs
            for f in parquet_files:
                meds_subject_ids.add(int(f.stem))
    print(f"Found {len(meds_subject_ids)} unique MEDS subject IDs")
else:
    print("MEDS path not provided - using sequential subject IDs")

# For each condition, load and align
conditions = [
    (0, "baseline"),
    (1, "metadata"),
    (2, "grimage_v2"),
    (3, "systemsage"),
    (4, "deep_embeddings"),
    (5, "maple_predictions"),
    (6, "dmi"),
    (7, "episcores"),
]

for cond_id, cond_name in conditions:
    csv_file = value_injection_path / f"condition_{cond_id}_{cond_name}.csv"

    if not csv_file.exists():
        print(f"⚠ Condition {cond_id} ({cond_name}): file not found")
        continue

    # Load value injection file
    df = pd.read_csv(csv_file, index_col=0)

    # If MEDS IDs provided, align to them
    if meds_subject_ids:
        # Get sample IDs from value injection file
        epimap_ids = set(int(idx.split('_')[-1]) for idx in df.index)

        # Create mapping: epimap sample number → MEDS subject ID
        meds_list = sorted(list(meds_subject_ids))
        if len(epimap_ids) <= len(meds_list):
            # Map sequentially
            id_mapping = {i: meds_list[i] for i in range(len(df))}
            df.index = [id_mapping[i] for i in range(len(df))]
        else:
            print(f"⚠ More samples ({len(epimap_ids)}) than MEDS subjects ({len(meds_list)})")

    # Save aligned file
    output_file = output_path / "value_injection_aligned" / f"condition_{cond_id}_{cond_name}_aligned.csv"
    df.to_csv(output_file)

    print(f"✓ Condition {cond_id} ({cond_name}): {len(df)} samples → {output_file.name}")

print("")
print("✓ Alignment complete")
print(f"  Output: {output_path}/value_injection_aligned/")

PYTHON_SCRIPT
