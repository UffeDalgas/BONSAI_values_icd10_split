#!/usr/bin/env python3
"""
Create finetuning configs for each value injection condition (0-7).

These configs define which biological features to inject during finetuning.

Conditions:
  0: baseline (empty - EHR only)
  1: metadata (age, sex, labs)
  2: grimage_v2 (aging markers)
  3: systemsage (11 organ systems)
  4: deep_embeddings (96-dim MAPLE)
  5: maple_predictions (CVD, T2D risk)
  6: dmi (methylation instability)
  7: episcores (103 proteins)

Usage:
    python create_value_injection_finetuning_configs.py \
      --output-path /cluster/outputs \
      --conditions 0 1 2 3 4 5 6 7 \
      --value-injection-path /cluster/value_injection_files
"""

import argparse
import logging
from pathlib import Path
import yaml
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("create_configs")

# Define which features are in each condition
CONDITION_FEATURES = {
    0: {
        "name": "baseline",
        "description": "EHR-only control (no biological features)",
        "features": []
    },
    1: {
        "name": "metadata",
        "description": "Patient demographics and labs (age, sex, BMI, etc.)",
        "features": ["age", "sex", "BMI", "disease_status"]
    },
    2: {
        "name": "grimage_v2",
        "description": "Epigenetic aging variants (GrimAge V2 + alternatives)",
        "features": ["altumage", "dunedinpace"]
    },
    3: {
        "name": "systemsage",
        "description": "Organ-system aging (11 systems)",
        "features": [
            "systemsage", "systemsageblood", "systemsagebrain",
            "systemsageheart", "systemsagehormone", "systemsageimmune",
            "systemsageinflammation", "systemsagekidney", "systemsageliver",
            "systemsagelung", "systemsagemetabolic"
        ]
    },
    4: {
        "name": "deep_embeddings",
        "description": "Deep embeddings from MAPLE (96-dim)",
        "features": ["maple_cvd", "maple_t2d", "maple_age"]  # 32 dims each
    },
    5: {
        "name": "maple_predictions",
        "description": "MAPLE disease risk predictions (CVD, T2D)",
        "features": ["cvd_risk", "t2d_risk"]
    },
    6: {
        "name": "dmi",
        "description": "DNA Methylation Instability (1 feature)",
        "features": ["dmi"]
    },
    7: {
        "name": "episcores",
        "description": "Epigenetic protein biomarkers (103 proteins)",
        "features": []  # 103 specific proteins, too many to list
    }
}


def create_config(condition_id: int, output_dir: Path, value_injection_path: Path):
    """Create finetuning config for a specific condition."""

    condition_info = CONDITION_FEATURES[condition_id]
    condition_name = condition_info["name"]

    # Load baseline config as template
    baseline_config_path = Path(__file__).parent / "outputs/finetune_models/ehr_only/finetune_config.yaml"

    if not baseline_config_path.exists():
        logger.error(f"Baseline config not found: {baseline_config_path}")
        return False

    with open(baseline_config_path) as f:
        cfg = yaml.safe_load(f)

    # Update for this condition
    cfg["biological_features"] = [
        {
            "condition": condition_id,
            "path": str(value_injection_path / f"condition_{condition_id}_{condition_name}.csv")
        }
    ]

    cfg["paths"]["model"] = f"./outputs/finetune_models/condition_{condition_id}_{condition_name}"

    # Create output directory for this condition
    condition_output_dir = output_dir / f"condition_{condition_id}_{condition_name}"
    condition_output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = condition_output_dir / "finetune_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    logger.info(f"✓ Condition {condition_id}: {condition_name}")
    logger.info(f"  Description: {condition_info['description']}")
    logger.info(f"  Features: {len(condition_info['features'])} features")
    logger.info(f"  Config: {config_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create finetuning configs for value injection conditions"
    )
    parser.add_argument(
        "--output-path",
        default="./outputs",
        help="Output directory (default: ./outputs)"
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        type=int,
        choices=range(8),
        default=list(range(8)),
        help="Condition IDs to create (default: 0 1 2 3 4 5 6 7)"
    )
    parser.add_argument(
        "--value-injection-path",
        default="./outputs/bonsai_value_injection",
        help="Path to value injection files (default: ./outputs/bonsai_value_injection)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_path) / "finetune_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    value_injection_path = Path(args.value_injection_path)
    if not value_injection_path.exists():
        logger.warning(f"Value injection path does not exist: {value_injection_path}")
        logger.info("Value injection files should be in this path")

    logger.info("="*80)
    logger.info("Creating Finetuning Configs for Value Injection Conditions")
    logger.info("="*80)

    success_count = 0
    for condition_id in sorted(args.conditions):
        if create_config(condition_id, output_dir, value_injection_path):
            success_count += 1

    logger.info("="*80)
    logger.info(f"✅ Created {success_count}/{len(args.conditions)} configs")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("Next step: Run finetuning for each condition")
    logger.info("")
    logger.info("Example:")
    logger.info("  sbatch submit_value_injection_finetuning.sh \\")
    logger.info("    --condition 3 \\")
    logger.info("    --output-path /cluster/outputs")


if __name__ == "__main__":
    main()
