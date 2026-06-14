#!/usr/bin/env python3
"""
Generate 40 ablation configuration files: 8 feature subsets × 5 injection methods.

This script creates YAML config files for systematic comparison of:
- Feature subsets: EHR, metadata, GrimAge, SystemsAge, CpGPT, MAPLE, MethylGPT, All
- Injection methods: concat, FiLM, discrete, comb, comb_binning

Usage:
    python scripts/generate_ablation_configs.py

Output:
    corebehrt/configs/ablation_*.yaml (40 files)
"""

import logging
import yaml
from pathlib import Path
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("generate_ablation_configs")


# Feature subsets
FEATURE_SUBSETS = [
    {
        "name": "ehr_only",
        "display_name": "EHR Only",
        "features": [],
        "description": "Baseline: EHR features only, no biological injection"
    },
    {
        "name": "metadata",
        "display_name": "Metadata + _preop",
        "features": ["metadata_preop", "metadata_surgical"],
        "description": "Surgical/preop indicators"
    },
    {
        "name": "grim_age",
        "display_name": "GrimAge v2 + Intermediate",
        "features": ["grim_age_v2", "grim_age_v2_dnam_glucose", "grim_age_v2_dnam_crp"],
        "description": "GrimAge accelerated aging + DNAm biomarkers"
    },
    {
        "name": "systems_age",
        "display_name": "SystemsAge (11 components)",
        "features": [
            "systems_age",
            "systems_age_glucose",
            "systems_age_crp",
            "systems_age_wbc",
            "systems_age_rbc",
            "systems_age_plate",
            "systems_age_bp_sys",
            "systems_age_bp_dia",
            "systems_age_chol",
            "systems_age_ldl",
            "systems_age_hdl"
        ],
        "description": "System-level aging across 11 domains"
    },
    {
        "name": "cpgt_proteins",
        "display_name": "CpGPTGrimAge v3 + Proteins",
        "features": ["cpgt_grim_v3", "protein_score_1", "protein_score_2"],
        "description": "CpGPT-derived protein scores"
    },
    {
        "name": "maple",
        "display_name": "MAPLE (32-dim embedding)",
        "features": ["maple_embedding"],
        "description": "Methylation patterns (32-dimensional)"
    },
    {
        "name": "cpgt_embed",
        "display_name": "CpGPT (64-dim embedding)",
        "features": ["cpgt_embedding"],
        "description": "CpGPT learned representations (64-dimensional)"
    },
    {
        "name": "methylgpt",
        "display_name": "MethylGPT (embeddings)",
        "features": ["methylgpt_embedding"],
        "description": "Methylation transformer embeddings"
    },
    {
        "name": "all_features",
        "display_name": "All Features (Full Ablation)",
        "features": [
            "grim_age_v2",
            "systems_age",
            "cpgt_grim_v3",
            "maple_embedding",
            "cpgt_embedding",
            "methylgpt_embedding",
        ],
        "description": "All biological features combined"
    },
]

# Injection methods
INJECTION_METHODS = [
    {
        "name": "concat",
        "display_name": "Concatenation",
        "description": "Simple concatenation of values to embedding",
        "mode": "concat"
    },
    {
        "name": "film",
        "display_name": "FiLM (Feature-wise Linear Modulation)",
        "description": "Feature-wise multiplicative modulation of embeddings",
        "mode": "film"
    },
    {
        "name": "discrete",
        "display_name": "Discretization (Binning)",
        "description": "Discretize values into quintile bins with learned embeddings",
        "mode": "discrete",
        "n_bins": 5
    },
    {
        "name": "comb",
        "display_name": "Combination/Fusion",
        "description": "Learned combination of values with EHR embedding",
        "mode": "comb"
    },
    {
        "name": "comb_binning",
        "display_name": "Combination + Binning",
        "description": "Combination of discretized values with learned fusion",
        "mode": "comb_binning",
        "n_bins": 5
    },
]

# Base configuration template
BASE_CONFIG = {
    "logging": {
        "level": "INFO",
        "path": "./outputs/logs"
    },
    "paths": {
        "prepared_data": "./outputs/finetuning/processed_data_with_values/",
        "pretrain_model": "./outputs/pretraining_dryrun",
        "model": None,  # Will be set per config
    },
    "model": {
        "cls": "default",
        "value_embedding_mode": "concat",  # Will be overridden
    },
    "trainer_args": {
        "batch_size": 16,
        "val_batch_size": 16,
        "effective_batch_size": 16,
        "epochs": 3,
        "info": True,
        "gradient_clip": {"clip_value": 1.0},
        "shuffle": True,
        "checkpoint_frequency": 1,
        "early_stopping": 3,
        "stopping_criterion": "roc_auc",
        "n_layers_to_freeze": 1,
        "unfreeze_on_plateau": False,
        "unfreeze_at_epoch": None,
        "plateau_threshold": 0.01,
    },
    "optimizer": {
        "lr": 5e-4,
        "eps": 1e-6,
    },
    "scheduler": {
        "_target_": "transformers.get_linear_schedule_with_warmup",
        "num_warmup_steps": 10,
        "num_training_steps": 50,
    },
    "metrics": {
        "accuracy": {
            "_target_": "corebehrt.modules.monitoring.metrics.Accuracy",
            "threshold": 0.5,
        },
        "roc_auc": {
            "_target_": "corebehrt.modules.monitoring.metrics.ROC_AUC",
        },
        "pr_auc": {
            "_target_": "corebehrt.modules.monitoring.metrics.PR_AUC",
        },
        "delong_roc_auc": {
            "_target_": "corebehrt.modules.monitoring.metrics.Delong_ROC_AUC",
        },
    },
    "evaluate": False,
}


def create_config(
    feature_subset: Dict,
    injection_method: Dict,
    output_dir: Path
) -> Dict:
    """Create a single config for a feature subset + injection method combination."""

    config = yaml.safe_load(yaml.dump(BASE_CONFIG))  # Deep copy

    # Set output path
    config_name = f"{feature_subset['name']}_{injection_method['name']}"
    config["paths"]["model"] = f"./outputs/ablation_models/{config_name}"

    # Set injection method
    config["model"]["value_embedding_mode"] = injection_method["mode"]

    # Add method-specific parameters
    if "n_bins" in injection_method:
        config["model"]["n_bins"] = injection_method["n_bins"]

    # Set biological features
    config["biological_features"] = feature_subset["features"]

    # Add metadata comment
    config["_metadata"] = {
        "config_name": config_name,
        "feature_subset": feature_subset["display_name"],
        "feature_count": len(feature_subset["features"]),
        "injection_method": injection_method["display_name"],
        "feature_description": feature_subset["description"],
        "method_description": injection_method["description"],
    }

    return config


def main():
    """Generate all 40 config files."""

    logger.info("\n" + "="*80)
    logger.info("GENERATING ABLATION CONFIGURATION FILES")
    logger.info("="*80 + "\n")

    config_dir = Path("./corebehrt/configs")
    config_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {config_dir}\n")

    # Generate all combinations
    total_configs = len(FEATURE_SUBSETS) * len(INJECTION_METHODS)
    created = 0

    logger.info(f"Creating {total_configs} configs:\n")
    logger.info(f"  Features:  {len(FEATURE_SUBSETS)} subsets")
    logger.info(f"  Methods:   {len(INJECTION_METHODS)} injection approaches")
    logger.info(f"  Total:     {total_configs} variants\n")

    # Create matrix
    logger.info("GENERATION MATRIX:")
    logger.info("─" * 80)

    for method_idx, injection_method in enumerate(INJECTION_METHODS):
        logger.info(f"\n{injection_method['display_name']} ({injection_method['name']}):")

        for feature_subset in FEATURE_SUBSETS:
            config = create_config(feature_subset, injection_method, config_dir)

            config_name = f"ablation_{feature_subset['name']}_{injection_method['name']}"
            config_path = config_dir / f"{config_name}.yaml"

            # Write config
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            created += 1

            # Log
            n_features = len(feature_subset["features"])
            feature_display = feature_subset["display_name"][:30]
            logger.info(f"  ✓ {config_name:50s} ({n_features:2d} features)")

    logger.info("\n" + "="*80)
    logger.info(f"✓ GENERATED {created} CONFIG FILES")
    logger.info("="*80 + "\n")

    # Summary table
    logger.info("SUMMARY: Feature Subsets × Injection Methods\n")
    logger.info(f"{'Feature Subset':<30} {'CONCAT':8} {'FiLM':8} {'DISCRETE':8} {'COMB':8} {'COMB_BIN':8}")
    logger.info("─" * 80)

    for feature_subset in FEATURE_SUBSETS:
        row = f"{feature_subset['display_name']:<30}"
        for method in INJECTION_METHODS:
            config_name = f"ablation_{feature_subset['name']}_{method['name']}.yaml"
            config_path = config_dir / config_name
            status = "✓" if config_path.exists() else "✗"
            row += f"  {status:6s} "
        logger.info(row)

    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS:")
    logger.info("="*80)
    logger.info("\n1. Create run script for batch training:")
    logger.info("   python scripts/run_full_ablation_with_methods.py\n")
    logger.info("2. Or train individual models:")
    logger.info("   python -m corebehrt.main.finetune_cv \\")
    logger.info("     --config_path corebehrt/configs/ablation_grim_age_film.yaml\n")
    logger.info("3. Expected training time:")
    logger.info(f"   Sequential: ~{40 * 45} minutes (30 hours)")
    logger.info(f"   Parallel (4 workers): ~{45 * 10} minutes (7.5 hours)\n")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
