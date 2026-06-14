#!/usr/bin/env python3
"""
Helper script: Create disease-stratified ablation cohorts from MEDS data.

This script demonstrates how to:
1. Load MEDS data from ehr2meds parquet files
2. Filter patients by disease code (e.g., "D74*" for diabetes)
3. Create disease-stratified cohorts (pretrain, finetune, eval)
4. Prepare data for ablation studies

Usage:
    python scripts/create_disease_stratified_ablation.py \
        --disease-code D74* \
        --meds-path ./outputs/tokenized/ \
        --finetune-patients data/diabetes_patients_with_proxies.txt \
        --output-dir ./outputs/ablation_cohorts/diabetes
"""

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd
import yaml

from corebehrt.main.ablation_pipeline import MEDSDiseasAblationStudy

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("create_disease_stratified_ablation")


def load_patient_ids_from_file(filepath: str) -> set:
    """Load patient IDs from a text file (one ID per line)."""
    with open(filepath) as f:
        patient_ids = {int(line.strip()) for line in f if line.strip()}
    return patient_ids


def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def create_ablation_cohorts_from_args(args):
    """Create ablation cohorts from command-line arguments."""

    # Parse finetune patient IDs
    finetune_patients = set()
    if args.finetune_patients:
        if args.finetune_patients.endswith('.txt'):
            finetune_patients = load_patient_ids_from_file(args.finetune_patients)
            logger.info(f"Loaded {len(finetune_patients)} finetune patient IDs from {args.finetune_patients}")
        elif args.finetune_patients.isdigit():
            finetune_patients = {int(args.finetune_patients)}
        else:
            finetune_patients = {int(pid) for pid in args.finetune_patients.split(',')}

    # Create ablation study
    study = MEDSDiseasAblationStudy(
        meds_data_path=args.meds_path,
        disease_code=args.disease_code,
        finetune_disease_patient_ids=finetune_patients,
        finetune_subset_size=args.finetune_size,
        eval_subset_size=args.eval_size,
        output_dir=args.output_dir,
    )

    # Load and create cohorts
    study.load_meds_data()
    pretrain, finetune, eval = study.create_disease_stratified_cohorts()

    # Print statistics
    stats = study.get_cohort_statistics()

    logger.info("\n" + "="*80)
    logger.info("✓ ABLATION COHORTS CREATED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"\nDisease code: {args.disease_code}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Create finetune config without features (baseline model)")
    logger.info(f"  2. Create finetune config with features (enhanced model)")
    logger.info(f"  3. Run both models on eval cohort")
    logger.info(f"  4. Compare ROC-AUC to measure biological signal")

    return study, stats


def create_ablation_cohorts_from_yaml(config_path: str):
    """Create ablation cohorts from YAML configuration."""

    config = load_config_from_yaml(config_path)

    # Extract relevant config sections
    meds_config = config.get("meds", {})
    ablation_config = config.get("ablation_study", {})
    output_config = config.get("output", {})
    patient_selection = config.get("patient_selection", {})

    # Load finetune patient IDs if specified
    finetune_patients = set()
    if patient_selection.get("method") == "explicit_ids":
        patient_file = patient_selection.get("finetune_patient_ids_file")
        if patient_file:
            finetune_patients = load_patient_ids_from_file(patient_file)
            logger.info(f"Loaded {len(finetune_patients)} finetune patient IDs from {patient_file}")

    # Create ablation study
    cohort_split = ablation_config.get("cohort_split", {})
    study = MEDSDiseasAblationStudy(
        meds_data_path=meds_config.get("data_path", "./outputs/tokenized/"),
        disease_code=meds_config.get("disease_code_pattern", "D74*"),
        finetune_disease_patient_ids=finetune_patients,
        finetune_subset_size=cohort_split.get("finetune_with_proxies", 150),
        eval_subset_size=cohort_split.get("eval_without_proxies", 75),
        output_dir=output_config.get("base_dir", "./outputs/ablation_results"),
    )

    # Load and create cohorts
    study.load_meds_data()
    pretrain, finetune, eval = study.create_disease_stratified_cohorts()

    # Print statistics
    stats = study.get_cohort_statistics()

    logger.info("\n" + "="*80)
    logger.info("✓ ABLATION COHORTS CREATED FROM CONFIG")
    logger.info("="*80)
    logger.info(f"\nConfig file: {config_path}")
    logger.info(f"Disease: {meds_config.get('disease_code_pattern')}")
    logger.info(f"Output directory: {output_config.get('base_dir')}")
    logger.info(f"\nBiological features to validate:")
    for feat in ablation_config.get("biological_features", []):
        logger.info(f"  - {feat}")

    return study, stats


def main():
    parser = argparse.ArgumentParser(
        description="Create disease-stratified ablation cohorts from MEDS data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # From command-line arguments
  python scripts/create_disease_stratified_ablation.py \\
    --disease-code D74* \\
    --meds-path ./outputs/tokenized/ \\
    --finetune-patients data/diabetes_patients.txt \\
    --output-dir ./outputs/ablation_cohorts/diabetes

  # From YAML configuration
  python scripts/create_disease_stratified_ablation.py \\
    --config corebehrt/configs/ablation_diabetes_meds.yaml

  # With explicit patient IDs
  python scripts/create_disease_stratified_ablation.py \\
    --disease-code I50* \\
    --meds-path ./outputs/tokenized/ \\
    --finetune-patients 101,102,103,105,108 \\
    --output-dir ./outputs/ablation_cohorts/heart_failure
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--config",
        type=str,
        help="YAML configuration file (e.g., corebehrt/configs/ablation_diabetes_meds.yaml)"
    )
    mode_group.add_argument(
        "--disease-code",
        type=str,
        help="Disease code pattern (e.g., 'D74*' for diabetes, 'I50*' for heart failure)"
    )

    # Disease-specific arguments (for --disease-code mode)
    parser.add_argument(
        "--meds-path",
        type=str,
        default="./outputs/tokenized/",
        help="Path to MEDS parquet data (default: ./outputs/tokenized/)"
    )
    parser.add_argument(
        "--finetune-patients",
        type=str,
        help="Patient IDs with biological proxies (file, comma-separated, or single ID)"
    )
    parser.add_argument(
        "--finetune-size",
        type=int,
        default=150,
        help="Number of disease patients for finetune (default: 150)"
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=75,
        help="Number of disease patients for eval (default: 75)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/ablation_cohorts",
        help="Output directory for cohorts (default: ./outputs/ablation_cohorts)"
    )

    args = parser.parse_args()

    try:
        if args.config:
            # Load from YAML config
            logger.info(f"Loading configuration from {args.config}")
            study, stats = create_ablation_cohorts_from_yaml(args.config)
        else:
            # Use command-line arguments
            logger.info(f"Creating ablation cohorts for disease code: {args.disease_code}")
            study, stats = create_ablation_cohorts_from_args(args)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
