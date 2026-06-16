#!/usr/bin/env python3
"""
Baseline finetuning pipeline with mortality outcomes.
Accepts cluster paths as command-line arguments.

Usage:
    python run_baseline_finetuning.py \
      --meds-path /cluster/path/meds_for_bonsai \
      --output-path /cluster/path/outputs \
      --features-path /cluster/path/features \
      --tokenized-path /cluster/path/tokenized
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
import yaml
import tempfile
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("baseline_finetuning")


def update_yaml_path(yaml_file: str, key_path: list, new_value: str) -> str:
    """Load YAML, update nested key, return updated content."""
    with open(yaml_file) as f:
        cfg = yaml.safe_load(f)

    # Navigate to the nested key and update
    d = cfg
    for k in key_path[:-1]:
        d = d[k]
    d[key_path[-1]] = new_value

    return yaml.dump(cfg, default_flow_style=False)


def create_temp_config(template_path: str, replacements: dict) -> str:
    """
    Create temporary config file with path replacements.

    Args:
        template_path: Path to template YAML
        replacements: Dict of {yaml_key_path: new_value}
                     where yaml_key_path is a tuple like ('paths', 'data')

    Returns:
        Path to temporary config file
    """
    with open(template_path) as f:
        cfg = yaml.safe_load(f)

    # Apply replacements
    for key_path, value in replacements.items():
        d = cfg
        for k in key_path[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[key_path[-1]] = value

    # Write to temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.yaml')
    with open(temp_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return temp_path


def run_step(step_name: str, command: list, config_path: str = None, cleanup_config: bool = False):
    """Run a step and clean up temporary config if needed."""
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP: {step_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Command: {' '.join(command)}")

    result = subprocess.run(command, cwd=str(Path(__file__).parent))

    if cleanup_config and config_path:
        Path(config_path).unlink()

    if result.returncode != 0:
        logger.error(f"{step_name} failed with return code {result.returncode}")
        sys.exit(1)

    logger.info(f"✓ {step_name} completed")


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline (EHR-only) finetuning with real mortality outcomes"
    )
    parser.add_argument(
        "--meds-path",
        required=True,
        help="Path to MEDS data directory (should contain train/, tuning/, held_out/)"
    )
    parser.add_argument(
        "--output-path",
        default="./outputs",
        help="Path for outputs (default: ./outputs)"
    )
    parser.add_argument(
        "--features-path",
        default="./outputs/features",
        help="Path to tokenized features (default: ./outputs/features)"
    )
    parser.add_argument(
        "--tokenized-path",
        default="./outputs/tokenized",
        help="Path to tokenized data (default: ./outputs/tokenized)"
    )
    parser.add_argument(
        "--pretrain-model",
        default="./outputs/pretraining_dryrun",
        help="Path to pretrained model (default: ./outputs/pretraining_dryrun)"
    )
    parser.add_argument(
        "--skip-outcomes",
        action="store_true",
        help="Skip outcome creation (if MORTALITY.csv already exists)"
    )
    parser.add_argument(
        "--skip-cohort",
        action="store_true",
        help="Skip cohort selection (if folds.pt already exists)"
    )

    args = parser.parse_args()

    # Validate paths
    meds_path = Path(args.meds_path)
    if not meds_path.exists():
        logger.error(f"MEDS path does not exist: {meds_path}")
        sys.exit(1)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"MEDS data path: {meds_path}")
    logger.info(f"Output path: {output_path}")

    # Step 1: Create outcomes (scan MEDS for DOD codes)
    if not args.skip_outcomes:
        logger.info("\n" + "="*80)
        logger.info("STEP 1: CREATE OUTCOMES (scan MEDS for DOD codes)")
        logger.info("="*80)

        outcomes_csv = output_path / "outcomes" / "MORTALITY.csv"
        if outcomes_csv.exists():
            logger.warning(f"MORTALITY.csv already exists: {outcomes_csv}")
            logger.info("Skipping outcome creation (use --skip-outcomes to suppress)")
        else:
            # Create temporary config with cluster paths
            config_replacements = {
                ('paths', 'data'): str(meds_path),
                ('paths', 'outcomes'): str(output_path / 'outcomes'),
                ('paths', 'features'): str(args.features_path),
            }
            temp_config = create_temp_config(
                "corebehrt/configs/create_outcomes_mortality.yaml",
                config_replacements
            )

            command = [
                "python", "-m", "corebehrt.main.create_outcomes",
                "--config", temp_config
            ]
            run_step("Create outcomes", command, temp_config, cleanup_config=True)

    # Step 2: Select cohort (create folds from tuning split)
    if not args.skip_cohort:
        logger.info("\n" + "="*80)
        logger.info("STEP 2: SELECT COHORT (create folds)")
        logger.info("="*80)

        folds_file = output_path / "cohort" / "finetune" / "folds.pt"
        if folds_file.exists():
            logger.warning(f"folds.pt already exists: {folds_file}")
            logger.info("Skipping cohort selection (use --skip-cohort to suppress)")
        else:
            # Create temporary config with cluster paths
            config_replacements = {
                ('paths', 'features'): str(args.features_path),
                ('paths', 'tokenized'): str(args.tokenized_path),
                ('paths', 'initial_pids'): str(args.tokenized_path / 'pids_tuning.pt'),
                ('paths', 'outcomes'): str(output_path / 'outcomes'),
                ('paths', 'cohort'): str(output_path / 'cohort' / 'finetune'),
            }
            temp_config = create_temp_config(
                "corebehrt/configs/select_cohort.yaml",
                config_replacements
            )

            command = [
                "python", "-m", "corebehrt.main.select_cohort",
                "--config", temp_config
            ]
            run_step("Select cohort", command, temp_config, cleanup_config=True)

    # Step 3: Prepare finetuning data
    logger.info("\n" + "="*80)
    logger.info("STEP 3: PREPARE FINETUNING DATA")
    logger.info("="*80)

    prepared_data_dir = output_path / "finetuning" / "processed_data_no_values"
    if (prepared_data_dir / "folds.pt").exists():
        logger.warning(f"Prepared data already exists: {prepared_data_dir}")
        logger.info("Skipping data preparation (delete folds.pt to re-run)")
    else:
        config_replacements = {
            ('paths', 'features'): str(args.features_path),
            ('paths', 'tokenized'): str(args.tokenized_path),
            ('paths', 'cohort'): str(output_path / 'cohort' / 'finetune'),
            ('paths', 'outcomes'): str(output_path / 'outcomes'),
            ('paths', 'prepared_data'): str(prepared_data_dir),
        }
        temp_config = create_temp_config(
            "corebehrt/configs/prepare_finetune.yaml",
            config_replacements
        )

        command = [
            "python", "-m", "corebehrt.main.prepare_training_data",
            "--config", temp_config
        ]
        run_step("Prepare finetuning data", command, temp_config, cleanup_config=True)

    # Step 4: Run baseline finetuning
    logger.info("\n" + "="*80)
    logger.info("STEP 4: RUN BASELINE FINETUNING (EHR-only, no biological features)")
    logger.info("="*80)

    finetune_output = output_path / "finetune_models" / "ehr_only"
    finetune_output.mkdir(parents=True, exist_ok=True)

    # Create temporary config with cluster paths
    config_replacements = {
        ('paths', 'model'): str(finetune_output),
        ('paths', 'prepared_data'): str(prepared_data_dir),
        ('paths', 'pretrain_model'): str(args.pretrain_model),
        ('logging', 'path'): str(output_path / 'logs'),
    }
    temp_config = create_temp_config(
        "outputs/finetune_models/ehr_only/finetune_config.yaml",
        config_replacements
    )

    command = [
        "python", "-m", "corebehrt.main.finetune_cv",
        "--config", temp_config
    ]
    run_step("Run finetuning", command, temp_config, cleanup_config=True)

    logger.info("\n" + "="*80)
    logger.info("✅ BASELINE FINETUNING COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {finetune_output}")
    logger.info(f"Checkpoints: {finetune_output / 'checkpoints'}")
    logger.info(f"Metrics: {finetune_output / 'fold_*/'}*")


if __name__ == "__main__":
    main()
