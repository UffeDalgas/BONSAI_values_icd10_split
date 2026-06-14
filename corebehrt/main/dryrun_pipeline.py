"""
End-to-end dry-run pipeline with synthetic value injection.

KEY DESIGN: Pretrain learns from raw EHR data, finetune learns to use biological features.

Pipeline:
1. Prepare data (pretrain) ← uses raw MEDS data
2. Pretrain (2 epochs, quick) ← no biological features
3. Inject biological features ← creates separate MEDS data copy with features
4. Create outcomes (binary mortality prediction)
5. Select cohort
6. Prepare data (finetune) ← uses MEDS data WITH injected features
7. Finetune with biological features (3 epochs)
8. Evaluate

Data Flow:
  ./example_data/example_MEDS_data/train/
        ↓
  [prepare for pretrain]
        ↓
  ./outputs/pretraining/processed_data/
        ↓
  [PRETRAIN: raw EHR representations]
        ↓
  ./outputs/pretraining_dryrun/  ← checkpoint
        ↓
  ./example_data/example_MEDS_data/train/ + inject features
        ↓
  ./example_data/example_MEDS_data_with_values/train/
        ↓
  [prepare for finetune]
        ↓
  ./outputs/finetuning/processed_data_with_values/
        ↓
  [FINETUNE: learn to use biological features]
        ↓
  ./outputs/finetuning_dryrun_values/  ← results

Usage:
    python -m corebehrt.main.dryrun_pipeline
"""

import logging
import os
import sys
from pathlib import Path
import shutil
from typing import Tuple

import torch
import numpy as np
import pandas as pd

from corebehrt.constants.paths import (
    PREPARED_ALL_PATIENTS,
    FOLDS_FILE,
    TEST_PIDS_FILE,
)
from corebehrt.functional.setup.args import get_args
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.modules.preparation.dataset import PatientDataset
from corebehrt.functional.features.value_injection import (
    SyntheticBiologicalFeatureGenerator,
    inject_values_into_meds,
    create_feature_concept_mapping,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dryrun_pipeline")

PRETRAIN_CONFIG = "./corebehrt/configs/pretrain_dryrun.yaml"
FINETUNE_CONFIG = "./corebehrt/configs/finetune_dryrun_values.yaml"
EXAMPLE_DATA = "./example_data/example_MEDS_data"


def setup_dryrun_data():
    """Setup example data for dry run."""
    logger.info("Setting up dry-run data...")

    # Use example MEDS data if it exists
    if Path(EXAMPLE_DATA).exists():
        logger.info(f"Using example MEDS data from {EXAMPLE_DATA}")
        return EXAMPLE_DATA

    logger.warning(f"Example data not found at {EXAMPLE_DATA}")
    logger.info("Please run: python -m corebehrt.synthetic_data.base_cohort.create_base_synthetic_cohort")
    return None


def inject_biological_features(data_dir: str, output_dir: str = None, feature_sets: list = None):
    """
    Inject synthetic biological features into MEDS data.

    Creates a separate copy with injected features, keeping original untouched.
    """
    if feature_sets is None:
        feature_sets = ["clocks", "proteins", "maple", "methylgpt"]

    if output_dir is None:
        output_dir = "./example_data/example_MEDS_data_with_values"

    logger.info(f"Injecting biological features: {feature_sets}")
    logger.info(f"Output directory: {output_dir}")

    # Count patients from original
    train_path = Path(data_dir) / "train"
    parquet_files = sorted(list(train_path.glob("*.parquet")))
    n_samples = len(parquet_files)

    logger.info(f"Found {n_samples} patient records in {train_path}")

    # Generate synthetic features once
    gen = SyntheticBiologicalFeatureGenerator(n_samples=n_samples)
    all_features = gen.generate_all_features()

    logger.info(f"Generated synthetic features:")
    for feature_name, feature_data in all_features.items():
        if isinstance(feature_data, dict):
            logger.info(f"  - {feature_name}: {len(feature_data)} sub-features")
        elif isinstance(feature_data, np.ndarray):
            logger.info(f"  - {feature_name}: shape {feature_data.shape}")

    # Prepare for injection
    patient_ids = list(range(n_samples))
    feature_concepts = create_feature_concept_mapping(all_features)

    # Create output directory
    output_train_path = Path(output_dir) / "train"
    output_train_path.mkdir(parents=True, exist_ok=True)

    # Inject into each patient's MEDS data (in separate location)
    injected_files = 0
    for parquet_file in parquet_files:
        try:
            # Read original MEDS data
            meds_df = pd.read_parquet(parquet_file)

            # Get patient index
            patient_idx = int(parquet_file.stem)

            # Create single-patient feature dict
            single_patient_features = {}
            for feature_name, feature_data in all_features.items():
                if isinstance(feature_data, dict):
                    single_patient_features[feature_name] = {
                        k: v[[patient_idx]] for k, v in feature_data.items()
                    }
                else:
                    single_patient_features[feature_name] = feature_data[[patient_idx]]

            # Inject features
            meds_df_injected = inject_values_into_meds(
                meds_df,
                single_patient_features,
                [patient_idx],
                feature_concepts,
            )

            # Save to new location (don't overwrite original)
            output_file = output_train_path / parquet_file.name
            meds_df_injected.to_parquet(output_file, index=False)
            injected_files += 1

        except Exception as e:
            logger.warning(f"Failed to inject features for {parquet_file}: {e}")
            continue

    logger.info(f"✓ Created {injected_files}/{len(parquet_files)} injected files in {output_dir}")
    return output_dir


def run_pretrain_step():
    """Run pretraining step."""
    logger.info("="*80)
    logger.info("STEP 1: CREATE DATA (tokenize raw MEDS)")
    logger.info("="*80)

    from corebehrt.main.create_data import main_data

    try:
        main_data("./corebehrt/configs/create_data.yaml")
    except Exception as e:
        logger.error(f"Data creation failed: {e}")
        return False

    logger.info("="*80)
    logger.info("STEP 2: PREPARE DATA FOR PRETRAINING")
    logger.info("="*80)

    # Run prepare_training_data for pretrain
    from corebehrt.main.prepare_training_data import main_prepare_data

    prepare_config = "./corebehrt/configs/prepare_pretrain.yaml"
    try:
        main_prepare_data(prepare_config)
    except Exception as e:
        logger.error(f"Pretrain preparation failed: {e}")
        return False

    logger.info("="*80)
    logger.info("STEP 3: PRETRAIN MODEL (2 EPOCHS)")
    logger.info("="*80)

    from corebehrt.main.pretrain import main_train

    try:
        result = main_train(PRETRAIN_CONFIG)
        logger.info(f"Pretraining completed with result: {result}")
    except Exception as e:
        import traceback
        logger.error(f"Pretraining failed: {e}")
        logger.error(traceback.format_exc())
        return False

    return True


def run_finetune_step():
    """Run finetuning step with value injection."""
    logger.info("="*80)
    logger.info("STEP 3: CREATE OUTCOMES")
    logger.info("="*80)

    from corebehrt.main.create_outcomes import main_data

    outcomes_config = "./corebehrt/configs/create_outcomes.yaml"
    try:
        main_data(outcomes_config)
    except Exception as e:
        logger.warning(f"Outcome creation failed (may be normal for dry run): {e}")
        # Continue anyway - outcomes may already exist

    logger.info("="*80)
    logger.info("STEP 4: SELECT COHORT")
    logger.info("="*80)

    from corebehrt.main.select_cohort import main_select_cohort

    cohort_config = "./corebehrt/configs/select_cohort.yaml"
    try:
        main_select_cohort(cohort_config)
    except Exception as e:
        logger.warning(f"Cohort selection failed (may be normal for dry run): {e}")

    logger.info("="*80)
    logger.info("STEP 5: PREPARE DATA FOR FINETUNING WITH VALUE INJECTION")
    logger.info("="*80)

    from corebehrt.main.prepare_training_data import main_prepare_data

    prepare_finetune_config = "./corebehrt/configs/prepare_finetune.yaml"
    try:
        main_prepare_data(prepare_finetune_config)
    except Exception as e:
        logger.error(f"Finetune preparation failed: {e}")
        return False

    logger.info("="*80)
    logger.info("STEP 6: FINETUNE WITH BIOLOGICAL FEATURES")
    logger.info("="*80)

    from corebehrt.main.finetune_cv import main_finetune

    try:
        main_finetune(FINETUNE_CONFIG)
    except Exception as e:
        logger.error(f"Finetuning failed: {e}")
        return False

    return True


def main():
    """Run complete dry-run pipeline."""
    logger.info("\n")
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*20 + "BONSAI DRY-RUN PIPELINE" + " "*35 + "║")
    logger.info("║" + " "*10 + "Pretrain: Raw EHR | Finetune: EHR + Biological Features" + " "*8 + "║")
    logger.info("╚" + "="*78 + "╝")
    logger.info("\n")

    # Setup data
    data_dir = setup_dryrun_data()
    if data_dir is None:
        logger.error("Failed to setup example data")
        return 1

    logger.info("="*80)
    logger.info("PHASE 1: PRETRAIN (without value injection)")
    logger.info("="*80)

    # Run pretrain with raw EHR data (NO injection)
    if not run_pretrain_step():
        logger.error("Pretraining pipeline failed")
        return 1

    logger.info("="*80)
    logger.info("PHASE 2: PREPARE FINETUNING DATA (with value injection)")
    logger.info("="*80)

    # Inject biological features into separate copy for finetuning
    try:
        injected_data_dir = inject_biological_features(data_dir)
        logger.info(f"✓ Created injected data at {injected_data_dir}")
    except Exception as e:
        logger.error(f"Value injection failed: {e}")
        return 1

    logger.info("="*80)
    logger.info("PHASE 3: FINETUNE (with value injection)")
    logger.info("="*80)

    # Run finetune with injected features
    if not run_finetune_step():
        logger.error("Finetuning pipeline failed")
        return 1

    logger.info("\n")
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*25 + "✓ DRY-RUN COMPLETE" + " "*35 + "║")
    logger.info("╚" + "="*78 + "╝")
    logger.info("\n")
    logger.info("Pipeline Architecture:")
    logger.info(f"  Raw EHR data:        {data_dir}")
    logger.info(f"  Injected data:       {injected_data_dir}")
    logger.info(f"  Pretrain (no values):{' '*5}./outputs/pretraining_dryrun")
    logger.info(f"  Finetune (w/ values):{' '*5}./outputs/finetuning_dryrun_values")

    return 0


if __name__ == "__main__":
    sys.exit(main())
