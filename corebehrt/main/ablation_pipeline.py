"""
Disease-specific ablation pipeline for validating biological proxy signal.

Experimental design:
  1. Pretrain on all data EXCEPT disease X (population control)
  2. Finetune on disease X subset WITH biological proxies
  3. Evaluate on disease X samples WITHOUT proxies (held-out test set)
  4. Compare ROC-AUC with/without features to quantify signal

This allows ablation of individual biological proxy sets (GrimAge2, SystemsAge, etc.)

Integration with MEDS format:
  - Loads disease codes from ehr2meds MEDS data
  - Filters patients by disease code patterns (e.g., "D74*" for diabetes)
  - Excludes target disease from pretraining cohort
  - Selects disease-specific subset for finetune/eval splits
"""

import logging
import sys
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from corebehrt.functional.data.disease_filtering import DiseaseFilter, DiseaseAwareDataPipeline

logger = logging.getLogger("ablation_pipeline")


class DiseaseAblationStudy:
    """Validates biological proxy signal for specific diseases."""

    def __init__(
        self,
        exclude_disease: str,
        finetune_disease: str,
        finetune_subset_size: int = 100,
        eval_subset_size: int = 50,
        feature_sets: List[str] = None,
    ):
        """
        Args:
            exclude_disease: Disease to exclude from pretraining (e.g., 'heart_failure')
            finetune_disease: Disease to focus finetune/eval on
            finetune_subset_size: Number of samples with biological proxies for finetuning
            eval_subset_size: Number of held-out samples for evaluation (no proxies)
            feature_sets: Biological features to ablate ['grim_age2', 'systems_age', 'maple', 'methylgpt']
        """
        self.exclude_disease = exclude_disease
        self.finetune_disease = finetune_disease
        self.finetune_subset_size = finetune_subset_size
        self.eval_subset_size = eval_subset_size
        self.feature_sets = feature_sets or ["grim_age2", "systems_age", "maple", "methylgpt"]

        self.results = {}


class MEDSDiseasAblationStudy:
    """
    Disease-specific ablation pipeline with integration to MEDS format data.

    This class handles:
    1. Loading MEDS data from parquet files
    2. Filtering patients by disease codes (e.g., "D74*" for ICD-10 diabetes)
    3. Creating disease-stratified cohorts (pretrain, finetune, eval)
    4. Preparing data for ablation studies (with/without biological features)
    """

    def __init__(
        self,
        meds_data_path: str,
        disease_code: str,
        finetune_disease_patient_ids: Optional[Set[int]] = None,
        finetune_subset_size: int = 100,
        eval_subset_size: int = 50,
        feature_sets: List[str] = None,
        output_dir: str = "./outputs/ablation_cohorts",
    ):
        """
        Args:
            meds_data_path: Path to MEDS parquet data
            disease_code: ICD code pattern (e.g., "D74*" for diabetes, "I50*" for heart failure)
            finetune_disease_patient_ids: Patient IDs with biological proxies
            finetune_subset_size: Number of disease patients for finetune with proxies
            eval_subset_size: Number of disease patients for evaluation (without proxies)
            feature_sets: Biological features to ablate
            output_dir: Where to save cohort data
        """
        self.meds_data_path = Path(meds_data_path)
        self.disease_code = disease_code
        self.finetune_disease_patient_ids = finetune_disease_patient_ids or set()
        self.finetune_subset_size = finetune_subset_size
        self.eval_subset_size = eval_subset_size
        self.feature_sets = feature_sets or ["grim_age2", "systems_age", "maple", "methylgpt"]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.disease_filter = DiseaseFilter(disease_code)
        self.meds_data = None
        self.cohorts = {}

    def load_meds_data(self) -> pd.DataFrame:
        """Load MEDS data from parquet files in meds_data_path."""
        logger.info(f"Loading MEDS data from {self.meds_data_path}")

        parquet_files = list(self.meds_data_path.glob("**/*.parquet"))
        if not parquet_files:
            logger.warning(f"No parquet files found in {self.meds_data_path}")
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in sorted(parquet_files)]
        self.meds_data = pd.concat(dfs, ignore_index=True)

        logger.info(f"✓ Loaded {len(self.meds_data)} events from {len(parquet_files)} files")
        logger.info(f"  Unique patients: {self.meds_data['person_id'].nunique()}")
        logger.info(f"  Date range: {self.meds_data['event_timestamp'].min()} to {self.meds_data['event_timestamp'].max()}")

        return self.meds_data

    def create_disease_stratified_cohorts(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create disease-stratified splits from MEDS data:

        Returns:
            (pretrain_cohort, finetune_cohort, eval_cohort)
        """
        if self.meds_data is None:
            self.load_meds_data()

        logger.info("\n" + "="*80)
        logger.info("CREATING DISEASE-STRATIFIED COHORTS FROM MEDS DATA")
        logger.info("="*80)

        # Split by disease
        pretrain, finetune, eval = self.disease_filter.split_by_disease(
            self.meds_data,
            self.finetune_disease_patient_ids
        )

        self.cohorts = {
            "pretrain": pretrain,
            "finetune": finetune,
            "eval": eval,
        }

        # Save cohorts
        logger.info(f"\nSaving cohorts to {self.output_dir}")
        pretrain.to_parquet(self.output_dir / "pretrain_cohort.parquet")
        finetune.to_parquet(self.output_dir / "finetune_cohort.parquet")
        eval.to_parquet(self.output_dir / "eval_cohort.parquet")

        # Save patient ID lists for reference
        pretrain_patients = sorted(pretrain["person_id"].unique().tolist())
        finetune_patients = sorted(finetune["person_id"].unique().tolist())
        eval_patients = sorted(eval["person_id"].unique().tolist())

        with open(self.output_dir / "pretrain_patients.txt", "w") as f:
            f.write("\n".join(map(str, pretrain_patients)))

        with open(self.output_dir / "finetune_patients.txt", "w") as f:
            f.write("\n".join(map(str, finetune_patients)))

        with open(self.output_dir / "eval_patients.txt", "w") as f:
            f.write("\n".join(map(str, eval_patients)))

        logger.info("✓ Cohorts saved successfully")

        return pretrain, finetune, eval

    def get_cohort_statistics(self) -> dict:
        """Get statistics about the created cohorts."""
        if not self.cohorts:
            logger.warning("No cohorts created yet. Call create_disease_stratified_cohorts() first.")
            return {}

        stats = {}
        for cohort_name, cohort_data in self.cohorts.items():
            n_patients = cohort_data["person_id"].nunique()
            n_events = len(cohort_data)
            avg_events_per_patient = n_events / n_patients if n_patients > 0 else 0

            stats[cohort_name] = {
                "n_patients": n_patients,
                "n_events": n_events,
                "avg_events_per_patient": avg_events_per_patient,
            }

        logger.info("\n" + "="*80)
        logger.info("COHORT STATISTICS")
        logger.info("="*80)
        for cohort_name, cohort_stats in stats.items():
            logger.info(f"{cohort_name}:")
            logger.info(f"  Patients: {cohort_stats['n_patients']}")
            logger.info(f"  Events: {cohort_stats['n_events']}")
            logger.info(f"  Avg events/patient: {cohort_stats['avg_events_per_patient']:.1f}")

        return stats

    def create_disease_stratified_cohorts(self, outcomes_df: pd.DataFrame, disease_column: str = 'disease'):
        """
        Split data into:
        - Pretrain: All except finetune_disease
        - Finetune: disease_subset with biological proxies
        - Eval: disease_subset without proxies (held-out test)
        """
        logger.info("="*80)
        logger.info("CREATING DISEASE-STRATIFIED COHORTS FOR ABLATION STUDY")
        logger.info("="*80)

        # Get disease-specific samples
        disease_samples = outcomes_df[outcomes_df[disease_column] == self.finetune_disease]
        logger.info(f"Total {self.finetune_disease} samples: {len(disease_samples)}")

        # Split disease samples
        n_disease = len(disease_samples)
        finetune_indices = np.random.choice(
            n_disease,
            size=min(self.finetune_subset_size, n_disease),
            replace=False
        )
        remaining_indices = np.setdiff1d(np.arange(n_disease), finetune_indices)
        eval_indices = remaining_indices[:min(self.eval_subset_size, len(remaining_indices))]

        finetune_cohort = disease_samples.iloc[finetune_indices]
        eval_cohort = disease_samples.iloc[eval_indices]

        # Pretrain: exclude this disease
        pretrain_cohort = outcomes_df[outcomes_df[disease_column] != self.finetune_disease]

        logger.info(f"\n✓ Pretrain cohort (no {self.finetune_disease}): {len(pretrain_cohort)}")
        logger.info(f"✓ Finetune cohort (with bio proxies): {len(finetune_cohort)}")
        logger.info(f"✓ Eval cohort (held-out, no proxies): {len(eval_cohort)}")
        logger.info(f"\nMortality rates:")
        logger.info(f"  Pretrain: {pretrain_cohort['mortality'].mean():.1%}")
        logger.info(f"  Finetune: {finetune_cohort['mortality'].mean():.1%}")
        logger.info(f"  Eval: {eval_cohort['mortality'].mean():.1%}")

        return pretrain_cohort, finetune_cohort, eval_cohort

    def run_feature_ablation(self, results_dir: str = "./outputs/ablation_results"):
        """Run ablation study comparing models with/without biological features."""
        logger.info("="*80)
        logger.info("ABLATION STUDY: BIOLOGICAL PROXY SIGNAL VALIDATION")
        logger.info("="*80)
        logger.info(f"\nDisease: {self.finetune_disease}")
        logger.info(f"Excluded from pretraining: {self.exclude_disease}")
        logger.info(f"Features to test: {self.feature_sets}")
        logger.info(f"\nExperimental design:")
        logger.info(f"  1. Pretrain on all samples except {self.exclude_disease}")
        logger.info(f"  2. Model A: Finetune on {self.finetune_disease} + EHR ONLY")
        logger.info(f"  3. Model B: Finetune on {self.finetune_disease} + EHR + Bio Features")
        logger.info(f"  4. Evaluate both on held-out {self.finetune_disease} samples")
        logger.info(f"  5. Delta ROC-AUC = feature contribution to mortality signal")

        logger.info(f"\n✓ This design allows you to:")
        logger.info(f"  - Validate GrimAge2, SystemsAge, MAPLE, MethylGPT signal")
        logger.info(f"  - Ablate features one at a time")
        logger.info(f"  - Quantify disease-specific contribution to mortality")
        logger.info(f"  - Compare across your epimap feature space")

        return {
            "disease": self.finetune_disease,
            "pretrain_strategy": f"exclude_{self.exclude_disease}",
            "finetune_samples": self.finetune_subset_size,
            "eval_samples": self.eval_subset_size,
            "features_tested": self.feature_sets,
            "status": "ready_for_implementation"
        }


def main():
    """Run ablation study for GrimAge2 + SystemsAge validation."""

    logger.info("")
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*15 + "BONSAI BIOLOGICAL PROXY ABLATION STUDY" + " "*25 + "║")
    logger.info("║" + " "*10 + "Validating GrimAge2, SystemsAge, MAPLE, MethylGPT Signal" + " "*12 + "║")
    logger.info("╚" + "="*78 + "╝")
    logger.info("")

    # Example: Study heart failure with GrimAge2 + SystemsAge
    ablation = DiseaseAblationStudy(
        exclude_disease="other_diseases",
        finetune_disease="heart_failure",
        finetune_subset_size=100,
        eval_subset_size=50,
        feature_sets=["grim_age2", "systems_age", "maple", "methylgpt"]
    )

    results = ablation.run_feature_ablation()

    logger.info("")
    logger.info("ABLATION STUDY CONFIGURATION:")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")

    logger.info("")
    logger.info("✓ Pipeline architecture ready for ablation studies!")
    logger.info("  Next: Implement feature-specific finetuning configs")
    logger.info("  Then: Run per-feature ablation and compare ROC-AUC deltas")
    logger.info("")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    )
    sys.exit(main())
