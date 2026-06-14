"""
Disease-specific data filtering for ablation studies.

This module handles disease-stratified filtering of MEDS data:
1. Exclude a disease from pretraining
2. Select disease-specific subset WITH biological proxies (finetune)
3. Select disease-specific subset WITHOUT proxies (evaluation)

Disease codes are identified by their prefix (e.g., "D" for diagnoses in Danish registers).
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Set
import pandas as pd
import numpy as np

logger = logging.getLogger("disease_filtering")


class DiseaseFilter:
    """Filter MEDS data by disease codes."""

    def __init__(self, disease_code_pattern: str, case_sensitive: bool = False):
        """
        Args:
            disease_code_pattern: Pattern to match disease codes (e.g., "D74*" for diabetes)
                                 Supports wildcards: "D74*" matches D74.0, D74.1, etc.
            case_sensitive: Whether to match case-sensitively
        """
        self.disease_code_pattern = disease_code_pattern
        self.case_sensitive = case_sensitive
        self.disease_codes: Set[str] = set()

    def _matches_pattern(self, code: str) -> bool:
        """Check if a code matches the disease pattern."""
        if not self.case_sensitive:
            code = code.upper()
            pattern = self.disease_code_pattern.upper()
        else:
            pattern = self.disease_code_pattern

        if "*" in pattern:
            # Wildcard matching: "D74*" matches anything starting with "D74"
            prefix = pattern.replace("*", "")
            return code.startswith(prefix)
        else:
            # Exact matching or substring
            return code == pattern or code.startswith(pattern)

    def identify_patients_with_disease(
        self,
        meds_data: pd.DataFrame,
        concept_column: str = "concept_id"
    ) -> Set[int]:
        """
        Identify all patients that have at least one event with the disease code.

        Args:
            meds_data: DataFrame with MEDS data (must have person_id and concept_column)
            concept_column: Column name containing medical concept codes

        Returns:
            Set of person_ids that have the disease
        """
        # Filter rows matching disease pattern
        disease_mask = meds_data[concept_column].apply(self._matches_pattern)
        disease_patients = set(meds_data[disease_mask]["person_id"].unique())

        logger.info(f"Found {len(disease_patients)} patients with disease pattern {self.disease_code_pattern}")
        return disease_patients

    def split_by_disease(
        self,
        meds_data: pd.DataFrame,
        finetune_disease_patients: Optional[Set[int]] = None,
        concept_column: str = "concept_id"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split MEDS data into three cohorts:
        1. Pretrain cohort: All patients EXCEPT those with the disease
        2. Finetune cohort: Disease patients WITH biological proxies
        3. Eval cohort: Disease patients WITHOUT biological proxies

        Args:
            meds_data: Full MEDS dataset
            finetune_disease_patients: Set of patient IDs with biological proxies
            concept_column: Column name for medical concepts

        Returns:
            (pretrain_data, finetune_data, eval_data) - three DataFrames
        """
        # Identify all patients with the disease
        all_disease_patients = self.identify_patients_with_disease(meds_data, concept_column)

        # Split disease patients into finetune (with proxies) and eval (without)
        if finetune_disease_patients is None:
            finetune_disease_patients = set()

        eval_disease_patients = all_disease_patients - finetune_disease_patients

        # Create cohorts
        pretrain_mask = ~meds_data["person_id"].isin(all_disease_patients)
        finetune_mask = meds_data["person_id"].isin(finetune_disease_patients)
        eval_mask = meds_data["person_id"].isin(eval_disease_patients)

        pretrain_data = meds_data[pretrain_mask].copy()
        finetune_data = meds_data[finetune_mask].copy()
        eval_data = meds_data[eval_mask].copy()

        logger.info(f"\n=== DISEASE-STRATIFIED COHORT SPLIT ===")
        logger.info(f"Disease pattern: {self.disease_code_pattern}")
        logger.info(f"Pretrain cohort (no disease): {pretrain_data['person_id'].nunique()} patients")
        logger.info(f"Finetune cohort (disease + proxies): {finetune_data['person_id'].nunique()} patients")
        logger.info(f"Eval cohort (disease, no proxies): {eval_data['person_id'].nunique()} patients")
        logger.info(f"\nPretrain events: {len(pretrain_data)}")
        logger.info(f"Finetune events: {len(finetune_data)}")
        logger.info(f"Eval events: {len(eval_data)}")

        return pretrain_data, finetune_data, eval_data


class DiseaseAwareDataPipeline:
    """
    End-to-end pipeline that applies disease filtering to MEDS data.

    Usage:
        pipeline = DiseaseAwareDataPipeline(
            meds_data_path="./outputs/tokenized/",
            disease_code="D74*",  # Diabetes codes
            finetune_disease_patients=set([101, 102, 103, ...])  # Patients with bio proxies
        )

        pretrain, finetune, eval = pipeline.create_disease_stratified_splits()
    """

    def __init__(
        self,
        meds_data_path: str,
        disease_code: str,
        finetune_disease_patients: Set[int],
        output_dir: str = "./outputs/disease_stratified",
    ):
        """
        Args:
            meds_data_path: Path to MEDS data directory
            disease_code: Disease pattern (e.g., "D74*" for diabetes)
            finetune_disease_patients: Patient IDs with biological proxies
            output_dir: Where to save stratified data
        """
        self.meds_data_path = Path(meds_data_path)
        self.disease_code = disease_code
        self.finetune_disease_patients = finetune_disease_patients
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.filter = DiseaseFilter(disease_code)

    def load_meds_data(self) -> pd.DataFrame:
        """Load MEDS data from parquet files."""
        logger.info(f"Loading MEDS data from {self.meds_data_path}")

        parquet_files = list(self.meds_data_path.glob("**/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.meds_data_path}")

        dfs = [pd.read_parquet(f) for f in parquet_files]
        meds_data = pd.concat(dfs, ignore_index=True)

        logger.info(f"Loaded {len(meds_data)} events from {len(parquet_files)} files")
        return meds_data

    def create_disease_stratified_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load MEDS data and create disease-stratified splits.

        Returns:
            (pretrain_data, finetune_data, eval_data)
        """
        meds_data = self.load_meds_data()
        pretrain, finetune, eval = self.filter.split_by_disease(
            meds_data,
            self.finetune_disease_patients
        )

        # Save splits
        logger.info(f"\nSaving splits to {self.output_dir}")

        pretrain.to_parquet(self.output_dir / "pretrain_cohort.parquet")
        finetune.to_parquet(self.output_dir / "finetune_cohort.parquet")
        eval.to_parquet(self.output_dir / "eval_cohort.parquet")

        logger.info("✓ Cohorts saved successfully")

        return pretrain, finetune, eval

    def create_cohort_metadata(self) -> dict:
        """
        Create metadata file describing the cohort split.

        Returns:
            Dictionary with cohort statistics
        """
        meds_data = self.load_meds_data()
        _, finetune, eval = self.filter.split_by_disease(
            meds_data,
            self.finetune_disease_patients
        )

        metadata = {
            "disease_code": self.disease_code,
            "finetune_patients": len(finetune["person_id"].unique()),
            "finetune_events": len(finetune),
            "eval_patients": len(eval["person_id"].unique()),
            "eval_events": len(eval),
            "finetune_patient_ids": sorted(list(self.finetune_disease_patients)),
            "eval_patient_ids": sorted(list(set(eval["person_id"].unique()))),
        }

        return metadata


# ============================================================================
# SIMPLE CONFIG-BASED INTERFACE
# ============================================================================

def create_disease_stratified_cohorts_from_config(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create disease-stratified cohorts from a config dictionary.

    Config format:
    {
        "disease_code": "D74*",  # Disease pattern (e.g., "D74*" for diabetes)
        "meds_data_path": "./outputs/tokenized/",
        "finetune_disease_patients": [101, 102, 103, ...],  # Patient IDs with bio proxies
        "output_dir": "./outputs/disease_stratified"
    }
    """
    pipeline = DiseaseAwareDataPipeline(
        meds_data_path=config["meds_data_path"],
        disease_code=config["disease_code"],
        finetune_disease_patients=set(config.get("finetune_disease_patients", [])),
        output_dir=config.get("output_dir", "./outputs/disease_stratified"),
    )

    return pipeline.create_disease_stratified_splits()
