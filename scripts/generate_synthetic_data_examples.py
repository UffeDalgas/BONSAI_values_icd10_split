#!/usr/bin/env python3
"""
Generate synthetic data examples for end-to-end pipeline testing.

Creates:
1. Synthetic MEDS data (EHR events)
2. Synthetic biological proxy files (GrimAge, SystemsAge, embeddings)
3. Synthetic outcomes (mortality)

All with proper alignment and realistic distributions.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_synthetic_data")


def generate_synthetic_meds_data(n_patients: int = 200, output_dir: str = "./data/synthetic"):
    """Generate synthetic MEDS format EHR data."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nGenerating synthetic MEDS data ({n_patients} patients)...")

    # Common medical concepts
    diagnosis_codes = [
        "D74.0", "D74.1", "D74.9",  # Diabetes variants
        "I50.0", "I50.1", "I50.9",  # Heart failure variants
        "J44.0", "J44.1", "J44.9",  # COPD variants
        "K74.0", "K74.1", "K74.9",  # Liver disease variants
    ]

    medication_codes = [
        "M10.001", "M10.002", "M10.003",  # Insulin variants
        "M20.001", "M20.002",  # ACE inhibitors
        "M30.001", "M30.002",  # Beta blockers
        "M40.001", "M40.002",  # Statins
    ]

    lab_codes = {
        "L100": ("glucose", "mg/dL", 70, 150),
        "L101": ("crp", "mg/L", 0.5, 10),
        "L102": ("wbc", "K/uL", 4, 11),
        "L103": ("rbc", "M/uL", 4, 6),
        "L104": ("platelets", "K/uL", 150, 400),
        "L105": ("cholesterol", "mg/dL", 150, 300),
        "L106": ("ldl", "mg/dL", 50, 200),
        "L107": ("hdl", "mg/dL", 20, 80),
    }

    # Generate events
    events = []
    base_date = datetime(2019, 1, 1)

    for person_id in range(1, n_patients + 1):
        # Number of events per patient (Poisson distributed)
        n_events = np.random.poisson(30)

        for _ in range(n_events):
            # Random date within 2 years
            days_offset = np.random.randint(0, 730)
            event_date = base_date + timedelta(days=days_offset)

            # Choose event type
            event_type = np.random.choice(["diagnosis", "medication", "lab"], p=[0.2, 0.3, 0.5])

            if event_type == "diagnosis":
                concept_id = np.random.choice(diagnosis_codes)
                value = None
            elif event_type == "medication":
                concept_id = np.random.choice(medication_codes)
                value = None
            else:  # lab
                concept_id = np.random.choice(list(lab_codes.keys()))
                lab_name, unit, min_val, max_val = lab_codes[concept_id]
                # Realistic values with some autocorrelation
                value = np.random.normal((min_val + max_val) / 2, (max_val - min_val) / 6)
                value = np.clip(value, min_val, max_val)

            events.append({
                "person_id": person_id,
                "event_timestamp": event_date,
                "concept_id": concept_id,
                "value": value,
                "unit": "N/A" if value is None else "varies",
            })

    # Create DataFrame and save
    meds_df = pd.DataFrame(events)
    meds_df = meds_df.sort_values(["person_id", "event_timestamp"]).reset_index(drop=True)

    # Save as parquet (mimicking epimap output)
    meds_parquet = output_dir / "meds_synthetic.parquet"
    meds_df.to_parquet(meds_parquet)

    logger.info(f"✓ Created {meds_parquet}")
    logger.info(f"  Events: {len(meds_df)}")
    logger.info(f"  Patients: {meds_df['person_id'].nunique()}")

    return meds_df


def generate_synthetic_biological_proxies(n_patients: int = 200, output_dir: str = "./data/synthetic", coverage: float = 0.6):
    """Generate synthetic biological proxy files from epimap.

    Args:
        coverage: fraction of patients with each measurement (default 60%)
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nGenerating synthetic biological proxies ({n_patients} patients, {coverage*100:.0f}% coverage)...")

    patients = np.arange(1, n_patients + 1)

    # ========================================================================
    # 1. GrimAge v2
    # ========================================================================
    n_grim = int(n_patients * coverage)
    grim_patients = np.random.choice(patients, n_grim, replace=False)

    grim_age_df = pd.DataFrame({
        "person_id": grim_patients,
        "grim_age_v2": np.random.uniform(0.3, 0.9, n_grim),
        "dnam_glucose": np.random.uniform(0.2, 0.8, n_grim),
        "dnam_crp": np.random.uniform(0.1, 0.9, n_grim),
        "measurement_date": pd.date_range("2020-01-01", periods=n_grim, freq="D"),
    })

    grim_age_file = output_dir / "GrimAge_v2.csv"
    grim_age_df.to_csv(grim_age_file, index=False)
    logger.info(f"✓ Created {grim_age_file} ({n_grim}/{n_patients} patients)")

    # ========================================================================
    # 2. SystemsAge Components (11 features)
    # ========================================================================
    n_systems = int(n_patients * coverage)
    systems_patients = np.random.choice(patients, n_systems, replace=False)

    systems_age_df = pd.DataFrame({
        "person_id": systems_patients,
        "systems_age": np.random.uniform(0.2, 0.8, n_systems),
        "glucose": np.random.normal(100, 20, n_systems),
        "crp": np.random.exponential(2, n_systems),
        "wbc": np.random.normal(7, 1.5, n_systems),
        "rbc": np.random.normal(5, 0.5, n_systems),
        "platelets": np.random.normal(250, 50, n_systems),
        "bp_systolic": np.random.normal(130, 15, n_systems),
        "bp_diastolic": np.random.normal(80, 10, n_systems),
        "cholesterol": np.random.normal(200, 40, n_systems),
        "ldl": np.random.normal(120, 30, n_systems),
        "hdl": np.random.normal(50, 10, n_systems),
        "measurement_date": pd.date_range("2020-02-01", periods=n_systems, freq="D"),
    })

    systems_age_file = output_dir / "SystemsAge_components.csv"
    systems_age_df.to_csv(systems_age_file, index=False)
    logger.info(f"✓ Created {systems_age_file} ({n_systems}/{n_patients} patients)")

    # ========================================================================
    # 3. MAPLE Embeddings (32-dimensional)
    # ========================================================================
    n_maple = int(n_patients * coverage)
    maple_patients = np.random.choice(patients, n_maple, replace=False)
    maple_data = np.random.normal(0, 0.5, (n_maple, 32))
    maple_df = pd.DataFrame(
        maple_data,
        columns=[f"maple_dim_{i+1}" for i in range(32)]
    )
    maple_df.insert(0, "person_id", maple_patients)
    maple_df["measurement_date"] = pd.date_range("2020-03-01", periods=n_maple, freq="D")

    maple_file = output_dir / "MAPLE_embeddings.csv"
    maple_df.to_csv(maple_file, index=False)
    logger.info(f"✓ Created {maple_file} (32-dimensional, {n_maple}/{n_patients} patients)")

    # ========================================================================
    # 4. CpGPT Embeddings (64-dimensional)
    # ========================================================================
    n_cpgt = int(n_patients * coverage)
    cpgt_patients = np.random.choice(patients, n_cpgt, replace=False)
    cpgt_data = np.random.normal(0, 0.5, (n_cpgt, 64))
    cpgt_df = pd.DataFrame(
        cpgt_data,
        columns=[f"cpgt_dim_{i+1}" for i in range(64)]
    )
    cpgt_df.insert(0, "person_id", cpgt_patients)
    cpgt_df["measurement_date"] = pd.date_range("2020-04-01", periods=n_cpgt, freq="D")

    cpgt_file = output_dir / "CpGPT_embeddings.csv"
    cpgt_df.to_csv(cpgt_file, index=False)
    logger.info(f"✓ Created {cpgt_file} (64-dimensional, {n_cpgt}/{n_patients} patients)")

    # ========================================================================
    # 5. MethylGPT Embeddings
    # ========================================================================
    n_methylgpt = int(n_patients * coverage)
    methylgpt_patients = np.random.choice(patients, n_methylgpt, replace=False)
    methylgpt_data = np.random.normal(0, 0.5, (n_methylgpt, 48))
    methylgpt_df = pd.DataFrame(
        methylgpt_data,
        columns=[f"methylgpt_dim_{i+1}" for i in range(48)]
    )
    methylgpt_df.insert(0, "person_id", methylgpt_patients)
    methylgpt_df["measurement_date"] = pd.date_range("2020-05-01", periods=n_methylgpt, freq="D")

    methylgpt_file = output_dir / "MethylGPT_embeddings.csv"
    methylgpt_df.to_csv(methylgpt_file, index=False)
    logger.info(f"✓ Created {methylgpt_file} ({n_methylgpt}/{n_patients} patients)")

    return grim_age_df, systems_age_df, maple_df, cpgt_df, methylgpt_df


def generate_synthetic_outcomes(n_patients: int = 200, disease_prevalence: float = 0.15,
                               outcome_dir: str = "./outputs/outcomes"):
    """Generate synthetic outcomes (mortality) with realistic distributions."""

    outcome_dir = Path(outcome_dir)
    outcome_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nGenerating synthetic outcomes ({n_patients} patients)...")

    patients = np.arange(1, n_patients + 1)

    # Disease assignment (ICD code: D74 = diabetes)
    has_disease = np.random.random(n_patients) < disease_prevalence

    # Mortality risk based on disease status and age
    # Higher risk if have disease
    base_mortality_rate = 0.05
    disease_multiplier = 3.0

    mortality_prob = np.where(
        has_disease,
        base_mortality_rate * disease_multiplier,
        base_mortality_rate
    )

    # Add some noise
    mortality_prob = np.clip(mortality_prob + np.random.normal(0, 0.02, n_patients), 0, 1)

    # Sample mortality outcomes
    outcomes = np.random.binomial(1, mortality_prob)

    # Follow-up days (longer if alive, shorter if dead)
    follow_up_days = np.where(
        outcomes == 1,
        np.random.randint(10, 100, n_patients),  # Dead: short follow-up
        np.random.randint(200, 365, n_patients)  # Alive: long follow-up
    )

    # Create outcomes dataframe
    outcomes_df = pd.DataFrame({
        "person_id": patients,
        "disease_indicator": has_disease.astype(int),
        "disease_code": np.where(has_disease, "D74", ""),
        "mortality": outcomes,
        "follow_up_days": follow_up_days,
        "outcome_date": pd.date_range("2020-06-01", periods=n_patients, freq="D"),
    })

    outcomes_file = outcome_dir / "TEST_OUTCOME.csv"
    outcomes_df.to_csv(outcomes_file, index=False)

    logger.info(f"✓ Created {outcomes_file}")
    logger.info(f"  Total patients: {len(outcomes_df)}")
    logger.info(f"  Disease patients: {has_disease.sum()} ({has_disease.sum()/n_patients*100:.1f}%)")
    logger.info(f"  Mortality rate: {outcomes.mean():.1%}")
    logger.info(f"  Mortality rate (disease): {outcomes[has_disease].mean():.1%}")
    logger.info(f"  Mortality rate (no disease): {outcomes[~has_disease].mean():.1%}")

    return outcomes_df


def generate_data_integration_example():
    """Show how to use the synthetic data with the pipeline."""

    example_script = """
# EXAMPLE: Using synthetic data with the pipeline

import pandas as pd
from corebehrt.functional.data.disease_filtering import DiseaseFilter
from corebehrt.ablation.batch_ablation_runner import BatchAblationRunner

# ========================================================================
# 1. Load MEDS data
# ========================================================================
meds_data = pd.read_parquet("./data/synthetic/meds_synthetic.parquet")
print(f"MEDS data: {len(meds_data)} events, {meds_data['person_id'].nunique()} patients")

# ========================================================================
# 2. Filter by disease (diabetes = "D74*")
# ========================================================================
disease_filter = DiseaseFilter("D74*")
disease_patients = disease_filter.identify_patients_with_disease(meds_data)
print(f"Patients with D74* disease: {len(disease_patients)}")

# ========================================================================
# 3. Load biological proxies and identify finetune set
# ========================================================================
grim_age = pd.read_csv("./data/synthetic/GrimAge_v2.csv")
systems_age = pd.read_csv("./data/synthetic/SystemsAge_components.csv")
maple = pd.read_csv("./data/synthetic/MAPLE_embeddings.csv")

# Finetune patients = disease patients with bio proxies
finetune_patients = set(grim_age['person_id']) & disease_patients
print(f"Finetune patients (disease + bio proxies): {len(finetune_patients)}")

# ========================================================================
# 4. Create disease-stratified splits
# ========================================================================
pretrain, finetune, eval = disease_filter.split_by_disease(
    meds_data,
    finetune_disease_patients=finetune_patients
)
print(f"Pretrain: {pretrain['person_id'].nunique()} patients")
print(f"Finetune: {finetune['person_id'].nunique()} patients")
print(f"Eval: {eval['person_id'].nunique()} patients")

# ========================================================================
# 5. Run multi-model ablation
# ========================================================================
runner = BatchAblationRunner(
    pretrain_checkpoint="./outputs/pretraining_dryrun/checkpoints/best.pt",
    output_dir="./outputs/ablation_results",
    n_workers=4
)

# Add configs for different feature subsets and injection methods
runner.add_ablation_config(
    name="diabetes_ehr_only",
    config_path="./corebehrt/configs/ablation_ehr_only_concat.yaml",
    features=[],
    description="EHR only (baseline)"
)

runner.add_ablation_config(
    name="diabetes_grim_age_film",
    config_path="./corebehrt/configs/ablation_grim_age_film.yaml",
    features=["grim_age_v2"],
    description="GrimAge v2 with FiLM injection"
)

# Train and evaluate
runner.train_all_models()
runner.evaluate_all_models()
comparison_df = runner.generate_comparison_report()

print(comparison_df)
"""

    return example_script


def main():
    """Generate all synthetic data."""

    logger.info("\n" + "="*80)
    logger.info("GENERATING SYNTHETIC DATA FOR END-TO-END TESTING")
    logger.info("="*80)

    # Generate all components
    meds_df = generate_synthetic_meds_data(n_patients=200)
    grim_age_df, systems_age_df, maple_df, cpgt_df, methylgpt_df = \
        generate_synthetic_biological_proxies(n_patients=200)
    outcomes_df = generate_synthetic_outcomes(n_patients=200, disease_prevalence=0.15)

    # Show example usage
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE: How to use synthetic data")
    logger.info("="*80)

    example = generate_data_integration_example()
    logger.info(example)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SYNTHETIC DATA GENERATED SUCCESSFULLY")
    logger.info("="*80)
    logger.info("""
Files created:
  Synthetic MEDS data:      ./data/synthetic/meds_synthetic.parquet
  Biological proxies:
    - GrimAge v2:           ./data/synthetic/GrimAge_v2.csv
    - SystemsAge:           ./data/synthetic/SystemsAge_components.csv
    - MAPLE embeddings:     ./data/synthetic/MAPLE_embeddings.csv
    - CpGPT embeddings:     ./data/synthetic/CpGPT_embeddings.csv
    - MethylGPT embeddings: ./data/synthetic/MethylGPT_embeddings.csv
  Outcomes:                 ./outputs/outcomes/TEST_OUTCOME.csv

Next steps:
  1. Verify data formats
  2. Create disease-stratified splits:
     python scripts/create_disease_stratified_ablation.py \\
       --disease-code D74* \\
       --meds-path ./data/synthetic/ \\
       --output-dir ./outputs/disease_stratified

  3. Run end-to-end pipeline:
     python scripts/run_full_ablation_with_methods.py
""")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
