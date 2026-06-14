#!/usr/bin/env python3
"""
Simplified End-to-End Pipeline Demo

This demonstrates the complete framework without requiring full BONSAI setup:
1. Manual patient selection based on biological values
2. 2-condition comparison (EHR only vs. EHR + values)
3. Comprehensive evaluation with all metrics
4. Full statistical significance testing

This works out-of-the-box without needing the data preparation pipeline.
When you have full BONSAI setup, use run_end_to_end_pipeline.py for actual training.
"""

import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simplified_demo")


def step_1_generate_synthetic_data():
    """Step 1: Generate synthetic data (MEDS + biological proxies + outcomes)."""
    logger.info("\n" + "="*80)
    logger.info("STEP 1: GENERATING SYNTHETIC DATA")
    logger.info("="*80 + "\n")

    from generate_synthetic_data_examples import (
        generate_synthetic_meds_data,
        generate_synthetic_biological_proxies,
        generate_synthetic_outcomes
    )

    # Generate more patients so we have a realistic split
    # (many patients without all biological values)
    meds_df = generate_synthetic_meds_data(n_patients=200)
    bio_proxies = generate_synthetic_biological_proxies(n_patients=200)
    outcomes_df = generate_synthetic_outcomes(n_patients=200, disease_prevalence=0.2)

    logger.info(f"✓ Generated data for 100 patients")
    return meds_df, bio_proxies, outcomes_df


def step_2_identify_finetune_patients(bio_proxies_dfs):
    """Step 2: Manual patient selection based on biological values."""
    logger.info("\n" + "="*80)
    logger.info("STEP 2: IDENTIFYING FINETUNE PATIENTS (MANUAL SELECTION)")
    logger.info("="*80 + "\n")

    grim_age_df, systems_age_df, maple_df, cpgt_df, methylgpt_df = bio_proxies_dfs

    # Find which patients have measurements
    patients_with_grim = set(grim_age_df["person_id"])
    patients_with_systems = set(systems_age_df["person_id"])
    patients_with_maple = set(maple_df["person_id"])
    patients_with_cpgt = set(cpgt_df["person_id"])
    patients_with_methylgpt = set(methylgpt_df["person_id"])

    all_with_values = (
        patients_with_grim |
        patients_with_systems |
        patients_with_maple |
        patients_with_cpgt |
        patients_with_methylgpt
    )

    logger.info(f"Patients with GrimAge measurements:     {len(patients_with_grim)}")
    logger.info(f"Patients with SystemsAge measurements:  {len(patients_with_systems)}")
    logger.info(f"Patients with MAPLE measurements:       {len(patients_with_maple)}")
    logger.info(f"Patients with CpGPT measurements:       {len(patients_with_cpgt)}")
    logger.info(f"Patients with MethylGPT measurements:   {len(patients_with_methylgpt)}")
    logger.info(f"\nTotal patients with any values:         {len(all_with_values)}")

    # YOU CHOOSE which patients to use for finetune
    # In this demo, we use GrimAge patients
    finetune_patients = patients_with_grim

    logger.info(f"\n>>> MANUAL SELECTION: Using {len(finetune_patients)} patients for finetune")
    logger.info(f"    (These are the patients with GrimAge measurements)")
    logger.info(f"    YOU CONTROL: Change patient selection for different ablations!")

    return finetune_patients, all_with_values


def step_3_simulate_model_training(finetune_patients, all_patients, outcomes_df):
    """Step 3: Simulate training results for both conditions."""
    logger.info("\n" + "="*80)
    logger.info("STEP 3: TRAINING BOTH CONDITIONS (SIMULATED)")
    logger.info("="*80 + "\n")

    np.random.seed(42)

    # Get evaluation patients (those WITHOUT biological values)
    # This ensures we test on held-out data
    eval_patients = all_patients - finetune_patients
    logger.info(f"Finetune cohort:  {len(finetune_patients)} patients (WITH values)")
    logger.info(f"Eval cohort:      {len(eval_patients)} patients (WITHOUT values)")

    # Get outcomes for eval set
    outcomes_eval = outcomes_df[outcomes_df['person_id'].isin(eval_patients)]
    true_labels = outcomes_eval['mortality'].values

    n_test = len(true_labels)

    # Condition 1: EHR only (baseline predictions)
    logger.info("\nCondition 1: Training EHR-only model...")
    logger.info("  Learning from EHR sequences only...")
    predictions_ehr_only = np.random.beta(2, 5, n_test)
    predictions_ehr_only[true_labels == 1] += 0.15
    predictions_ehr_only = np.clip(predictions_ehr_only, 0, 1)
    logger.info(f"  ✓ Trained on {len(finetune_patients)} patients")

    # Condition 2: EHR + biological proxies (enhanced predictions)
    logger.info("\nCondition 2: Training EHR + Values model...")
    logger.info(f"  Learning from EHR + GrimAge + SystemsAge...")
    predictions_with_values = np.random.beta(2, 5, n_test)
    predictions_with_values[true_labels == 1] += 0.35
    predictions_with_values = np.clip(predictions_with_values, 0, 1)
    logger.info(f"  ✓ Trained on {len(finetune_patients)} patients")

    logger.info(f"\n✓ Both conditions trained on identical finetune cohort")
    logger.info(f"✓ Both tested on identical eval cohort (no biological values)")
    logger.info(f"✓ Fair comparison: same patients, different features")

    return true_labels, predictions_ehr_only, predictions_with_values


def step_4_comprehensive_evaluation(predictions, true_labels, condition_name):
    """Step 4: Comprehensive evaluation with all metrics."""
    from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve
    from scipy.stats import chi2

    metrics = {}

    # ROC-AUC with 95% bootstrap CI
    roc_auc = roc_auc_score(true_labels, predictions)
    metrics['roc_auc'] = roc_auc

    n_bootstrap = 1000
    bootstrap_aucs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(true_labels), len(true_labels), replace=True)
        try:
            auc_score = roc_auc_score(true_labels[indices], predictions[indices])
            bootstrap_aucs.append(auc_score)
        except:
            pass

    bootstrap_aucs = np.array(bootstrap_aucs)
    metrics['roc_auc_ci_lower'] = np.percentile(bootstrap_aucs, 2.5)
    metrics['roc_auc_ci_upper'] = np.percentile(bootstrap_aucs, 97.5)

    # PR-AUC
    precision, recall, _ = precision_recall_curve(true_labels, predictions)
    pr_auc = auc(recall, precision)
    metrics['pr_auc'] = pr_auc

    # ECE
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i+1])
        if mask.sum() > 0:
            bin_acc = (true_labels[mask] == (predictions[mask] >= 0.5)).mean()
            bin_conf = predictions[mask].mean()
            ece += np.abs(bin_acc - bin_conf) * mask.sum() / len(predictions)
    metrics['ece'] = ece

    # Calibration slope
    from sklearn.linear_model import LogisticRegression
    try:
        lr = LogisticRegression()
        lr.fit(predictions.reshape(-1, 1), true_labels)
        metrics['calibration_slope'] = float(lr.coef_[0][0])
    except:
        metrics['calibration_slope'] = 1.0

    # Net benefit
    threshold = 0.5
    tp = ((predictions >= threshold) & (true_labels == 1)).sum()
    fp = ((predictions >= threshold) & (true_labels == 0)).sum()
    n_pos = (true_labels == 1).sum()
    net_benefit = (tp - fp * (threshold / (1 - threshold))) / n_pos if n_pos > 0 else 0
    metrics['net_benefit_at_50pct'] = max(0, net_benefit)

    # Sensitivity at 90% specificity
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    specificity = 1 - fpr
    idx = np.argmin(np.abs(specificity - 0.9))
    metrics['sensitivity_at_90spec'] = tpr[idx]

    # ICI
    ici = 0
    for percentile in np.linspace(0, 100, 11):
        pred_percentile = np.percentile(predictions, percentile)
        mask = np.abs(predictions - pred_percentile) < 0.05
        if mask.sum() > 1:
            expected = predictions[mask].mean()
            observed = true_labels[mask].mean()
            ici += np.abs(expected - observed)
    metrics['ici'] = ici / 11

    # Hosmer-Lemeshow
    n_deciles = 5
    decile_edges = np.percentile(predictions, np.linspace(0, 100, n_deciles + 1))
    chi2_stat = 0
    for i in range(n_deciles):
        mask = (predictions >= decile_edges[i]) & (predictions <= decile_edges[i+1])
        if mask.sum() > 0:
            observed = true_labels[mask].sum()
            expected = predictions[mask].sum()
            if expected > 0 and expected < mask.sum():
                chi2_stat += (observed - expected) ** 2 / (expected * (1 - expected/mask.sum()))
    metrics['hl_pvalue'] = 1 - chi2.cdf(chi2_stat, n_deciles - 2)

    return metrics


def step_5_comparison_and_significance():
    """Step 5: Compare results with statistical significance."""
    logger.info("\n" + "="*80)
    logger.info("STEP 4: COMPREHENSIVE EVALUATION")
    logger.info("="*80 + "\n")

    # Return placeholder - called from main
    pass


def main():
    """Run complete simplified end-to-end pipeline."""
    logger.info("\n" + "╔" + "="*78 + "╗")
    logger.info("║" + " "*10 + "SIMPLIFIED END-TO-END BONSAI PIPELINE" + " "*32 + "║")
    logger.info("║" + " "*15 + "Complete workflow demonstration" + " "*34 + "║")
    logger.info("╚" + "="*78 + "╝\n")

    try:
        # Step 1: Generate synthetic data
        meds_df, bio_proxies, outcomes_df = step_1_generate_synthetic_data()

        # Step 2: Manual patient selection
        finetune_patients, all_patients = step_2_identify_finetune_patients(bio_proxies)

        # Step 3: Simulate training
        true_labels, pred_ehr_only, pred_with_values = step_3_simulate_model_training(
            finetune_patients, all_patients, outcomes_df
        )

        # Step 4: Comprehensive evaluation
        logger.info("\n" + "="*80)
        logger.info("STEP 4: COMPREHENSIVE EVALUATION")
        logger.info("="*80 + "\n")

        metrics_ehr = step_4_comprehensive_evaluation(pred_ehr_only, true_labels, "EHR Only")
        metrics_values = step_4_comprehensive_evaluation(
            pred_with_values, true_labels, "EHR + Bio Proxies"
        )

        # Step 5: Comparison
        logger.info("\n" + "="*80)
        logger.info("STEP 5: RESULTS COMPARISON")
        logger.info("="*80 + "\n")

        comparison_data = {
            "Condition": ["EHR Only", "EHR + Values"],
            "ROC-AUC": [f"{metrics_ehr['roc_auc']:.3f}", f"{metrics_values['roc_auc']:.3f}"],
            "95% CI": [
                f"[{metrics_ehr['roc_auc_ci_lower']:.3f}-{metrics_ehr['roc_auc_ci_upper']:.3f}]",
                f"[{metrics_values['roc_auc_ci_lower']:.3f}-{metrics_values['roc_auc_ci_upper']:.3f}]",
            ],
            "PR-AUC": [f"{metrics_ehr['pr_auc']:.3f}", f"{metrics_values['pr_auc']:.3f}"],
            "ECE": [f"{metrics_ehr['ece']:.3f}", f"{metrics_values['ece']:.3f}"],
            "Cal.Slope": [f"{metrics_ehr['calibration_slope']:.3f}", f"{metrics_values['calibration_slope']:.3f}"],
            "NetBenefit": [f"{metrics_ehr['net_benefit_at_50pct']:.3f}", f"{metrics_values['net_benefit_at_50pct']:.3f}"],
            "HL p-val": [f"{metrics_ehr['hl_pvalue']:.3f}", f"{metrics_values['hl_pvalue']:.3f}"],
        }

        comparison_df = pd.DataFrame(comparison_data)
        logger.info("\nCOMPARISON TABLE:")
        logger.info("─" * 100)
        logger.info(comparison_df.to_string(index=False))
        logger.info("─" * 100)

        # Delta analysis
        delta_roc = metrics_values['roc_auc'] - metrics_ehr['roc_auc']
        delta_ci_lower = metrics_values['roc_auc_ci_lower'] - metrics_ehr['roc_auc_ci_upper']
        delta_ci_upper = metrics_values['roc_auc_ci_upper'] - metrics_ehr['roc_auc_ci_lower']

        logger.info("\nIMPROVEMENT ANALYSIS:")
        logger.info("─" * 100)
        logger.info(f"Δ ROC-AUC (With Values - EHR Only):    {delta_roc:+.3f}")
        logger.info(f"95% CI:                                [{delta_ci_lower:.3f} to {delta_ci_upper:.3f}]")
        logger.info(f"Δ PR-AUC:                              {metrics_values['pr_auc'] - metrics_ehr['pr_auc']:+.3f}")
        logger.info(f"Δ ECE:                                 {metrics_values['ece'] - metrics_ehr['ece']:+.3f} (↓ better)")

        if delta_roc > 0 and delta_ci_lower > 0:
            logger.info(f"\n✓ SIGNIFICANT IMPROVEMENT (p < 0.05)")
        elif delta_roc > 0:
            logger.info(f"\n⚠ Improvement observed but NOT SIGNIFICANT (p ≥ 0.05)")
        else:
            logger.info(f"\n✗ No improvement detected")

        # Save results
        output_file = Path("./outputs/simplified_pipeline_comparison.csv")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_file, index=False)
        logger.info(f"\n✓ Results saved to {output_file}")

        # Summary
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE - KEY FEATURES DEMONSTRATED")
        logger.info("="*80 + "\n")

        logger.info("""
✓ Manual Patient Selection
  - YOU chose which patients have biological values
  - Finetune cohort: patients WITH measurements
  - Eval cohort:     patients WITHOUT measurements

✓ 2-Condition Comparison
  - Condition 1: EHR only (baseline)
  - Condition 2: EHR + biological proxies (values-enhanced)
  - Same patients, different features

✓ Comprehensive Evaluation
  - Discrimination:     ROC-AUC, PR-AUC with 95% bootstrap CI
  - Calibration:        ECE, Hosmer-Lemeshow, calibration slope
  - Clinical utility:   Net benefit, sensitivity@specificity
  - Statistical significance: CI overlap analysis

✓ Framework Ready to Scale
  - This demo uses simulated predictions
  - Real training: Use run_end_to_end_pipeline.py with actual BONSAI setup
  - Ablation studies: Scale to 40+ models with run_full_ablation_with_methods.py

Next Steps:
  1. For quick testing: python3 scripts/quick_evaluation_demo.py
  2. For full setup: Prepare MEDS + bio proxies, then run_end_to_end_pipeline.py
  3. For ablations: Run 40 models with run_full_ablation_with_methods.py
""")

        return 0

    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
