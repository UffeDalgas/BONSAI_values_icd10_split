#!/usr/bin/env python3
"""
Quick End-to-End Demo: 2 Conditions with Evaluation Metrics

This simplified demo shows:
1. How to manually select finetune patients (those with values)
2. How to prepare data for 2 conditions
3. How to evaluate both with comprehensive metrics

For demonstration, we use:
- Synthetic MEDS data
- Synthetic biological proxies
- Synthetic outcomes
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("demo")


def demo():
    """Run quick demonstration."""
    
    logger.info("\n" + "="*80)
    logger.info("QUICK END-TO-END DEMO: 2 CONDITIONS WITH EVALUATION")
    logger.info("="*80 + "\n")

    # ========================================================================
    # SETUP: Generate synthetic data
    # ========================================================================
    logger.info("SETUP: Generating synthetic data...")
    
    # Synthetic patients with and without biological values
    n_total = 100
    n_with_values = 60  # 60 patients have biological measurements
    
    all_patients = np.arange(1, n_total + 1)
    patients_with_values = np.arange(1, n_with_values + 1)
    
    logger.info(f"  Total patients: {n_total}")
    logger.info(f"  Patients with biological measurements: {n_with_values}")
    logger.info(f"  Patients without measurements: {n_total - n_with_values}\n")
    
    # Create synthetic biological values (GrimAge, SystemsAge)
    bio_values = pd.DataFrame({
        'person_id': patients_with_values,
        'grim_age_v2': np.random.uniform(0.3, 0.9, n_with_values),
        'systems_age': np.random.uniform(0.2, 0.8, n_with_values),
    })
    
    # Create synthetic outcomes
    outcomes = pd.DataFrame({
        'person_id': all_patients,
        'mortality': np.random.binomial(1, 0.15, n_total),
    })
    
    logger.info("✓ Synthetic data created\n")
    
    # ========================================================================
    # STEP 1: Manual patient selection for finetune
    # ========================================================================
    logger.info("STEP 1: MANUAL PATIENT SELECTION FOR FINETUNE")
    logger.info("-" * 80)
    
    # Identify patients with biological values
    finetune_patients = set(bio_values['person_id'])
    eval_patients = set(all_patients) - finetune_patients
    
    logger.info(f"Patients available for finetune (have values): {len(finetune_patients)}")
    logger.info(f"Patients for evaluation (no values, test set): {len(eval_patients)}")
    logger.info(f"Person IDs with values: {sorted(finetune_patients)[:10]}... (showing first 10)\n")
    
    # ========================================================================
    # STEP 2: Create mock model outputs for 2 conditions
    # ========================================================================
    logger.info("STEP 2: SIMULATING MODEL OUTPUTS FOR 2 CONDITIONS")
    logger.info("-" * 80)
    
    # For demonstration, create mock predictions for eval set
    eval_patient_list = sorted(list(eval_patients))
    n_eval = len(eval_patient_list)
    
    # Condition 1: EHR only (baseline)
    predictions_ehr_only = np.random.uniform(0.3, 0.7, n_eval)
    
    # Condition 2: EHR + values (slightly better)
    # Correlate with actual outcomes where possible
    outcome_vals = outcomes[outcomes['person_id'].isin(eval_patient_list)]['mortality'].values
    predictions_with_values = predictions_ehr_only + 0.15 * outcome_vals + np.random.normal(0, 0.05, n_eval)
    predictions_with_values = np.clip(predictions_with_values, 0, 1)
    
    logger.info(f"Condition 1 (EHR only) predictions created")
    logger.info(f"  Mean prediction: {predictions_ehr_only.mean():.3f}")
    logger.info(f"  Std: {predictions_ehr_only.std():.3f}")
    
    logger.info(f"\nCondition 2 (EHR + values) predictions created")
    logger.info(f"  Mean prediction: {predictions_with_values.mean():.3f}")
    logger.info(f"  Std: {predictions_with_values.std():.3f}\n")
    
    # ========================================================================
    # STEP 3: Evaluate both conditions
    # ========================================================================
    logger.info("STEP 3: COMPREHENSIVE EVALUATION")
    logger.info("-" * 80 + "\n")
    
    from corebehrt.functional.evaluation.comprehensive_metrics import ComprehensiveModelEvaluation
    
    true_labels = outcomes[outcomes['person_id'].isin(eval_patient_list)]['mortality'].values
    
    results = []
    
    for condition_name, predictions in [
        ("EHR Only (Baseline)", predictions_ehr_only),
        ("EHR + Values", predictions_with_values)
    ]:
        logger.info(f"Evaluating: {condition_name}")
        logger.info("-" * 60)
        
        evaluator = ComprehensiveModelEvaluation(predictions, true_labels)
        metrics = evaluator.evaluate_all()
        
        logger.info(evaluator.summary_report())
        
        results.append({
            'Condition': condition_name,
            'ROC-AUC': f"{metrics['roc_auc']:.3f}",
            'ROC-AUC 95% CI': f"[{metrics['roc_auc_ci_lower']:.3f}-{metrics['roc_auc_ci_upper']:.3f}]",
            'PR-AUC': f"{metrics['pr_auc']:.3f}",
            'ECE': f"{metrics['ece']:.3f}",
            'Calibration Slope': f"{metrics['calibration_slope']:.3f}",
            'HL p-value': f"{metrics['hl_pvalue']:.4f}",
            'Net Benefit @50%': f"{metrics['net_benefit_at_50pct']:.3f}",
        })
    
    # ========================================================================
    # STEP 4: Compare results
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("COMPARISON: WITHOUT VS WITH VALUE INJECTION")
    logger.info("="*80 + "\n")
    
    comparison_df = pd.DataFrame(results)
    logger.info(comparison_df.to_string(index=False))
    
    # Calculate improvements
    roc_auc_1 = float(results[0]['ROC-AUC'])
    roc_auc_2 = float(results[1]['ROC-AUC'])
    delta = roc_auc_2 - roc_auc_1
    
    logger.info(f"\n{'─'*80}")
    logger.info(f"Δ ROC-AUC (With Values - EHR Only): {delta:+.4f}")
    logger.info(f"{'─'*80}\n")
    
    if delta > 0:
        improvement_pct = (delta / roc_auc_1) * 100
        logger.info(f"✓ Value injection improved ROC-AUC by {improvement_pct:.1f}%")
    else:
        logger.info(f"✗ Value injection decreased ROC-AUC by {abs(delta)/roc_auc_1*100:.1f}%")
    
    # ========================================================================
    # KEY INSIGHT: Manual Patient Selection
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("KEY INSIGHT: MANUAL PATIENT SELECTION FOR FINETUNE")
    logger.info("="*80 + "\n")
    
    logger.info(f"""
How to manually select finetune patients:

1. Load biological proxy CSVs (from epimap):
   - GrimAge_v2.csv (60 patients have measurements)
   - SystemsAge_components.csv (maybe 55 patients)
   - MAPLE_embeddings.csv (maybe 58 patients)

2. Find intersection (patients with ANY measurements):
   finetune_patients = set(grim.person_id) | set(systems.person_id) | ...
   finetune_patients = {len(finetune_patients)} patients

3. Split cohorts:
   - Finetune: patients with biological values ({n_with_values} patients)
   - Eval (test): patients without values ({len(eval_patients)} patients)
   
4. This ensures:
   ✓ Clean separation: finetune data has features, test data doesn't
   ✓ No data leakage: biological values only in finetune
   ✓ Fair evaluation: test set represents real deployment scenario
   ✓ Manual control: you choose who has measurements based on actual data
""")
    
    logger.info("="*80)
    logger.info("✓ DEMO COMPLETE")
    logger.info("="*80 + "\n")
    
    return comparison_df


if __name__ == "__main__":
    demo()
