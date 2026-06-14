#!/usr/bin/env python3
"""
Quick Evaluation Demo: Show comprehensive metrics comparison without full training.

This demonstrates:
1. 2-condition comparison (EHR only vs. EHR + bio proxies)
2. Comprehensive evaluation metrics
3. Statistical significance testing

No environment setup or full training required.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("quick_demo")


def generate_mock_predictions():
    """Generate realistic mock predictions for 2 conditions."""
    np.random.seed(42)

    n_test = 100

    # True labels: mortality outcomes
    base_mortality_rate = 0.15
    true_labels = np.random.binomial(1, base_mortality_rate, n_test)

    # Condition 1: EHR only (baseline, weaker predictions)
    # Random performance, slightly biased toward mortality prevalence
    predictions_ehr_only = np.random.beta(2, 5, n_test)  # Concentrated toward 0-0.4
    predictions_ehr_only[true_labels == 1] += 0.15  # Slight boost for positive class
    predictions_ehr_only = np.clip(predictions_ehr_only, 0, 1)

    # Condition 2: EHR + bio proxies (better predictions)
    # Better separation due to added features
    predictions_with_values = np.random.beta(2, 5, n_test)
    predictions_with_values[true_labels == 1] += 0.35  # Stronger boost
    predictions_with_values = np.clip(predictions_with_values, 0, 1)

    return true_labels, predictions_ehr_only, predictions_with_values


def evaluate_predictions(predictions, true_labels, condition_name):
    """Compute comprehensive evaluation metrics."""
    from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve

    logger.info(f"\n{'='*80}")
    logger.info(f"EVALUATING: {condition_name}")
    logger.info('='*80)

    metrics = {}

    # ROC-AUC
    roc_auc = roc_auc_score(true_labels, predictions)
    metrics['roc_auc'] = roc_auc

    # Bootstrap CI for ROC-AUC
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
    ci_lower = np.percentile(bootstrap_aucs, 2.5)
    ci_upper = np.percentile(bootstrap_aucs, 97.5)
    metrics['roc_auc_ci_lower'] = ci_lower
    metrics['roc_auc_ci_upper'] = ci_upper

    # PR-AUC
    precision, recall, _ = precision_recall_curve(true_labels, predictions)
    pr_auc = auc(recall, precision)
    metrics['pr_auc'] = pr_auc

    # Expected Calibration Error
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

    # Calibration slope (logistic regression coefficient)
    from sklearn.linear_model import LogisticRegression
    try:
        lr = LogisticRegression()
        lr.fit(predictions.reshape(-1, 1), true_labels)
        metrics['calibration_slope'] = float(lr.coef_[0][0])
    except:
        metrics['calibration_slope'] = 1.0

    # Net benefit at 50% threshold
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

    # Integrated Calibration Index (ICI)
    ici = 0
    for percentile in np.linspace(0, 100, 11):
        pred_percentile = np.percentile(predictions, percentile)
        mask = np.abs(predictions - pred_percentile) < 0.05
        if mask.sum() > 1:
            expected = predictions[mask].mean()
            observed = true_labels[mask].mean()
            ici += np.abs(expected - observed)
    metrics['ici'] = ici / 11

    # Hosmer-Lemeshow p-value (simplified)
    from scipy.stats import chi2
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
    hl_pvalue = 1 - chi2.cdf(chi2_stat, n_deciles - 2)
    metrics['hl_pvalue'] = hl_pvalue

    # Print summary
    logger.info(f"\n{condition_name} Metrics:")
    logger.info(f"  ROC-AUC:              {roc_auc:.3f} [95% CI: {ci_lower:.3f}-{ci_upper:.3f}]")
    logger.info(f"  PR-AUC:               {pr_auc:.3f}")
    logger.info(f"  ECE:                  {ece:.3f}")
    logger.info(f"  Calibration slope:    {metrics['calibration_slope']:.3f}")
    logger.info(f"  Hosmer-Lemeshow p:    {hl_pvalue:.3f}")
    logger.info(f"  Net Benefit (50%):    {metrics['net_benefit_at_50pct']:.3f}")
    logger.info(f"  Sensitivity @ 90% Sp: {metrics['sensitivity_at_90spec']:.3f}")
    logger.info(f"  ICI:                  {metrics['ici']:.3f}")

    return metrics


def main():
    """Run quick demo."""
    logger.info("\n" + "╔" + "="*78 + "╗")
    logger.info("║" + " "*20 + "QUICK EVALUATION DEMO" + " "*37 + "║")
    logger.info("║" + " "*15 + "2-Condition Comparison with Metrics" + " "*28 + "║")
    logger.info("╚" + "="*78 + "╝\n")

    # Generate mock predictions
    logger.info("Generating mock predictions...")
    true_labels, pred_ehr_only, pred_with_values = generate_mock_predictions()
    logger.info(f"✓ Generated {len(true_labels)} test samples")
    logger.info(f"  Mortality rate: {true_labels.mean():.1%}")

    # Evaluate both conditions
    metrics_ehr = evaluate_predictions(pred_ehr_only, true_labels, "EHR Only")
    metrics_values = evaluate_predictions(pred_with_values, true_labels, "EHR + Bio Proxies")

    # Comparison table
    logger.info("\n" + "="*80)
    logger.info("COMPARISON TABLE: WITHOUT VS WITH VALUE INJECTION")
    logger.info("="*80 + "\n")

    comparison_data = {
        "Condition": ["EHR Only", "EHR + Values"],
        "ROC-AUC": [f"{metrics_ehr['roc_auc']:.3f}", f"{metrics_values['roc_auc']:.3f}"],
        "95% CI": [
            f"[{metrics_ehr['roc_auc_ci_lower']:.3f}-{metrics_ehr['roc_auc_ci_upper']:.3f}]",
            f"[{metrics_values['roc_auc_ci_lower']:.3f}-{metrics_values['roc_auc_ci_upper']:.3f}]"
        ],
        "PR-AUC": [f"{metrics_ehr['pr_auc']:.3f}", f"{metrics_values['pr_auc']:.3f}"],
        "ECE": [f"{metrics_ehr['ece']:.3f}", f"{metrics_values['ece']:.3f}"],
        "Cal. Slope": [f"{metrics_ehr['calibration_slope']:.3f}", f"{metrics_values['calibration_slope']:.3f}"],
        "Net Benefit": [f"{metrics_ehr['net_benefit_at_50pct']:.3f}", f"{metrics_values['net_benefit_at_50pct']:.3f}"],
        "HL p-value": [f"{metrics_ehr['hl_pvalue']:.3f}", f"{metrics_values['hl_pvalue']:.3f}"],
    }

    comparison_df = pd.DataFrame(comparison_data)
    logger.info(comparison_df.to_string(index=False))

    # Delta ROC-AUC
    delta_roc = metrics_values['roc_auc'] - metrics_ehr['roc_auc']
    delta_ci_lower = metrics_values['roc_auc_ci_lower'] - metrics_ehr['roc_auc_ci_upper']
    delta_ci_upper = metrics_values['roc_auc_ci_upper'] - metrics_ehr['roc_auc_ci_lower']

    logger.info("\n" + "="*80)
    logger.info("IMPROVEMENT ANALYSIS")
    logger.info("="*80)
    logger.info(f"\nΔ ROC-AUC (With Values - EHR Only):  {delta_roc:+.3f}")
    logger.info(f"95% CI:                             [{delta_ci_lower:.3f} to {delta_ci_upper:.3f}]")
    logger.info(f"Δ PR-AUC:                           {metrics_values['pr_auc'] - metrics_ehr['pr_auc']:+.3f}")
    logger.info(f"Δ ECE:                              {metrics_values['ece'] - metrics_ehr['ece']:+.3f} (lower is better)")
    logger.info(f"Δ Net Benefit:                      {metrics_values['net_benefit_at_50pct'] - metrics_ehr['net_benefit_at_50pct']:+.3f}")

    # Significance
    if delta_roc > 0 and delta_ci_lower > 0:
        logger.info(f"\n✓ Improvement is SIGNIFICANT (p < 0.05)")
    elif delta_roc > 0:
        logger.info(f"\n⚠ Improvement observed but NOT SIGNIFICANT (p ≥ 0.05)")
    else:
        logger.info(f"\n✗ No improvement detected")

    # Save comparison
    output_file = Path("./outputs/quick_demo_comparison.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Comparison saved to {output_file}")

    logger.info("\n" + "="*80)
    logger.info("DEMO COMPLETE")
    logger.info("="*80)
    logger.info("""
This demo shows:
  ✓ How comprehensive evaluation compares 2 conditions
  ✓ Statistical significance testing (ROC-AUC with bootstrap CI)
  ✓ Multiple metrics (discrimination, calibration, clinical utility)
  ✓ Impact quantification (Δ ROC-AUC = {:.3f})

To run with your actual data:
  1. Prepare MEDS format EHR data (ehr2meds pipeline output)
  2. Prepare biological proxy CSVs (epimap pipeline output)
  3. Run: python scripts/run_end_to_end_pipeline.py

The framework is fully set up and ready to use!
""".format(delta_roc))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
