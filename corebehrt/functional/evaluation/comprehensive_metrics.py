"""
Comprehensive model evaluation metrics for ablation studies.

Metrics computed for each model:
- Discrimination: ROC-AUC, PR-AUC, rank correlation
- Calibration: Expected calibration error, Hosmer-Lemeshow test, calibration slope
- Clinical utility: Decision curves, net benefit, threshold analysis
- Risk stratification: Quintile analysis, integrated calibration index
- Subgroup analysis: Performance across demographic groups
- Statistical significance: Bootstrap CI, permutation tests
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    auc, precision_recall_curve,
    confusion_matrix, accuracy_score
)

logger = logging.getLogger("comprehensive_metrics")


class ComprehensiveModelEvaluation:
    """Comprehensive evaluation suite for binary classification models."""

    def __init__(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        features: Optional[pd.DataFrame] = None,
        feature_names: Optional[list] = None,
    ):
        """
        Args:
            predictions: Model predictions (probabilities, 0-1)
            labels: Binary labels (0 or 1)
            features: Optional feature matrix for feature importance
            feature_names: Names of features
        """
        self.predictions = np.asarray(predictions)
        self.labels = np.asarray(labels)
        self.features = features
        self.feature_names = feature_names or (
            [f"Feature_{i}" for i in range(features.shape[1])] if features is not None else []
        )

        # Validation
        assert len(self.predictions) == len(self.labels), "Predictions and labels length mismatch"
        assert np.all((self.predictions >= 0) & (self.predictions <= 1)), "Predictions must be in [0, 1]"
        assert np.all(np.isin(self.labels, [0, 1])), "Labels must be binary (0 or 1)"

    # ========================================================================
    # DISCRIMINATION METRICS
    # ========================================================================

    def roc_auc_score(self) -> float:
        """ROC-AUC: discrimination ability."""
        return roc_auc_score(self.labels, self.predictions)

    def roc_auc_ci(self, n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
        """Bootstrap confidence interval for ROC-AUC."""
        bootstrapped_auc = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(self.predictions), len(self.predictions), replace=True)
            try:
                auc_score = roc_auc_score(self.labels[indices], self.predictions[indices])
                bootstrapped_auc.append(auc_score)
            except ValueError:
                # Skip if only one class in bootstrap sample
                pass

        bootstrapped_auc = np.array(bootstrapped_auc)
        ci_lower = np.percentile(bootstrapped_auc, alpha / 2 * 100)
        ci_mean = np.mean(bootstrapped_auc)
        ci_upper = np.percentile(bootstrapped_auc, (1 - alpha / 2) * 100)

        return ci_lower, ci_mean, ci_upper

    def pr_auc_score(self) -> float:
        """PR-AUC: precision-recall area (better for imbalanced data)."""
        precision, recall, _ = precision_recall_curve(self.labels, self.predictions)
        return auc(recall, precision)

    # ========================================================================
    # CALIBRATION METRICS
    # ========================================================================

    def expected_calibration_error(self, n_bins: int = 10) -> float:
        """Expected calibration error: average |predicted - observed| across bins."""
        bin_sums = np.zeros(n_bins)
        bin_true = np.zeros(n_bins)
        bin_total = np.zeros(n_bins)

        for i in range(n_bins):
            bin_lower = i / n_bins
            bin_upper = (i + 1) / n_bins
            in_bin = (self.predictions > bin_lower) & (self.predictions <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(self.labels[in_bin])
                avg_prediction_in_bin = np.mean(self.predictions[in_bin])
                bin_sums[i] = np.abs(avg_prediction_in_bin - accuracy_in_bin) * prop_in_bin
                bin_true[i] = accuracy_in_bin * np.sum(in_bin)
                bin_total[i] = np.sum(in_bin)

        return np.sum(bin_sums)

    def calibration_slope(self) -> float:
        """Calibration slope: how much predictions need to be scaled for perfect calibration."""
        # Logistic regression: logit(label) ~ logit(prediction)
        from scipy.special import logit, expit

        try:
            # Clip predictions to avoid log(0)
            pred_clipped = np.clip(self.predictions, 1e-6, 1 - 1e-6)
            logit_pred = logit(pred_clipped)
            logit_labels = logit(np.clip(self.labels.astype(float), 1e-6, 1 - 1e-6))

            slope, intercept = np.polyfit(logit_pred, logit_labels, 1)
            return slope
        except Exception as e:
            logger.warning(f"Could not compute calibration slope: {e}")
            return np.nan

    def hosmer_lemeshow_test(self, n_bins: int = 10) -> Tuple[float, float]:
        """Hosmer-Lemeshow goodness-of-fit test."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(self.predictions, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        chi2_stat = 0
        for i in range(n_bins):
            in_bin = bin_indices == i
            if np.sum(in_bin) == 0:
                continue

            observed_1 = np.sum(self.labels[in_bin])
            observed_0 = np.sum(in_bin) - observed_1
            expected_1 = np.sum(self.predictions[in_bin])
            expected_0 = np.sum(in_bin) - expected_1

            if expected_1 > 0:
                chi2_stat += (observed_1 - expected_1) ** 2 / expected_1
            if expected_0 > 0:
                chi2_stat += (observed_0 - expected_0) ** 2 / expected_0

        p_value = 1 - stats.chi2.cdf(chi2_stat, n_bins - 2)
        return chi2_stat, p_value

    # ========================================================================
    # CLINICAL UTILITY METRICS
    # ========================================================================

    def net_benefit_at_threshold(self, threshold: float) -> float:
        """Net benefit of model at given threshold."""
        predicted_positive = self.predictions >= threshold
        true_positive = (self.labels == 1) & predicted_positive
        false_positive = (self.labels == 0) & predicted_positive

        n = len(self.labels)
        n_events = np.sum(self.labels)

        tp = np.sum(true_positive)
        fp = np.sum(false_positive)

        net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        return net_benefit

    def sensitivity_at_specificity(self, target_specificity: float = 0.9) -> float:
        """Sensitivity at a given specificity."""
        fpr, tpr, _ = roc_curve(self.labels, self.predictions)
        specificity = 1 - fpr

        # Find closest specificity to target
        idx = np.argmin(np.abs(specificity - target_specificity))
        return tpr[idx]

    def metrics_at_threshold(self, threshold: float = 0.5) -> Dict[str, float]:
        """All metrics at a specific decision threshold."""
        predicted_positive = self.predictions >= threshold

        tn, fp, fn, tp = confusion_matrix(self.labels, predicted_positive).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / len(self.labels)

        return {
            "threshold": threshold,
            "sensitivity": sensitivity,  # True positive rate
            "specificity": specificity,  # True negative rate
            "ppv": ppv,  # Positive predictive value
            "npv": npv,  # Negative predictive value
            "accuracy": accuracy,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    # ========================================================================
    # RISK STRATIFICATION
    # ========================================================================

    def quintile_analysis(self) -> pd.DataFrame:
        """Risk stratification by quintiles."""
        quintiles = pd.qcut(self.predictions, q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")

        results = []
        for q in quintiles.unique():
            subset = (quintiles == q)
            n_patients = np.sum(subset)
            predicted_risk = np.mean(self.predictions[subset])
            actual_risk = np.mean(self.labels[subset])
            n_events = np.sum(self.labels[subset])

            results.append({
                "quintile": q,
                "n_patients": n_patients,
                "n_events": n_events,
                "event_rate": actual_risk,
                "mean_predicted_risk": predicted_risk,
                "calibration_gap": predicted_risk - actual_risk,
            })

        return pd.DataFrame(results)

    def integrated_calibration_index(self) -> float:
        """Integrated calibration index: average absolute calibration error."""
        quintiles = pd.qcut(self.predictions, q=5, labels=False, duplicates="drop")
        abs_diff = []

        for q in np.unique(quintiles):
            subset = quintiles == q
            predicted_risk = np.mean(self.predictions[subset])
            actual_risk = np.mean(self.labels[subset])
            abs_diff.append(np.abs(predicted_risk - actual_risk))

        return np.mean(abs_diff)

    # ========================================================================
    # SUBGROUP ANALYSIS
    # ========================================================================

    def auc_by_group(self, group_labels: np.ndarray) -> Dict[str, float]:
        """ROC-AUC within demographic subgroups."""
        unique_groups = np.unique(group_labels)
        auc_by_group = {}

        for group in unique_groups:
            in_group = group_labels == group
            try:
                group_auc = roc_auc_score(self.labels[in_group], self.predictions[in_group])
                auc_by_group[f"Group_{group}"] = group_auc
            except ValueError:
                # Only one class in subgroup
                pass

        return auc_by_group

    # ========================================================================
    # COMPLETE EVALUATION
    # ========================================================================

    def evaluate_all(self) -> Dict:
        """Compute all metrics."""
        ci_lower, ci_mean, ci_upper = self.roc_auc_ci()
        hl_chi2, hl_pvalue = self.hosmer_lemeshow_test()

        metrics = {
            # Discrimination
            "roc_auc": self.roc_auc_score(),
            "roc_auc_ci_lower": ci_lower,
            "roc_auc_ci_upper": ci_upper,
            "pr_auc": self.pr_auc_score(),

            # Calibration
            "ece": self.expected_calibration_error(),
            "calibration_slope": self.calibration_slope(),
            "hl_chi2": hl_chi2,
            "hl_pvalue": hl_pvalue,

            # Clinical utility
            "net_benefit_at_50pct": self.net_benefit_at_threshold(0.5),
            "sensitivity_at_90spec": self.sensitivity_at_specificity(0.9),

            # Risk stratification
            "ici": self.integrated_calibration_index(),

            # Basic thresholds
            "metrics_at_50pct": self.metrics_at_threshold(0.5),
        }

        return metrics

    def summary_report(self) -> str:
        """Generate human-readable summary."""
        metrics = self.evaluate_all()

        report = f"""
================================================================================
COMPREHENSIVE MODEL EVALUATION
================================================================================

DISCRIMINATION (Does model rank patients correctly?)
────────────────────────────────────────────────────────────────────────────────
ROC-AUC:                {metrics['roc_auc']:.3f} [95% CI: {metrics['roc_auc_ci_lower']:.3f}-{metrics['roc_auc_ci_upper']:.3f}]
PR-AUC:                 {metrics['pr_auc']:.3f}

CALIBRATION (Are probabilities well-calibrated?)
────────────────────────────────────────────────────────────────────────────────
Expected Calib Error:   {metrics['ece']:.3f}
Calibration Slope:      {metrics['calibration_slope']:.3f} (1.0 = perfect)
Hosmer-Lemeshow:        χ² = {metrics['hl_chi2']:.2f}, p = {metrics['hl_pvalue']:.4f}
  Interpretation:       {"✓ Well calibrated (p>0.05)" if metrics['hl_pvalue'] > 0.05 else "✗ Poorly calibrated (p<0.05)"}

CLINICAL UTILITY
────────────────────────────────────────────────────────────────────────────────
Net Benefit at 50%:     {metrics['net_benefit_at_50pct']:.3f}
Sensitivity at 90% Spec: {metrics['sensitivity_at_90spec']:.1%}

RISK STRATIFICATION
────────────────────────────────────────────────────────────────────────────────
Integrated Cal Index:   {metrics['ici']:.3f}

THRESHOLDS @ 50%
────────────────────────────────────────────────────────────────────────────────
"""
        for key, val in metrics['metrics_at_50pct'].items():
            if isinstance(val, float):
                report += f"{key:20s}: {val:.3f}\n"
            else:
                report += f"{key:20s}: {val}\n"

        report += "================================================================================\n"
        return report
