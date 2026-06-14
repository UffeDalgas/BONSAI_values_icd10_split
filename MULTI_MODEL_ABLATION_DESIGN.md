# Multi-Model Ablation Design: From Single Pretrain to Targeted Finetuned Models

## Overview

Your architecture is sound: **1 pretrain → N specialized finetune models**. But we can significantly improve both the ablation strategy AND the evaluation rigor.

---

## Part 1: Current Design

### Your 8 Finetune Models

```
Pretrain (no features) 
  ↓
├─ Model 1: Baseline (EHR only, no injection)
├─ Model 2: Metadata + _preop indicators
├─ Model 3: GrimAge v2 + intermediate values
├─ Model 4: SystemsAge (11 features)
├─ Model 5: CpGPTGrimAge v3 + protein proxies
├─ Model 6: MAPLE embeddings
├─ Model 7: CpGPT embeddings
└─ Model 8: MethylGPT embeddings
```

**Strengths:**
- ✓ Single pretrain (efficient)
- ✓ Systematic feature comparison
- ✓ Each model learns to USE its features

**Gaps:**
- ✗ No feature interaction testing
- ✗ No combined models (e.g., GrimAge + SystemsAge together)
- ✗ No hierarchical ablation (what's the minimal set?)
- ✗ Evaluation is just ROC-AUC (no calibration, decision utility, etc.)
- ✗ No statistical significance testing
- ✗ No confidence intervals
- ✗ No analysis of why features help

---

## Part 2: Enhanced Ablation Strategy

### A. Hierarchical/Stepwise Ablation

**Goal:** Understand feature value incrementally

```
Model 1:  EHR only
Model 2:  EHR + GrimAge v2
Model 3:  EHR + GrimAge v2 + SystemsAge
Model 4:  EHR + GrimAge v2 + SystemsAge + MAPLE
Model 5:  EHR + GrimAge v2 + SystemsAge + MAPLE + CpGPT
Model 6:  EHR + GrimAge v2 + SystemsAge + MAPLE + CpGPT + MethylGPT
```

**Interpretation:**
- ΔROCₐᵤcₛ(1→2) = GrimAge v2 contribution alone
- ΔROCₐᵤcₛ(2→3) = SystemsAge marginal contribution (given GrimAge exists)
- ΔROCₐᵤcₛ(3→4) = MAPLE marginal contribution
- etc.

**Reveals:** Redundancy, synergies, diminishing returns

---

### B. Feature Combination Matrix

Test all meaningful combinations:

```
Individual Features:
  • GrimAge v2 alone          → Model A
  • SystemsAge alone          → Model B
  • MAPLE alone               → Model C
  • CpGPT alone               → Model D
  • MethylGPT alone          → Model E

Pairwise Combinations:
  • GrimAge + SystemsAge      → Model F
  • GrimAge + MAPLE           → Model G
  • GrimAge + CpGPT           → Model H
  • GrimAge + MethylGPT       → Model I
  • SystemsAge + MAPLE        → Model J
  • SystemsAge + CpGPT        → Model K
  ... (10 models for C(5,2))

All Combined:
  • EHR + All 5 features      → Model X
```

**Interpretation Matrix:**
```
                  SystemsAge  MAPLE  CpGPT  MethylGPT  ΔROCₐᵤc
GrimAge           +           -      -      -          +0.087
GrimAge+SystemsAge +          -      -      -          +0.110
GrimAge+SystemsAge+MAPLE +    +      -      -          +0.115
...
All               +           +      +      +          +0.131
```

**Reveals:** Synergies, redundancies, optimal minimal sets

---

### C. Temporal Ablation

Test how predictions change over time:

```
Model T1: Trained on all data, evaluate at 30 days
Model T2: Trained on all data, evaluate at 90 days
Model T3: Trained on all data, evaluate at 1 year
Model T4: Trained on all data, evaluate at 3 years
```

**Reveals:** Whether biological signal strengthens/weakens over time

---

### D. Domain-Specific Groupings

Test biological logic groupings:

```
Aging Clocks:
  Model A1: GrimAge v2 + SystemsAge (pure aging signal)
  
Protein Proxies:
  Model A2: CpGPTGrimAge v3 (protein scores only)
  
Methylation:
  Model A3: MAPLE + CpGPT + MethylGPT (all epigenetic)
  
Metabolic:
  Model A4: SystemsAge subset (glucose, lipids, etc.)
  
Inflammatory:
  Model A5: SystemsAge subset (CRP, inflammatory markers)
```

**Reveals:** Which biological domains matter most

---

## Part 3: Enhanced Evaluation Rigor

Your evaluation is currently:
```python
roc_auc = compute_roc_auc(predictions, labels)
```

This is **necessary but insufficient**. Here's what to add:

### 1. Statistical Confidence

```python
# Bootstrap confidence intervals (1000 resamples)
ci_lower, ci_mean, ci_upper = bootstrap_roc_auc(predictions, labels, n_bootstrap=1000)
print(f"ROC-AUC: {ci_mean:.3f} [95% CI: {ci_lower:.3f}-{ci_upper:.3f}]")

# Permutation test for significance
p_value = permutation_test(model_a_auc, model_b_auc, n_permutations=1000)
print(f"Δ ROC-AUC: {model_b_auc - model_a_auc:.3f}, p={p_value:.4f}")
```

**Output Example:**
```
Baseline (EHR only):       ROC-AUC = 0.612 [95% CI: 0.551-0.671]
Enhanced (EHR + all bio):  ROC-AUC = 0.743 [95% CI: 0.688-0.797]
Δ ROC-AUC = +0.131 [95% CI: 0.049-0.213], p=0.002 **
```

### 2. Calibration Analysis

```python
# Calibration curve: predicted prob vs. observed freq
plot_calibration_curve(predictions, labels, name="Model A")

# Hosmer-Lemeshow goodness-of-fit test
hl_stat, hl_pvalue = hosmer_lemeshow_test(predictions, labels)
print(f"Hosmer-Lemeshow p-value: {hl_pvalue:.4f}")

# Expected calibration error (ECE)
ece = expected_calibration_error(predictions, labels, n_bins=10)
print(f"ECE: {ece:.3f}")
```

**Interpretation:**
- Well-calibrated: predicted 70% mortality → actually ~70% mortality
- Overconfident: predicts 70% but only 50% actual
- Underconfident: predicts 50% but actually 70%

### 3. Decision Curve Analysis (Clinical Utility)

```python
# Decision curves: net benefit vs. threshold
plot_decision_curve(model_a_probs, model_b_probs, labels)

# Example: Should we intervene if model predicts >50% mortality?
threshold = 0.5
sensitivity, specificity, npv, ppv = metrics_at_threshold(predictions, labels, threshold)

print(f"At {threshold} threshold:")
print(f"  Sensitivity: {sensitivity:.1%} (true positive rate)")
print(f"  Specificity: {specificity:.1%} (true negative rate)")
print(f"  PPV: {ppv:.1%} (if model says positive, actually positive)")
print(f"  NPV: {npv:.1%} (if model says negative, actually negative)")
```

**Interpretation:**
- Is the improvement in ROC-AUC clinically meaningful?
- At what threshold does the model add value?
- What's the cost of false positives vs. false negatives?

### 4. Risk Stratification Analysis

```python
# Quintile analysis
quintiles = pd.qcut(predictions, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
stratified = pd.DataFrame({
    'quintile': quintiles,
    'predicted_risk': predictions,
    'actual_outcome': labels
})

for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
    subset = stratified[stratified['quintile'] == q]
    actual_rate = subset['actual_outcome'].mean()
    pred_rate = subset['predicted_risk'].mean()
    print(f"{q}: predicted={pred_rate:.1%}, actual={actual_rate:.1%}")

# Plot: Predicted vs. Actual by Quintile
```

**Example Output:**
```
Q1 (lowest risk):  predicted=10%, actual=8%    ✓ Well calibrated
Q2:                predicted=25%, actual=22%   ✓
Q3:                predicted=50%, actual=48%   ✓
Q4:                predicted=75%, actual=78%   ~ Close
Q5 (highest risk): predicted=90%, actual=92%   ✓
```

### 5. Feature Importance Analysis

```python
# SHAP values: which features drive predictions?
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Mean absolute SHAP: average impact per feature
feature_importance = np.abs(shap_values).mean(axis=0)
feature_names = ['age', 'grim_age2', 'systems_age', 'maple', ...]

for fname, importance in zip(feature_names, feature_importance):
    print(f"{fname:20s}: {importance:.4f}")

# Plot: Force plot for individual predictions
shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])
```

### 6. Subgroup Performance Analysis

```python
# Does model work equally well for men vs. women?
subgroups = {'Male': data[data['sex']=='M'], 'Female': data[data['sex']=='F']}

for subgroup_name, subgroup_data in subgroups.items():
    auc = compute_roc_auc(subgroup_data['predictions'], subgroup_data['labels'])
    print(f"{subgroup_name:20s}: ROC-AUC = {auc:.3f}")

# Statistical test for interaction
interaction_pvalue = test_subgroup_interaction(subgroups)
print(f"Subgroup interaction p-value: {interaction_pvalue:.4f}")
```

### 7. Precision-Recall Curve (for imbalanced data)

```python
# PR-AUC often more informative when outcomes are rare
pr_auc = compute_pr_auc(predictions, labels)
plot_pr_curve(predictions, labels)

print(f"ROC-AUC: {roc_auc:.3f}  (overall discrimination)")
print(f"PR-AUC:  {pr_auc:.3f}  (for rare outcomes)")
```

---

## Part 4: Additional Feature Subsets to Test

Beyond your 8 models, consider:

### Temporal/Trajectory Features

```
Model T1: Acceleration rates
  - Rate of GrimAge increase (years of aging per calendar year)
  - Rate of SystemsAge increase
  
Model T2: Volatility
  - Stability of biological markers over time
  - High volatility → poor health trajectory
```

### Interaction Terms

```
Model I1: Age × Biological Age
  - Interaction: chronological age + GrimAge acceleration

Model I2: Disease Burden × Biological Age
  - Interaction: comorbidities + epigenetic age

Model I3: Sex-Specific Models
  - Train separate models for M/F with sex-specific features
```

### Principal Component Reductions

```
Model P1: MAPLE PCA (reduce 32 dims → 5 PCs)
  - What if we compress embeddings?

Model P2: CpGPT PCA (reduce 64 dims → 5 PCs)

Model P3: All embeddings combined PCA
  - Project MAPLE + CpGPT + MethylGPT into shared space
```

### Disease-Specific Variants

```
Heart Failure Models:
  Model HF1: GrimAge + cardiac-specific markers (troponin trajectory)
  Model HF2: SystemsAge + cardiac markers
  Model HF3: MAPLE + cardiac remodeling genes

Diabetes Models:
  Model DM1: GrimAge + glucose control markers
  Model DM2: SystemsAge + metabolic features
  
Sepsis Models:
  Model SEP1: GrimAge + inflammatory markers
  Model SEP2: SystemsAge + infection trajectory
```

### Embedding Space Explorations

```
Model E1: MAPLE only (validate its signal)
Model E2: CpGPT only
Model E3: MethylGPT only
Model E4: MAPLE + CpGPT (epigenetic + protein space)
Model E5: All embeddings (multi-modal fusion)
```

### Minimal Sets (Model Efficiency)

```
Model M1: Top 1 feature (which is most important?)
Model M2: Top 1+2 features (incremental value)
Model M3: Top 1+2+3 features
Model M4: Optimal feature set (best ROC-AUC with fewest features)
```

---

## Part 5: Recommended Evaluation Pipeline

### Standard Metrics Across All Models

For EACH of your N models, compute:

```python
class ComprehensiveModelEvaluation:
    def __init__(self, predictions, labels, features=None):
        self.predictions = predictions
        self.labels = labels
        self.features = features
    
    def compute_all_metrics(self):
        """Comprehensive evaluation suite."""
        
        # Discrimination (does model rank patients correctly?)
        metrics = {
            'roc_auc': self.compute_roc_auc(),
            'roc_auc_ci': self.bootstrap_ci(self.compute_roc_auc),
            'pr_auc': self.compute_pr_auc(),
            
            # Calibration (are probabilities well-calibrated?)
            'ece': self.expected_calibration_error(),
            'hl_pvalue': self.hosmer_lemeshow_pvalue(),
            'calibration_slope': self.calibration_slope(),
            
            # Clinical utility
            'decision_curve_nba': self.net_benefit_at_threshold(0.5),
            'sensitivity_at_90spec': self.sensitivity_at_specificity(0.9),
            
            # Risk stratification
            'stratification_ici': self.integrated_calibration_index(),
            
            # Feature importance (if model supports)
            'feature_importance': self.compute_feature_importance() if self.features else None,
            
            # Subgroup analysis
            'age_stratified_auc': self.auc_by_age_group(),
            'sex_stratified_auc': self.auc_by_sex(),
        }
        
        return metrics
    
    def bootstrap_ci(self, metric_fn, n_bootstrap=1000, alpha=0.05):
        """Compute 95% CI via bootstrap."""
        bootstrapped_metrics = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(self.predictions), len(self.predictions), replace=True)
            bootstrapped_metrics.append(metric_fn(self.predictions[indices], self.labels[indices]))
        
        ci_lower = np.percentile(bootstrapped_metrics, alpha/2 * 100)
        ci_mean = np.mean(bootstrapped_metrics)
        ci_upper = np.percentile(bootstrapped_metrics, (1-alpha/2) * 100)
        
        return ci_lower, ci_mean, ci_upper
```

### Summary Comparison Table

```
Model                          ROC-AUC    ΔROCₐᵤc  p-value  ECE   PPV@50%  Calibration
───────────────────────────────────────────────────────────────────────────────────
1. Baseline (EHR only)         0.612      —        —        0.12  0.45     Good
2. + GrimAge v2                0.699      +0.087   0.003    0.09  0.58     Good
3. + SystemsAge                0.644      +0.032   0.042    0.11  0.52     Fair
4. + GrimAge + Systems         0.721      +0.109   <0.001   0.08  0.64     Good
5. + MAPLE                     0.622      +0.010   0.451    0.12  0.47     Good
6. + CpGPT                     0.658      +0.046   0.018    0.10  0.55     Good
7. + MethylGPT                 0.626      +0.014   0.328    0.12  0.48     Good
8. + All features              0.743      +0.131   <0.001   0.07  0.72     Excellent
```

---

## Part 6: Architecture Implementation

### Directory Structure

```
./outputs/ablation_results/
├── model_configs/
│   ├── ablation_step1_ehr_only.yaml
│   ├── ablation_step2_grim_age.yaml
│   ├── ablation_step3_systems_age.yaml
│   ├── ablation_combined_all.yaml
│   └── ...
│
├── model_checkpoints/
│   ├── finetune_ehr_only/
│   │   └── best_model.pt
│   ├── finetune_grim_age/
│   │   └── best_model.pt
│   └── ...
│
├── predictions/
│   ├── ehr_only_predictions.csv
│   ├── grim_age_predictions.csv
│   └── ...
│
├── evaluation_metrics/
│   ├── metrics_summary.json
│   ├── roc_curves.png
│   ├── calibration_plots.png
│   ├── decision_curves.png
│   └── feature_importance.png
│
└── comparison_reports/
    ├── ablation_summary.md
    ├── statistical_significance.csv
    └── clinical_utility_analysis.md
```

### Batch Runner

```python
from corebehrt.ablation.batch_runner import BatchAblationRunner

ablation_configs = [
    'ablation_step1_ehr_only.yaml',
    'ablation_step2_grim_age.yaml',
    'ablation_step3_systems_age.yaml',
    # ... etc
]

runner = BatchAblationRunner(
    configs=ablation_configs,
    pretrain_checkpoint='./outputs/pretraining_dryrun/checkpoints/best.pt',
    output_dir='./outputs/ablation_results'
)

# Train all models
runner.train_all_models(n_workers=4)  # Parallel training

# Evaluate all models
runner.evaluate_all_models()

# Generate comparison report
runner.generate_comparison_report()
```

---

## Part 7: Expected Insights

Running this comprehensive ablation, you'll discover:

### ROC-AUC Insights
- Which features contribute most signal
- Where we hit diminishing returns
- Synergies vs. redundancy

### Calibration Insights
- Whether GrimAge overconfident in high-risk populations
- If MAPLE needs recalibration
- Whether combining features maintains calibration

### Clinical Utility Insights
- At what mortality risk threshold is intervention justified?
- Does biological signal translate to actionable predictions?
- Cost-benefit of each feature addition

### Disease-Specific Insights
- Which proxies work best in heart failure, diabetes, sepsis?
- Are features disease-agnostic or domain-specific?

### Feature Importance Insights
- Via SHAP: which biological markers drive decisions?
- Are predictions driven by rare events or common patterns?
- Can we simplify the model?

---

## Summary

Your 8-model design is good. This framework makes it **excellent** by:

1. ✅ Adding hierarchical ablation (understand feature value)
2. ✅ Testing feature combinations (find synergies)
3. ✅ Rigorous evaluation (confidence intervals, calibration, utility)
4. ✅ More feature subsets (combinations, domain-specific)
5. ✅ Automated reporting (batch runner + comparison tables)

**Next step:** Would you like me to implement the batch runner and comprehensive evaluation module?
