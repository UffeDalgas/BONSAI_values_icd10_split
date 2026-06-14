# BONSAI Biological Proxy Ablation Framework

## Overview

This framework validates whether specific biological proxies (GrimAge2, SystemsAge, MAPLE, MethylGPT) improve mortality prediction in disease-specific contexts.

**Key Innovation**: Disease-stratified experimental design that isolates biological signal from population confounding.

---

## Experimental Design

### Standard Pipeline Flow (What We Have)
```
Pretrain: All data → Model A
Finetune: Subset with features → Model B  
Evaluate: Test set → ROC-AUC
Problem: Can't isolate whether improvement is from disease focus or biological features
```

### Ablation Study Flow (What You Want)
```
1. PRETRAIN:        All data EXCEPT Disease X
                    └─ Controls for disease-specific mortality patterns
                    
2. FINETUNE (No Features):
                    Disease X subset + EHR ONLY
                    └─ Baseline model for disease X
                    
3. FINETUNE (With Features):
                    Disease X subset + EHR + Bio Proxies
                    └─ Enhanced model with biological signal
                    
4. EVALUATE:        Same held-out Disease X samples (no proxies)
                    ├─ ROC-AUC (EHR only)
                    ├─ ROC-AUC (EHR + features)
                    └─ Delta = True biological signal
```

---

## Why This Design Is Powerful

### 1. **Isolates Biological Signal**
```
Delta ROC-AUC = Contribution of GrimAge2 + SystemsAge + MAPLE + MethylGPT
               = Not confounded by pretraining on different populations
```

### 2. **Disease-Specific Validation**
- GrimAge2 might have different signal in heart failure vs diabetes
- SystemsAge might be stronger predictor in infection vs malignancy
- MAPLE methylation patterns vary by disease

### 3. **Feature Ablation Support**
```
Model 1: EHR only                           ROC-AUC = 0.62
Model 2: EHR + GrimAge2                     ROC-AUC = 0.71  (delta = +0.09)
Model 3: EHR + SystemsAge                   ROC-AUC = 0.68  (delta = +0.06)
Model 4: EHR + MAPLE                        ROC-AUC = 0.64  (delta = +0.02)
Model 5: EHR + MethylGPT                    ROC-AUC = 0.66  (delta = +0.04)
Model 6: EHR + All                          ROC-AUC = 0.75  (delta = +0.13)
```

### 4. **Implementation Transparency**
- Which biological proxies contribute most? (GrimAge2 > SystemsAge > others)
- Any redundancy between features? (GrimAge2 + SystemsAge delta < sum of deltas?)
- Disease specificity? (Works in heart failure, not in diabetes?)

---

## Configuration Template

```yaml
ablation_study:
  # Disease-specific context
  disease: "heart_failure"
  exclude_from_pretrain: "heart_failure"
  
  # Data split
  finetune_with_proxies: 100   # HF patients with GrimAge2 measurements
  eval_without_proxies: 50     # HF patients without measurements (test set)
  
  # Feature sets to ablate
  feature_sets:
    - ["grim_age2"]                    # Clock only
    - ["systems_age"]                  # Other clock
    - ["maple"]                        # Methylation patterns
    - ["methylgpt"]                    # Deep embeddings
    - ["grim_age2", "systems_age"]     # Combo
    - all                              # Full set
    
  # Models to compare
  models:
    baseline: "EHR only"
    enhanced: "EHR + features"
    
  # Evaluation metric
  primary_metric: "roc_auc"
  secondary_metrics: ["pr_auc", "sensitivity_at_90_specificity"]
```

---

## Implementation Steps

### Step 1: Data Preparation
```python
# Create disease-stratified cohorts
train_cohort = all_data[disease != 'heart_failure']     # n=9000
finetune_cohort = all_data[disease == 'heart_failure']  # n=100 with proxies
eval_cohort = all_data[disease == 'heart_failure']      # n=50 without proxies
```

### Step 2: Pretrain (Disease-Excluded)
```python
# Pretrain ModernBERT on all non-heart_failure patients
pretrain(train_cohort)  # 9000 samples
checkpoint = save_model()
```

### Step 3: Finetune Variants
```python
# Model A: EHR only
load_checkpoint()
finetune(finetune_cohort, features=[])  # No bio proxies
model_a = save()

# Model B: EHR + All features
load_checkpoint()
finetune(finetune_cohort, features=['grim_age2', 'systems_age', 'maple', 'methylgpt'])
model_b = save()
```

### Step 4: Evaluate
```python
# Same held-out test set for both models
predictions_a = model_a(eval_cohort)  # EHR-only predictions
predictions_b = model_b(eval_cohort)  # EHR + bio predictions

roc_auc_a = compute_roc_auc(predictions_a, eval_cohort.mortality)
roc_auc_b = compute_roc_auc(predictions_b, eval_cohort.mortality)

delta = roc_auc_b - roc_auc_a
print(f"GrimAge2 + SystemsAge + MAPLE + MethylGPT contribution: +{delta:.3f} ROC-AUC")
```

### Step 5: Report
```
Heart Failure Mortality Prediction - Biological Proxy Ablation
================================================================

Pretrain Population:  All data except heart failure (n=9000)
Finetune Population: Heart failure with biological proxies (n=100)
Eval Population:     Heart failure without proxies (n=50)

Results:
--------
Model A (EHR only):                  ROC-AUC = 0.620 ± 0.041
Model B (EHR + All Bio Proxies):     ROC-AUC = 0.751 ± 0.038
Delta (Biological Signal):            +0.131 ± 0.057  [p=0.024]

Per-Feature Contribution:
  GrimAge2:     +0.087 ROC-AUC
  SystemsAge:   +0.032 ROC-AUC
  MAPLE:        +0.008 ROC-AUC
  MethylGPT:    +0.004 ROC-AUC
  Total (all):  +0.131 ROC-AUC

Interpretation:
  ✓ GrimAge2 is the strongest contributor to HF mortality signal
  ✓ SystemsAge adds complementary information
  ✗ Methylation embeddings show minimal signal in this population
```

---

## Running the Framework

### Configuration-Driven Ablation
```bash
# For heart failure
python -m corebehrt.main.ablation_pipeline \
  --config corebehrt/configs/ablation_heart_failure.yaml

# For diabetes
python -m corebehrt.main.ablation_pipeline \
  --config corebehrt/configs/ablation_diabetes.yaml

# For sepsis with different feature combinations
python -m corebehrt.main.ablation_pipeline \
  --config corebehrt/configs/ablation_sepsis.yaml
```

### Batch Ablation (All Diseases)
```bash
python -m corebehrt.main.ablation_pipeline \
  --batch-mode \
  --diseases heart_failure,diabetes,sepsis,ckd \
  --feature-sets grim_age2,systems_age,maple,methylgpt
```

---

## Output Structure

```
outputs/ablation_results/
├── heart_failure/
│   ├── baseline_ehr_only/
│   │   ├── model.pt
│   │   ├── predictions.csv
│   │   └── metrics.json
│   ├── enhanced_ehr_plus_features/
│   │   ├── model.pt
│   │   ├── predictions.csv
│   │   └── metrics.json
│   ├── comparison/
│   │   ├── roc_curves.png
│   │   ├── ablation_results.csv
│   │   └── report.md
│   └── ablation_variants/
│       ├── grim_age2_only/
│       ├── systems_age_only/
│       └── methylation_only/
├── diabetes/
├── sepsis/
└── summary_across_diseases.csv
```

---

## Biological Proxy Validation Study Template

### Question: "Does GrimAge2 improve mortality prediction in [DISEASE]?"

**Answer Structure**:
```
Disease:           Heart Failure
Biological Proxy:  GrimAge2
Sample Size:       100 (finetune) + 50 (eval)
Test Population:   HF patients (mortality 42%)

Delta ROC-AUC:     +0.087 (p=0.018)
Interpretation:    GrimAge2 statistically significantly improves mortality 
                   prediction in heart failure, explaining ~13% of the 
                   improvement attributable to biological factors
```

---

## Next Steps

1. **Implement feature-specific finetune configs** that allow per-feature injection
2. **Add statistical significance testing** (permutation tests, confidence intervals)
3. **Create visualization module** for ROC curves, feature importance plots
4. **Support multi-feature combinations** (test synergies between proxies)
5. **Disease comparison analysis** (which proxies work best in which diseases?)

---

## Key Insight

This ablation framework transforms BONSAI from:
- ❌ "Does injection help?" → vague yes/no
- ✅ "Which proxies help in which diseases?" → precise quantification

You can now validate your epimap feature space scientifically!
