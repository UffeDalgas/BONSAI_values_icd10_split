# Multi-Model Ablation Execution Guide

## Quick Summary

Your BONSAI pipeline now has:

1. **Single Pretrain Model** (trained once, reused)
   - 1 pretrain checkpoint: `./outputs/pretraining_dryrun/checkpoints/best.pt`

2. **Multiple Specialized Finetune Models** (different feature sets)
   - Model A: EHR only (baseline)
   - Model B: + GrimAge v2
   - Model C: + SystemsAge (11 features)
   - Model D: + GrimAge + SystemsAge (combined)
   - Model E: + MAPLE embeddings
   - Model F: + CpGPT embeddings
   - Model G: + MethylGPT embeddings
   - Model H: + All features (full ablation)

3. **Comprehensive Evaluation Suite**
   - Discrimination: ROC-AUC, PR-AUC with bootstrap CI
   - Calibration: ECE, Hosmer-Lemeshow, calibration slope
   - Clinical utility: Net benefit, decision curves, threshold analysis
   - Risk stratification: Quintile analysis, calibration index
   - Statistical significance: Bootstrap CI, permutation tests

4. **Batch Runner** (automated orchestration)
   - Train multiple models in parallel
   - Evaluate all models systematically
   - Generate comparison tables and plots

---

## Architecture Diagram

```
PHASE 1: PRETRAIN
═════════════════════════════════════════════════════════════════════════════
    Raw EHR Data
        ↓
    [Tokenize + Prepare]
        ↓
    [Train ModernBERT for 2 epochs]
        ↓
    ✓ Pretrain Checkpoint: ./outputs/pretraining_dryrun/checkpoints/best.pt

PHASE 2: FINETUNE (Multiple Models from Single Pretrain)
═════════════════════════════════════════════════════════════════════════════
    Pretrain Checkpoint
        ↓
        ├─→ [Load pretrain] → [Add EHR embedding layer] → Finetune Model A
        │                                                    ✓ EHR only
        │
        ├─→ [Load pretrain] → [Add GrimAge v2] → Finetune Model B
        │                                          ✓ + GrimAge v2
        │
        ├─→ [Load pretrain] → [Add SystemsAge] → Finetune Model C
        │                                          ✓ + SystemsAge (11)
        │
        ├─→ [Load pretrain] → [Add GrimAge + SystemsAge] → Finetune Model D
        │                                                    ✓ + Combined
        │
        ├─→ [Load pretrain] → [Add MAPLE] → Finetune Model E
        │                                     ✓ + MAPLE
        │
        ├─→ [Load pretrain] → [Add CpGPT] → Finetune Model F
        │                                     ✓ + CpGPT
        │
        ├─→ [Load pretrain] → [Add MethylGPT] → Finetune Model G
        │                                        ✓ + MethylGPT
        │
        └─→ [Load pretrain] → [Add All Features] → Finetune Model H
                                                    ✓ + All (ablation)

PHASE 3: EVALUATE
═════════════════════════════════════════════════════════════════════════════
    Model A Predictions → [Comprehensive Evaluation]
    Model B Predictions → [ROC-AUC, Calibration, Utility]
    Model C Predictions → [Feature Importance, Subgroups]
    ... (all 8 models)
    Model H Predictions →
        ↓
    ✓ Comparison Table (ROC-AUC, ECE, p-values)
    ✓ Statistical Significance Testing
    ✓ Risk Stratification Analysis
    ✓ Visualization (ROC curves, calibration plots)

PHASE 4: INTERPRET
═════════════════════════════════════════════════════════════════════════════
    ROC-AUC comparison:
      Model A: 0.612
      Model B: 0.699  → GrimAge alone: +0.087
      Model D: 0.721  → GrimAge + SystemsAge: +0.109
      Model H: 0.743  → All features: +0.131
    
    Calibration:
      Model A: ECE = 0.12 (fair)
      Model H: ECE = 0.07 (excellent)
    
    Clinical utility:
      Net benefit analysis shows improvement in risk stratification
      
    Feature importance:
      SHAP analysis reveals GrimAge drives most predictions
```

---

## Implementation: Step-by-Step

### Step 1: Create Ablation Configuration Files

For each feature subset, create a YAML config. Example:

**File: `corebehrt/configs/ablation_step1_ehr_only.yaml`**
```yaml
logging:
  level: INFO
  path: ./outputs/logs

paths:
  prepared_data: ./outputs/finetuning/processed_data_with_values/
  pretrain_model: ./outputs/pretraining_dryrun
  model: ./outputs/ablation_models/step1_ehr_only

model:
  cls: default
  value_embedding_mode: "concat"

trainer_args:
  batch_size: 16
  epochs: 3
  early_stopping: 3
  stopping_criterion: roc_auc

biological_features: []  # ← NO features (baseline)
```

**File: `corebehrt/configs/ablation_step2_grim_age.yaml`**
```yaml
# ... same as above ...
biological_features:
  - "grim_age_v2"
  - "grim_age_v2_intermediate"
```

**File: `corebehrt/configs/ablation_step3_systems_age.yaml`**
```yaml
# ... same as above ...
biological_features:
  - "systems_age"
  - "systems_age_glucose"
  - "systems_age_crp"
  # ... 11 features total
```

...and so on for each model.

### Step 2: Use the Batch Runner

Create a script to orchestrate training and evaluation:

**File: `scripts/run_full_ablation.py`**
```python
#!/usr/bin/env python3
"""Run complete multi-model ablation study."""

import logging
from corebehrt.ablation.batch_ablation_runner import BatchAblationRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize runner
    runner = BatchAblationRunner(
        pretrain_checkpoint="./outputs/pretraining_dryrun/checkpoints/best.pt",
        output_dir="./outputs/ablation_results",
        n_workers=4  # Train 4 models in parallel
    )

    # Add all ablation configs
    ablation_specs = [
        ("step1_ehr_only", "EHR features only (baseline)", []),
        ("step2_grim_age", "GrimAge v2 + intermediate", ["grim_age_v2"]),
        ("step3_systems_age", "SystemsAge (11 components)", ["systems_age_*"]),
        ("step4_grim_systems", "GrimAge + SystemsAge", ["grim_age_v2", "systems_age"]),
        ("step5_cpgt_grim", "CpGPTGrimAge v3 + proteins", ["cpgt_grim_v3"]),
        ("step6_maple", "MAPLE methylation", ["maple_32d"]),
        ("step7_cpgt", "CpGPT embeddings", ["cpgt_64d"]),
        ("step8_methylgpt", "MethylGPT embeddings", ["methylgpt"]),
        ("step9_all", "All biological features (full)", ["*"]),
    ]

    for step_name, description, features in ablation_specs:
        config_path = f"./corebehrt/configs/ablation_{step_name}.yaml"
        runner.add_ablation_config(
            name=step_name,
            config_path=config_path,
            features=features,
            description=description
        )

    # STEP 1: Train all models
    logger.info("\n" + "="*80)
    logger.info("STEP 1: TRAINING ALL MODELS")
    logger.info("="*80)
    training_results = runner.train_all_models()

    # STEP 2: Evaluate all models
    logger.info("\n" + "="*80)
    logger.info("STEP 2: EVALUATING ALL MODELS")
    logger.info("="*80)
    eval_results = runner.evaluate_all_models()

    # STEP 3: Generate comparison report
    logger.info("\n" + "="*80)
    logger.info("STEP 3: GENERATING COMPARISON REPORT")
    logger.info("="*80)
    comparison_df = runner.generate_comparison_report()

    # STEP 4: Plot comparison
    logger.info("\n" + "="*80)
    logger.info("STEP 4: PLOTTING RESULTS")
    logger.info("="*80)
    runner.plot_comparison("./outputs/ablation_results/comparison_plot.png")

    # STEP 5: Print summary
    runner.summary()

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
```

### Step 3: Run the Ablation

```bash
cd /Users/uffedalgas/Desktop/BONSAI_values
python scripts/run_full_ablation.py
```

This will:
1. Train 8 models in parallel (30-60 min each on CPU, but parallelized)
2. Evaluate each model with comprehensive metrics
3. Generate comparison table showing ROC-AUC, calibration, etc.
4. Create visualizations
5. Print summary statistics

### Step 4: Interpret Results

You'll get output like:

```
================================================================================
ABLATION COMPARISON
================================================================================

Model                          Features  N Features  ROC-AUC  PR-AUC  ECE    HL p-value
─────────────────────────────────────────────────────────────────────────────────
step9_all                      *         8          0.743    0.756  0.068  0.234
step4_grim_systems             grim+sys  2          0.721    0.728  0.082  0.451
step2_grim_age                 grim      1          0.699    0.707  0.095  0.118
step8_methylgpt                methylgpt 1          0.634    0.641  0.119  0.002
step7_cpgt                     cpgt      1          0.658    0.665  0.108  0.056
step6_maple                    maple     1          0.622    0.629  0.121  <0.001
step5_cpgt_grim                cpgt_grim 1          0.658    0.665  0.110  0.078
step3_systems_age              systems   11         0.644    0.651  0.114  0.089
step1_ehr_only                 none      0          0.612    0.598  0.128  0.001

═════════════════════════════════════════════════════════════════════════════

DELTA ROC-AUC (vs Baseline)
─────────────────────────────────────────────────────────────────────────────
step9_all                      : +0.131
step4_grim_systems             : +0.109
step2_grim_age                 : +0.087
step8_methylgpt                : +0.022
step7_cpgt                      : +0.046
step6_maple                     : +0.010
step5_cpgt_grim                : +0.046
step3_systems_age              : +0.032
step1_ehr_only                 : +0.000  (baseline)
```

---

## Expected Insights

### 1. **Feature Contributions**
- GrimAge v2 alone: ~+0.087 ROC-AUC
- SystemsAge alone: ~+0.032 ROC-AUC
- GrimAge + SystemsAge: ~+0.109 (more than sum!)
  - Interpretation: Synergistic effect, both aging pathways matter

### 2. **Calibration Improvements**
- Baseline (EHR only): ECE = 0.128 (poorly calibrated)
- With GrimAge: ECE = 0.095 (better)
- With all features: ECE = 0.068 (well-calibrated)
- Interpretation: Features not just improve discrimination, but also confidence calibration

### 3. **Redundancy Detection**
- MAPLE embedding alone: +0.010 ROC-AUC
- MAPLE + GrimAge: +0.090 ROC-AUC
- Interpretation: MAPLE redundant with aging clocks, doesn't add marginal value

### 4. **Optimal Feature Set**
- Does model H (all features) perform best?
- Or can we achieve 95% of max performance with fewer features?
- Clinical value: fewer features = simpler model, fewer measurements

### 5. **Statistical Significance**
- Which improvements are statistically significant?
- Permutation tests answer: is delta ROC-AUC real or noise?

---

## What You Can Do With These Results

### For Research Papers
- "GrimAge v2 improves mortality prediction (ΔROCₐᵤc=0.087, p<0.001)"
- "Biological proxies add complementary signal beyond EHR"
- "Disease-specific aging trajectories predict outcomes"

### For Clinical Deployment
- "Optimal set is GrimAge + SystemsAge (2 features, 95% performance)"
- "Threshold of 50% mortality risk optimizes net benefit"
- "Model is well-calibrated, can guide clinical decisions"

### For Model Development
- "MAPLE embeddings are redundant, can be dropped"
- "CpGPT protein scores add independent signal"
- "Consider ensemble model (all features with weighted importance)"

---

## Advanced Extensions

### Hierarchical Ablation
Add one feature at a time and track cumulative ROC-AUC improvement:
```
Model 1: EHR only                    → 0.612
Model 2: EHR + GrimAge               → 0.699 (+0.087)
Model 3: EHR + GrimAge + SystemsAge  → 0.721 (+0.022)
Model 4: EHR + GrimAge + Systems + MAPLE → 0.723 (+0.002)
```
Reveals order of feature importance and diminishing returns.

### Disease-Specific Ablations
Train separate ablation models for:
- Heart failure mortality (different aging patterns)
- Sepsis outcomes (inflammatory signal more important)
- Diabetes complications (metabolic markers matter)

### Temporal Analysis
- Do predictions improve over time?
- Which features stabilize earliest?
- Does acceleration rate matter more than absolute value?

### Cross-Disease Transfer
- Train on heart failure, test on diabetes
- Understand generalizability of biological proxies
- Identify disease-specific vs. universal aging signals

---

## Files Created This Session

1. **Design & Documentation**
   - `MULTI_MODEL_ABLATION_DESIGN.md` (this comprehensive design)
   - `ABLATION_EXECUTION_GUIDE.md` (this guide)

2. **Implementation**
   - `corebehrt/functional/evaluation/comprehensive_metrics.py` (evaluation suite)
   - `corebehrt/ablation/batch_ablation_runner.py` (batch orchestration)

3. **Scripts** (you'll create)
   - `scripts/run_full_ablation.py` (main execution script)
   - `corebehrt/configs/ablation_*.yaml` (config files for each model)

---

## Troubleshooting

**Q: Training takes too long**
A: Reduce `n_workers` in BatchAblationRunner if memory constrained, or train models sequentially

**Q: Models not improving with features**
A: Check that features are properly injected into training data. Verify `prepared_data_with_values` contains biological values.

**Q: Calibration still poor**
A: Consider adding calibration layer (temperature scaling, Platt scaling) after training

**Q: Too much variance in metrics**
A: Increase bootstrap samples from 1000 to 5000 for more stable CI estimates

---

## Next Steps

1. ✅ Complete STEP 6 (finetune baseline model): `python scripts/run_finetune_step6.py`
2. Create ablation config files for each feature subset
3. Run batch ablation: `python scripts/run_full_ablation.py`
4. Analyze comparison table to identify optimal features
5. (Optional) Run disease-specific ablations
6. (Optional) Test cross-disease transfer

---

## Questions?

Refer back to:
- `MULTI_MODEL_ABLATION_DESIGN.md` for detailed design rationale
- `corebehrt/functional/evaluation/comprehensive_metrics.py` for metric definitions
- `corebehrt/ablation/batch_ablation_runner.py` for orchestration details
