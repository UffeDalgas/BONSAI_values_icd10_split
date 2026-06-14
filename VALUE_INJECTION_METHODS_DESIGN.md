# Value Injection Methods: Complete Design

## Overview

Your BONSAI framework supports **5 distinct value injection approaches**. Each has different properties:

```
Biological Features (8 subsets)
    ↓
    ├─ concat (concatenation)
    ├─ FiLM (Feature-wise Linear Modulation)
    ├─ discrete (discretization)
    ├─ comb (combination/fusion)
    └─ comb_binning (combination with binning)
    ↓
5 Injection Methods × 8 Feature Subsets = 40 Model Variants
```

---

## The 5 Injection Methods

### 1. CONCAT: Simple Concatenation

**How it works:**
```
Input EHR embedding: [e1, e2, e3, ..., e128]  (from pretrain)
Biological values:   [v1, v2, v3, ..., v8]    (GrimAge, SystemsAge, etc.)
                     ↓
Output to finetune:  [e1, e2, ..., e128, v1, v2, ..., v8]
                     └─────────────────────┬──────────────┘
                              136 dimensions total
```

**Properties:**
- ✓ Simplest, interpretable
- ✓ Allows model to learn feature weights naturally
- ✗ Treats values same as embeddings (scale mismatch)
- ✗ May need normalization
- **Best for:** Tabular features, clear feature importance needed

**When to use:**
- Baseline approach
- When you want explicit feature weights
- Limited data (fewer parameters to learn)

---

### 2. FiLM: Feature-wise Linear Modulation

**How it works:**
```
EHR embedding: [e1, e2, ..., e128]

For each biological value vi:
  γ = Linear(vi) → [γ1, γ2, ..., γ128]
  β = Linear(vi) → [β1, β2, ..., β128]
  
  e'_j = γ_j * e_j + β_j    ∀ j ∈ [1..128]

Result: Modulated embedding [e'1, e'2, ..., e'128]
```

**Properties:**
- ✓ Values act as gates/modulators, not raw inputs
- ✓ Each value controls all embedding dimensions
- ✓ Multiplicative interaction (more expressive)
- ✓ Computationally efficient
- ✗ More parameters to learn
- **Best for:** Learning complex feature interactions

**When to use:**
- Strong biological signal (confident in feature values)
- Want multiplicative interactions
- Have sufficient data
- Value ranges well understood

**Intuition:**
- If GrimAge=0.8 (high aging), scale up "frailty dimensions" of EHR
- If GrimAge=0.2 (low aging), scale down "frailty dimensions"

---

### 3. DISCRETE: Discretization into Bins

**How it works:**
```
Biological values → Binned representation
  GrimAge = 0.65 → [0, 0, 1, 0, 0]  (bin 3 of 5)
  SystemsAge = 0.42 → [0, 1, 0, 0, 0]  (bin 2 of 5)

Binned embeddings → Learned bin embeddings
  Bin 1: [w11, w12, ..., w1_d]
  Bin 2: [w21, w22, ..., w2_d]
  ...
  Bin 5: [w51, w52, ..., w5_d]

Final embedding = concatenate bin embeddings
```

**Properties:**
- ✓ Reduces continuous value complexity
- ✓ Captures non-linear relationships
- ✓ Less sensitive to value scaling
- ✓ Interpretable (quintiles, quartiles)
- ✗ Loses fine-grained information
- ✗ More embeddings to learn
- **Best for:** Noisy data, non-linear signals

**When to use:**
- Values may be noisy or imprecise
- Non-linear relationships expected
- Want quintile-based clinical interpretation
- Have abundant data

**Number of bins:**
- Standard: 5 bins (quintiles) - clinical interpretation
- Conservative: 3 bins (low/medium/high)
- Detailed: 10 bins (deciles)

---

### 4. COMB: Direct Fusion/Combination

**How it works:**
```
EHR embedding: [e1, e2, ..., e128]
Values: [v1, v2, ..., v8]

Fusion layer: 
  z = Linear([e1, e2, ..., e128, v1, v2, ..., v8])
  
Or element-wise:
  z_j = w_j * e_j + Σ c_ij * v_i * e_j
        └─────────┬──────────┘
          EHR      Cross-interaction
```

**Properties:**
- ✓ Explicit fusion/combination logic
- ✓ Can model cross-feature interactions
- ✓ More expressive than concat
- ✗ More parameters
- **Best for:** Multi-feature fusion, learned combinations

**When to use:**
- Multiple values interact (not additive)
- Want to learn feature combinations
- Have good theoretical understanding

---

### 5. COMB_BINNING: Discretization + Combination

**How it works:**
```
Step 1: Bin values as in DISCRETE
  GrimAge → bin embedding [g1, g2, ..., g_d]
  SystemsAge → bin embedding [s1, s2, ..., s_d]
  
Step 2: Fuse bin embeddings with EHR
  z = Linear([e1, ..., e128, g1, ..., g_d, s1, ..., s_d])
  Or: element-wise combinations
```

**Properties:**
- ✓ Combines benefits of DISCRETE (interpretability) and COMB (expressiveness)
- ✓ Reduced noise (binning) + learned interactions (fusion)
- ✓ Most expressive, most complex
- ✗ Most parameters, needs most data
- **Best for:** Large datasets, complex signal

**When to use:**
- Large cohorts available
- Complex interactions suspected
- Want both interpretability and expressiveness
- Can afford extra parameters

---

## Comparison Table

| Method | Simplicity | Expressiveness | Parameters | Interpretability | Best Use Case |
|--------|-----------|-----------------|-----------|-----------------|---------------|
| concat | ★★★★★ | ★★☆☆☆ | Few | High | Baseline, limited data |
| FiLM | ★★★☆☆ | ★★★★☆ | Medium | Medium | Multiplicative signal |
| discrete | ★★★☆☆ | ★★★☆☆ | Medium | High | Noisy data, quintiles |
| comb | ★★★☆☆ | ★★★★☆ | Medium | Medium | Fusion learning |
| comb_binning | ★★☆☆☆ | ★★★★★ | High | High | Large data, complex |

---

## Multi-Value Support

**Yes, all 5 methods support multiple values at once!**

### How It Works

With your 8 feature subsets (e.g., GrimAge v2 + intermediate values):

```
SUBSET: GrimAge v2 + Intermediate Values
  v1: GrimAge (biological age)
  v2: DNAm-based glucose
  v3: DNAm-based CRP
  v4: DNAm-based WBC
  ...
  v8: GrimAge intermediate composite

CONCAT METHOD:
  embedding = [e1, ..., e128, v1, v2, v3, v4, v5, v6, v7, v8]
  
FILM METHOD:
  For each vi (i=1..8):
    γi = Linear_γ(vi)
    βi = Linear_β(vi)
    e = γi * e + βi
  Result: Modulated by all 8 values

DISCRETE METHOD:
  For each vi (i=1..8):
    bin_i = digitize(vi, bins=5)
    embed_i = Embedding(bin_i)
  Result: Concatenate all 8 embeddings
  
COMB METHOD:
  z = Linear([e1, ..., e128, v1, v2, ..., v8])
  All values fused in single layer
  
COMB_BINNING METHOD:
  For each vi: bin_embed_i = get_bin_embedding(vi, bin=5)
  z = Linear([e1, ..., e128, bin_embed_1, ..., bin_embed_8])
  All binned embeddings fused together
```

### Multiple Values in Parallel

You can also apply different injection methods to different features:

```
GrimAge subset: Use FiLM (multiplicative aging signal)
SystemsAge subset: Use discrete (quintile interpretation)
MAPLE embeddings: Use concat (already learned embeddings)
CpGPT embeddings: Use comb_binning (fusion with binning)
```

---

## The Full Experimental Grid

Your ablation now expands to:

```
8 Feature Subsets
× 5 Injection Methods
= 40 Model Variants

From single pretrain checkpoint:
```

### Grid Layout

```
                  concat    FiLM    discrete    comb    comb_binning
              ──────────────────────────────────────────────────────
EHR only        M1-1     M1-2      M1-3       M1-4       M1-5
Metadata        M2-1     M2-2      M2-3       M2-4       M2-5
GrimAge         M3-1     M3-2      M3-3       M3-4       M3-5
SystemsAge      M4-1     M4-2      M4-3       M4-4       M4-5
CpGPT+Proteins  M5-1     M5-2      M5-3       M5-4       M5-5
MAPLE           M6-1     M6-2      M6-3       M6-4       M6-5
CpGPT Embed     M7-1     M7-2      M7-3       M7-4       M7-5
MethylGPT       M8-1     M8-2      M8-3       M8-4       M8-5
All Features    M9-1     M9-2      M9-3       M9-4       M9-5
```

### Analysis Questions Per Cell

For each model variant (M_ij), compute:

1. **ROC-AUC** — discrimination ability
2. **ECE** — calibration quality
3. **Training speed** — method efficiency
4. **Parameter count** — model complexity
5. **Convergence** — training stability

### Expected Insights

```
Question 1: Which method works best overall?
  Answer: Comparison across all 40 models
  
  Possible outcome:
    concat:         0.698 ± 0.032
    FiLM:           0.712 ± 0.028  ← Best overall
    discrete:       0.701 ± 0.035
    comb:           0.709 ± 0.031
    comb_binning:   0.715 ± 0.029
    
Question 2: Does method vary by feature type?
  Answer: Which method works best for each feature subset?
  
  Possible outcome:
    GrimAge (continuous clock): FiLM best (+0.087)
    SystemsAge (11 components): discrete best (+0.045)
    MAPLE (32-dim embedding): concat best (+0.010)
    CpGPT (64-dim embedding): comb_binning best (+0.035)
    
Question 3: What's the computational tradeoff?
  Answer: ROC-AUC vs. training time vs. parameters
  
  Possible outcome:
    concat:         0.698, 10 min, 130K params
    FiLM:           0.712, 12 min, 145K params
    discrete:       0.701, 11 min, 160K params
    comb:           0.709, 13 min, 155K params
    comb_binning:   0.715, 15 min, 180K params
    
  → FiLM best efficiency (ROC-AUC / parameters / time)

Question 4: Is there a best combination (feature + method)?
  Answer: Which (feature, method) pair is optimal?
  
  Possible outcome:
    (GrimAge, FiLM):              0.712
    (GrimAge+SystemsAge, FiLM):   0.724
    (All, comb_binning):          0.742
    → Recommend: All features with comb_binning
```

---

## Implementation Strategy

### Configuration Files

Create YAML configs for each combination:

```yaml
# corebehrt/configs/ablation_grim_age_film.yaml
paths:
  prepared_data: ./outputs/finetuning/processed_data_with_values/
  pretrain_model: ./outputs/pretraining_dryrun
  model: ./outputs/ablation_models/grim_age_film

model:
  cls: default
  value_embedding_mode: "film"  # ← Different for each!
  
biological_features:
  - "grim_age_v2"
  - "grim_age_v2_intermediate"

trainer_args:
  batch_size: 16
  epochs: 3
```

### Batch Runner Update

```python
from corebehrt.ablation.batch_ablation_runner import BatchAblationRunner

runner = BatchAblationRunner(
    pretrain_checkpoint="./outputs/pretraining_dryrun/checkpoints/best.pt",
    output_dir="./outputs/ablation_results",
    n_workers=4  # Train 4 in parallel
)

# Define feature subsets
feature_subsets = [
    ("ehr_only", []),
    ("grim_age", ["grim_age_v2"]),
    ("systems_age", ["systems_age_*"]),
    ("all_features", ["*"]),
]

# Define injection methods
injection_methods = ["concat", "film", "discrete", "comb", "comb_binning"]

# Create all 40 configs
for feat_name, feat_list in feature_subsets:
    for method in injection_methods:
        config_name = f"{feat_name}_{method}"
        runner.add_ablation_config(
            name=config_name,
            config_path=f"./corebehrt/configs/ablation_{feat_name}_{method}.yaml",
            features=feat_list,
            description=f"{feat_name} with {method} injection"
        )

# Train all 40 models
runner.train_all_models(n_workers=4)  # Parallel!

# Evaluate all 40 models
runner.evaluate_all_models()

# Generate comparison report (will be much richer!)
runner.generate_comparison_report()
```

---

## Expected Performance Profile

Based on typical deep learning patterns:

### By Injection Method

```
concat:           Simple baseline
  └─ Good for: small data, interpretability
  
FiLM:             Multiplicative interactions
  └─ Good for: features that modulate embeddings
  └─ Expected: +0.02-0.05 over concat
  
discrete:         Non-linear binning
  └─ Good for: noisy data, quintile interpretation
  └─ Expected: +0.01-0.03 over concat
  
comb:             Learned fusion
  └─ Good for: explicit combinations
  └─ Expected: +0.02-0.06 over concat
  
comb_binning:     Best expressiveness
  └─ Good for: large data, complex signal
  └─ Expected: +0.03-0.07 over concat
```

### By Feature Type × Method Interaction

```
Aging Clocks (GrimAge, SystemsAge):
  concat: 0.685
  FiLM: 0.715  ← Best (multiplicative aging signal)
  discrete: 0.698
  comb: 0.712
  comb_binning: 0.708

Embeddings (MAPLE, CpGPT):
  concat: 0.622  ← Best (already learned)
  FiLM: 0.618 (overfitting risk)
  discrete: 0.615
  comb: 0.625
  comb_binning: 0.623

Protein Proxies (CpGPT Proteins):
  concat: 0.658
  FiLM: 0.667
  discrete: 0.662
  comb: 0.673  ← Best (fusion signal)
  comb_binning: 0.671
```

---

## Key Insights You'll Get

### Insight 1: Universal Best Method
- Is one method consistently better (e.g., FiLM > all)?
- Or does it depend on feature type?

### Insight 2: Method × Feature Interaction
- Which combinations are synergistic?
- Any combinations to avoid?

### Insight 3: Practical Tradeoffs
- FiLM: better performance, minimal overhead
- comb_binning: best performance, highest complexity

### Insight 4: Clinical Interpretability
- Which method gives clearest feature importance?
- Can clinicians understand how features contribute?

---

## Execution Plan

### Phase 1: Generate Configurations
```bash
# Create all 40 YAML config files
python scripts/generate_ablation_configs.py
```

### Phase 2: Train All Models
```bash
# Takes ~40 × 30-60 min = 20-40 hours on CPU
# But parallelizable: ~5-10 hours with n_workers=4
python scripts/run_full_ablation_with_methods.py
```

### Phase 3: Comprehensive Comparison
```
Output: 
  ✓ 40-row comparison table
  ✓ Method performance plot
  ✓ Feature × Method interaction heatmap
  ✓ Efficiency frontier (ROC-AUC vs. complexity)
```

---

## Files to Create

1. **Config Generator**
   - `scripts/generate_ablation_configs.py`
   - Creates 40 YAML configs from templates

2. **Extended Batch Runner**
   - Update `batch_ablation_runner.py` to handle methods
   - Add method-specific comparison logic

3. **Advanced Visualizations**
   - 2D heatmap: Features × Methods
   - Efficiency frontier plot
   - Method performance distribution

---

## Summary

Your experimental scope expands significantly:

**Before:** 8 models × 1 injection method = 8 variants
**After:** 8 features × 5 methods = 40 variants

**New insights gained:**
- Which injection method is universally best?
- Do methods work better for specific feature types?
- What's the efficiency frontier (performance vs. complexity)?
- Which (feature, method) combination is optimal?

**Investment:** ~10-20 hours of computation
**Return:** Deep understanding of value injection mechanics + optimal deployment config
