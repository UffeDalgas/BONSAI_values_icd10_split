# Using Epimap Outputs with BONSAI Pipeline

Complete guide for integrating epimap biological values into BONSAI finetuning.

---

## Quick Start

### 1. Generate Epimap CSVs (Already Done ✅)

Epimap has created 8 condition-specific CSVs in:
```
/Users/uffedalgas/Desktop/epimap/outputs/bonsai_inputs/
```

Files created:
```
bonsai_condition_1_ehr_only.csv
bonsai_condition_2_metadata_preop.csv
bonsai_condition_3_grimage_v2.csv
bonsai_condition_4_systems_age.csv
bonsai_condition_5_cpgt_grimage3.csv
bonsai_condition_6_maple_embeddings.csv
bonsai_condition_7_cpgpt_embeddings.csv
bonsai_condition_8_methylgpt_embeddings.csv
```

All files:
- Row-aligned by sample_id
- Include disease_risk and DMI fields
- Ready to use with BONSAI

### 2. Copy to BONSAI Project

```bash
# Copy epimap CSVs to BONSAI data directory
cp /Users/uffedalgas/Desktop/epimap/outputs/bonsai_inputs/*.csv \
   /Users/uffedalgas/Desktop/BONSAI_values/data/biological_proxies/
```

### 3. Prepare EHR Data (Your Part)

You need to provide:
- **MEDS parquet files** (from ehr2meds pipeline)
  - Location: `./outputs/tokenized/`
  - Format: Parquet with columns [person_id, event_timestamp, concept_id, value]

- **Outcomes data** (mortality, complications, etc.)
  - Location: `./outputs/outcomes/TEST_OUTCOME.csv`
  - Format: CSV with [person_id, outcome_label]

### 4. Run BONSAI Pipeline

```bash
cd /Users/uffedalgas/Desktop/BONSAI_values
conda activate bonsai_dryrun

# Option A: Quick demo with mock predictions
python3 scripts/quick_evaluation_demo.py

# Option B: Simplified end-to-end with simulated training
python3 scripts/simplified_end_to_end_demo.py

# Option C: Full BONSAI pipeline (requires complete setup)
python3 scripts/run_end_to_end_pipeline.py
```

---

## Understanding the 8 Conditions

Each CSV represents one finetuning condition:

### 1. EHR Only (Baseline)
- **Features:** None (EHR sequences only)
- **Purpose:** Control condition - establishes baseline performance
- **Expected:** Moderate ROC-AUC

### 2. Metadata (_preop values)
- **Features:** Clinical preoperative values
- **Includes:** Age, ASA score, Charlson index, BMI, blood values, etc.
- **Purpose:** Test contribution of standard clinical variables
- **Expected:** Incremental improvement over baseline

### 3. GrimAge v2
- **Features:** Epigenetic age acceleration (pcdnamtl)
- **Purpose:** Test aging clock as predictor
- **Expected:** Moderate improvement

### 4. SystemsAge (11 components)
- **Features:** All 11 physiological system ages
- **Includes:** Aging of immune, inflammatory, kidney, liver, lung, etc. systems
- **Purpose:** Comprehensive biological system assessment
- **Expected:** Significant improvement

### 5. CpGPTGrimAge v3 + Proteins
- **Features:** Foundation model outputs + protein proxies
- **Purpose:** High-dimensional learned representations
- **Expected:** Strongest improvement (if model available)
- **Note:** Currently sparse (foundation model not local)

### 6. MAPLE Embeddings
- **Features:** 32-dimensional embeddings from MAPLE
- **Purpose:** Task-specific age prediction embeddings
- **Expected:** Strong improvement

### 7. CpGPT Embeddings
- **Features:** 64+ dimensional embeddings from CpGPT
- **Purpose:** DNA-based learned representations
- **Expected:** Very strong improvement
- **Note:** Requires foundation model (not local)

### 8. MethylGPT Embeddings
- **Features:** 256-dimensional embeddings from MethylGPT
- **Purpose:** High-dimensional methylation-based embeddings
- **Expected:** Strongest improvement among available models

---

## CSV Structure

### Common Columns (All Files)
```
sample_id        : Patient identifier (e.g., "SURG_000083")
dmi_raw          : Disease Mortality Index (BASRAI, raw)
dmi_residual     : Disease Mortality Index (BASRAI, residual)
[features...]    : Condition-specific biological values
condition        : Label for the condition
```

### Example: bonsai_condition_4_systems_age.csv

```csv
sample_id,dmi_raw,dmi_residual,systemsage,systemsageblood,systemsagebrain,systemsageheart,systemsagehormone,systemsageimmune,systemsageinflammation,systemsagekidney,systemsageliver,systemsagelung,systemsagemetabolic,condition
SURG_000083,0.234,0.123,45.2,43.1,46.5,44.8,47.2,42.1,48.3,45.6,44.2,46.1,47.8,systems_age
SURG_000053,0.189,0.087,42.1,40.5,43.2,41.8,44.1,39.5,45.2,42.8,41.5,43.6,44.3,systems_age
...
```

---

## Integration with BONSAI Pipeline

### How BONSAI Uses These CSVs

1. **Patient Selection** (Manual)
   ```python
   finetune_patients = set(df['sample_id'])  # Your choice!
   ```

2. **Load Biological Values**
   ```python
   df = pd.read_csv('bonsai_condition_3_grimage_v2.csv')
   biological_features = df[['sample_id', 'pcdnamtl', 'dmi_raw', 'dmi_residual']]
   ```

3. **Combine with EHR Data**
   ```python
   combined = ehr_data.merge(biological_features, on='sample_id')
   ```

4. **Train Model**
   ```python
   # BONSAI trains with biological features injected
   # (concat, FiLM, discrete, comb, or comb_binning methods)
   ```

5. **Evaluate**
   ```python
   # Compare all 8 conditions
   # Measure: ROC-AUC, calibration, clinical utility, statistical significance
   ```

---

## Expected Results

### Baseline Performance
```
Condition 1 (EHR only):           ROC-AUC ≈ 0.60-0.65
```

### With Biological Features
```
Condition 2 (Metadata):           ROC-AUC ≈ 0.62-0.68
Condition 3 (GrimAge):            ROC-AUC ≈ 0.65-0.72
Condition 4 (SystemsAge):         ROC-AUC ≈ 0.68-0.75
Condition 5 (CpGPT+Proteins):     ROC-AUC ≈ 0.70-0.78 ⭐
Condition 6 (MAPLE):              ROC-AUC ≈ 0.67-0.75
Condition 7 (CpGPT embed):        ROC-AUC ≈ 0.72-0.80 ⭐
Condition 8 (MethylGPT):          ROC-AUC ≈ 0.71-0.79 ⭐
```

**⭐ Most promising** = Complex embeddings + epigenetic information

---

## Troubleshooting

### Problem: Missing Values in CSV

**If some conditions have sparse data:**
- Condition 5 (CpGPT+Proteins): Requires foundation model (may be mostly NaN)
- Condition 7 (CpGPT embed): Requires foundation model (may be missing)

**Solution:** These are expected. The script gracefully handles missing columns.

### Problem: Sample Alignment Issues

**Check that all CSVs have same sample order:**
```bash
cut -d',' -f1 bonsai_condition_1_*.csv > ids1.txt
cut -d',' -f1 bonsai_condition_8_*.csv > ids8.txt
diff ids1.txt ids8.txt  # Should be empty
```

### Problem: Merging with EHR Data

**Ensure sample IDs match:**
```python
# EHR data must have 'sample_id' or 'person_id' column
ehr_df['sample_id'] = ehr_df['person_id']

# Then merge
combined = ehr_df.merge(bio_df, on='sample_id')
```

---

## Advanced Usage

### Subset to Specific Samples

```python
import pandas as pd

# Load epimap data
df = pd.read_csv('bonsai_condition_4_systems_age.csv')

# Subset to samples with all measurements
df_complete = df.dropna()

# Or subset to your manual selection
my_samples = ['SURG_000083', 'SURG_000053', 'SURG_000070']
df_subset = df[df['sample_id'].isin(my_samples)]
```

### Combine Multiple Conditions

```python
# Create a "maximum features" condition with all non-sparse data
df_list = [pd.read_csv(f) for f in ['bonsai_condition_*.csv']]
combined = df_list[0][['sample_id', 'dmi_raw', 'dmi_residual']]

for df in df_list:
    # Add non-ID columns
    feature_cols = [c for c in df.columns if c not in ['sample_id', 'condition', 'dmi_raw', 'dmi_residual']]
    combined = combined.merge(df[['sample_id'] + feature_cols], on='sample_id', how='left')
```

### Create Disease-Specific Subsets

```python
# If you want to filter by diagnosis (from EHR data)
disease_patients = ehr_df[ehr_df['diagnosis'] == 'D74'].sample_id

df = pd.read_csv('bonsai_condition_4_systems_age.csv')
df_disease = df[df['sample_id'].isin(disease_patients)]
```

---

## Scalability

### Current Setup
- **Samples:** 100 (50 used in dry run)
- **Time to generate:** ~1 minute
- **File sizes:** 1KB - 160KB per CSV

### Full Production Setup
- **Samples:** 50,000+
- **Estimated time:** 30-60 minutes (parallel processing on cluster)
- **File sizes:** 200KB - 20MB per CSV

**The script scales linearly. Just provide more samples!**

---

## Files & Locations

### Epimap Inputs
```
/Users/uffedalgas/Desktop/epimap/
├── outputs/
│   ├── geo_metadata.csv
│   ├── clock_predictions.csv
│   ├── episcore_disease_risk.csv
│   ├── dmi_results.csv
│   ├── maple_embeddings_age.csv
│   ├── methylgpt_embeddings.csv
│   └── episcore_protein_levels.csv
└── scripts/
    └── prepare_bonsai_inputs.py
```

### BONSAI Integration Points
```
/Users/uffedalgas/Desktop/BONSAI_values/
├── data/
│   └── biological_proxies/         ← Copy epimap CSVs here
├── outputs/tokenized/              ← MEDS format EHR data
├── outputs/outcomes/               ← Outcomes data
└── scripts/
    ├── simplified_end_to_end_demo.py
    └── run_end_to_end_pipeline.py
```

---

## Quick Reference Commands

```bash
# Generate epimap CSVs (50 samples for testing)
cd /Users/uffedalgas/Desktop/epimap
python3 scripts/prepare_bonsai_inputs.py 50

# Generate all samples (full production run)
python3 scripts/prepare_bonsai_inputs.py

# Copy to BONSAI
cp outputs/bonsai_inputs/*.csv \
   /Users/uffedalgas/Desktop/BONSAI_values/data/biological_proxies/

# Run BONSAI pipeline
cd /Users/uffedalgas/Desktop/BONSAI_values
python3 scripts/simplified_end_to_end_demo.py
```

---

## Next Steps

1. ✅ **Phase 1 Complete:** Epimap biological values generated
2. ⏳ **Phase 2:** Prepare MEDS format EHR data
3. ⏳ **Phase 3:** Run BONSAI finetuning with 8 conditions
4. ⏳ **Phase 4:** Comprehensive evaluation & comparison

**The framework is ready. Just add your EHR data!**
