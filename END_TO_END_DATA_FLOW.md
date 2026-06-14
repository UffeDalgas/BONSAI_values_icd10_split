# End-to-End Data Flow: From Raw EHR to Model Evaluation

## The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ INPUTS (Your responsibility)                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Raw EHR Data                                                             │
│    └─ Patient encounters, diagnoses, medications, procedures               │
│    └─ Format: CSV, OMOP, FHIR, or any structured format                   │
│                                                                             │
│ 2. EHR2MEDS Converter                                                      │
│    └─ Your ehr2meds repository                                            │
│    └─ Produces: MEDS parquet files (standardized EHR format)              │
│                                                                             │
│ 3. Disease ICD Code                                                        │
│    └─ E.g., "D74*" for diabetes, "I50*" for heart failure                 │
│    └─ Used to filter pretrain/finetune/eval cohorts                       │
│                                                                             │
│ 4. Biological Proxy Files (from epimap)                                   │
│    └─ GrimAge v2 measurements: patient_id, grim_age, timestamp            │
│    └─ SystemsAge components: patient_id, sys_age, glucose, crp, ...      │
│    └─ MAPLE embeddings: patient_id, maple_dim1, maple_dim2, ...          │
│    └─ CpGPT embeddings: patient_id, cpgt_dim1, cpgt_dim2, ...            │
│    └─ All formats: CSV or Parquet with patient_id index                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: MEDS DATA PREPARATION (Codebase handles)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ STEP 1a: Load MEDS parquet files                                           │
│          Input:  ./outputs/tokenized/ (from epimap ehr2meds pipeline)     │
│          Output: MEDS DataFrame with columns:                              │
│                  [person_id, event_timestamp, concept_id, value, ...]    │
│                                                                             │
│ STEP 1b: Tokenize EHR events → patient sequences                          │
│          Input:  MEDS DataFrame                                            │
│          Output: ./outputs/tokenized/features_*.parquet                    │
│                  Tokenized sequences ready for model                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: DISEASE-STRATIFIED COHORT CREATION (Codebase handles)            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input:  MEDS data + disease ICD code (e.g., "D74*")                       │
│         Biological proxy files with patient_ids                            │
│                                                                             │
│ Process:                                                                    │
│  └─ DiseaseFilter.identify_patients_with_disease("D74*")                  │
│     → Find all patients with diabetes diagnosis                            │
│     → Result: Set of patient_ids with disease                             │
│                                                                             │
│  └─ Match with biological_proxy_patients (from CSV files)                 │
│     → Which disease patients have bio measurements?                        │
│     → Result: finetune_patients = disease ∩ has_proxies                   │
│                                  eval_patients = disease ∖ has_proxies    │
│                                  pretrain_patients = all ∖ disease        │
│                                                                             │
│ Output: Three cohorts (parquet files)                                      │
│  ├─ pretrain_cohort.parquet   (all except disease)                        │
│  ├─ finetune_cohort.parquet   (disease + bio proxies)                     │
│  └─ eval_cohort.parquet        (disease, no proxies - test set)           │
│                                                                             │
│ Output: Patient ID lists                                                   │
│  ├─ pretrain_patients.txt                                                  │
│  ├─ finetune_patients.txt                                                  │
│  └─ eval_patients.txt                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: BIOLOGICAL PROXY INTEGRATION (Codebase handles)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input:  finetune_patients.txt + biological proxy CSVs                      │
│                                                                             │
│ Process:                                                                    │
│  └─ Load GrimAge_v2.csv                                                    │
│     ├─ Columns: [patient_id, grim_age_value, timestamp]                   │
│     └─ Filter to finetune_patients only                                    │
│                                                                             │
│  └─ Load SystemsAge_components.csv                                         │
│     ├─ Columns: [patient_id, sys_age, glucose, crp, wbc, ...]           │
│     └─ Filter to finetune_patients only                                    │
│                                                                             │
│  └─ Load MAPLE_embeddings.csv                                              │
│     ├─ Columns: [patient_id, maple_dim_1, ..., maple_dim_32]             │
│     └─ Filter to finetune_patients only                                    │
│                                                                             │
│  └─ Merge all into single "values" table                                   │
│     ├─ Index: patient_id                                                   │
│     └─ Columns: all bio proxy measurements                                 │
│                                                                             │
│ Output: ./outputs/features/                                                │
│  ├─ patient_values.parquet    (finetune patients + all bio proxies)       │
│  ├─ patient_info.parquet       (metadata, demographics)                    │
│  └─ data_config.yaml           (feature definitions)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: OUTCOME DEFINITION (User responsibility)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input:  Outcome data (mortality, readmission, complication, etc.)          │
│         Format: CSV with [patient_id, outcome, date]                       │
│                                                                             │
│ Process:                                                                    │
│  └─ Filter to cohort patients only                                         │
│  └─ Align with MEDS timeline                                               │
│  └─ Create binary labels for finetuning                                    │
│                                                                             │
│ Output: ./outputs/outcomes/                                                │
│  └─ TEST_OUTCOME.csv                                                       │
│     ├─ Columns: [person_id, mortality, follow_up_days]                    │
│     └─ Only finetune + eval patients                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: PRETRAIN DATA PREPARATION (Codebase handles)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input:  pretrain_cohort.parquet (EHR sequences only)                       │
│                                                                             │
│ Process:                                                                    │
│  └─ prepare_training_data.main_prepare_data()                             │
│     ├─ Load tokenized sequences                                            │
│     ├─ Create train/val splits                                             │
│     └─ Normalize sequence lengths                                          │
│                                                                             │
│ Output: ./outputs/pretraining/processed_data/                              │
│  ├─ patients_train.pt                                                      │
│  ├─ patients_val.pt                                                        │
│  └─ splits config                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: PRETRAIN MODEL (Codebase handles)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input:  Pretrain data (EHR only, no biological features)                   │
│         Config: pretrain_dryrun.yaml                                       │
│                                                                             │
│ Process:                                                                    │
│  └─ Train ModernBERT on raw EHR sequences                                  │
│     ├─ 2 epochs (dryrun) or more for production                            │
│     ├─ Masked language modeling objective                                  │
│     └─ Generic EHR representation learned                                  │
│                                                                             │
│ Output: ./outputs/pretraining_dryrun/checkpoints/                          │
│  └─ checkpoint_epoch999_end.pt  (8.0 MB)                                   │
│     → Reused for all 40 finetuning variants                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: FINETUNE DATA PREPARATION (Codebase handles)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input:  finetune_cohort.parquet (EHR sequences)                            │
│         patient_values.parquet (biological proxies)                        │
│         outcomes.csv (mortality labels)                                    │
│                                                                             │
│ Process:                                                                    │
│  └─ prepare_training_data.main_prepare_data(mode="finetune")              │
│     ├─ Load tokenized sequences for finetune patients                      │
│     ├─ Load biological values from CSVs                                    │
│     ├─ Align values with sequence timestamps                              │
│     ├─ Create train/val/test splits                                        │
│     └─ Combine into single dataset                                         │
│                                                                             │
│ Output: ./outputs/finetuning/processed_data_with_values/                   │
│  ├─ patients_train.pt   (with value tensors embedded)                     │
│  ├─ patients_val.pt     (with value tensors embedded)                     │
│  ├─ patients_test.pt    (with value tensors embedded)                     │
│  └─ data_config.yaml                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 8: FINETUNING (40 VARIANTS) (Codebase handles)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ For each (feature_subset, injection_method) combination:                   │
│                                                                             │
│  Input:  Pretrain checkpoint (shared)                                      │
│          Finetune data with biological values                              │
│          Config: ablation_[feature]_[method].yaml                          │
│          - biological_features: [grim_age, systems_age, ...]              │
│          - value_embedding_mode: concat|film|discrete|comb|comb_binning   │
│                                                                             │
│  Process:                                                                    │
│   └─ Load pretrain checkpoint                                              │
│   └─ Add feature-specific embedding layer                                  │
│   └─ Finetune on mortality prediction task (3 epochs)                      │
│   └─ Evaluate on held-out test set                                         │
│                                                                             │
│  Output: ./outputs/ablation_models/[feature]_[method]/                     │
│   ├─ best_model.pt                                                         │
│   ├─ metrics.json (ROC-AUC, calibration, etc.)                            │
│   └─ predictions.csv (test predictions)                                    │
│                                                                             │
│ × 40 variants total                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 9: COMPREHENSIVE EVALUATION (Codebase handles)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ For each of 40 models:                                                     │
│                                                                             │
│  Inputs:  predictions.csv (from finetuning)                                │
│           true_labels (from outcomes.csv)                                  │
│                                                                             │
│  Metrics Computed:                                                         │
│   ├─ Discrimination: ROC-AUC, PR-AUC (with 95% CI via bootstrap)         │
│   ├─ Calibration: ECE, Hosmer-Lemeshow test, calibration slope           │
│   ├─ Clinical Utility: Net benefit, decision thresholds                    │
│   ├─ Risk Stratification: Quintile analysis, ICI                           │
│   └─ Subgroup Analysis: Performance by age, sex, comorbidities            │
│                                                                             │
│ Output: ./outputs/ablation_results/                                        │
│  ├─ ablation_comparison.csv (40 rows × 12 metrics)                        │
│  ├─ method_feature_matrix.csv (methods vs. features ROC-AUC)              │
│  ├─ comparison_plot.png (visualization)                                    │
│  ├─ roc_curves.png (all 40 models)                                         │
│  ├─ calibration_plots.png (all 40 models)                                  │
│  └─ feature_importance.csv (SHAP analysis)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                         ✓ ANALYSIS COMPLETE
```

---

## Input Data Format Specifications

### 1. MEDS Data (Output of EHR2MEDS)

**Location:** `./outputs/tokenized/` (parquet files from your ehr2meds pipeline)

**Format:** Parquet files with columns:
```python
person_id: int64              # Patient identifier
event_timestamp: datetime64   # When the event occurred
concept_id: string            # Medical concept (e.g., "D74.1", "M10.001")
value: float64 (optional)     # Numeric value (lab result, measurement)
unit: string (optional)       # Unit of measurement
source_table: string          # Original data source
```

**Example Structure:**
```
person_id | event_timestamp      | concept_id | value | unit
----------|----------------------|------------|-------|------
101       | 2020-01-15 09:30:00 | D74.1      | NaN   | -
101       | 2020-01-15 10:15:00 | L123       | 115.0 | mg/dL
101       | 2020-02-10 14:00:00 | M01.02     | 1     | count
102       | 2019-12-01 08:00:00 | I50.0      | NaN   | -
102       | 2019-12-01 08:30:00 | L456       | 0.8   | mmol/L
...
```

---

### 2. Biological Proxy Files (From epimap)

**Location:** `./data/biological_proxies/` (you create these from epimap outputs)

**Format 1: GrimAge v2**
```
File: GrimAge_v2.csv

person_id | grim_age_value | dnam_glucose | dnam_crp | measurement_date
----------|----------------|--------------|----------|------------------
101       | 0.65           | 0.58         | 0.72     | 2020-06-15
102       | 0.42           | 0.35         | 0.48     | 2020-06-20
103       | 0.78           | 0.81         | 0.85     | 2020-06-25
...
```

**Format 2: SystemsAge (11 Components)**
```
File: SystemsAge_components.csv

person_id | systems_age | glucose | crp | wbc | rbc | platelets | sbp | dbp | cholesterol | ldl | hdl
----------|-------------|---------|-----|-----|-----|-----------|-----|-----|-------------|-----|-----
101       | 0.55        | 105     | 2.1 | 7.2 | 4.8 | 250       | 128 | 82  | 220         | 140 | 45
102       | 0.42        | 95      | 1.5 | 6.8 | 5.1 | 280       | 118 | 75  | 190         | 120 | 50
103       | 0.72        | 145     | 3.8 | 8.5 | 4.5 | 220       | 145 | 92  | 250         | 165 | 38
...
```

**Format 3: MAPLE Embeddings (32-dimensional)**
```
File: MAPLE_embeddings.csv

person_id | maple_dim_1 | maple_dim_2 | ... | maple_dim_32
----------|-------------|-------------|-----|---------------
101       | 0.123       | -0.456      | ... | 0.789
102       | -0.234      | 0.567       | ... | -0.890
103       | 0.345       | -0.678      | ... | 0.901
...
```

**Format 4: CpGPT Embeddings (64-dimensional)**
```
File: CpGPT_embeddings.csv

person_id | cpgt_dim_1 | cpgt_dim_2 | ... | cpgt_dim_64
----------|------------|------------|-----|---------------
101       | 0.234      | -0.567     | ... | 0.123
102       | -0.345     | 0.678      | ... | -0.234
103       | 0.456      | -0.789     | ... | 0.345
...
```

---

### 3. Outcomes File (Your responsibility)

**Location:** `./outputs/outcomes/` (create from EHR data or clinical records)

**Format:** CSV with outcomes aligned to cohort

```
person_id | mortality | follow_up_days | outcome_date
----------|-----------|----------------|---------------
101       | 1         | 180            | 2021-01-15
102       | 0         | 365            | 2021-06-20
103       | 1         | 45             | 2020-07-25
...
```

---

### 4. Disease ICD Codes (Standard)

**Used for:** Filtering pretrain/finetune/eval cohorts

**Examples:**
```yaml
Diabetes:           "D74*"   # ICD-10 E10-E14
Heart Failure:      "I50*"   # ICD-10 I50.0-I50.9
COPD:               "J44*"   # ICD-10 J44.0-J44.9
Sepsis:             "A40*"   # ICD-10 A40.0-A40.9
Chronic Kidney Disease: "N18*"  # ICD-10 N18.0-N18.9
Liver Disease:      "K74*"   # ICD-10 K74.0-K74.9
```

---

## Current Codebase Status

### ✅ IMPLEMENTED

1. **Disease Filtering** (`disease_filtering.py`)
   - Reads MEDS parquet files
   - Filters by disease ICD code
   - Creates pretrain/finetune/eval splits
   - ✓ Handles multiple biological proxies
   - ✓ Aligns patient IDs across data sources

2. **Multi-Model Ablation** (`batch_ablation_runner.py`)
   - Trains N models from single pretrain
   - Evaluates comprehensively
   - Compares across dimensions

3. **Value Injection Methods**
   - Config supports: concat, FiLM, discrete, comb, comb_binning
   - Automatic method switching via YAML

### ⚠️ MISSING/NEEDS VERIFICATION

1. **Biological Proxy Integration**
   - Current assumption: values already in correct format
   - Need: Validation script to check CSV structure
   - Need: Alignment logic (patient_id + timestamp matching)

2. **End-to-End Data Pipeline**
   - Have pieces, but not fully integrated into single script
   - Need: Master orchestration script that runs all steps

3. **Synthetic Data Examples**
   - Needed for testing without real data

---

## What You Need to Provide

### Input Files (Minimal Example)

1. **MEDS Data** (from your ehr2meds)
   ```
   ./outputs/tokenized/
   ├── features_train.parquet
   ├── features_tuning.parquet
   └── features_held_out.parquet
   ```

2. **Biological Proxies**
   ```
   ./data/biological_proxies/
   ├── GrimAge_v2.csv
   ├── SystemsAge_components.csv
   ├── MAPLE_embeddings.csv
   ├── CpGPT_embeddings.csv
   └── MethylGPT_embeddings.csv
   ```

3. **Outcomes**
   ```
   ./outputs/outcomes/
   └── TEST_OUTCOME.csv
   ```

4. **Disease Code**
   ```yaml
   disease_code: "D74*"  # Diabetes example
   ```

---

## Does the Codebase Support This End-to-End?

### Current State: **PARTIALLY** ✓

**What works:**
- ✓ Disease filtering from MEDS data
- ✓ Cohort creation (pretrain/finetune/eval)
- ✓ Multi-model training
- ✓ Comprehensive evaluation

**What's manual:**
- ⚠️ Biological proxy CSV → model input tensor conversion
- ⚠️ Timestamp alignment between MEDS and bio proxies
- ⚠️ Missing validation/QA scripts

**Recommendation:**
Create a **master integration script** that:
1. Takes MEDS parquet + bio CSVs + outcomes CSV
2. Validates data formats
3. Runs disease filtering
4. Creates value tensors for finetune data
5. Launches multi-model ablation

Would you like me to create this integration script and synthetic examples?
