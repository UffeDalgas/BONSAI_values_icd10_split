# Disease-Specific Filtering for MEDS Data

## Quick Start

Your disease-specific filtering module integrates **MEDS format data** (from ehr2meds) with disease-stratified cohort creation. This enables you to:

1. **Exclude a disease from pretraining** - Train general EHR model on all non-target-disease patients
2. **Select disease-specific subset with biological proxies** - Finetune on patients with epigenetic measurements
3. **Hold-out disease patients without proxies** - Evaluate on unseen samples (true test set)
4. **Measure biological signal** - Compare ROC-AUC with/without features

---

## How It Works

### Disease Code Filtering

Disease codes come from ehr2meds MEDS data (concept_id column):

```
ICD-10 → ehr2meds prefix → MEDS concept_id
----------------------------------------
E10-E14 (Diabetes)    → D    → D74*
I50 (Heart Failure)   → I    → I50*
J44 (COPD)            → J    → J44*
K74 (Liver disease)   → K    → K74*
```

### The Three Cohorts

```
STEP 1: Load MEDS data (all patients)
        ├─ Disease patients: those with ANY event matching "D74*"
        └─ Non-disease patients: all others

STEP 2: Split disease patients into two groups
        ├─ Finetune: patients WITH biological proxies (e.g., GrimAge2)
        └─ Eval: patients WITHOUT proxies (held-out test)

STEP 3: Create pretrain cohort
        └─ All non-disease patients + eval cohort (no feature leakage)

RESULT:
        Pretrain: ~8,500 patients (all except diabetes)
        Finetune: ~150 diabetes patients (with epigenetic data)
        Eval:     ~75 diabetes patients (without epigenetic data)
```

---

## Usage Examples

### 1. Simple Usage with Disease Filter

```python
from corebehrt.functional.data.disease_filtering import DiseaseFilter

# Create filter for diabetes patients
filter = DiseaseFilter("D74*")

# Load your MEDS data
meds_data = pd.read_parquet("./outputs/tokenized/part-*.parquet")

# Identify all diabetes patients
diabetes_patients = filter.identify_patients_with_disease(meds_data)
print(f"Found {len(diabetes_patients)} diabetes patients")

# Split into cohorts
pretrain, finetune, eval = filter.split_by_disease(
    meds_data,
    finetune_disease_patients={101, 102, 103, ...}  # Patients with bio proxies
)

print(f"Pretrain: {len(pretrain)} events from {pretrain['person_id'].nunique()} patients")
print(f"Finetune: {len(finetune)} events from {finetune['person_id'].nunique()} patients")
print(f"Eval:     {len(eval)} events from {eval['person_id'].nunique()} patients")
```

### 2. Full Pipeline with Disease-Aware Split

```python
from corebehrt.main.ablation_pipeline import MEDSDiseasAblationStudy

# Create ablation study
study = MEDSDiseasAblationStudy(
    meds_data_path="./outputs/tokenized/",
    disease_code="D74*",  # Diabetes
    finetune_disease_patient_ids={101, 102, 103, ...},  # Patients with measurements
    finetune_subset_size=150,
    eval_subset_size=75,
    output_dir="./outputs/ablation_cohorts/diabetes"
)

# Load and stratify
study.load_meds_data()
pretrain, finetune, eval = study.create_disease_stratified_cohorts()

# Get statistics
stats = study.get_cohort_statistics()
```

### 3. Config-Based Approach

```python
import yaml

# Load config
with open("corebehrt/configs/ablation_diabetes_meds.yaml") as f:
    config = yaml.safe_load(f)

# Extract disease code and paths
disease_code = config["meds"]["disease_code_pattern"]
meds_path = config["meds"]["data_path"]

# Create study from config
study = MEDSDiseasAblationStudy(
    meds_data_path=meds_path,
    disease_code=disease_code,
    finetune_subset_size=config["ablation_study"]["cohort_split"]["finetune_with_proxies"],
    eval_subset_size=config["ablation_study"]["cohort_split"]["eval_without_proxies"],
    output_dir=config["output"]["base_dir"]
)

study.create_disease_stratified_cohorts()
```

---

## Common Disease Code Patterns

```yaml
# Danish ICD-10 in ehr2meds format
Diabetes:
  code: "D74*"  # E10-E14
  prevalence: "~10% of population"

Heart Failure:
  code: "I50*"  # I50.0-I50.9
  prevalence: "~5% of population"

COPD:
  code: "J44*"  # J44.0-J44.9
  prevalence: "~4% of population"

Liver Disease:
  code: "K74*"  # K74.0-K74.9
  prevalence: "~2% of population"

Hypertension:
  code: "I10*"  # I10.0-I10.9
  prevalence: "~30% of population"

Sepsis:
  code: "A40*"  # A40.0-A40.9
  prevalence: "~0.5% of population"
```

---

## Integration with Your Pipeline

### STEP 1: Load and Filter MEDS Data

```python
# From dryrun_pipeline.py, after tokenization
from corebehrt.functional.data.disease_filtering import DiseaseFilter

meds_data = load_meds_data("./outputs/tokenized/")
filter = DiseaseFilter("D74*")

pretrain, finetune, eval = filter.split_by_disease(
    meds_data,
    finetune_disease_patients={...}  # Your diabetes patients with proxies
)
```

### STEP 2: Select Subsets for Finetune/Eval

```python
# From create_outcomes.py
# Filter your outcomes to match the disease-stratified cohorts

finetune_outcomes = outcomes[outcomes['person_id'].isin(finetune['person_id'].unique())]
eval_outcomes = outcomes[outcomes['person_id'].isin(eval['person_id'].unique())]
```

### STEP 3: Ablate Biological Features

```python
# From finetune configs
feature_sets = [
    [],                                      # Baseline: EHR only
    ["grim_age2"],                          # Individual features
    ["systems_age"],
    ["maple"],
    ["methylgpt"],
    ["grim_age2", "systems_age"],           # Combinations
    ["grim_age2", "systems_age", "maple", "methylgpt"]  # All
]

for features in feature_sets:
    finetune_with_features = inject_features(finetune, features)
    model = train_finetune(finetune_with_features)
    metrics[str(features)] = evaluate(model, eval)
```

### STEP 4: Compare Results

```python
# Calculate per-feature contributions
baseline_auc = metrics["[]"]["roc_auc"]
enhanced_auc = metrics["['grim_age2', 'systems_age', 'maple', 'methylgpt']"]["roc_auc"]
delta = enhanced_auc - baseline_auc

print(f"Biological signal (all features): +{delta:.3f} ROC-AUC")

# Per-feature breakdown
for features in [["grim_age2"], ["systems_age"], ["maple"], ["methylgpt"]]:
    feature_auc = metrics[str(features)]["roc_auc"]
    feature_delta = feature_auc - baseline_auc
    print(f"{features[0]}: +{feature_delta:.3f} ROC-AUC")
```

---

## Disease-Specific Variations

### Heart Failure Study

```python
study = MEDSDiseasAblationStudy(
    meds_data_path="./outputs/tokenized/",
    disease_code="I50*",  # Heart failure
    finetune_disease_patient_ids=hf_patients_with_proxies,
    output_dir="./outputs/ablation_cohorts/heart_failure"
)
```

### COPD Study

```python
study = MEDSDiseasAblationStudy(
    meds_data_path="./outputs/tokenized/",
    disease_code="J44*",  # COPD
    finetune_disease_patient_ids=copd_patients_with_proxies,
    output_dir="./outputs/ablation_cohorts/copd"
)
```

### Sepsis Study

```python
study = MEDSDiseasAblationStudy(
    meds_data_path="./outputs/tokenized/",
    disease_code="A40*",  # Sepsis
    finetune_disease_patient_ids=sepsis_patients_with_proxies,
    output_dir="./outputs/ablation_cohorts/sepsis"
)
```

---

## FAQ

### Q: How do I specify which patients have biological proxies?

**A:** You provide a `Set[int]` of patient IDs:
```python
finetune_disease_patients = {101, 102, 103, 105, 108, ...}  # Your patient IDs
study = MEDSDiseasAblationStudy(..., finetune_disease_patient_ids=finetune_disease_patients)
```

### Q: Can I exclude multiple diseases from pretraining?

**A:** Currently, the filter handles one disease code pattern at a time. For multiple exclusions:
```python
# Exclude diabetes AND heart failure
filter_diabetes = DiseaseFilter("D74*")
filter_hf = DiseaseFilter("I50*")

# Filter sequentially
pretrain = meds_data[
    ~meds_data["person_id"].isin(filter_diabetes.identify_patients_with_disease(meds_data)) &
    ~meds_data["person_id"].isin(filter_hf.identify_patients_with_disease(meds_data))
]
```

### Q: What if I have missing disease codes for some patients?

**A:** The filter only identifies patients with at least one event matching the disease pattern. Patients without any matching event are implicitly included in the pretrain cohort. This is intentional - you're pretraining on patients where the disease is not coded (either absent or not recorded).

### Q: Can I stratify by multiple disease codes (e.g., all cardiac diseases)?

**A:** Extend the DiseaseFilter class:
```python
class MultiDiseaseFilter:
    def __init__(self, disease_code_patterns: List[str]):
        self.filters = [DiseaseFilter(p) for p in disease_code_patterns]
    
    def identify_patients(self, meds_data):
        patients = set()
        for f in self.filters:
            patients.update(f.identify_patients_with_disease(meds_data))
        return patients
```

---

## Next Steps

1. **Identify your disease cohorts**: Determine which disease codes you want to study
2. **Prepare biological proxy data**: Collect patient IDs with GrimAge2, SystemsAge, etc.
3. **Create disease-specific configs**: Copy `ablation_diabetes_meds.yaml` for each disease
4. **Run ablation studies**: Execute per-disease ablations to measure biological signal
5. **Compare across diseases**: Which proxies work best in which diseases?

---

## Files Reference

- **`corebehrt/functional/data/disease_filtering.py`** - DiseaseFilter + DiseaseAwareDataPipeline classes
- **`corebehrt/main/ablation_pipeline.py`** - MEDSDiseasAblationStudy class
- **`corebehrt/configs/ablation_diabetes_meds.yaml`** - Example configuration for diabetes study
- **`ABLATION_FRAMEWORK.md`** - Full theoretical background and experimental design
