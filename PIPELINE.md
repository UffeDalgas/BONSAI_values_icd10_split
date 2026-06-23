# BONSAI value-injection pipeline

Mortality finetuning of a pretrained EHR model with **biological value injection** from epimap
(`condition_*.csv`). Pretraining uses EHR only; biomarker values are injected **only at
finetuning** (the model's concept-embedding table is grown for the new `BIO/*` concepts);
evaluation runs on EHR-only held-out samples.

## Environment
Use a Python ≤3.11 env with `corebehrt`, `torch` (2.2–2.6) and `transformers` that has ModernBert
(e.g. the `bonsai_dryrun` conda env). Python 3.12 + torch<2.4 fails (`Dynamo not supported`).

## Key design rules (leakage-safe)
- **Index = blood-draw date** (the prediction anchor). EHR censored at the draw; `BIO/*` injected
  at the draw; mortality counted strictly after the draw.
- Biomarker **draw date ≤ index** always (enforced by anchoring `BIO/*` at the draw).
- Mortality (`DOD`) must come from a source covering **post-draw** follow-up (e.g. a registry).
- `train` / `tuning` / `held_out` must be **subject-disjoint** (asserted by `run_condition.sh`).
- The **eval data must be tokenized with the finetune vocabulary** (the one containing `BIO/*`).

## Inputs you provide
- MEDS at `data/meds_for_bonsai/{train,tuning,held_out}/*.parquet`
  (`subject_id, time, code, numeric_value`; mortality as a `DOD` code).
- A pretrained model (EHR-only) → set `paths.pretrain_model` in each `finetune_<cond>.yaml`.
- epimap condition CSVs `condition_<n>_<name>.csv` (index = sample_id).
- `metadata.csv` with a blood-draw-date column.
- If `sample_id != subject_id`: a 2-col `sample_id,subject_id` map (`--id-map`).

## Steps

### 0. Per-subject index from the blood-draw date
```bash
python build_index_from_drawdate.py --metadata metadata.csv \
  --id-col sample_id --date-col blood_draw_date --out ./outputs/exposures_drawdate.csv [--id-map map.csv]
```
Then in `select_cohort_shared.yaml` / `select_cohort_held_out_shared.yaml`:
```yaml
index_date: {mode: relative, relative: {n_hours_from_exposure: 0}}
paths: {exposure: ./outputs/exposures_drawdate.csv}
```
Set `n_hours_censoring: 0` in the prepare/finetune configs so the draw-anchored `BIO/*` survives censoring.

### 1. One-time, shared across all arms
```bash
python -m corebehrt.main.create_data     --config corebehrt/configs/create_data_baseline.yaml
python -m corebehrt.main.create_outcomes --config corebehrt/configs/create_outcomes_mortality.yaml   # DOD -> MORTALITY.csv
python -m corebehrt.main.select_cohort   --config corebehrt/configs/select_cohort_shared.yaml          # CV folds
python -m corebehrt.main.select_cohort   --config corebehrt/configs/select_cohort_held_out_shared.yaml # held-out
```

### 2. Per arm — one command
```bash
bash run_condition.sh baseline
bash run_condition.sh grimage_v2 --id-map map.csv --sample-date-csv metadata_drawdates.csv
```
`run_condition.sh` does, per arm: split-overlap guard → inject (`BIO/*` at draw date) → `create_data`
→ **BIO/\* vocab check** → prepare_finetune → finetune (embedding grows for `BIO/*`) →
prepare_held_out → evaluate_finetune. Metrics land in
`outputs/testing/held_out/predictions_<cond>/metrics.csv` (ROC-AUC, PR-AUC, DeLong).

### 3. EHR recency gap (optional)
```bash
python inject_conditions_into_meds.py --meds-in data/meds_for_bonsai \
  --condition condition_2_grimage_v2.csv --meds-out data/meds_grimage_v2 \
  --sample-date-csv metadata_drawdates.csv \
  --add-gap-feature \                         # injects BIO/ehr_gap_days
  --max-gap-days 365 --exclude-out outputs/exclude_gap.csv   # drop stale-EHR subjects
```
Point `select_cohort` `paths.exclude_pids` at `exclude_gap.csv` to drop over-gap subjects.

## How value injection works under the hood
- The injector adds rows `subject_id, time=draw, code=BIO/<col>, numeric_value` to a MEDS copy.
- `create_data` tokenizes `BIO/*` into the vocabulary; the model's `value_embedding_mode: concat`
  folds each numeric value into its concept embedding.
- At finetune, `corebehrt/modules/setup/initializer.py::_grow_concept_embeddings` extends the
  pretrained embedding table to cover the new `BIO/*` tokens (pretrained rows kept, new rows
  learned). `loader.py` matches the table to the checkpoint when reloading.
- EHR-only eval samples simply never contain `BIO/*` tokens — the value path is masked, so the
  model degrades gracefully to EHR-only for them.

## Conditions
`baseline, metadata, grimage_v2, systemsage, deep_embeddings, maple_predictions, dmi, episcores`
— each with `create_data_<cond>.yaml`, `prepare_finetune_<cond>.yaml`, `finetune_<cond>.yaml`,
`prepare_held_out_<cond>.yaml`, `evaluate_finetune_<cond>.yaml`.

## Compare arms
```bash
for c in baseline metadata grimage_v2 systemsage deep_embeddings maple_predictions dmi episcores; do
  echo -n "$c: "; tail -1 outputs/testing/held_out/predictions_$c/metrics.csv 2>/dev/null || echo "(not run)"
done
```

## Fixed-horizon mortality (1–5 year)
`n_hours_end_follow_up: null` = "ever died" → label depends on follow-up length (bias). Use a fixed
horizon and drop subjects with insufficient follow-up:

```bash
python build_mortality_horizon.py \
  --outcomes ./outputs/outcomes/MORTALITY.csv \
  --index    ./outputs/exposures_drawdate.csv \
  --admin-censor-date 2024-12-31 \        # registry/study end (preferred); or --meds for per-subject last event
  --horizons 1,2,3,4,5 --out-dir ./outputs/horizons
```
It prints the label balance per horizon and writes `exclude_<H>y.csv`. For horizon H:
- `select_cohort` (shared + held_out): `paths.exclude_pids: ./outputs/horizons/exclude_<H>y.csv`
- prepare configs: `outcome.n_hours_end_follow_up:` the printed hours
  (1y=8766, 2y=17532, 3y=26298, 4y=35064, 5y=43830)
- use horizon-specific output dirs so runs don't clobber.

Horizon is an experiment axis alongside condition (so you get a model per condition × horizon).
CoxPH loss + time-to-event evaluation will replace the fixed-horizon binary setup later.

## Leakage diagnostics
After `prepare` (finetune or held-out), before trusting any AUC:
```bash
python check_leakage.py --prepared ./outputs/.../processed_data_<cond> --vocab ./outputs/tokenized_<cond>
```
Reports: (1) % sequences containing a death token in input (must be ~0), (2) sequence-length gap by
class, (3) **the input tokens that most separate deaths vs survivors** (transparency). Pre-index
terminal-care/ICU codes showing up in (3) is *expected legitimate signal* — only a token flagged
`death-like?` there is a leak.

`archive/` holds earlier-iteration helpers (superseded by this flow); kept for reference.
