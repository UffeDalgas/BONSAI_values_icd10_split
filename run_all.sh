#!/bin/bash
###############################################################################
# run_all.sh — full pipeline, ALL conditions, sweeping 1..5y horizons cleanly.
#
# Tokenization + pretraining are horizon-INDEPENDENT → done once.
# Horizon-DEPENDENT steps (cohorts/prepare/finetune/eval) go under
# outputs/sweep/<H>y/ so horizons never clobber each other.
#
# Edit CONFIG, then:  bash run_all.sh
###############################################################################
set -euo pipefail

# ============================== CONFIG =======================================
PYTHON=/opt/anaconda3/envs/bonsai_dryrun/bin/python
MEDS=./data/meds_for_bonsai
METADATA=./metadata.csv
ID_MAP=                                   # sample_id,subject_id (empty if identical)
DAYFIRST=--dayfirst                       # metadata dates DD-MM-YYYY (else "")
HORIZONS="1 2 3 4 5"                      # years to sweep
TOLERANCE_DAYS=30
PRETRAINED_MODEL=                         # blank => pretrain here
CONDITIONS="baseline metadata grimage_v2 systemsage deep_embeddings maple_predictions dmi episcores"
# =============================================================================
C=corebehrt/configs
idmap_arg=""; [ -n "$ID_MAP" ] && idmap_arg="--id-map $ID_MAP"
HSWEEP=$(echo "$HORIZONS" | tr ' ' ',')
INDEX=./outputs/labels/index.csv

echo "==== sweep horizons: $HORIZONS | conditions: $CONDITIONS ===="

# ---- Phase 1 (once): labels, index(opdate2), per-horizon exclusions ----------
echo "== Phase 1: outcomes / index / horizons =="
$PYTHON build_outcomes_from_survival.py --metadata "$METADATA" --meds "$MEDS" \
   --out-dir ./outputs/labels --tolerance-days "$TOLERANCE_DAYS" $DAYFIRST $idmap_arg
$PYTHON build_index_from_drawdate.py --metadata "$METADATA" --id-col sample_id \
   --date-col opdate2 --out "$INDEX" $DAYFIRST $idmap_arg
$PYTHON build_mortality_horizon.py --outcomes ./outputs/labels/MORTALITY.csv \
   --index "$INDEX" --censor-csv ./outputs/labels/censor.csv \
   --horizons "$HSWEEP" --out-dir ./outputs/labels/horizons

# ---- Phase 2 (once): baseline tokenize + pretrain ---------------------------
if [ -z "$PRETRAINED_MODEL" ]; then
  echo "== Phase 2: tokenize baseline + pretrain =="
  $PYTHON -m corebehrt.main.create_data           --config $C/create_data_baseline.yaml
  $PYTHON -m corebehrt.main.prepare_training_data --config $C/prepare_pretrain_main.yaml
  $PYTHON -m corebehrt.main.pretrain              --config $C/pretrain_main.yaml
  PRETRAINED_MODEL=./outputs/pretrained_model
fi

# ---- Phase 3 (once): inject + tokenize each condition (horizon-independent) --
echo "== Phase 3: inject + tokenize conditions =="
declare -A CSV=([baseline]=condition_0_baseline.csv [metadata]=condition_1_metadata.csv \
 [grimage_v2]=condition_2_grimage_v2.csv [systemsage]=condition_3_systemsage.csv \
 [deep_embeddings]=condition_4_deep_embeddings.csv [maple_predictions]=condition_5_maple_predictions.csv \
 [dmi]=condition_6_dmi.csv [episcores]=condition_7_episcores.csv)
for cond in $CONDITIONS; do
  if [ "$cond" = baseline ]; then continue; fi          # baseline already tokenized in Phase 2
  if [ -f "outputs/tokenized_${cond}/vocabulary.pt" ]; then echo "  $cond: tokenized ✓ (skip)"; continue; fi
  $PYTHON inject_conditions_into_meds.py --meds-in "$MEDS" --condition "./${CSV[$cond]}" \
     --meds-out "./data/meds_${cond}" --sample-date-csv "$INDEX" --add-gap-feature $DAYFIRST $idmap_arg
  $PYTHON -m corebehrt.main.create_data --config $C/create_data_${cond}.yaml
done

# ---- Phase 4: per horizon × condition (isolated output dirs) -----------------
gen_cfg() {  # $1=horizon  -> writes patched configs into $C/h$1y/
  $PYTHON - "$1" "$PRETRAINED_MODEL" <<'PY'
import sys, glob, re, pathlib
H=sys.argv[1]; pretrain=sys.argv[2]; hours=int(round(float(H)*365.25*24))
root=f"./outputs/sweep/{H}y"; cfgdir=pathlib.Path(f"corebehrt/configs/h{H}y"); cfgdir.mkdir(parents=True,exist_ok=True)
excl=f"./outputs/labels/horizons/exclude_{int(float(H))}y.csv"
def cohort(initial, codir):
    return (f"logging: {{level: INFO, path: ./outputs/logs}}\npaths:\n"
        f"  features: ./outputs/features_baseline\n  tokenized: ./outputs/tokenized_baseline\n"
        f"  initial_pids: {initial}\n  outcomes: ./outputs/labels/\n  outcome: MORTALITY.csv\n"
        f"  exposure: {sys.argv[0] and './outputs/labels/index.csv'}\n  exclude_pids: {excl}\n  cohort: {root}/{codir}/\n"
        f"selection: {{exclude_prior_outcomes: false, exposed_only: false, age: {{min_years: 18, max_years: 120}}}}\n"
        f"index_date: {{mode: relative, relative: {{n_hours_from_exposure: 0}}}}\ncv_folds: 2\nval_ratio: 0.1\n")
(cfgdir/"select_cohort_shared.yaml").write_text(cohort("pids_tuning.pt","cohort/finetune")+"test_ratio: 0.0\n")
(cfgdir/"select_cohort_held_out_shared.yaml").write_text(cohort("pids_held_out.pt","cohort/held_out")+"test_ratio: 1.0\n")
conds=[p.name.split("prepare_finetune_")[1][:-5] for p in pathlib.Path("corebehrt/configs").glob("prepare_finetune_*.yaml") if "smoke" not in p.name]
for cond in conds:
    (cfgdir/f"prepare_finetune_{cond}.yaml").write_text(
      f"logging: {{level: INFO, path: ./outputs/logs}}\npaths:\n  features: ./outputs/features_{cond}\n"
      f"  tokenized: ./outputs/tokenized_{cond}\n  cohort: {root}/cohort/finetune/\n  outcomes: ./outputs/labels\n"
      f"  outcome: MORTALITY.csv\n  prepared_data: {root}/processed_{cond}/\n"
      f"data: {{type: finetune, truncation_len: 30, min_len: 2}}\n"
      f"outcome: {{n_hours_censoring: 0, n_hours_start_follow_up: 1, n_hours_end_follow_up: {hours}}}\n"
      f'concept_pattern_hours_delay: {{"^D(?!OD)": 72}}\n')
    (cfgdir/f"finetune_{cond}.yaml").write_text(
      f"logging: {{level: INFO, path: ./outputs/logs}}\npaths:\n  prepared_data: {root}/processed_{cond}/\n"
      f"  pretrain_model: {pretrain}\n  tokenized: ./outputs/tokenized_{cond}\n  model: {root}/models/{cond}\n"
      f'model: {{cls: default, value_embedding_mode: "concat"}}\n'
      f"trainer_args: {{batch_size: 8, val_batch_size: 8, effective_batch_size: 8, epochs: 2, info: true, "
      f"gradient_clip: {{clip_value: 1.0}}, shuffle: true, checkpoint_frequency: 1, early_stopping: 2, "
      f"stopping_criterion: roc_auc, n_layers_to_freeze: 1}}\noptimizer: {{lr: 5e-4, eps: 1e-6}}\n"
      f"scheduler: {{_target_: transformers.get_linear_schedule_with_warmup, num_warmup_steps: 5, num_training_steps: 20}}\n"
      f"metrics: {{roc_auc: {{_target_: corebehrt.modules.monitoring.metrics.ROC_AUC}}, "
      f"pr_auc: {{_target_: corebehrt.modules.monitoring.metrics.PR_AUC}}}}\nevaluate: false\n")
    (cfgdir/f"prepare_held_out_{cond}.yaml").write_text(
      f"logging: {{level: INFO, path: ./outputs/logs}}\npaths:\n  features: ./outputs/features_{cond}\n"
      f"  tokenized: ./outputs/tokenized_{cond}\n  cohort: {root}/cohort/held_out/\n  outcomes: ./outputs/labels\n"
      f"  outcome: MORTALITY.csv\n  prepared_data: {root}/test_{cond}/\n"
      f"data: {{type: test, test_modes: [held_out], truncation_len: 30, min_len: 2}}\n"
      f"outcome: {{n_hours_censoring: 0, n_hours_start_follow_up: 1, n_hours_end_follow_up: {hours}}}\n"
      f'concept_pattern_hours_delay: {{"^D(?!OD)": 72}}\n')
    (cfgdir/f"evaluate_finetune_{cond}.yaml").write_text(
      f"logging: {{level: INFO, path: ./outputs/logs}}\npaths:\n  model: {root}/models/{cond}\n"
      f"  folds_dir: {root}/processed_{cond}\n  test_data_dir: {root}/test_{cond}\n  run_name: eval_{cond}\n"
      f"  predictions: {root}/pred_{cond}\ntest_batch_size: 128\nreturn_embeddings: false\n"
      f"metrics: {{roc_auc: {{_target_: sklearn.metrics.roc_auc_score}}, "
      f"pr_auc: {{_target_: sklearn.metrics.average_precision_score}}, "
      f"delong_roc_auc: {{_target_: corebehrt.functional.evaluation.delong.delong_roc_auc}}}}\n")
print(f"  generated configs for horizon {H}y -> {cfgdir}")
PY
}

for H in $HORIZONS; do
  echo "== Phase 4: horizon ${H}y =="
  gen_cfg "$H"; HC="$C/h${H}y"
  $PYTHON -m corebehrt.main.select_cohort --config $HC/select_cohort_shared.yaml
  $PYTHON -m corebehrt.main.select_cohort --config $HC/select_cohort_held_out_shared.yaml
  for cond in $CONDITIONS; do
    echo "  ---- ${H}y / $cond ----"
    $PYTHON -m corebehrt.main.prepare_training_data --config $HC/prepare_finetune_${cond}.yaml
    $PYTHON -m corebehrt.main.finetune_cv           --config $HC/finetune_${cond}.yaml
    $PYTHON -m corebehrt.main.prepare_training_data --config $HC/prepare_held_out_${cond}.yaml
    $PYTHON -m corebehrt.main.evaluate_finetune     --config $HC/evaluate_finetune_${cond}.yaml
  done
done

# ---- Phase 5: comparison table ---------------------------------------------
echo "== Phase 5: held-out comparison =="
for H in $HORIZONS; do
  echo "--- ${H}y mortality ---"
  for cond in $CONDITIONS; do
    m=outputs/sweep/${H}y/pred_${cond}/metrics.csv
    echo -n "  $cond: "; [ -f "$m" ] && tail -1 "$m" || echo "(no metrics)"
  done
done
