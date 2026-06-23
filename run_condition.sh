#!/bin/bash
###############################################################################
# run_condition.sh <name> [--id-map map.csv]
#
# Chains the full per-condition finetuning + evaluation flow for one value-
# injection arm, reusing the SHARED cohort/folds + MORTALITY outcomes.
#
#   <name> in: baseline metadata grimage_v2 systemsage deep_embeddings
#              maple_predictions dmi episcores
#
# Prereqs (run ONCE before any condition):
#   python -m corebehrt.main.create_data     --config corebehrt/configs/create_data_baseline.yaml
#   python -m corebehrt.main.create_outcomes --config corebehrt/configs/create_outcomes_mortality.yaml
#   python -m corebehrt.main.select_cohort   --config corebehrt/configs/select_cohort_shared.yaml
#   python -m corebehrt.main.select_cohort   --config corebehrt/configs/select_cohort_held_out_shared.yaml
#
# Example:
#   bash run_condition.sh baseline
#   bash run_condition.sh grimage_v2 --id-map sampleid_to_subjectid.csv
###############################################################################
set -euo pipefail

NAME="${1:?usage: run_condition.sh <name> [--id-map map.csv] [--sample-date-csv dates.csv] [--dayfirst] [--skip-validate]}"
shift || true
INJECT_ARGS=""; ID_MAP=""; DRAW_DATES=""; DAYFIRST=""; SKIP_VALIDATE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --id-map)          INJECT_ARGS="$INJECT_ARGS --id-map $2"; ID_MAP="$2"; shift 2;;
    --sample-date-csv) INJECT_ARGS="$INJECT_ARGS --sample-date-csv $2"; DRAW_DATES="$2"; shift 2;;
    --dayfirst)        INJECT_ARGS="$INJECT_ARGS --dayfirst"; DAYFIRST="--dayfirst"; shift;;
    --skip-validate)   SKIP_VALIDATE=1; shift;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

# ---- pre-flight input validation (MEDS columns, DOD, split-disjointness, condition/id/date overlap) ----
if [ -z "$SKIP_VALIDATE" ]; then
  echo ">>> validating inputs"
  VAL_ARGS="--meds data/meds_for_bonsai --conditions ."
  [ -n "$ID_MAP" ]     && VAL_ARGS="$VAL_ARGS --id-map $ID_MAP"
  [ -n "$DRAW_DATES" ] && VAL_ARGS="$VAL_ARGS --draw-dates $DRAW_DATES"
  [ -n "$DAYFIRST" ]   && VAL_ARGS="$VAL_ARGS $DAYFIRST"
  python validate_inputs.py $VAL_ARGS || { echo "input validation failed — aborting (use --skip-validate to override)"; exit 1; }
fi

# name -> condition CSV
declare -A CSV=(
  [baseline]=condition_0_baseline.csv
  [metadata]=condition_1_metadata.csv
  [grimage_v2]=condition_2_grimage_v2.csv
  [systemsage]=condition_3_systemsage.csv
  [deep_embeddings]=condition_4_deep_embeddings.csv
  [maple_predictions]=condition_5_maple_predictions.csv
  [dmi]=condition_6_dmi.csv
  [episcores]=condition_7_episcores.csv
)
[ -n "${CSV[$NAME]:-}" ] || { echo "unknown condition: $NAME"; exit 1; }

C="corebehrt/configs"
run(){ echo -e "\n>>> $*"; "$@"; }

echo "================ CONDITION: $NAME ================"

# 1. Inject (baseline = EHR-only, no injection; tokenizes the original MEDS)
if [ "$NAME" != "baseline" ]; then
  run python inject_conditions_into_meds.py \
      --meds-in ./data/meds_for_bonsai \
      --condition "./${CSV[$NAME]}" \
      --meds-out "./data/meds_${NAME}" $INJECT_ARGS
fi

# 2. Tokenize this arm (baseline already tokenized in prereqs; skip to save time)
if [ "$NAME" != "baseline" ]; then
  run python -m corebehrt.main.create_data --config "$C/create_data_${NAME}.yaml"
  # sanity: confirm BIO/* tokens landed
  python - "$NAME" <<'PYCHK'
import sys, torch
n=sys.argv[1]
v=torch.load(f"outputs/tokenized_{n}/vocabulary.pt", weights_only=False)
bio=[k for k in v if str(k).startswith("BIO/")]
print(f"  [check] BIO/* concepts in vocab: {len(bio)} -> {bio[:6]}")
assert bio, "NO BIO/* concepts — injection/id-map failed; this arm == baseline. Fix --id-map."
PYCHK
fi

# 3. Finetune
run python -m corebehrt.main.prepare_training_data --config "$C/prepare_finetune_${NAME}.yaml"
run python -m corebehrt.main.finetune_cv           --config "$C/finetune_${NAME}.yaml"

# 4. Held-out evaluation
run python -m corebehrt.main.prepare_training_data --config "$C/prepare_held_out_${NAME}.yaml"
run python -m corebehrt.main.evaluate_finetune     --config "$C/evaluate_finetune_${NAME}.yaml"

echo -e "\n================ DONE: $NAME ================"
echo "metrics:     outputs/testing/held_out/predictions_${NAME}/metrics.csv"
echo "predictions: outputs/testing/held_out/predictions_${NAME}/predictions.csv"
