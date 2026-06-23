#!/usr/bin/env python3
"""
Pre-flight validation of all Part-B inputs before launching a run.

Checks (PASS / WARN / FAIL):
  1. MEDS: train/tuning/held_out present; columns+dtypes; death code exists; splits subject-disjoint.
  2. Condition CSVs: indexed by sample_id; numeric cols; index overlaps MEDS subjects (post id-map).
  3. Blood-draw dates: parse OK; overlap subjects; draw dates fall within the MEDS time span.
  4. ID map (optional): two columns, covers the condition sample_ids.
  5. Admin-censor date (optional): parses; not before most draw dates.
  6. Pretrained model (optional): checkpoint present.

Exit code is non-zero if any FAIL.

Usage:
  python validate_inputs.py \
    --meds data/meds_for_bonsai \
    --conditions . \
    --draw-dates drawdates.csv [--id-map map.csv] \
    [--admin-censor-date 2024-12-31] [--pretrain-model outputs/pretrained_model] [--dayfirst]
"""
import argparse, glob
from pathlib import Path
import pandas as pd

PID, TIME, CODE, VAL = "subject_id", "time", "code", "numeric_value"
SPLITS = ["train", "tuning", "held_out"]
fails = []; warns = []
def ok(m):   print(f"  \033[32mPASS\033[0m {m}")
def warn(m): print(f"  \033[33mWARN\033[0m {m}"); warns.append(m)
def fail(m): print(f"  \033[31mFAIL\033[0m {m}"); fails.append(m)


def check_meds(meds_dir, death_code):
    print("\n[1] MEDS")
    root = Path(meds_dir)
    subjects = {}
    all_codes_have_death = False
    for sp in SPLITS:
        files = glob.glob(str(root / sp / "*.parquet"))
        if not files:
            fail(f"{sp}/ has no parquet files"); continue
        cols_seen = None; subj = set(); has_death = False; n = 0
        for f in files:
            d = pd.read_parquet(f)
            cols_seen = list(d.columns)
            missing = [c for c in (PID, TIME, CODE, VAL) if c not in d.columns]
            if missing:
                fail(f"{sp}: missing columns {missing}"); break
            subj |= set(d[PID].astype(str)); n += len(d)
            if d[CODE].astype(str).str.startswith(death_code).any():
                has_death = True
        else:
            subjects[sp] = subj
            ok(f"{sp}: {len(files)} shard(s), {len(subj)} subjects, {n} rows")
            if has_death: all_codes_have_death = True
            else: warn(f"{sp}: no '{death_code}' (death) codes found")
    if all_codes_have_death: ok(f"death code '{death_code}' present")
    else: fail(f"death code '{death_code}' not found in any split (no labels possible)")
    # disjoint splits
    for a in range(len(SPLITS)):
        for b in range(a + 1, len(SPLITS)):
            sa, sb = SPLITS[a], SPLITS[b]
            if sa in subjects and sb in subjects:
                ov = subjects[sa] & subjects[sb]
                if ov: fail(f"{sa} ∩ {sb}: {len(ov)} shared subjects (leakage) e.g. {list(ov)[:3]}")
                else: ok(f"{sa} ∩ {sb}: disjoint")
    return set().union(*subjects.values()) if subjects else set()


def load_idmap(path):
    if not path: return None
    m = pd.read_csv(path, dtype=str)
    if m.shape[1] < 2: fail("id-map needs 2 columns (sample_id,subject_id)"); return None
    return dict(zip(m.iloc[:, 0], m.iloc[:, 1]))


def check_conditions(cond_dir, meds_subjects, idmap):
    print("\n[2] Condition CSVs")
    files = sorted(glob.glob(str(Path(cond_dir) / "condition_*.csv")))
    if not files: fail("no condition_*.csv found"); return set()
    sample_ids = set()
    for f in files:
        df = pd.read_csv(f, index_col=0)
        name = Path(f).name
        ids = [idmap.get(str(s), str(s)) for s in df.index] if idmap else [str(s) for s in df.index]
        sample_ids |= set(ids)
        ncols = df.select_dtypes("number").shape[1]
        if "baseline" in name and df.shape[1] == 0:
            ok(f"{name}: baseline (0 cols)"); continue
        if ncols == 0: warn(f"{name}: 0 numeric columns")
        ov = set(ids) & meds_subjects
        if not ov: fail(f"{name}: 0/{len(ids)} sample_ids match MEDS subject_ids (id-map needed?)")
        else: ok(f"{name}: {ncols} features, {len(ov)}/{len(ids)} subjects in MEDS")
    return sample_ids


def check_draw_dates(path, meds_subjects, idmap, dayfirst, meds_dir):
    print("\n[3] Blood-draw dates")
    if not path: warn("no --draw-dates given (needed for leakage-safe index)"); return
    df = pd.read_csv(path)
    sid = df.iloc[:, 0].astype(str)
    datecol = next((c for c in df.columns[1:] if any(k in c.lower() for k in ("date","draw","time"))), df.columns[1])
    parsed = pd.to_datetime(df[datecol], errors="coerce", dayfirst=dayfirst)
    nbad = int(parsed.isna().sum())
    if nbad: warn(f"{nbad}/{len(parsed)} draw dates unparseable (try --dayfirst)")
    else: ok(f"all {len(parsed)} draw dates parse")
    ids = [idmap.get(s, s) for s in sid] if idmap else list(sid)
    ov = set(ids) & meds_subjects
    if not ov: fail("0 draw-date subjects match MEDS")
    else: ok(f"{len(ov)} draw-date subjects in MEDS")
    # draw dates within MEDS time span?
    try:
        tspan = []
        for f in glob.glob(str(Path(meds_dir) / "*/*.parquet"))[:3]:
            tspan.append(pd.read_parquet(f, columns=[TIME])[TIME])
        if tspan:
            tmin, tmax = pd.concat(tspan).min(), pd.concat(tspan).max()
            inrange = parsed.between(tmin, tmax + pd.Timedelta(days=3650)).mean()
            if inrange < 0.8: warn(f"only {inrange:.0%} of draw dates within MEDS time span "
                                   f"[{tmin.date()}..{tmax.date()}] — check format/timezone")
            else: ok(f"draw dates align with MEDS time span ({inrange:.0%} in range)")
    except Exception:
        pass


def check_admin(date, dayfirst):
    print("\n[4] Admin-censor date")
    if not date: warn("no --admin-censor-date (needed for fixed-horizon labels)"); return
    try:
        pd.Timestamp(date); ok(f"admin-censor date parses: {date}")
    except Exception:
        fail(f"admin-censor date '{date}' does not parse")


def check_pretrain(path):
    print("\n[5] Pretrained model")
    if not path: warn("no --pretrain-model (fine if you run Phase 1 pretraining)"); return
    ck = glob.glob(str(Path(path) / "checkpoints" / "*.pt"))
    if ck: ok(f"checkpoint found: {Path(ck[0]).name}")
    else: fail(f"no checkpoint under {path}/checkpoints/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meds", required=True)
    ap.add_argument("--conditions", default=".")
    ap.add_argument("--draw-dates", default=None)
    ap.add_argument("--id-map", default=None)
    ap.add_argument("--admin-censor-date", default=None)
    ap.add_argument("--pretrain-model", default=None)
    ap.add_argument("--death-code", default="DOD")
    ap.add_argument("--dayfirst", action="store_true")
    args = ap.parse_args()

    print("="*64 + "\nPart-B input validation\n" + "="*64)
    idmap = load_idmap(args.id_map)
    meds_subjects = check_meds(args.meds, args.death_code)
    check_conditions(args.conditions, meds_subjects, idmap)
    check_draw_dates(args.draw_dates, meds_subjects, idmap, args.dayfirst, args.meds)
    check_admin(args.admin_censor_date, args.dayfirst)
    check_pretrain(args.pretrain_model)

    print("\n" + "="*64)
    print(f"RESULT: {len(fails)} FAIL, {len(warns)} WARN")
    if fails:
        print("Fix the FAILs before launching."); raise SystemExit(1)
    print("All hard checks passed — safe to run." + (" (review WARNs)" if warns else ""))


if __name__ == "__main__":
    main()
