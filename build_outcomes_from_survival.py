#!/usr/bin/env python3
"""
Build the mortality outcome + per-subject censor from the gold-standard survival data,
coalescing with EHR DOD, and run consistency QC.

Index = surgery date (opdate2) = blood-draw = survival-clock origin.

Label source priority:
  - methylation cohort (rows in --metadata): use survival (survival_/deathdate/opdate2+Survtime_months)
  - everyone else (EHR-only, e.g. the eval cohort): use EHR DOD codes from the MEDS

Outputs (in --out-dir):
  - MORTALITY.csv : death EVENTS only -> subject_id, time, abspos   (corebehrt outcome file)
  - censor.csv    : per-subject last-known-alive date -> subject_id, time   (for horizons)
  - qc_report.csv : flagged inconsistencies

Usage:
  python build_outcomes_from_survival.py --metadata metadata.csv --meds data/meds_for_bonsai \
     --out-dir ./outputs --tolerance-days 30 --dayfirst [--id-map map.csv]
"""
import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd
from corebehrt.functional.utils.time import get_hours_since_epoch

PID, TIME, CODE = "subject_id", "time", "code"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--meds", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--id-col", default="sample_id")
    ap.add_argument("--survival-col", default="survival_")
    ap.add_argument("--deathdate-col", default="deathdate")
    ap.add_argument("--opdate-col", default="opdate2")
    ap.add_argument("--survtime-months-col", default="Survtime_months")
    ap.add_argument("--death-code", default="DOD")
    ap.add_argument("--tolerance-days", type=float, default=30)
    ap.add_argument("--id-map", default=None)
    ap.add_argument("--dayfirst", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dt = lambda s: pd.to_datetime(s, errors="coerce", dayfirst=args.dayfirst)

    md = pd.read_csv(args.metadata)
    sid = md[args.id_col].astype(str)
    if args.id_map:
        m = pd.read_csv(args.id_map, dtype=str); mp = dict(zip(m.iloc[:, 0], m.iloc[:, 1]))
        sid = sid.map(lambda s: mp.get(s, s))
    md = md.assign(_sid=sid.values)
    opdate = dt(md[args.opdate_col])
    deathdate = dt(md[args.deathdate_col])
    survival = pd.to_numeric(md[args.survival_col], errors="coerce")
    survmon = pd.to_numeric(md[args.survtime_months_col], errors="coerce")

    # survival-derived death + censor (index = opdate)
    surv = pd.DataFrame({"sid": md["_sid"].values, "index": opdate.values,
                         "survival": survival.values, "deathdate": deathdate.values,
                         "survmon": survmon.values})
    last_fu = surv["index"] + pd.to_timedelta(surv["survmon"] * 30.4375, unit="D")
    surv["death_time"] = surv["deathdate"].where(surv["survival"] == 1)
    surv["censor_time"] = surv["death_time"].fillna(last_fu)   # alive -> opdate+survtime
    surv = surv.set_index("sid")

    # EHR DOD events
    ehr = {}
    for f in glob.glob(f"{args.meds}/*/*.parquet"):
        d = pd.read_parquet(f, columns=[PID, TIME, CODE]); d[PID] = d[PID].astype(str)
        dd = d[d[CODE].astype(str).str.startswith(args.death_code)]
        for s, t in dd.groupby(PID)[TIME].min().items():
            ehr[s] = min(pd.to_datetime(t), ehr.get(s, pd.Timestamp.max))
    ehr_dod = pd.Series(ehr, name="ehr_dod")
    # last EHR event per subject (censor for EHR-only subjects)
    last_ehr = {}
    for f in glob.glob(f"{args.meds}/*/*.parquet"):
        d = pd.read_parquet(f, columns=[PID, TIME]); d[PID] = d[PID].astype(str)
        for s, t in d.groupby(PID)[TIME].max().items():
            last_ehr[s] = max(pd.to_datetime(t), last_ehr.get(s, pd.Timestamp.min))
    last_ehr = pd.Series(last_ehr, name="last_ehr")
    all_meds_subjects = set(last_ehr.index)

    # ---- coalesce deaths (survival priority where subject is in metadata) ----
    deaths = {}
    for s, r in surv.iterrows():
        if pd.notna(r["death_time"]):
            deaths[s] = r["death_time"]
    for s, t in ehr_dod.items():                       # EHR-only subjects (not in metadata)
        if s not in surv.index and pd.notna(t):
            deaths[s] = t
    dser = pd.Series(deaths, name="time").dropna()
    mort = pd.DataFrame({PID: dser.index, TIME: dser.values})
    mort["abspos"] = get_hours_since_epoch(mort[TIME]).astype(int)
    mort.to_csv(out / "MORTALITY.csv", index=False)

    # ---- per-subject censor (last known alive) ----
    cens = {}
    for s, r in surv.iterrows():                        # methylation cohort: survival censor
        cens[s] = r["censor_time"]
    for s in all_meds_subjects:                         # EHR-only: last EHR event
        if s not in cens:
            cens[s] = last_ehr.get(s)
    cser = pd.Series(cens, name="time").dropna()
    pd.DataFrame({PID: cser.index, "time": cser.values}).to_csv(out / "censor.csv", index=False)

    # ---- QC ----
    tol = pd.Timedelta(days=args.tolerance_days)
    flags = []
    both = surv.index.intersection(ehr_dod.index)
    for s in both:
        sd, ed = surv.loc[s, "death_time"], ehr_dod[s]
        surv_alive = surv.loc[s, "survival"] == 0
        if surv_alive and pd.notna(ed):
            flags.append((s, "alive_in_survival_but_EHR_DOD", str(ed.date())))
        elif pd.notna(sd) and pd.notna(ed) and abs(sd - ed) > tol:
            flags.append((s, "death_date_mismatch_gt_tol", f"surv={sd.date()} ehr={ed.date()} diff={abs((sd-ed).days)}d"))
    # death at/before index (surgery)
    for s, r in surv.iterrows():
        if pd.notna(r["death_time"]) and pd.notna(r["index"]) and r["death_time"] <= r["index"]:
            flags.append((s, "death_at_or_before_index", f"index={r['index'].date()} death={r['death_time'].date()}"))
    # internal: deathdate vs opdate+survtime
    for s, r in surv.iterrows():
        if r["survival"] == 1 and pd.notna(r["deathdate"]):
            implied = r["index"] + pd.to_timedelta((r["survmon"] or 0) * 30.4375, unit="D")
            if pd.notna(implied) and abs(r["deathdate"] - implied) > tol:
                flags.append((s, "deathdate_vs_survtime_mismatch", f"deathdate={r['deathdate'].date()} implied={implied.date()}"))
    qc = pd.DataFrame(flags, columns=[PID, "issue", "detail"])
    qc.to_csv(out / "qc_report.csv", index=False)

    n_surv_death = int((surv["death_time"].notna()).sum())
    n_ehr_only_death = sum(1 for s in deaths if s not in surv.index)
    print(f"MORTALITY.csv: {len(mort)} death events "
          f"({n_surv_death} from survival, {n_ehr_only_death} EHR-only)")
    print(f"censor.csv: {len(cser)} subjects")
    print(f"qc_report.csv: {len(qc)} flags")
    if len(qc):
        print(qc["issue"].value_counts().to_string())
        print("  → review qc_report.csv (labels use survival where present; tolerance "
              f"{args.tolerance_days:.0f}d)")
    print("\nNext: index from opdate2:  build_index_from_drawdate.py --date-col opdate2 ...")
    print("      horizons:             build_mortality_horizon.py --outcomes "
          f"{out}/MORTALITY.csv --censor-csv {out}/censor.csv ...")


if __name__ == "__main__":
    main()
