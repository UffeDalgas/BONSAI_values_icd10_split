#!/usr/bin/env python3
"""
Build fixed-horizon mortality labels' EXCLUSION lists (1/2/3/4/5-year, etc.).

corebehrt labels `1` if death falls in (index+start, index+horizon] and `0` otherwise — but it
does NOT drop survivors whose follow-up ends before the horizon (their status at the horizon is
unknown, so labelling them `0` is wrong). This script computes, per horizon, the subjects to
exclude (alive-but-insufficient-follow-up) and reports the resulting label balance.

A subject is:
  - POSITIVE  if index < death <= index + horizon
  - NEGATIVE  if known alive at the horizon: death > index+horizon, OR admin-censor >= index+horizon
  - EXCLUDED  otherwise (no death and follow-up shorter than the horizon)

Follow-up end (admin censor) = a global registry/study end date (--admin-censor-date, recommended)
or, as a fallback, each subject's last event time in the MEDS (--meds).

Usage:
  python build_mortality_horizon.py \
    --outcomes ./outputs/outcomes/MORTALITY.csv \
    --index    ./outputs/exposures_drawdate.csv \
    --admin-censor-date 2024-12-31 \
    --horizons 1,2,3,4,5 \
    --out-dir  ./outputs/horizons
Outputs per horizon H: ./outputs/horizons/exclude_<H>y.csv  (set select_cohort paths.exclude_pids)
and prints the n_hours_end_follow_up to use in the prepare/outcome config.
"""
import argparse, glob
from pathlib import Path
import pandas as pd

PID, TIME = "subject_id", "time"
HOURS_PER_YEAR = 365.25 * 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", required=True, help="MORTALITY.csv (subject_id, time[, abspos])")
    ap.add_argument("--index", required=True, help="exposures_drawdate.csv (subject_id, time)")
    ap.add_argument("--censor-csv", default=None,
                    help="per-subject censor (subject_id,time) — e.g. censor.csv from "
                         "build_outcomes_from_survival.py. Preferred over a global date.")
    ap.add_argument("--admin-censor-date", default=None, help="global study/registry end, e.g. 2024-12-31")
    ap.add_argument("--meds", default=None, help="MEDS dir; fallback admin censor = per-subject last event")
    ap.add_argument("--horizons", default="1,2,3,4,5")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dayfirst", action="store_true")
    args = ap.parse_args()

    idx = pd.read_csv(args.index)
    idx[PID] = idx[PID].astype(str)
    idx_t = idx.set_index(PID)[TIME].pipe(pd.to_datetime, errors='coerce', format='mixed', dayfirst=args.dayfirst)

    dod = pd.read_csv(args.outcomes)
    dod[PID] = dod[PID].astype(str)
    death_t = dod.groupby(PID)[TIME].min().pipe(pd.to_datetime, errors='coerce', format='mixed', dayfirst=args.dayfirst)

    # admin censor per subject: per-subject csv (best) > global date > per-subject last MEDS event
    if args.censor_csv:
        c = pd.read_csv(args.censor_csv); c[PID] = c[PID].astype(str)
        admin = c.set_index(PID)[TIME].pipe(pd.to_datetime, errors='coerce', format='mixed', dayfirst=args.dayfirst).reindex(idx_t.index)
        print(f"using per-subject censor dates from {args.censor_csv} "
              f"({admin.notna().sum()}/{len(admin)} present)")
    elif args.admin_censor_date:
        admin = pd.Series(pd.Timestamp(args.admin_censor_date), index=idx_t.index)
    elif args.meds:
        last = {}
        for f in glob.glob(f"{args.meds}/*/*.parquet"):
            d = pd.read_parquet(f, columns=[PID, TIME]); d[PID] = d[PID].astype(str)
            for s, t in d.groupby(PID)[TIME].max().items():
                last[s] = max(pd.to_datetime(t), last.get(s, pd.Timestamp.min))
        admin = pd.Series(last).reindex(idx_t.index)
        print("⚠ using per-subject last MEDS event as admin censor (a registry end date is safer).")
    else:
        raise SystemExit("provide --censor-csv (best), --admin-censor-date, or --meds")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(idx_t)} subjects | {death_t.notna().sum()} with a death record\n")
    print(f"{'horizon':>8} {'pos(1)':>8} {'neg(0)':>8} {'excluded':>9} {'prevalence':>11}  n_hours_end_follow_up")
    for h in [float(x) for x in args.horizons.split(",")]:
        H = pd.Timedelta(hours=h * HOURS_PER_YEAR)
        d = death_t.reindex(idx_t.index)
        died_within = d.notna() & (d > idx_t) & (d <= idx_t + H)
        alive_known = (d.notna() & (d > idx_t + H)) | (admin >= idx_t + H)
        excluded = ~died_within & ~alive_known
        pos, neg, exc = int(died_within.sum()), int((~died_within & alive_known).sum()), int(excluded.sum())
        prev = pos / max(pos + neg, 1)
        pd.DataFrame({PID: idx_t.index[excluded.values]}).to_csv(
            out_dir / f"exclude_{int(h)}y.csv", index=False)
        print(f"{int(h):>7}y {pos:>8} {neg:>8} {exc:>9} {prev:>11.4f}  {int(round(h*HOURS_PER_YEAR)):>}")
    print(f"\nexclude lists -> {out_dir}/exclude_<H>y.csv")
    print("For horizon H: set select_cohort paths.exclude_pids to exclude_<H>y.csv and "
          "prepare/outcome n_hours_end_follow_up to the printed value.")


if __name__ == "__main__":
    main()
