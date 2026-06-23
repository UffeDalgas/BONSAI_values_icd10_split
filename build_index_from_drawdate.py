#!/usr/bin/env python3
"""
Build the per-subject index (= blood-draw date) as a corebehrt exposures file, so that
select_cohort anchors the prediction at the methylation draw:

    index_date = exposure_time + n_hours_from_exposure   (use n_hours_from_exposure: 0)

Then EHR is censored at the draw, BIO/* (injected at the draw) is the last pre-prediction
token, and mortality is counted strictly after the draw — no temporal leakage.

Usage:
    python build_index_from_drawdate.py \
        --metadata metadata.csv \
        --id-col sample_id --date-col blood_draw_date \
        --out ./outputs/exposures_drawdate.csv \
        [--id-map sampleid_to_subjectid.csv]

Output columns: subject_id, time   (consumed by select_cohort as `exposure`/`exposures`)
"""
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--id-col", default="sample_id")
    ap.add_argument("--date-col", default="blood_draw_date")
    ap.add_argument("--out", required=True)
    ap.add_argument("--id-map", default=None)
    ap.add_argument("--dayfirst", action="store_true",
                    help="Parse dates as day-first (DD-MM-YYYY / DD/MM/YYYY), e.g. 23-06-2026.")
    args = ap.parse_args()

    md = pd.read_csv(args.metadata)
    parsed = pd.to_datetime(md[args.date_col], errors="coerce", dayfirst=args.dayfirst)
    n_bad = parsed.isna().sum()
    if n_bad:
        print(f"  ⚠ {n_bad}/{len(parsed)} draw dates could not be parsed (dropped). "
              f"Check format / try --dayfirst.")
    out = pd.DataFrame({
        "subject_id": md[args.id_col].astype(str),
        "time": parsed,
    }).dropna(subset=["time"])

    if args.id_map:
        m = pd.read_csv(args.id_map, dtype=str)
        mp = dict(zip(m.iloc[:, 0], m.iloc[:, 1]))
        out["subject_id"] = [mp.get(s, s) for s in out["subject_id"]]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} subjects with a draw-date index")
    print(f"  date range: {out['time'].min()} .. {out['time'].max()}")
    print("Next: in select_cohort config set")
    print(f"  index_date: {{mode: relative, relative: {{n_hours_from_exposure: 0}}}}")
    print(f"  paths.exposure: {args.out}   (per-subject index = blood-draw date)")


if __name__ == "__main__":
    main()
