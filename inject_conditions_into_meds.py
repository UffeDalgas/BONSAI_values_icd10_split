#!/usr/bin/env python3
"""
Inject epimap value-injection condition CSVs into a copy of the MEDS dataset.

Each numeric column of a condition CSV becomes a MEDS concept row per subject:
    subject_id, time, code="BIO/<COLUMN>", numeric_value=<value>
placed at the subject's earliest timestamp so it sits at the start of the sequence
(inside the observation window, before the index date). create_data then tokenizes
these BIO/* concepts and the finetuning model consumes their numeric values via
value_embedding_mode: concat.

condition_0_baseline.csv has 0 feature columns → nothing injected (EHR-only); just
point create_data at the original MEDS for that arm.

Usage:
    python inject_conditions_into_meds.py \
        --meds-in   ./data/meds_for_bonsai \
        --condition ./condition_2_grimage_v2.csv \
        --meds-out  ./data/meds_grimage_v2 \
        [--id-map   ./sampleid_to_subjectid.csv]   # if the CSV index isn't MEDS subject_id

MEDS layout expected: <meds-in>/{train,tuning,held_out}/*.parquet
"""
import argparse
from pathlib import Path
import pandas as pd

PID, TIME, CODE, VAL = "subject_id", "time", "code", "numeric_value"
SPLITS = ["train", "tuning", "held_out"]


def load_condition(path: Path, id_map: Path | None) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    if id_map is not None:                       # map epimap sample_id -> MEDS subject_id
        m = pd.read_csv(id_map, dtype=str)
        mapping = dict(zip(m.iloc[:, 0], m.iloc[:, 1]))
        df.index = [mapping.get(s, s) for s in df.index]
    df = df.apply(pd.to_numeric, errors="coerce")   # numeric feature columns only
    df = df.dropna(axis=1, how="all")
    return df


def inject_split(in_dir: Path, out_dir: Path, cond: pd.DataFrame,
                 draw_dates: dict | None = None, add_gap: bool = False,
                 max_gap_days: float | None = None, excluded: set | None = None) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows, n_subj = 0, 0
    for shard in sorted(in_dir.glob("*.parquet")):
        meds = pd.read_parquet(shard)
        meds[PID] = meds[PID].astype(str)
        # Anchor the BIO concepts at the blood-draw date (= the prediction index) when provided,
        # so censoring keeps them iff draw <= index and they reflect state AT the decision time.
        # Fall back to the subject's earliest timestamp only if no draw date is given.
        first_time = meds.groupby(PID)[TIME].min()
        rows = []
        present = [s for s in meds[PID].unique() if s in cond.index]
        for sid in present:
            t0 = first_time[sid]
            if draw_dates is not None and sid in draw_dates:
                t0 = draw_dates[sid]
            # EHR recency gap: days between the last EHR event at/before the draw and the draw.
            if (add_gap or max_gap_days is not None) and draw_dates and sid in draw_dates:
                draw = draw_dates[sid]
                pre = meds.loc[(meds[PID] == sid) & (meds[TIME] <= draw)
                               & (~meds[CODE].astype(str).str.startswith("BIO/")), TIME]
                if len(pre):
                    gap_days = (draw - pre.max()).days
                    if max_gap_days is not None and gap_days > max_gap_days and excluded is not None:
                        excluded.add(sid)
                    if add_gap:
                        rows.append({PID: sid, TIME: t0, CODE: "BIO/ehr_gap_days",
                                     VAL: float(gap_days)})
            for col, value in cond.loc[sid].items():
                if pd.isna(value):
                    continue
                rows.append({PID: sid, TIME: t0, CODE: f"BIO/{col}", VAL: float(value)})
        if rows:
            inj = pd.DataFrame(rows).astype({PID: meds[PID].dtype})
            meds = pd.concat([meds, inj], ignore_index=True)
            meds = meds.sort_values([PID, TIME], kind="stable")
            n_rows += len(rows); n_subj += len(present)
        # restore subject_id dtype if original was int
        meds.to_parquet(out_dir / shard.name, index=False)
    return n_rows, n_subj


def load_draw_dates(path: Path | None, id_map: Path | None) -> dict | None:
    """Read a CSV of subject_id,draw_date -> {subject_id: Timestamp} (the prediction anchor)."""
    if path is None:
        return None
    df = pd.read_csv(path)
    sid_col = df.columns[0]
    date_col = next((c for c in df.columns[1:]
                     if "date" in c.lower() or "draw" in c.lower() or "time" in c.lower()),
                    df.columns[1])
    s = df.set_index(df[sid_col].astype(str))[date_col]
    if id_map is not None:
        m = pd.read_csv(id_map, dtype=str); mp = dict(zip(m.iloc[:, 0], m.iloc[:, 1]))
        s.index = [mp.get(x, x) for x in s.index]
    return {k: pd.to_datetime(v) for k, v in s.items() if pd.notna(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meds-in", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--meds-out", required=True)
    ap.add_argument("--id-map", default=None)
    ap.add_argument("--sample-date-csv", default=None,
                    help="CSV subject_id,draw_date — anchor BIO/* at the true blood-draw date "
                         "(= prediction index). Strongly recommended to avoid temporal leakage.")
    ap.add_argument("--add-gap-feature", action="store_true",
                    help="Inject BIO/ehr_gap_days = days from last pre-draw EHR event to the draw "
                         "(EHR recency; needs --sample-date-csv).")
    ap.add_argument("--max-gap-days", type=float, default=None,
                    help="Write subjects whose EHR->draw gap exceeds this to --exclude-out "
                         "(for select_cohort paths.exclude_pids). Needs --sample-date-csv.")
    ap.add_argument("--exclude-out", default=None,
                    help="Where to write the over-max-gap exclude list (CSV of subject_id).")
    args = ap.parse_args()

    idmap = Path(args.id_map) if args.id_map else None
    cond = load_condition(Path(args.condition), idmap)
    draw_dates = load_draw_dates(Path(args.sample_date_csv) if args.sample_date_csv else None, idmap)
    print(f"condition: {Path(args.condition).name} -> {cond.shape[1]} feature cols, {len(cond)} subjects")
    print(f"  columns -> concepts: {', '.join('BIO/'+c for c in cond.columns)}")
    print(f"  BIO anchor: {'blood-draw date (per subject)' if draw_dates else 'EARLIEST timestamp (⚠ provide --sample-date-csv to avoid leakage)'}")
    if (args.add_gap_feature or args.max_gap_days is not None) and not draw_dates:
        print("  ⚠ --add-gap-feature/--max-gap-days require --sample-date-csv; skipping gap logic.")

    if cond.shape[1] == 0 and not args.add_gap_feature:
        print("  0 feature columns (baseline) — nothing to inject; use the original MEDS for this arm.")
        return

    excluded: set = set()
    in_root, out_root = Path(args.meds_in), Path(args.meds_out)
    total_rows = total_subj = 0
    for split in SPLITS:
        sd = in_root / split
        if not sd.exists():
            print(f"  [skip] {split}/ not found"); continue
        r, s = inject_split(sd, out_root / split, cond, draw_dates,
                            add_gap=args.add_gap_feature, max_gap_days=args.max_gap_days,
                            excluded=excluded)
        print(f"  {split}: injected {r} rows across {s} subjects")
        total_rows += r; total_subj += s
    print(f"done -> {out_root}  ({total_rows} BIO rows, {total_subj} subject-injections)")
    if args.max_gap_days is not None and args.exclude_out:
        pd.DataFrame({PID: sorted(excluded)}).to_csv(args.exclude_out, index=False)
        print(f"  excluded {len(excluded)} subjects with EHR->draw gap > {args.max_gap_days}d "
              f"-> {args.exclude_out} (set select_cohort paths.exclude_pids to this)")


if __name__ == "__main__":
    main()
