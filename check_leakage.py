#!/usr/bin/env python3
"""
Leakage diagnostics for a prepared finetune/eval dataset.

Checks the three most common mortality-prediction leaks:
  1. Outcome/death tokens present in the INPUT sequences (the model literally sees the label).
  2. Sequence-length separating the classes (informative-censoring artifact).
  3. Label distribution (sanity).

Usage:
    python check_leakage.py --prepared ./outputs/finetuning/processed_data_grimage_v2 \
                            --vocab    ./outputs/tokenized_grimage_v2 \
                            [--death-substrings DOD,DEATH,DOD/]
"""
import argparse
from pathlib import Path
import numpy as np
import torch

from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.constants.paths import PREPARED_ALL_PATIENTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True, help="dir with patients.pt")
    ap.add_argument("--vocab", required=True, help="tokenized dir with vocabulary.pt")
    ap.add_argument("--death-substrings", default="DOD,DEATH,DECEAS,MORT")
    args = ap.parse_args()

    vocab = load_vocabulary(args.vocab)                      # {token: id}
    id2tok = {v: k for k, v in vocab.items()}
    subs = [s.strip().upper() for s in args.death_substrings.split(",") if s.strip()]
    death_ids = {i for t, i in vocab.items() if any(s in str(t).upper() for s in subs)}
    death_toks = sorted(str(t) for t in vocab if any(s in str(t).upper() for s in subs))
    print(f"death-like tokens in vocab ({len(death_toks)}): {death_toks[:20]}")

    patients = torch.load(Path(args.prepared) / PREPARED_ALL_PATIENTS, weights_only=False)
    patients = getattr(patients, "patients", patients)

    n = len(patients)
    y = np.array([int(getattr(p, "outcome", 0) or 0) for p in patients])
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    print(f"\nLabel distribution: total={n} | 0={n_neg} | 1={n_pos} | prevalence={n_pos/max(n,1):.4f}")

    # 1. death tokens inside input sequences
    contains = np.array([len(death_ids.intersection(p.concepts)) > 0 for p in patients])
    print(f"\n[1] sequences containing a death-like token: {contains.sum()}/{n} "
          f"({contains.mean()*100:.1f}%)")
    if contains.sum():
        among_pos = contains[y == 1].mean() if n_pos else 0
        among_neg = contains[y == 0].mean() if n_neg else 0
        print(f"    among positives: {among_pos*100:.1f}%  | among negatives: {among_neg*100:.1f}%")
        print("    ⚠ LEAKAGE: the outcome token appears in the input — censoring is not removing it.")
    else:
        print("    ✓ no death-like tokens in input sequences")

    # 2. sequence-length separation by class
    lens = np.array([len(p.concepts) for p in patients])
    if n_pos and n_neg:
        lp, ln = lens[y == 1].mean(), lens[y == 0].mean()
        print(f"\n[2] mean sequence length: positives={lp:.1f}  negatives={ln:.1f}  "
              f"ratio={lp/ln:.2f}")
        if abs(np.log(max(lp, 1) / max(ln, 1))) > np.log(1.5):
            print("    ⚠ large length gap between classes — possible informative-censoring leak.")
        else:
            print("    ✓ sequence lengths comparable across classes")

    # 3. which INPUT tokens most separate deaths vs survivors (transparency, NOT for removal)
    if n_pos and n_neg:
        from collections import Counter
        cpos, cneg = Counter(), Counter()
        for p, yi in zip(patients, y):
            toks = set(p.concepts)                       # presence (not count) per sequence
            (cpos if yi == 1 else cneg).update(toks)
        rows = []
        for tid in set(cpos) | set(cneg):
            ppos = cpos.get(tid, 0) / n_pos
            pneg = cneg.get(tid, 0) / n_neg
            # log odds ratio with Laplace smoothing → signed separation strength
            lor = np.log(((cpos.get(tid,0)+0.5)/(n_pos-cpos.get(tid,0)+0.5)) /
                         ((cneg.get(tid,0)+0.5)/(n_neg-cneg.get(tid,0)+0.5)))
            rows.append((abs(lor), lor, id2tok.get(tid, tid), ppos, pneg))
        rows.sort(reverse=True)
        print(f"\n[3] top input tokens separating deaths(1) vs survivors(0)  "
              f"[pos_rate | neg_rate | logOR]:")
        for _, lor, tok, ppos, pneg in rows[:25]:
            flag = "  ← death-like?" if any(s in str(tok).upper() for s in subs) else ""
            print(f"    {str(tok):22} {ppos:5.2f} | {pneg:5.2f} | {lor:+.2f}{flag}")
        print("    (legitimate pre-index terminal/ICU codes will appear here — that's expected "
              "signal, not leakage. A token flagged 'death-like?' here WOULD be a leak.)")

    print("\nIf [1] flags, or a death-like token appears in [3], fix censoring/index before trusting the AUC.")


if __name__ == "__main__":
    main()
