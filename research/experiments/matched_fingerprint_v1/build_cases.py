#!/usr/bin/env python3
"""Select four paired score-change examples without exporting raw commands."""
import csv
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "research/results/matched_fingerprint_v1"
cases = []
for dataset, fullvariant in (("cth", "BS"), ("ait", "BST")):
    frames = []
    for variant in ("evidence", fullvariant):
        paths = sorted((OUT / dataset).glob(f"fold-*-xgb-{variant}.csv.gz"))
        frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        if dataset == "cth":
            frame = frame[frame.fraction == .5]
        frames.append(frame)
    b, f = frames
    keys = ["fold", "pair_id"]
    cols = keys + ["score", "threshold", "predicted_link", "fallback"]
    joined = b.merge(f[cols], on=keys, suffixes=("_before", "_after"), validate="one_to_one")
    for truth, before, after, label in ((1, 0, 1, "missed_link_corrected"),
                                        (1, 1, 0, "new_missed_link"),
                                        (0, 1, 0, "false_merge_corrected"),
                                        (0, 0, 1, "new_false_merge")):
        x = joined[(joined.same_task == truth) & (joined.predicted_link_before == before)
                   & (joined.predicted_link_after == after) & (joined.fallback_after == 0)]
        # Prefer a mixed-mode example where it exists; tie-break by anonymized IDs.
        if (x.pair_type == "HA").any():
            x = x[x.pair_type == "HA"]
        if x.empty:
            continue
        row = x.sort_values(["query_id", "candidate_id", "replicate"]).iloc[0]
        cases.append({
            "dataset": dataset, "error_transition": label, "fold": row["fold"],
            "pair_id": row["pair_id"], "pair_type": row["pair_type"],
            "query_id": row["query_id"], "candidate_id": row["candidate_id"],
            "query_task": row["query_task"], "candidate_task": row["candidate_task"],
            "same_task": truth,
            "evidence_score": row["score_before"], "evidence_threshold": row["threshold_before"],
            "evidence_link": before, "full_score": row["score_after"],
            "full_threshold": row["threshold_after"], "full_link": after,
            "interpretation": "score/threshold decision; not calibrated probability or causal attribution",
        })
path = OUT / "paired_score_cases.csv"
with path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(cases[0]))
    w.writeheader()
    w.writerows(cases)
print("Wrote", len(cases), "paired score cases")
