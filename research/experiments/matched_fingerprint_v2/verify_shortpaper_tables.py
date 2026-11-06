#!/usr/bin/env python3
"""Recompute the paper's principal point estimates from bundled pair predictions.

This verifier uses only Python's standard library. It does not refit models or
recreate source-derived traces; it checks aggregation from the frozen predictions
included in the reproducibility archive.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

METRICS = ("fmr", "fsr", "ba", "hit1", "gallery_auc")


def discover_layout(script: Path) -> tuple[Path, bool]:
    repo = script.resolve().parents[3]
    if (repo / "research/results").is_dir():
        return repo, False
    bundle = script.resolve().parents[1]
    if (bundle / "results").is_dir() and (bundle / "paper").is_dir():
        return bundle, True
    raise FileNotFoundError("cannot locate repository or extracted bundle data roots")


def input_paths(root: Path, bundled: bool) -> dict[str, object]:
    if bundled:
        base = root / "results"
        return {
            "action": [base / "frozen/action_bigram_cosine_pair_scores.csv.gz"],
            "cth_e": [base / f"cth_mixed/fold-{i}-xgb-evidence.csv.gz" for i in range(3)],
            "cth_bs": [base / f"cth_mixed/fold-{i}-xgb-BS.csv.gz" for i in range(3)],
            "cth_aa": base / "cth_aa/predictions.csv.gz",
            "nyu_aa": base / "nyu_aa/predictions.csv.gz",
            "mechanism": base / "mechanism/predictions.csv.gz",
            "pair_support": base / "pair_support/pair_predictions.csv.gz",
        }
    base = root / "research/results"
    short = base / "matched_fingerprint_v2/shortpaper_frozen_20261105"
    v1 = base / "matched_fingerprint_v1/cth"
    return {
        "action": [short / "action_bigram_cosine_pair_scores.csv.gz"],
        "cth_e": [v1 / f"fold-{i}-xgb-evidence.csv.gz" for i in range(3)],
        "cth_bs": [v1 / f"fold-{i}-xgb-BS.csv.gz" for i in range(3)],
        "cth_aa": base / "matched_fingerprint_v2/independent_cth_agent/predictions.csv.gz",
        "nyu_aa": base / "matched_fingerprint_v2/independent_nyu_agent/predictions.csv.gz",
        "mechanism": base / "matched_fingerprint_v2/shortpaper_mechanism_20261105/predictions.csv.gz",
        "pair_support": base / "matched_fingerprint_v2/pair_support/pair_predictions.csv.gz",
    }


def read_galleries(paths: list[Path] | Path, *, filters: dict[str, str] | None = None,
                   seed_filter: int | None = None) -> list[dict]:
    if isinstance(paths, Path):
        paths = [paths]
    grouped: dict[tuple, list[tuple[int, float, int]]] = defaultdict(list)
    for path in paths:
        with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if filters and any(row.get(k) != v for k, v in filters.items()):
                    continue
                if seed_filter is not None and int(row.get("shuffle_seed", 0)) != seed_filter:
                    continue
                key = (row.get("fold", "0"), row.get("replicate", "0"),
                       float(row.get("fraction", "1")), row["query_id"], row["query_task"])
                grouped[key].append((int(row["same_task"]), float(row["score"]),
                                     int(row["predicted_link"])))
    out = []
    for key, records in grouped.items():
        positives = [r for r in records if r[0] == 1]
        negatives = [r for r in records if r[0] == 0]
        if len(positives) != 1 or not negatives:
            continue
        ps = positives[0][1]
        neg_scores = [r[1] for r in negatives]
        ties = sum(s == ps for s in neg_scores)
        greater = sum(s > ps for s in neg_scores)
        fmr = mean(r[2] for r in negatives)
        fsr = float(positives[0][2] == 0)
        out.append({"fraction": key[2], "task": key[4],
                    "fmr": fmr, "fsr": fsr, "ba": 1 - (fmr + fsr) / 2,
                    "hit1": 1 / (1 + greater + ties),
                    "gallery_auc": mean(float(s < ps) + .5 * float(s == ps)
                                         for s in neg_scores)})
    return out


def task_share_means(galleries: list[dict]) -> dict[tuple[float, str], dict[str, float]]:
    raw: dict[tuple[float, str], dict[str, list[float]]] = defaultdict(
        lambda: {m: [] for m in METRICS})
    for g in galleries:
        key = (g["fraction"], g["task"])
        for metric in METRICS:
            raw[key][metric].append(g[metric])
    return {key: {m: mean(values[m]) for m in METRICS}
            for key, values in raw.items()}


def share_macro(galleries: list[dict], share: float) -> dict[str, float]:
    by_task = task_share_means([g for g in galleries if g["fraction"] == share])
    tasks = sorted({task for _, task in by_task})
    return {m: mean(by_task[(share, task)][m] for task in tasks) for m in METRICS}


def overall_macro(galleries: list[dict]) -> dict[str, float]:
    by_task_share = task_share_means(galleries)
    tasks = sorted({task for _, task in by_task_share})
    shares = sorted({share for share, _ in by_task_share})
    values = []
    for task in tasks:
        observed = [s for s in shares if (s, task) in by_task_share]
        values.append({m: mean(by_task_share[(s, task)][m] for s in observed)
                       for m in METRICS})
    return {m: mean(row[m] for row in values) for m in METRICS}


def endpoint_delta(galleries: list[dict], metric: str, low: float = 0.0,
                   high: float = 1.0) -> float:
    by = task_share_means(galleries)
    tasks = sorted({task for _, task in by})
    paired = [by[(high, task)][metric] - by[(low, task)][metric]
              for task in tasks if (low, task) in by and (high, task) in by]
    return mean(paired)


def contrast(left: list[dict], right: list[dict]) -> dict[str, float]:
    a, b = task_share_means(left), task_share_means(right)
    common = sorted(set(a) & set(b))
    tasks = sorted({task for _, task in common})
    return {m: mean(mean(a[(s, t)][m] - b[(s, t)][m]
                         for s in sorted({s for s, _ in common if _ == t}))
                     for t in tasks) for m in METRICS}


def check(label: str, actual: float, expected: float, tolerance: float = 0.00006):
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: got {actual:.8f}, expected {expected:.8f}")


def pct(row: dict[str, float]) -> dict[str, float]:
    return {k: 100 * v for k, v in row.items()}


def pair_support_increments(path: Path) -> dict[str, dict[str, float]]:
    """Reaggregate the four XGBoost B/S increments from frozen pair rows."""
    # Each cell stores positive count/accepted and negative count/accepted.
    cells: dict[tuple[str, str, str, float], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["condition"], row["version"], row["query_task"],
                   float(row["fraction"]))
            target, accepted = int(row["same_task"]), int(row["predicted_link"])
            if target:
                cells[key][0] += 1
                cells[key][1] += accepted
            else:
                cells[key][2] += 1
                cells[key][3] += accepted

    result = {}
    comparisons = (
        ("sampled_8_per_task", "E_BS", "E"),
        ("sampled_8_per_task", "E_args_BS", "E_args"),
        ("all_pairs", "E_BS", "E"),
        ("all_pairs", "E_args_BS", "E_args"),
    )
    for condition, after, before in comparisons:
        name = f"{condition}:{after}_minus_{before}"
        task_deltas: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {m: [] for m in ("fmr", "fsr", "ba")})
        task_ids = sorted({key[2] for key in cells})
        for task in task_ids:
            for fraction in (0., .25, .5, .75, 1.):
                metrics = {}
                for version in (after, before):
                    pos_n, pos_yes, neg_n, neg_yes = cells[(condition, version, task, fraction)]
                    if not pos_n or not neg_n:
                        raise AssertionError(f"incomplete pair-support cell: {condition}/{version}/{task}/{fraction}")
                    fmr = neg_yes / neg_n
                    fsr = 1 - pos_yes / pos_n
                    metrics[version] = {"fmr": fmr, "fsr": fsr,
                                        "ba": 1 - (fmr + fsr) / 2}
                for metric in ("fmr", "fsr", "ba"):
                    task_deltas[task][metric].append(
                        metrics[after][metric] - metrics[before][metric])
        result[name] = {
            metric: 100 * mean(mean(values) for values in
                               (task_deltas[task][metric] for task in task_ids))
            for metric in ("fmr", "fsr", "ba")
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="write the recomputed point estimates as JSON")
    args = parser.parse_args()
    root, bundled = discover_layout(Path(__file__))
    paths = input_paths(root, bundled)

    action = read_galleries(paths["action"])
    action_rows = {s: pct(share_macro(action, s)) for s in (0., .25, .5, .75, 1.)}
    for s, expected in {
        0.: {"fmr": 32.27, "fsr": 52.94, "ba": 57.40, "hit1": 53.13, "gallery_auc": 63.79},
        1.: {"fmr": 9.18, "fsr": 97.87, "ba": 46.48, "hit1": 53.08, "gallery_auc": 52.31},
    }.items():
        for metric, value in expected.items():
            check(f"action-bigram {s:g} {metric}", action_rows[s][metric] / 100, value / 100)

    cth_e = read_galleries(paths["cth_e"])
    cth_bs = read_galleries(paths["cth_bs"])
    cth_mixed = {"E": pct(overall_macro(cth_e)), "E+B/S": pct(overall_macro(cth_bs))}
    for arm, expected in {
        "E": {"fmr": 5.97, "fsr": 81.91, "ba": 56.06},
        "E+B/S": {"fmr": 15.43, "fsr": 77.21, "ba": 53.68},
    }.items():
        for metric, value in expected.items():
            check(f"CTH mixed {arm} {metric}", cth_mixed[arm][metric] / 100, value / 100)

    cth_aa_raw = paths["cth_aa"]
    nyu_aa_raw = paths["nyu_aa"]
    cth_aa = {arm: read_galleries(cth_aa_raw, filters={"learner": "xgb", "variant": variant})
              for arm, variant in (("E", "E"), ("E+B/S", "E_BS"))}
    nyu_aa = {arm: read_galleries(nyu_aa_raw, filters={"learner": "xgb", "variant": variant})
              for arm, variant in (("E", "E"), ("E+B/S", "E_BS"))}
    external = {}
    for source, values, expected in (
        ("CTH AA", cth_aa, {"E": (11.78, 78.04, 55.09), "E+B/S": (5.63, 91.06, 51.65)}),
        ("NYU AA", nyu_aa, {"E": (10.72, 86.96, 51.16), "E+B/S": (25.39, 60.86, 56.87)}),
    ):
        external[source] = {arm: pct(overall_macro(g)) for arm, g in values.items()}
        for arm, triplet in expected.items():
            for metric, value in zip(("fmr", "fsr", "ba"), triplet):
                check(f"{source} {arm} {metric}", external[source][arm][metric] / 100,
                      value / 100)

    mechanism_raw = paths["mechanism"]
    mechanism = read_galleries(mechanism_raw)
    # The five non-shuffled arms have one fixed seed marker (0); each shuffled arm
    # carries one row set per predeclared seed.
    mech_contrasts = {}
    arms = ("E", "E+records", "E+records+LCS", "E+records+B/S")
    by_arm = {arm: read_galleries(mechanism_raw, filters={"arm": arm}) for arm in arms}
    mech_contrasts["E+records - E"] = pct(contrast(by_arm["E+records"], by_arm["E"]))
    mech_contrasts["E+records+LCS - E+records"] = pct(
        contrast(by_arm["E+records+LCS"], by_arm["E+records"]))
    mech_contrasts["E+records+B/S - E+records"] = pct(
        contrast(by_arm["E+records+B/S"], by_arm["E+records"]))
    shuffled = {}
    for seed in (20261105, 20261106, 20261107, 20261108, 20261109):
        shuffled_g = read_galleries(mechanism_raw,
                                    filters={"arm": "E+records+shuffled-B/S"},
                                    seed_filter=seed)
        shuffled[str(seed)] = pct(contrast(by_arm["E+records+B/S"], shuffled_g))

    support = pair_support_increments(paths["pair_support"])
    expected_support = {
        "sampled_8_per_task:E_BS_minus_E": {"fmr": 6.83, "fsr": -8.93, "ba": 1.05},
        "sampled_8_per_task:E_args_BS_minus_E_args": {"fmr": 1.38, "fsr": -5.06, "ba": 1.84},
        "all_pairs:E_BS_minus_E": {"fmr": 0.70, "fsr": -7.29, "ba": 3.29},
        "all_pairs:E_args_BS_minus_E_args": {"fmr": 0.56, "fsr": -13.06, "ba": 6.25},
    }
    for contrast_name, expected in expected_support.items():
        for metric, value in expected.items():
            check(f"pair-support {contrast_name} {metric}",
                  support[contrast_name][metric] / 100, value / 100)

    output = {
        "scope": "frozen pair prediction re-aggregation; no refitting",
        "cth_action_bigram_by_agent_share_percent": action_rows,
        "cth_action_bigram_endpoint_delta_pp": {
            m: 100 * endpoint_delta(action, m) for m in METRICS
        },
        "cth_mixed_xgb_equal_share_percent": cth_mixed,
        "cth_aa_xgb_percent": external["CTH AA"],
        "nyu_aa_xgb_percent": external["NYU AA"],
        "mechanism_equal_share_contrasts_pp": mech_contrasts,
        "real_minus_shuffled_bs_by_seed_pp": shuffled,
        "pair_support_xgb_increment_pp": support,
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
