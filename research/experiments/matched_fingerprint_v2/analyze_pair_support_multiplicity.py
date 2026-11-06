#!/usr/bin/env python3
"""Family-wise max-T intervals for the four exploratory CTH B/S contrasts."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if (REPOSITORY_ROOT / "research/results").is_dir():
    ROOT = REPOSITORY_ROOT
    DATA_DIR = ROOT / "research/results/matched_fingerprint_v2/pair_support"
    INPUT_NAMES = {
        "pair_predictions.csv.gz": DATA_DIR / "pair_predictions.csv.gz",
        "analysis.json": DATA_DIR / "analysis.json",
    }
    SCRIPT_NAME = "analyze_pair_support_multiplicity.py"
    OUTPUT_LABEL = "research/results/matched_fingerprint_v2/pair_support/multiplicity_adjustment.json"
elif (Path(__file__).resolve().parents[1] / "results/pair_support").is_dir():
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "results/pair_support"
    INPUT_NAMES = {
        "pair_predictions.csv.gz": DATA_DIR / "pair_predictions.csv.gz",
        "analysis.json": DATA_DIR / "analysis.json",
    }
    SCRIPT_NAME = "analyze_pair_support_multiplicity.py"
    OUTPUT_LABEL = "results/pair_support/multiplicity_adjustment.json"
else:
    raise FileNotFoundError("cannot locate repository or extracted bundle pair-support files")
PREDICTIONS = INPUT_NAMES["pair_predictions.csv.gz"]
ANALYSIS = INPUT_NAMES["analysis.json"]
OUT = DATA_DIR / "multiplicity_adjustment.json"
N_BOOT = 10_000
SEED = 20261105
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
CONTRASTS = (
    ("sampled_8_per_task", "E_BS", "E"),
    ("sampled_8_per_task", "E_args_BS", "E_args"),
    ("all_pairs", "E_BS", "E"),
    ("all_pairs", "E_args_BS", "E_args"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def task_share_ba(frame: pd.DataFrame, task_ids: list[str]) -> np.ndarray:
    """Return one task x share matrix of task-cell balanced accuracy."""
    values = np.full((len(task_ids), len(FRACTIONS)), np.nan)
    for ti, task in enumerate(task_ids):
        for fi, fraction in enumerate(FRACTIONS):
            x = frame[(frame.query_task == task) & (frame.fraction == fraction)]
            pos, neg = x[x.same_task == 1], x[x.same_task == 0]
            if len(pos) and len(neg):
                fmr = float(neg.predicted_link.mean())
                fsr = float(1.0 - pos.predicted_link.mean())
                values[ti, fi] = 1.0 - (fmr + fsr) / 2.0
    return values


def main() -> None:
    frame = pd.read_csv(PREDICTIONS, compression="gzip")
    source = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    task_ids = sorted(frame.query_task.unique())
    assert len(task_ids) == 17
    assert len(frame) == 435_200
    assert set(frame.learner) == {"xgb"}
    assert set(frame.condition) == {"sampled_8_per_task", "all_pairs"}

    # All model arms share the exact query/candidate rows within each condition.
    pair_keys = ["fold", "pair_id", "replicate", "fraction", "query_id",
                 "candidate_id", "query_task", "candidate_task", "pair_type",
                 "same_task", "candidate_position"]
    ref = frame[(frame.condition == "sampled_8_per_task") & (frame.version == "E")][pair_keys]
    for condition in ("sampled_8_per_task", "all_pairs"):
        for version in ("E", "E_args", "E_BS", "E_args_BS"):
            current = frame[(frame.condition == condition) & (frame.version == version)][pair_keys]
            pd.testing.assert_frame_equal(ref.reset_index(drop=True),
                                          current.reset_index(drop=True), check_dtype=False)

    # Each draw resamples task clusters once and applies the same multiplicities
    # to all four contrasts, preserving their paired covariance.
    task_vectors = {}
    point = []
    for condition, after, before in CONTRASTS:
        after_frame = frame[(frame.condition == condition) & (frame.version == after)]
        before_frame = frame[(frame.condition == condition) & (frame.version == before)]
        after_ba = task_share_ba(after_frame, task_ids)
        before_ba = task_share_ba(before_frame, task_ids)
        assert np.array_equal(np.isfinite(after_ba), np.isfinite(before_ba))
        per_task = np.nanmean(after_ba - before_ba, axis=1)
        assert np.isfinite(per_task).all()
        name = f"{condition}:{after}_minus_{before}"
        task_vectors[name] = per_task
        point.append(float(per_task.mean()))

    point = np.asarray(point)
    rng = np.random.default_rng(SEED)
    indexes = rng.integers(0, len(task_ids), size=(N_BOOT, len(task_ids)))
    boots = np.column_stack([
        values[indexes].mean(axis=1) for values in task_vectors.values()
    ])
    nominal = np.quantile(boots, [0.025, 0.975], axis=0).T
    standard_errors = boots.std(axis=0, ddof=1)
    max_abs_t = np.max(np.abs((boots - point[None, :]) / standard_errors[None, :]), axis=1)
    critical = float(np.quantile(max_abs_t, 0.95))
    simultaneous = np.column_stack((point - critical * standard_errors,
                                    point + critical * standard_errors))

    contrasts = {}
    for i, name in enumerate(task_vectors):
        contrasts[name] = {
            "delta_ba_percentage_points": float(point[i] * 100),
            "nominal_percentile_ci95_percentage_points": [float(x * 100) for x in nominal[i]],
            "max_t_simultaneous_ci95_percentage_points": [float(x * 100) for x in simultaneous[i]],
            "bootstrap_se_percentage_points": float(standard_errors[i] * 100),
        }
    result = {
        "status": "completed_exploratory_familywise_adjustment",
        "estimand": "XGBoost BA increment from adding B/S, averaged equally over five role shares and 17 query-task clusters",
        "family": "four inspected contrasts: two command baselines by two training-pair coverage conditions",
        "bootstrap_unit": "query task; paired common task resamples across all four contrasts",
        "bootstrap_draws": N_BOOT,
        "bootstrap_seed": SEED,
        "adjustment": "two-sided max-T simultaneous 95% intervals using the bootstrap covariance across four contrasts",
        "max_t_critical_value": critical,
        "tasks": task_ids,
        "input_hashes": {
            "pair_predictions.csv.gz": sha256(PREDICTIONS),
            "analysis.json": sha256(ANALYSIS),
            SCRIPT_NAME: sha256(Path(__file__)),
        },
        "contrasts": contrasts,
        "point_estimate_crosscheck": {
            "all_pairs_E_BS_minus_E_ba": source["fingerprint_increment"]["all_pairs"]["ALL"]["E_BS_minus_E"]["all_shares"]["metrics"]["ba"] * 100,
            "all_pairs_E_args_BS_minus_E_args_ba": source["fingerprint_increment"]["all_pairs"]["ALL"]["E_args_BS_minus_E_args"]["all_shares"]["metrics"]["ba"] * 100,
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({"output": OUTPUT_LABEL, "max_t_critical_value": critical,
                      "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
