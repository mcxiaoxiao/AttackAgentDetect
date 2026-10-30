#!/usr/bin/env python3
"""Paired task-level contrasts for sequence budgets and order randomization."""
import collections, csv, datetime as dt, hashlib, json, pathlib, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK_CSV = ROOT / "results" / "feature_ablation_task_metrics_v1.csv"
OUT_JSON = ROOT / "results" / "feature_ablation_contrasts_v1.json"
FIG = ROOT / "results" / "figures" / "feature_budget_order_v1"
SITE_FIG = ROOT / "site" / "assets" / "feature_budget_order_v1.png"
SHUFFLE_SEEDS = (17, 29, 43, 71, 101)
METRIC = "balanced_accuracy"


def load_rows():
    with TASK_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def records_for(rows, cohort, feature, classifier="logistic"):
    by_task = {}
    for row in rows:
        if row["cohort"] == cohort and row["feature"] == feature and row["classifier"] == classifier:
            by_task[(row["benchmark"], row["task"])] = float(row[METRIC])
    return by_task


def average_records_for(rows, cohort, feature, classifier="logistic"):
    grouped = collections.defaultdict(list)
    for row in rows:
        if row["cohort"] == cohort and row["feature"] == feature and row["classifier"] == classifier:
            grouped[(row["benchmark"], row["task"])].append(float(row[METRIC]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def grouped_mean(task_values):
    by_benchmark = collections.defaultdict(list)
    for (benchmark, _task), value in task_values.items():
        by_benchmark[benchmark].append(float(value))
    return float(np.mean([np.mean(values) for values in by_benchmark.values()])) if by_benchmark else None


def paired_contrast(left, right, *, draws=5000, seed=20261030):
    common = sorted(set(left) & set(right))
    by_benchmark = collections.defaultdict(list)
    for key in common:
        by_benchmark[key[0]].append(float(left[key] - right[key]))
    if not by_benchmark:
        return {"estimate": None, "ci_95": None, "task_groups": 0}
    estimate = float(np.mean([np.mean(values) for values in by_benchmark.values()]))
    rng = np.random.default_rng(seed)
    sample = np.empty(draws)
    for i in range(draws):
        sample[i] = np.mean([np.mean(rng.choice(values, len(values), replace=True))
                             for values in by_benchmark.values()])
    return {"estimate": estimate, "ci_95": [float(x) for x in np.quantile(sample, [0.025, 0.975])],
            "task_groups": len(common), "bootstrap_unit": "paired benchmark-qualified task, resampled within benchmark",
            "bootstrap_draws": draws, "direction": "left minus right"}


def summarize(rows, draws=5000):
    contrasts = {"feature_increment": {}, "observation_budget": {}, "order_randomization": {}}
    for classifier in ("logistic", "extra_trees"):
        f1 = average_records_for(rows, "full_177", "F1_command_tfidf", classifier)
        f3 = average_records_for(rows, "full_177", "F3_command_plus_structure", classifier)
        contrasts["feature_increment"][classifier] = {
            "F3_minus_F1": paired_contrast(f3, f1, draws=draws,
                                            seed=20261030 + len(classifier)),
            "F1_balanced_accuracy": grouped_mean(f1), "F3_balanced_accuracy": grouped_mean(f3),
            "task_groups": len(set(f1) & set(f3)), "split_seeds_averaged": 6,
        }
    for min_count, full_name, prefix_name in (
        (2, "matched_ge2_full_same_cohort", "matched_ge2_prefix_2"),
        (4, "matched_ge4_full_same_cohort", "matched_ge4_prefix_4"),
    ):
        label = f"minimum_{min_count}"
        contrasts["observation_budget"][label] = {"eligible_task_groups": 40 if min_count == 2 else 25,
                                                   "eligible_records": 175 if min_count == 2 else 99,
                                                   "features": {}}
        for feature in ("F1_command_tfidf", "F3_command_plus_structure"):
            full = records_for(rows, full_name, feature)
            prefix = records_for(rows, prefix_name, feature)
            contrasts["observation_budget"][label]["features"][feature] = {
                "full_balanced_accuracy": grouped_mean(full),
                "prefix_balanced_accuracy": grouped_mean(prefix),
                "full_minus_prefix": paired_contrast(full, prefix, draws=draws,
                                                       seed=20261030 + min_count + len(feature)),
            }

    shuffle_cohorts = [f"prefix4_order_shuffle_{seed}" for seed in SHUFFLE_SEEDS]
    for feature in ("F1_command_tfidf", "F3_command_plus_structure"):
        ordered = records_for(rows, "matched_ge4_prefix_4", feature)
        shuffled_by_task = collections.defaultdict(list)
        for cohort in shuffle_cohorts:
            for task, value in records_for(rows, cohort, feature).items():
                shuffled_by_task[task].append(value)
        shuffled = {task: float(np.mean(values)) for task, values in shuffled_by_task.items()}
        contrasts["order_randomization"][feature] = {
            "ordered_balanced_accuracy": grouped_mean(ordered),
            "five_shuffle_mean_balanced_accuracy": grouped_mean(shuffled),
            "ordered_minus_shuffle": paired_contrast(ordered, shuffled, draws=draws,
                                                       seed=20262000 + len(feature)),
            "shuffle_seeds": list(SHUFFLE_SEEDS), "task_groups": len(set(ordered) & set(shuffled)),
        }
    return contrasts


def draw(contrasts):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
                         "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 10,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "savefig.dpi": 320,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), gridspec_kw={"width_ratios": [1.45, 1]})
    budget_labels, budget_values, budget_intervals, budget_colors = [], [], [], []
    for cohort, title, color_f1, color_f3 in (
        ("minimum_2", "At least 2 commands", "#99CDFC", "#173F68"),
        ("minimum_4", "At least 4 commands", "#99CDFC", "#173F68"),
    ):
        for feature, label, color in (("F1_command_tfidf", "F1 · command TF-IDF", color_f1),
                                      ("F3_command_plus_structure", "F3 · command + structure", color_f3)):
            item = contrasts["observation_budget"][cohort]["features"][feature]["full_minus_prefix"]
            budget_labels.append(f"{title} · {label}")
            budget_values.append(item["estimate"]); budget_intervals.append(item["ci_95"]); budget_colors.append(color)
    y = np.arange(len(budget_labels))
    low = np.asarray([v - ci[0] for v, ci in zip(budget_values, budget_intervals)])
    high = np.asarray([ci[1] - v for v, ci in zip(budget_values, budget_intervals)])
    axes[0].barh(y, budget_values, color=budget_colors, height=0.58, edgecolor="white", linewidth=0.8)
    axes[0].errorbar(budget_values, y, xerr=[low, high], fmt="none", ecolor="#142A3B", capsize=2.5, linewidth=1)
    axes[0].axvline(0, color="#536476", linewidth=0.8)
    axes[0].set_yticks(y, budget_labels); axes[0].invert_yaxis()
    axes[0].set_xlabel("Balanced-accuracy change (full − prefix)")
    axes[0].set_title("Observation budget", loc="left", fontweight="semibold")
    axes[0].grid(axis="x", color="#E1EAF2", linewidth=0.7); axes[0].set_axisbelow(True)
    axes[0].set_xlim(-0.10, 0.22)
    for yi, value in zip(y, budget_values):
        axes[0].text(0.195, yi, f"{value:+.3f}", ha="left", va="center", fontsize=7.5, color="#142A3B")

    methods = [("F1_command_tfidf", "F1 · command TF-IDF", "#45A6FC"),
               ("F3_command_plus_structure", "F3 · command + structure", "#173F68")]
    for yi, (feature, label, color) in enumerate(methods):
        item = contrasts["order_randomization"][feature]["ordered_minus_shuffle"]
        value = item["estimate"]; ci = item["ci_95"]
        axes[1].barh(yi, value, color=color, height=0.55, edgecolor="white", linewidth=0.8)
        axes[1].errorbar(value, yi, xerr=[[value - ci[0]], [ci[1] - value]], fmt="none",
                         ecolor="#142A3B", capsize=2.5, linewidth=1)
        axes[1].text(0.065, yi, f"{value:+.3f}", ha="left", va="center", fontsize=8, color="#142A3B")
    axes[1].axvline(0, color="#536476", linewidth=0.8)
    axes[1].set_yticks([0, 1], [label for _, label, _ in methods]); axes[1].invert_yaxis()
    axes[1].set_xlabel("BA change (ordered − shuffled)")
    axes[1].set_title("Order randomization · prefix 4", loc="left", fontweight="semibold")
    axes[1].grid(axis="x", color="#E1EAF2", linewidth=0.7); axes[1].set_axisbelow(True)
    axes[1].set_xlim(-0.08, 0.14)
    fig.suptitle("What extra commands and order contribute", x=0.02, ha="left", fontsize=12, fontweight="bold", color="#142A3B")
    fig.text(0.02, 0.005, "Paired task-level differences; 95% intervals bootstrap benchmark-qualified tasks. Prefix comparisons keep the same eligible cohort.", fontsize=7.5, color="#536476")
    fig.tight_layout(rect=(0, 0.04, 1, 0.91), w_pad=1.4)
    FIG.parent.mkdir(parents=True, exist_ok=True); SITE_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(FIG.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(SITE_FIG, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    rows = load_rows()
    contrasts = summarize(rows)
    result = {
        "status": "derived paired contrasts from task-held-out predictions; no model refit",
        "protocol_version": "feature-contrasts-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "results/feature_ablation_task_metrics_v1.csv; task groups matched across conditions",
        "metric": "task-macro balanced accuracy, then equal benchmark weighting",
        "contrasts": contrasts,
        "provenance": {"git_head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                       "input_sha256": hashlib.sha256(TASK_CSV.read_bytes()).hexdigest(),
                       "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                       "bootstrap_unit": "benchmark-qualified task, resampled within benchmark", "draws": 5000},
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    draw(contrasts)
    print(json.dumps(contrasts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
