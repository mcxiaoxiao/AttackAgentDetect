#!/usr/bin/env python3
"""Nested development-only test of rare-sequence and phase-gated association."""
from __future__ import annotations

import collections
import csv
import gzip
import difflib
import hashlib
import io
import json
import math
import pathlib
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT.parent), str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments")]
sys.path.insert(0, str(ROOT / "experiments" / "multidimensional_rerun_v1"))

import run_association_error_analysis as error_eval  # noqa: E402
import run_feature_ablation as split_utils  # noqa: E402
import run_mixed_gallery as gallery  # noqa: E402
import multidimensional_rerun_v1.run as base  # noqa: E402
from multidimensional_rerun_v1.trace_adapter import _load_development  # noqa: E402
from attackagent import traces as trace_source  # noqa: E402

OUT = ROOT / "results" / "balanced_innovation_v2"
SEED = 20261031
BOOTSTRAPS = 5000
GRID_SIZE = 101
BASELINE = "command_tfidf"
NEW_METHODS = ("idf_lcs_coverage", "phase_gated_lcs", "mutual_rank_fusion")
METHODS = (*base.METHODS, *NEW_METHODS)
RANK_METRICS = base.RANK_METRICS
RATE_METRICS = (*error_eval.RATE_METRICS, "balanced_accuracy", "query_has_accepted_match")
SIMPLICITY = {method: i for i, method in enumerate(METHODS)}
ACTION_LABELS = tuple(base.ACTION_LABELS)
ACTION_INDEX = {name: i for i, name in enumerate(ACTION_LABELS)}


def file_hash(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_scoped_development_rows():
    """Read only the frozen development sources and selected Sonnet model."""
    intercode_manifest_path = trace_source.DATA / "download_manifest.json"
    cross_manifest_path = ROOT / "data" / "manifests" / "crossbenchmark_public_subset.json"
    cache_path = pathlib.Path("/tmp/attackagentdetect-balanced-innovation-v1-dev-cache.json")
    source_fingerprint = hashlib.sha256(
        intercode_manifest_path.read_bytes() + cross_manifest_path.read_bytes()
    ).hexdigest()
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_schema") == 2 and cached.get("source_fingerprint") == source_fingerprint:
            print("loaded scoped development observation cache", flush=True)
            return [
                trace_source.Observation(
                    row["trace_id"], row["task"], row["label"], row["source"],
                    tuple(row["names"]), tuple(row["groups"]),
                    row.get("skipped_python_calls", 0), row["benchmark"],
                )
                for row in cached["rows"]
            ]
    observations = []
    source_manifest = json.loads(intercode_manifest_path.read_text(encoding="utf-8"))
    intercode_evalset = "eval-set-83d0a1kx3i5t3u6b"
    for record in source_manifest["files"]:
        relative = record["path"]
        path = trace_source.DATA / relative
        if "/human/eval_logs/intercode-ctf/" in f"/{relative}":
            print(f"reading development source: {pathlib.PurePosixPath(relative).name}", flush=True)
            with gzip.open(path, "rb") as handle:
                for member, sample in trace_source.samples(io.BytesIO(handle.read())):
                    logs = sample.get("store", {}).get("HumanAgentState:logs", {})
                    lines = []
                    for key, value in sorted(logs.items()):
                        if key.endswith(".input"):
                            lines.extend(trace_source.terminal_lines(value))
                    names = trace_source.command_heads("\n".join(lines))
                    trace_id = hashlib.sha256((relative + "/" + member).encode()).hexdigest()[:16]
                    observations.append(trace_source.Observation(
                        trace_id, str(sample["id"]), "human", "expert", tuple(names),
                        tuple(trace_source.ATOM_GROUP[name] for name in names),
                        benchmark="intercode-ctf",
                    ))
        elif pathlib.PurePosixPath(relative).parts[2:3] == (intercode_evalset,):
            model = source_manifest["model_sets"][path.parent.name]
            if model != "claude-sonnet-4.6":
                continue
            print(f"reading development source: {pathlib.PurePosixPath(relative).name}", flush=True)
            for member, sample in trace_source.samples(path):
                names, skipped = [], 0
                for message in sample.get("messages", []):
                    for call in (message.get("tool_calls", []) or []):
                        arguments = call.get("arguments", {})
                        if call.get("function") == "bash" and isinstance(arguments.get("cmd"), str):
                            names.extend(trace_source.command_heads(arguments["cmd"]))
                        elif call.get("function") == "python":
                            skipped += 1
                trace_id = hashlib.sha256((relative + "/" + member).encode()).hexdigest()[:16]
                observations.append(trace_source.Observation(
                    trace_id, str(sample["id"]), "agent", model, tuple(names),
                    tuple(trace_source.ATOM_GROUP[name] for name in names), skipped,
                    "intercode-ctf",
                ))

    cross_manifest = json.loads(cross_manifest_path.read_text(encoding="utf-8"))
    latest_agents = {}
    for record in cross_manifest["files"]:
        benchmark = record.get("benchmark")
        if benchmark not in {"cybench", "nyuctf"}:
            continue
        path = trace_source.DATA / record["path"]
        if record["kind"] == "human":
            print(f"reading development source: {pathlib.PurePosixPath(record['path']).name}", flush=True)
            with gzip.open(path, "rb") as handle:
                for member, sample in trace_source.samples(io.BytesIO(handle.read())):
                    logs = sample.get("store", {}).get("HumanAgentState:logs", {})
                    lines = []
                    for key, value in sorted(logs.items()):
                        if key.endswith(".input"):
                            lines.extend(trace_source.terminal_lines(value))
                    names = trace_source.command_heads("\n".join(lines))
                    trace_id = hashlib.sha256((record["path"] + "/" + member).encode()).hexdigest()[:16]
                    observations.append(trace_source.Observation(
                        trace_id, f"{benchmark}/{record['task_id']}", "human", "expert",
                        tuple(names), tuple(trace_source.ATOM_GROUP[name] for name in names),
                        0, benchmark,
                    ))
        elif record.get("model") == "claude-sonnet-4.6":
            print(f"reading development source: {pathlib.PurePosixPath(record['path']).name}", flush=True)
            for member, sample in trace_source.samples(path):
                raw_task = str(sample.get("id", "")).strip().split("/")[-1]
                if not raw_task:
                    continue
                key = (benchmark, raw_task)
                order = record.get("attempt_filename", pathlib.PurePosixPath(record["path"]).name)
                if key not in latest_agents or order > latest_agents[key][0]:
                    latest_agents[key] = (order, record, member, sample)

    for (benchmark, raw_task), (_order, record, member, sample) in sorted(latest_agents.items()):
        names, skipped = [], 0
        for message in sample.get("messages", []):
            for call in message.get("tool_calls", []) or []:
                arguments = call.get("arguments", {}) or {}
                if call.get("function") == "bash" and isinstance(arguments.get("cmd"), str):
                    names.extend(trace_source.command_heads(arguments["cmd"]))
                elif call.get("function") == "python":
                    skipped += 1
        trace_id = hashlib.sha256((record["path"] + "/" + member).encode()).hexdigest()[:16]
        observations.append(trace_source.Observation(
            trace_id, f"{benchmark}/{raw_task}", "agent", "claude-sonnet-4.6",
            tuple(names), tuple(trace_source.ATOM_GROUP[name] for name in names), skipped,
            benchmark,
        ))
    cache_path.write_text(json.dumps({
        "cache_schema": 2,
        "source_fingerprint": source_fingerprint,
        "rows": [row.__dict__ for row in observations],
    }, ensure_ascii=False), encoding="utf-8")
    return observations


def idf_weights(rows: list) -> dict[str, float]:
    document_frequency = collections.Counter()
    for row in rows:
        document_frequency.update(set(row.names))
    n = max(1, len(rows))
    return {
        token: math.log((n + 1) / (count + 1)) + 1.0
        for token, count in document_frequency.items()
    }


def weighted_jaccard(left, right, weights: dict[str, float]) -> float:
    keys = left.keys() | right.keys()
    denominator = numerator = 0.0
    for token in keys:
        weight = weights.get(token, 1.0)
        a, b = left.get(token, 0), right.get(token, 0)
        numerator += min(a, b) * weight
        denominator += max(a, b) * weight
    return numerator / denominator if denominator else 0.0


def weighted_lcs_coverage(a: tuple[str, ...], b: tuple[str, ...], weights) -> float:
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    matched = sum(
        sum(weights.get(token, 1.0) for token in a[block.a:block.a + block.size])
        for block in matcher.get_matching_blocks()
    )
    shorter_mass = min(
        sum(weights.get(token, 1.0) for token in a),
        sum(weights.get(token, 1.0) for token in b),
    )
    return matched / shorter_mass if shorter_mass else 0.0


def phase_action_profile(row, bins: int = 4) -> np.ndarray:
    actions = row.groups
    profile = np.zeros((bins, len(ACTION_LABELS)), dtype=float)
    for phase in range(bins):
        start = len(actions) * phase // bins
        end = len(actions) * (phase + 1) // bins
        for action in actions[start:end]:
            if action in ACTION_INDEX:
                profile[phase, ACTION_INDEX[action]] += 1.0
        norm = np.linalg.norm(profile[phase])
        if norm:
            profile[phase] /= norm
    return profile


def phase_similarity(left: np.ndarray, right: np.ndarray) -> float:
    # Empty phases count as no evidence, not a perfect match.
    values = []
    for a, b in zip(left, right, strict=True):
        if np.linalg.norm(a) and np.linalg.norm(b):
            values.append(float(np.dot(a, b)))
        else:
            values.append(0.0)
    return float(np.mean(values)) if values else 0.0


def add_innovative_scores(pair_rows: list[dict], train_rows: list, all_rows: list) -> None:
    weights = idf_weights(train_rows)
    by_id = {row.trace_id: row for row in all_rows}
    phase = {row.trace_id: phase_action_profile(row) for row in all_rows}
    command_counts = {row.trace_id: collections.Counter(row.names) for row in all_rows}
    bigram_counts = {
        row.trace_id: collections.Counter(zip(row.names, row.names[1:]))
        for row in all_rows
    }
    for pair in pair_rows:
        left = by_id[pair["query_trace_id"]]
        right = by_id[pair["candidate_trace_id"]]
        coverage = weighted_lcs_coverage(left.names, right.names, weights)
        ratio = min(len(left.names), len(right.names)) / max(1, max(len(left.names), len(right.names)))
        length_adjusted = coverage * (0.70 + 0.30 * math.sqrt(ratio))
        command_overlap = weighted_jaccard(
            command_counts[left.trace_id], command_counts[right.trace_id], weights
        )
        bigram_overlap = weighted_jaccard(
            bigram_counts[left.trace_id], bigram_counts[right.trace_id], weights
        )
        local_score = 0.55 * length_adjusted + 0.25 * command_overlap + 0.20 * bigram_overlap
        stage_score = phase_similarity(phase[left.trace_id], phase[right.trace_id])
        pair["idf_lcs_coverage"] = float(np.clip(local_score, 0.0, 1.0))
        pair["phase_gated_lcs"] = float(np.sqrt(max(0.0, local_score * stage_score)))


def add_mutual_rank_fusion(pair_rows: list[dict]) -> None:
    """Fuse query-relative ranks across profile modalities, then require reciprocity."""
    modalities = ("command_tfidf", "profile_24d", "tensor_4x8x3", "partial_sequence")
    by_query = collections.defaultdict(list)
    for row in pair_rows:
        by_query[(row["benchmark"], row["query_trace_id"])].append(row)

    directional_rrf = {}
    for (benchmark, query_trace_id), rows in by_query.items():
        ranks = {}
        for method in modalities:
            ordered = sorted(rows, key=lambda row: (-float(row[method]), row["candidate_trace_id"]))
            method_ranks = {}
            index = 0
            while index < len(ordered):
                end = index + 1
                score = float(ordered[index][method])
                while end < len(ordered) and float(ordered[end][method]) == score:
                    end += 1
                average_rank = (index + 1 + end) / 2.0
                for tied in ordered[index:end]:
                    method_ranks[tied["candidate_trace_id"]] = average_rank
                index = end
            ranks[method] = method_ranks
        for row in rows:
            candidate_id = row["candidate_trace_id"]
            fused = sum(1.0 / (5.0 + ranks[method][candidate_id]) for method in modalities)
            directional_rrf[(benchmark, query_trace_id, candidate_id)] = fused

    for row in pair_rows:
        benchmark = row["benchmark"]
        query_id = row["query_trace_id"]
        candidate_id = row["candidate_trace_id"]
        forward = directional_rrf[(benchmark, query_id, candidate_id)]
        reverse = directional_rrf[(benchmark, candidate_id, query_id)]
        row["mutual_rank_fusion"] = float(np.sqrt(forward * reverse))


def task_rank_rows(pair_rows: list[dict], methods: tuple[str, ...]) -> list[dict]:
    by_query = collections.defaultdict(list)
    for row in pair_rows:
        by_query[(row["benchmark"], row["query_task"], row["query_trace_id"])].append(row)
    task_values = collections.defaultdict(list)
    for (benchmark, task, _query_id), rows in by_query.items():
        relevance = np.asarray(
            [int(row["same_task_proxy_for_evaluation_only"]) for row in rows], dtype=int
        )
        if not relevance.any():
            continue
        for method in methods:
            metrics = gallery.expected_rank_metrics([row[method] for row in rows], relevance, k=5)
            if metrics is not None:
                task_values[(benchmark, task, method)].append(metrics)
    return [
        {
            "benchmark": benchmark,
            "task": task,
            "method": method,
            **{
                key: float(np.mean([row[key] for row in values]))
                for key in RANK_METRICS
            },
            "query_count": len(values),
        }
        for (benchmark, task, method), values in sorted(task_values.items())
    ]


def macro(rows: list[dict], metric: str) -> float | None:
    return error_eval._macro_metric(rows, metric)[0]


def policy_values(pair_rows, ranks, method, threshold):
    rates = error_eval._query_rates(pair_rows, method, float(threshold))
    coverage = {
        (row["benchmark"], row["task"]): row["query_has_accepted_match"]
        for row in base.query_coverage(pair_rows, method, float(threshold))
    }
    for row in rates:
        row["query_has_accepted_match"] = coverage.get((row["benchmark"], row["task"]))
    balanced_accuracy = macro(rates, "balanced_accuracy")
    rank_values = [macro(ranks_for_method(ranks, method), metric) for metric in RANK_METRICS]
    rank_quality = float(np.mean([value for value in rank_values if value is not None]))
    query_coverage = macro(rates, "query_has_accepted_match")
    utility = float(np.mean([balanced_accuracy, rank_quality, query_coverage]))
    return {
        "false_split_rate": macro(rates, "false_split_rate"),
        "false_merge_rate": macro(rates, "false_merge_rate"),
        "balanced_accuracy": balanced_accuracy,
        "query_coverage": query_coverage,
        **{metric: value for metric, value in zip(RANK_METRICS, rank_values, strict=True)},
        "rank_quality": rank_quality,
        "balanced_utility": utility,
    }


def ranks_for_method(rank_rows, method):
    return [row for row in rank_rows if row["method"] == method]


def select_policies(pair_rows: list[dict], rank_rows: list[dict]) -> dict[str, dict]:
    policies = {}
    for method in METHODS:
        scores = np.asarray([row[method] for row in pair_rows], dtype=float)
        thresholds = np.unique(np.r_[
            np.nextafter(scores.min(), -np.inf),
            np.quantile(scores, np.linspace(0.0, 1.0, GRID_SIZE)),
            np.nextafter(scores.max(), np.inf),
        ])
        candidates = []
        for threshold in thresholds:
            values = policy_values(pair_rows, rank_rows, method, float(threshold))
            candidates.append((values["balanced_utility"], values["balanced_accuracy"],
                               -values["false_merge_rate"], -values["false_split_rate"],
                               values["ndcg_at_5"], float(threshold), values))
        best = max(candidates, key=lambda item: item[:-1])
        policies[method] = {
            "threshold": best[5],
            "training_metrics": best[6],
            "candidate_threshold_count": len(candidates),
            "selection_rule": "maximum equal-weight mean of balanced accuracy, four-metric ranking mean, and query coverage",
        }
    return policies


def select_method(policies: dict[str, dict]) -> tuple[str, dict]:
    method = max(
        METHODS,
        key=lambda name: (
            policies[name]["training_metrics"]["balanced_utility"],
            policies[name]["training_metrics"]["balanced_accuracy"],
            -policies[name]["training_metrics"]["false_merge_rate"],
            -policies[name]["training_metrics"]["false_split_rate"],
            policies[name]["training_metrics"]["ndcg_at_5"],
            -SIMPLICITY[name],
        ),
    )
    return method, policies[method]


def evaluate_fold(pair_rows, ranks, policies, outer_fold, methods=METHODS):
    rate_rows = []
    for method in methods:
        threshold = policies[method]["threshold"]
        rates = error_eval._query_rates(pair_rows, method, threshold)
        coverage = {
            (row["benchmark"], row["task"]): row["query_has_accepted_match"]
            for row in base.query_coverage(pair_rows, method, threshold)
        }
        for row in rates:
            rate_rows.append({
                **row,
                "method": method,
                "threshold": float(threshold),
                "outer_fold": outer_fold,
                "query_has_accepted_match": coverage.get((row["benchmark"], row["task"])),
            })
    rank_rows = [
        {**row, "outer_fold": outer_fold}
        for row in ranks
    ]
    return rate_rows, rank_rows


def bootstrap_macro(rows: list[dict], metric: str, seed: int, draws: int = BOOTSTRAPS):
    grouped = collections.defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None:
            grouped[row["benchmark"]].append(float(value))
    if not grouped:
        return {"estimate": None, "ci95": None, "task_count": 0}
    estimate = float(np.mean([np.mean(values) for values in grouped.values()]))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        benchmark_means = []
        for values in grouped.values():
            selected = rng.choice(values, size=len(values), replace=True)
            benchmark_means.append(float(np.mean(selected)))
        samples[draw] = float(np.mean(benchmark_means))
    return {
        "estimate": estimate,
        "ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "task_count": sum(len(values) for values in grouped.values()),
        "bootstrap_draws": draws,
    }


def paired_delta(rate_rows, rank_rows, metric, method, baseline, seed):
    rows = rate_rows if metric in RATE_METRICS else rank_rows
    return base._paired_delta(rows, metric, method, baseline, BOOTSTRAPS, seed)


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_plots(results: dict, rate_rows: list[dict], rank_rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial Unicode MS", "Hiragino Sans GB", "STHeiti", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    colors = {
        "command_tfidf": "#0072B2",
        "profile_24d": "#E69F00",
        "tensor_4x8x3": "#009E73",
        "tensor_tucker": "#CC79A7",
        "partial_sequence": "#56B4E9",
        "learned_fusion": "#D55E00",
        "idf_lcs_coverage": "#000000",
        "phase_gated_lcs": "#9467BD",
        "mutual_rank_fusion": "#B279A2",
    }
    labels = {
        "command_tfidf": ("Command TF–IDF", "命令 TF–IDF"),
        "profile_24d": ("24D profile", "24 维画像"),
        "tensor_4x8x3": ("4×8×3 tensor", "4×8×3 张量"),
        "tensor_tucker": ("Tucker tensor", "Tucker 张量"),
        "partial_sequence": ("IDF partial sequence", "IDF 局部序列"),
        "learned_fusion": ("Learned fusion", "学习式融合"),
        "idf_lcs_coverage": ("IDF-LCS", "IDF-LCS"),
        "phase_gated_lcs": ("Phase-gated LCS", "阶段门控 LCS"),
        "mutual_rank_fusion": ("Mutual rank fusion", "互惠多视角排序融合"),
        "balanced_selected": ("Selected per fold", "逐折入选策略"),
    }
    core_methods = [*METHODS, "balanced_selected"]
    ci = {}
    for index, method in enumerate(core_methods):
        subset = [row for row in rate_rows if row["method"] == method]
        ci[method] = {
            metric: bootstrap_macro(subset, metric, SEED + 200 + index * 7 + offset)
            for offset, metric in enumerate(("false_merge_rate", "false_split_rate"))
        }

    for lang in ("en",):
        english = lang == "en"
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), gridspec_kw={"width_ratios": [1.02, 1.25]})
        fig.patch.set_facecolor("white")
        ax = axes[0]
        legend_handles = []
        for method in core_methods:
            rates = [row for row in rate_rows if row["method"] == method]
            fmr = bootstrap_macro(rates, "false_merge_rate", SEED + 100 + SIMPLICITY.get(method, 9))
            fsr = bootstrap_macro(rates, "false_split_rate", SEED + 150 + SIMPLICITY.get(method, 9))
            if fmr["estimate"] is None or fsr["estimate"] is None:
                continue
            x, y = fmr["estimate"] * 100, fsr["estimate"] * 100
            xerr = np.array([[x - fmr["ci95"][0] * 100], [fmr["ci95"][1] * 100 - x]])
            yerr = np.array([[y - fsr["ci95"][0] * 100], [fsr["ci95"][1] * 100 - y]])
            marker = "*" if method == "balanced_selected" else "o"
            size = 140 if method == "balanced_selected" else 62
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker, ms=9 if marker == "*" else 6,
                        color=colors.get(method, "#333333"), ecolor=colors.get(method, "#333333"),
                        elinewidth=1.2, capsize=3, alpha=.95, zorder=4 if marker == "*" else 2)
            legend_handles.append(Line2D(
                [], [], color=colors.get(method, "#333333"), marker=marker, linestyle="none",
                markersize=7 if marker == "*" else 5,
                label=labels[method][1 if english else 0],
            ))
        ax.set_xlim(65, 105)
        ax.set_ylim(-3, 40)
        ax.set_xlabel("False-merge rate FMR (%)" if english else "误合并率 FMR（%）", fontsize=10)
        ax.set_ylabel("Missed-link rate FSR (%)" if english else "漏关联率 FSR（%）", fontsize=10)
        ax.set_title("Error trade-off" if english else "误差权衡", loc="left", fontsize=12, weight="bold")
        ax.grid(True, color="#dbe3ea", linewidth=.7, alpha=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(.02, .02, "lower-left is better" if english else "左下角较好", transform=ax.transAxes,
                fontsize=8, color="#526575")
        fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(.5, .015),
                   ncol=3, frameon=False, fontsize=7.5, columnspacing=1.5,
                   handletextpad=.5)

        ax = axes[1]
        metrics = [
            ("false_merge_rate", "False merges" if english else "误合并"),
            ("false_split_rate", "Missed links" if english else "漏关联"),
            ("false_merge_human_agent", "H–A false merges" if english else "H–A 误合并"),
            ("false_split_human_agent", "H–A missed links" if english else "H–A 漏关联"),
            ("balanced_accuracy", "Balanced accuracy" if english else "平衡准确率"),
            ("query_has_accepted_match", "Query coverage" if english else "查询覆盖"),
            ("hit_at_1", "Hit@1"), ("mrr", "MRR"), ("recall_at_5", "Recall@5"), ("ndcg_at_5", "NDCG@5"),
        ]
        selected_rates = [row for row in rate_rows if row["method"] == "balanced_selected"]
        baseline_rates = [row for row in rate_rows if row["method"] == BASELINE]
        selected_ranks = [row for row in rank_rows if row["method"] == "balanced_selected"]
        baseline_ranks = [row for row in rank_rows if row["method"] == BASELINE]
        deltas = []
        for index, (metric, label) in enumerate(metrics):
            if metric in RATE_METRICS:
                selected, baseline = selected_rates, baseline_rates
            else:
                selected, baseline = selected_ranks, baseline_ranks
            delta = base._paired_delta([*selected, *baseline],
                                       metric, "balanced_selected" if metric in RATE_METRICS else "balanced_selected",
                                       BASELINE, BOOTSTRAPS, SEED + 500 + index)
            # Error rates are reversed so positive always means improvement.
            estimate = float(delta["estimate"])
            interval = delta.get("ci_95")
            if metric in error_eval.RATE_METRICS:
                estimate = -estimate
                interval = [-interval[1], -interval[0]] if interval else None
            deltas.append((label, estimate * 100, [x * 100 for x in interval] if interval else None))
        deltas.reverse()
        y_positions = np.arange(len(deltas))
        for y, (label, estimate, interval) in zip(y_positions, deltas, strict=True):
            if interval:
                ax.errorbar(estimate, y, xerr=[[estimate - interval[0]], [interval[1] - estimate]],
                            fmt="o", color="#0072B2", ecolor="#0072B2", capsize=2.5, markersize=4)
            else:
                ax.plot(estimate, y, "o", color="#0072B2", markersize=4)
        ax.axvline(0, color="#465867", linewidth=1, linestyle="--")
        ax.set_yticks(y_positions, [item[0] for item in deltas], fontsize=8)
        ax.set_xlabel("Direction-adjusted change (percentage points)" if english else "方向校正变化（百分点）", fontsize=9)
        ax.set_title("Selected policy vs command baseline" if english else "逐折入选策略 vs 命令基线", loc="left", fontsize=12, weight="bold")
        ax.grid(axis="x", color="#dbe3ea", linewidth=.7, alpha=.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        fig.suptitle(
            "Exploratory, task-held-out development results · 40 task groups · 95% task-cluster bootstrap"
            if english else "探索性任务留出结果 · 40 个任务组 · 95% 任务簇 Bootstrap",
            x=.02, ha="left", fontsize=9, color="#526575",
        )
        fig.tight_layout(rect=(0, .12, 1, .92), w_pad=3.2)
        suffix = "en" if english else "zh"
        fig.savefig(OUT / f"balanced-innovation-v2-{suffix}.png", dpi=260, bbox_inches="tight", facecolor="white")
        fig.savefig(OUT / "balanced-innovation-v2.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)


def run() -> dict:
    # The broad gallery loader parses every benchmark/model before filtering.
    # This scoped adapter preserves the cohort while avoiding the locked external data.
    gallery.load_rows = load_scoped_development_rows
    development, _by_benchmark, _unmatched = _load_development()
    task_keys = {(row.benchmark, row.task) for row in development}
    if len(development) != 97 or len(task_keys) != 40:
        raise RuntimeError(f"Expected the frozen 97-trace / 40-task development cohort; got {len(development)} / {len(task_keys)}")

    outer_folds = split_utils.grouped_folds(development, SEED, n_folds=3)
    all_pair_rows, all_rate_rows, all_rank_rows = [], [], []
    fold_audit = []
    threshold_rows = []

    for outer_index, (train_keys, test_keys) in enumerate(outer_folds, start=1):
        outer_train = [row for row in development if (row.benchmark, row.task) in train_keys]
        outer_test = [row for row in development if (row.benchmark, row.task) in test_keys]
        inner_oof = []
        inner_folds = split_utils.grouped_folds(outer_train, SEED + outer_index * 100, n_folds=3)
        for inner_index, (inner_train_keys, inner_test_keys) in enumerate(inner_folds, start=1):
            inner_train = [row for row in outer_train if (row.benchmark, row.task) in inner_train_keys]
            inner_test = [row for row in outer_train if (row.benchmark, row.task) in inner_test_keys]
            scorer, _ = base.fit_methods(inner_train, inner_test, SEED + outer_index * 1000 + inner_index)
            pairs = base.make_pairs(inner_test, scorer, fold=outer_index, split="balanced_inner_oof")
            add_innovative_scores(pairs, inner_train, [*inner_train, *inner_test])
            add_mutual_rank_fusion(pairs)
            inner_oof.extend(pairs)
        inner_ranks = task_rank_rows(inner_oof, METHODS)
        policies = select_policies(inner_oof, inner_ranks)
        selected_method, selected_policy = select_method(policies)

        scorer, dimensions = base.fit_methods(outer_train, outer_test, SEED + outer_index)
        outer_pairs = base.make_pairs(outer_test, scorer, fold=outer_index, split="development_outer_test")
        add_innovative_scores(outer_pairs, outer_train, [*outer_train, *outer_test])
        add_mutual_rank_fusion(outer_pairs)
        outer_ranks = task_rank_rows(outer_pairs, METHODS)

        # Save each preselected method/threshold, plus the method selected inside the training fold.
        for row in outer_pairs:
            enriched = dict(row)
            enriched["outer_fold"] = outer_index
            for method in METHODS:
                enriched[f"threshold_{method}"] = policies[method]["threshold"]
                enriched[f"accepted_{method}"] = int(row[method] >= policies[method]["threshold"])
            enriched["selected_method"] = selected_method
            enriched["selected_score"] = row[selected_method]
            enriched["selected_threshold"] = selected_policy["threshold"]
            enriched["selected_accepted"] = int(row[selected_method] >= selected_policy["threshold"])
            enriched["balanced_selected"] = row[selected_method]
            all_pair_rows.append(enriched)

        fold_rates, fold_ranks = evaluate_fold(outer_pairs, outer_ranks, policies, outer_index)
        all_rate_rows.extend(fold_rates)
        all_rank_rows.extend(fold_ranks)

        selected_rows = [dict(row, balanced_selected=row[selected_method]) for row in outer_pairs]
        selected_policy_by_method = dict(policies)
        selected_policy_by_method["balanced_selected"] = selected_policy
        selected_rates, selected_ranks = evaluate_fold(
            selected_rows, task_rank_rows(selected_rows, ("balanced_selected",)),
            {"balanced_selected": selected_policy}, outer_index, methods=("balanced_selected",),
        )
        all_rate_rows.extend(selected_rates)
        all_rank_rows.extend(selected_ranks)

        for method, policy in policies.items():
            threshold_rows.append({
                "outer_fold": outer_index,
                "method": method,
                "threshold": float(policy["threshold"]),
                "inner_balanced_utility": policy["training_metrics"]["balanced_utility"],
                "inner_balanced_accuracy": policy["training_metrics"]["balanced_accuracy"],
                "inner_fmr": policy["training_metrics"]["false_merge_rate"],
                "inner_fsr": policy["training_metrics"]["false_split_rate"],
            })
        fold_audit.append({
            "outer_fold": outer_index,
            "train_task_groups": len(train_keys),
            "test_task_groups": len(test_keys),
            "train_traces": len(outer_train),
            "test_traces": len(outer_test),
            "selected_method": selected_method,
            "selected_threshold": float(selected_policy["threshold"]),
            "selected_inner_metrics": selected_policy["training_metrics"],
            "candidate_policies": policies,
            "fitted_dimensions": dimensions,
        })
        print(f"outer fold {outer_index}/3: selected {selected_method}; test tasks={len(test_keys)}", flush=True)

    methods_for_summary = (*METHODS, "balanced_selected")
    metric_summary = {}
    for method_index, method in enumerate(methods_for_summary):
        rates = [row for row in all_rate_rows if row["method"] == method]
        ranks = [row for row in all_rank_rows if row["method"] == method]
        metric_summary[method] = {
            metric: bootstrap_macro(
                rates if metric in RATE_METRICS else ranks,
                metric,
                SEED + method_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate((*RATE_METRICS, *RANK_METRICS))
        }

    selected_deltas = {
        metric: paired_delta(
            all_rate_rows,
            all_rank_rows,
            metric,
            "balanced_selected",
            BASELINE,
            SEED + 500 + index,
        )
        for index, metric in enumerate((*RATE_METRICS, *RANK_METRICS))
    }
    improvements = {}
    for metric, result in selected_deltas.items():
        estimate = result["estimate"]
        improvements[metric] = (
            None if estimate is None
            else float(-estimate if metric in error_eval.RATE_METRICS else estimate)
        )
    estimable_improvements = [value for value in improvements.values() if value is not None]

    selected_counts = collections.Counter(row["selected_method"] for row in fold_audit)
    result = {
        "status": "completed exploratory nested task-held-out development evaluation",
        "created_utc": "2026-10-31",
        "protocol": "experiments/balanced_innovation_v2/program.md",
        "dataset": {
            "records": len(development),
            "task_groups": len(task_keys),
            "benchmarks": sorted({row.benchmark for row in development}),
            "revision": "178c0da1800f9c07311e1b8d6c61dd8104e7b4ae",
            "label_scope": "same task retrieval proxy, not actor or organization identity",
            "reuse_boundary": "The same task groups were inspected by earlier experiments; results are exploratory, not independent confirmation.",
            "locked_external_data_read": False,
        },
        "selection_rule": "equal-weight mean of task-macro balanced accuracy, mean rank metric, and query coverage; thresholds and methods selected on inner OOF tasks only",
        "new_methods": {
            "idf_lcs_coverage": "IDF-weighted command sequence alignment coverage plus weighted command and bigram overlap, with mild length-mismatch penalty",
            "phase_gated_lcs": "geometric mean of IDF-LCS coverage and four-phase action-profile similarity",
            "mutual_rank_fusion": "query-relative reciprocal-rank fusion over command, 24D profile, 4x8x3 tensor, and local-sequence scores; symmetrized by geometric mean of both directional fused ranks",
        },
        "selected_method_by_fold": dict(selected_counts),
        "outer_folds": fold_audit,
        "metrics": metric_summary,
        "paired_deltas_vs_command": selected_deltas,
        "direction_adjusted_selected_changes": improvements,
        "all_estimable_metrics_pointwise_nonworse": bool(
            estimable_improvements
            and all(value >= -1e-12 for value in estimable_improvements)
            and any(value > 1e-12 for value in estimable_improvements)
        ),
        "provenance": {
            "runner_sha256": file_hash(pathlib.Path(__file__)),
            "base_runner_sha256": file_hash(ROOT / "experiments" / "multidimensional_rerun_v1" / "run.py"),
            "source_manifest_sha256": file_hash(ROOT / "experiments" / "multidimensional_rerun_v1" / "source_manifest.json"),
            "seed": SEED,
            "outer_folds": 3,
            "inner_folds": 3,
            "bootstrap_draws": BOOTSTRAPS,
        },
        "interpretation": "No result from this reused development cohort is an independent confirmation. All indicators and task-cluster intervals must be read together; a low false-split rate alone is not a success criterion.",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(OUT / "outer_pair_scores.csv", all_pair_rows)
    write_csv(OUT / "task_metrics.csv", [*all_rate_rows, *all_rank_rows])
    write_csv(OUT / "thresholds.csv", threshold_rows)
    make_plots(result, all_rate_rows, all_rank_rows)
    print(f"Wrote balanced development results to {OUT.relative_to(ROOT)}", flush=True)
    print(json.dumps({
        "selected_method_by_fold": result["selected_method_by_fold"],
        "direction_adjusted_selected_changes": improvements,
        "all_estimable_metrics_pointwise_nonworse": result["all_estimable_metrics_pointwise_nonworse"],
    }, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    run()


if __name__ == "__main__":
    main()
