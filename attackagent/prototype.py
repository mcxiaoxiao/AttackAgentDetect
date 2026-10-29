"""Research prototype for mode-aware behavior matching.

This prototype is intentionally trained on labeled traces supplied by the
caller. Its match score estimates *same benchmark activity/task* under the
training distribution; it is not a campaign or APT attribution probability.
"""

from __future__ import annotations

import collections
import itertools
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class TraceRecord:
    """Normalized trace input. Commands/actions should be ordered observations."""

    trace_id: str
    benchmark: str
    activity_id: str
    mode: str  # currently supported training labels: "human" and "agent"
    commands: tuple[str, ...]
    actions: tuple[str, ...]
    source: str = "unknown"  # model/toolchain/operator identifier, if known


def _sample(values, n, rng):
    values = list(values)
    if len(values) <= n:
        return values
    return [values[i] for i in rng.choice(len(values), size=n, replace=False)]


def _cosine_rows(matrix, pairs):
    if not pairs:
        return np.empty(0, dtype=float)
    left = np.asarray([a for a, _ in pairs], dtype=int)
    right = np.asarray([b for _, b in pairs], dtype=int)
    return np.asarray(matrix[left].multiply(matrix[right]).sum(axis=1)).ravel()


class ModeAwareAssociationPrototype:
    """Two-head baseline: execution-mode score plus a mode-conditioned match score.

    Fit only on the training split when evaluating. `match_score` is a ranking
    score from a class-balanced pair model, not a prevalence-calibrated
    probability. The current public data support only human-vs-Agent labels;
    scripts and unknown modes must be treated as unsupported/abstain.
    """

    def __init__(self, seed=20261029, max_pairs_per_task=16):
        self.seed = int(seed)
        self.max_pairs_per_task = int(max_pairs_per_task)
        self._fitted = False

    @staticmethod
    def _text(records, attr):
        return [" ".join(getattr(row, attr)) for row in records]

    @staticmethod
    def _fit_mode(records, train_indices, predict_indices, seed):
        texts = [" ".join(row.commands) for row in records]
        vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=1)
        vec.fit([texts[i] for i in train_indices])
        X_train = vec.transform([texts[i] for i in train_indices])
        X_predict = vec.transform([texts[i] for i in predict_indices])
        y = np.asarray([int(records[i].mode == "agent") for i in train_indices])
        if len(np.unique(y)) != 2:
            raise ValueError("Mode training requires both human and agent labels")
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=seed)
        clf.fit(X_train, y)
        return clf.predict_proba(X_predict)[:, 1], vec, clf

    def _crossfit_mode(self, records):
        tasks = np.asarray([f"{r.benchmark}/{r.activity_id}" for r in records])
        task_ids = sorted(set(tasks))
        n_splits = min(4, len(task_ids))
        if n_splits < 2:
            raise ValueError("Mode OOF scores require at least two activity groups")
        output = np.full(len(records), np.nan)
        splitter = GroupKFold(n_splits=n_splits)
        for fold, (train_pos, test_pos) in enumerate(splitter.split(np.zeros(len(records)), groups=tasks)):
            scores, _, _ = self._fit_mode(records, train_pos, test_pos, self.seed + fold + 1)
            output[test_pos] = scores
        if np.isnan(output).any():
            raise AssertionError("Missing cross-fitted mode scores")
        return output

    @staticmethod
    def _pair_types(records, i, j):
        modes = {records[i].mode, records[j].mode}
        if modes == {"human"}:
            return "human-human"
        if modes == {"agent"}:
            return "agent-agent"
        return "human-agent"

    def _training_pairs(self, records):
        rng = np.random.default_rng(self.seed)
        by_group = collections.defaultdict(lambda: {"human": [], "agent": []})
        for i, row in enumerate(records):
            if row.mode not in ("human", "agent"):
                continue
            by_group[(row.benchmark, row.activity_id)][row.mode].append(i)
        by_benchmark = collections.defaultdict(list)
        for key in by_group:
            by_benchmark[key[0]].append(key)

        pairs = []
        seen = set()

        def add(a, b, label, group, relation):
            key = (min(a, b), max(a, b), label)
            if key not in seen:
                seen.add(key)
                pairs.append((key[0], key[1], label, group, relation))

        relations = ("human-human", "human-agent", "agent-agent")
        for group, modes in sorted(by_group.items()):
            benchmark, activity = group
            for relation in relations:
                if relation == "human-human":
                    positives = list(itertools.combinations(modes["human"], 2))
                elif relation == "human-agent":
                    positives = list(itertools.product(modes["human"], modes["agent"]))
                else:
                    agents = modes["agent"]
                    positives = [(i, j) for i, j in itertools.combinations(agents, 2)
                                 if records[i].source != records[j].source]
                positives = _sample(positives, self.max_pairs_per_task, rng)
                if not positives:
                    continue
                for a, b in positives:
                    add(a, b, 1, group, relation)

                negatives = []
                for other in by_benchmark[benchmark]:
                    if other == group:
                        continue
                    other_modes = by_group[other]
                    if relation == "human-human":
                        candidates = itertools.product(modes["human"], other_modes["human"])
                    elif relation == "human-agent":
                        candidates = itertools.chain(
                            itertools.product(modes["human"], other_modes["agent"]),
                            itertools.product(other_modes["human"], modes["agent"]),
                        )
                    else:
                        candidates = ((i, j) for i, j in itertools.product(modes["agent"], other_modes["agent"])
                                      if records[i].source != records[j].source)
                    negatives.extend(candidates)
                negatives = _sample(negatives, len(positives), rng)
                for a, b in negatives:
                    add(a, b, 0, group, relation)

        if not pairs or len({p[2] for p in pairs}) < 2:
            raise ValueError("Could not form both same-activity and different-activity training pairs")
        return pairs

    def _representations(self, records, train_indices):
        command_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 1), min_df=1)
        action_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(2, 2), min_df=1)
        command_text = self._text(records, "commands")
        action_text = self._text(records, "actions")
        command_vec.fit([command_text[i] for i in train_indices])
        command_matrix = command_vec.transform(command_text)
        try:
            action_vec.fit([action_text[i] for i in train_indices])
            action_matrix = action_vec.transform(action_text)
        except ValueError:
            from scipy.sparse import csr_matrix
            action_matrix = csr_matrix((len(records), 0), dtype=float)
        self.command_vectorizer_ = command_vec
        self.action_vectorizer_ = action_vec
        self.command_matrix_ = command_matrix
        self.action_matrix_ = action_matrix

    @staticmethod
    def _pair_features(records, pairs, agent_probs, command_matrix, action_matrix, include_mode):
        command_sim = _cosine_rows(command_matrix, [(a, b) for a, b, *_ in pairs])
        action_sim = _cosine_rows(action_matrix, [(a, b) for a, b, *_ in pairs])
        base = np.column_stack([command_sim, action_sim])
        if not include_mode:
            return base
        p1 = np.asarray([agent_probs[a] for a, *_ in pairs])
        p2 = np.asarray([agent_probs[b] for _, b, *_ in pairs])
        gap = np.abs(p1 - p2)
        jointly_agent = p1 * p2
        jointly_human = (1 - p1) * (1 - p2)
        return np.column_stack([
            base, gap, jointly_agent, jointly_human,
            command_sim * gap, action_sim * gap,
            command_sim * jointly_agent, action_sim * jointly_agent,
            command_sim * jointly_human, action_sim * jointly_human,
        ])

    def fit(self, records):
        records = list(records)
        if len(records) < 8:
            raise ValueError("At least eight labeled traces are required for this research prototype")
        if any(row.mode not in ("human", "agent") for row in records):
            raise ValueError("Training labels must currently be human or agent")
        groups = [f"{r.benchmark}/{r.activity_id}" for r in records]
        if len(set(groups)) < 2:
            raise ValueError("Association training requires multiple activity groups")

        self.training_mode_oof_ = self._crossfit_mode(records)
        indices = list(range(len(records)))
        mode_prob, mode_vec, mode_clf = self._fit_mode(records, indices, indices, self.seed)
        self.mode_vectorizer_ = mode_vec
        self.mode_classifier_ = mode_clf
        self.records_ = records
        self.final_training_mode_prob_ = mode_prob
        self._representations(records, indices)

        pairs = self._training_pairs(records)
        pair_x_base = self._pair_features(
            records, pairs, self.training_mode_oof_, self.command_matrix_, self.action_matrix_, False
        )
        pair_x_mode = self._pair_features(
            records, pairs, self.training_mode_oof_, self.command_matrix_, self.action_matrix_, True
        )
        true_mode_prob = np.asarray([int(row.mode == "agent") for row in records], dtype=float)
        pair_x_oracle = self._pair_features(
            records, pairs, true_mode_prob, self.command_matrix_, self.action_matrix_, True
        )
        y = np.asarray([p[2] for p in pairs])
        self.match_baseline_ = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=self.seed)
        self.match_mode_aware_ = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=self.seed)
        self.match_oracle_ = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=self.seed)
        self.match_baseline_.fit(pair_x_base, y)
        self.match_mode_aware_.fit(pair_x_mode, y)
        self.match_oracle_.fit(pair_x_oracle, y)
        self.training_pair_counts_ = {
            "pairs": int(len(pairs)),
            "same_activity": int(sum(p[2] == 1 for p in pairs)),
            "different_activity": int(sum(p[2] == 0 for p in pairs)),
            "groups": int(len(set(groups))),
            "by_pair_mode": {
                relation: {
                    "same_activity": int(sum(p[4] == relation and p[2] == 1 for p in pairs)),
                    "different_activity": int(sum(p[4] == relation and p[2] == 0 for p in pairs)),
                }
                for relation in ("human-human", "human-agent", "agent-agent")
            },
        }
        self._fitted = True
        return self

    def predict_mode(self, records):
        """Score supported evidence; empty or entirely unseen input abstains.

        This is an evidence-availability gate, not a validated out-of-distribution
        or script detector. A nonempty known vocabulary does not establish that
        the input follows the training distribution. ``confidence`` retains the
        uncalibrated classifier score for compatibility.
        """
        self._check_fitted()
        records = list(records)
        if not records:
            return []
        text = [" ".join(row.commands) for row in records]
        X = self.mode_vectorizer_.transform(text)
        probabilities = self.mode_classifier_.predict_proba(X)[:, 1]
        feature_counts = np.asarray(X.getnnz(axis=1)).ravel()
        result = []
        for row, prob, feature_count in zip(records, probabilities, feature_counts):
            reason = None
            if row.mode == "script":
                reason = "unsupported_mode_class"
            elif not row.commands:
                reason = "empty_commands"
            elif not feature_count:
                reason = "out_of_vocabulary"
            result.append({
                "trace_id": row.trace_id,
                "agent_like_score": float(prob) if reason is None else None,
                "score_calibrated": False,
                "score_semantics": "uncalibrated logistic-regression score",
                "supported_prediction": ("agent-like" if prob >= 0.5 else "human-like") if reason is None else "unknown",
                "confidence": float(max(prob, 1 - prob)) if reason is None else None,
                "confidence_calibrated": False,
                "abstention_reason": reason,
                "command_count": len(row.commands),
                "recognized_feature_count": int(feature_count),
                "script_class_supported": False,
            })
        return result

    @staticmethod
    def _validate_probabilities(values, n_records, name):
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric vector with one value per record") from exc
        if array.shape != (n_records,):
            raise ValueError(f"{name} must have exactly one value per record ({n_records})")
        if not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 1)):
            raise ValueError(f"{name} must contain finite values in [0, 1]")
        return array

    def predict_pair_scores(self, records, pairs, agent_probabilities=None, oracle_probabilities=None):
        """Return baseline and mode-aware ranking scores for index pairs.

        `agent_probabilities` may be supplied as an out-of-fold vector for
        evaluation; otherwise the fitted mode classifier is used.
        """
        self._check_fitted()
        records = list(records)
        pairs = list(pairs)
        for pair in pairs:
            if (not isinstance(pair, (tuple, list, np.ndarray)) or len(pair) != 2
                    or any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_))
                           or i < 0 or i >= len(records) for i in pair)):
                raise ValueError("Each pair must contain two valid non-negative record indices")
        if agent_probabilities is not None:
            agent_probabilities = self._validate_probabilities(agent_probabilities, len(records), "agent_probabilities")
        if oracle_probabilities is not None:
            oracle_probabilities = self._validate_probabilities(oracle_probabilities, len(records), "oracle_probabilities")
        if not pairs:
            result = {"similarity_only": np.empty(0), "mode_aware": np.empty(0)}
            if oracle_probabilities is not None:
                result["oracle_mode_upper_bound"] = np.empty(0)
            return result

        mode_results = self.predict_mode(records)
        used_indices = sorted({int(i) for pair in pairs for i in pair})
        unsupported = [i for i in used_indices if mode_results[i]["abstention_reason"] is not None]
        if unsupported:
            raise ValueError(
                "Cannot score pairs with insufficient command evidence at record indices "
                + ", ".join(map(str, unsupported))
                + "; inspect predict_mode abstention_reason before matching"
            )
        if agent_probabilities is None:
            # Unused unsupported records never enter a pair; the placeholder
            # only keeps the index-aligned vector numeric.
            agent_probabilities = np.asarray([
                r["agent_like_score"] if r["agent_like_score"] is not None else 0.5 for r in mode_results
            ])
        command_text = self._text(records, "commands")
        action_text = self._text(records, "actions")
        cm = self.command_vectorizer_.transform(command_text)
        am = self.action_vectorizer_.transform(action_text) if hasattr(self.action_vectorizer_, "vocabulary_") else self.action_matrix_.__class__((len(records), 0))
        pair_rows = [(a, b, 0, "", "") for a, b in pairs]
        x_base = self._pair_features(records, pair_rows, agent_probabilities, cm, am, False)
        x_mode = self._pair_features(records, pair_rows, agent_probabilities, cm, am, True)
        result = {
            "similarity_only": self.match_baseline_.predict_proba(x_base)[:, 1],
            "mode_aware": self.match_mode_aware_.predict_proba(x_mode)[:, 1],
        }
        if oracle_probabilities is not None:
            x_oracle = self._pair_features(records, pair_rows, oracle_probabilities, cm, am, True)
            result["oracle_mode_upper_bound"] = self.match_oracle_.predict_proba(x_oracle)[:, 1]
        return result

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit(records) before prediction")


def from_observation(row):
    """Adapt the repository's conservative public-trace parser output."""
    return TraceRecord(
        trace_id=row.trace_id,
        benchmark=row.benchmark,
        activity_id=row.task,
        mode=row.label,
        commands=tuple(row.names),
        actions=tuple(row.groups),
        source=row.source,
    )
