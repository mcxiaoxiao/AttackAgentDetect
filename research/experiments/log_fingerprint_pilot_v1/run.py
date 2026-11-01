"""Fixed exploratory common-log pilot; raw commands never enter published outputs."""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import itertools
import json
import math
import pathlib
import platform
import re
import sys
import time
import zipfile

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from attackagent.auditd import _decode_hex, _event_id, _parse_fields  # noqa: E402
from attackagent.traces import ATOM_GROUP  # noqa: E402

EXP = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "research/results/log_fingerprint_pilot_v1"
RAW = ROOT / "research/data/external/ait-shell-pilot-v1"
ARCHIVE_SHA = "f3ff78d7fcb7db9068d6b0c6057659412c725830d0b05fb64cea319f0e638914"
SEED = 20261101
FEATURES = ("K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8")
METHODS = ("length", "lexical", "core8")
VIEWS = ("login_exec", "all_exec", "login_no_wrappers", "login_first50")
WRAPPERS = {"sudo", "su", "sh", "bash", "dash", "env", "sshd", "systemd-executor"}
UNSET = {"", "-1", "4294967295", "unset", "4294967295L"}
RUN_RE = re.compile(r"data/(Nextcloud|WordPress)/LOGS_Result_([^/]+)/(exp\d+)/audit\.log")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stats(commands):
    n = len(commands)
    if not n:
        return dict.fromkeys(FEATURES), {"n": 0, "known": 0, "eligible_pairs": 0}
    counts = collections.Counter(commands)
    actions = [ATOM_GROUP.get(c) for c in commands]
    pairs = [(a, b) for a, b in zip(actions, actions[1:]) if a is not None and b is not None]
    pair_counts = collections.Counter(pairs)
    outgoing = collections.Counter(a for a, _ in pairs)
    m = len(pairs)
    longest = run = 0
    previous = None
    for action in actions:
        if action is None:
            run = 0
        else:
            run = run + 1 if action == previous else 1
            longest = max(longest, run)
        previous = action
    values = (
        len(counts) / n,
        -sum((v / n) * math.log2(v / n) for v in counts.values()),
        max(counts.values()) / n,
        sum(a != b for a, b in pairs) / m if m else None,
        -sum((v / m) * math.log2(v / outgoing[a]) for (a, _), v in pair_counts.items()) if m else None,
        longest / n if longest else None,
        1 - len(pair_counts) / m if m else None,
        sum(a == b for a, b in zip(commands, commands[1:])) / (n - 1) if n > 1 else None,
    )
    assert all(v is None or math.isfinite(v) for v in values)
    return dict(zip(FEATURES, values)), {"n": n, "known": sum(a is not None for a in actions), "eligible_pairs": m}


def parse_exec(stream):
    """Join launch records; do not promote unrelated syscall records to commands."""
    events = {}
    audit = collections.Counter()
    for raw in stream:
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        audit["lines"] += 1
        if not line.startswith(("type=EXECVE ", "type=SYSCALL ")):
            continue
        kind, fields = _parse_fields(line)
        event_id = _event_id(fields)
        if event_id is None:
            audit["malformed_launch_or_syscall_records"] += 1
            continue
        ts_text, serial, timestamp, host = event_id
        key = (host, ts_text, serial)
        event = events.setdefault(key, {"time": timestamp, "serial": int(serial), "syscall": [], "exec": []})
        bucket = event["syscall" if kind == "SYSCALL" else "exec"]
        if fields in bucket:
            audit["duplicate_records"] += 1
        else:
            bucket.append(fields)
    output = []
    for key, event in events.items():
        if not event["exec"]:
            continue
        audit["execve_event_bundles"] += 1
        if len(event["syscall"]) != 1:
            audit["missing_or_ambiguous_syscall"] += 1
            continue
        s = event["syscall"][0]
        if s.get("success") != "yes":
            audit["unsuccessful_or_unconfirmed_launch"] += 1
            continue
        if s.get("arch") == "c000003e" and s.get("syscall") not in {"59", "322"}:
            raise ValueError("EXECVE bundle has a non-launch x86_64 syscall")
        argv0 = next((x["a0"] for x in event["exec"] if x.get("a0")), "")
        image = s.get("exe", "")
        if image in {"", "?", "(null)"}:
            image = _decode_hex(argv0)
            audit["argv0_fallback"] += 1
        name = pathlib.PurePosixPath(image).name.lower()
        if name in {"", "?", "(null)"} or event["time"] is None:
            audit["missing_executable_or_time"] += 1
            continue
        login = s.get("auid", "") not in UNSET and s.get("ses", "") not in UNSET
        # Values below support quality diagnostics only, never model inputs.
        event_fingerprint = digest(json.dumps([key, s, event["exec"]], sort_keys=True).encode())
        output.append({"command": name, "time": event["time"], "serial": event["serial"],
                       "login": login, "event_hash": event_fingerprint,
                       "auid": s.get("auid", ""), "session": s.get("ses", "")})
    output.sort(key=lambda x: (x["time"], x["serial"], x["event_hash"]))
    audit["successful_launches"] = len(output)
    audit["login_attributable_launches"] = sum(x["login"] for x in output)
    audit["tied_start_times"] = len(output) - len({x["time"] for x in output})
    return output, dict(audit)


def checks():
    f, q = stats(["ls", "ls", "cat", "cat", "ls"])
    expected = {"K1": .4, "K2": -(3/5)*math.log2(3/5)-(2/5)*math.log2(2/5),
                "K3": .6, "K4": .5, "K5": 1., "K6": .4, "K7": 0., "K8": .5}
    assert all(math.isclose(f[k], v, abs_tol=1e-12) for k, v in expected.items())
    u, quality = stats(["ls", "unmapped", "cat"])
    assert quality["eligible_pairs"] == 0 and all(u[k] is None for k in ("K4", "K5", "K7"))
    assert u["K6"] == 1/3
    assert stats(["ls", "unmapped", "ls", "ls"])[0]["K6"] == .5
    assert all(v is None for v in stats([])[0].values())
    assert stats(["unmapped"])[0]["K6"] is None
    assert stats(["ls"])[0]["K8"] is None
    assert stats(["ls"]*10)[0]["K7"] == 8/9
    lines = [
        'type=SYSCALL msg=audit(1.000:1): arch=c000003e syscall=59 success=yes auid=0 ses=1 exe="/usr/bin/ls"\n',
        'type=EXECVE msg=audit(1.000:1): argc=1 a0="spoofed"\n',
        'type=SYSCALL msg=audit(2.000:2): arch=c000003e syscall=1 success=yes auid=0 ses=1 exe="/usr/bin/cat"\n',
        'type=SYSCALL msg=audit(3.000:3): arch=c000003e syscall=59 success=no auid=0 ses=1 exe="/usr/bin/cat"\n',
        'type=EXECVE msg=audit(3.000:3): argc=1 a0="cat"\n',
        'type=SYSCALL msg=audit(4.000:4): arch=c000003e syscall=59 success=yes auid=4294967295 ses=4294967295\n',
        'type=EXECVE msg=audit(4.000:4): argc=1 a0=2F62696E2F636174\n',
    ]
    events, audit = parse_exec(lines + lines[:2])
    assert [(e["command"], e["login"]) for e in events] == [("ls", True), ("cat", False)]
    assert audit["duplicate_records"] == 2 and audit["unsuccessful_or_unconfirmed_launch"] == 1
    return {"hand_checked_feature_cases": 7, "parser_checks": ["event_join", "duplicates", "nonexec_syscall_exclusion", "failed_launch_exclusion", "actual_image_preference", "hex_argv0_fallback", "login_id_filter"]}


def load_runs():
    archive = RAW / "data.zip"
    if digest(archive.read_bytes()) != ARCHIVE_SHA:
        raise ValueError("Dataset content hash differs from the frozen protocol")
    output = []
    with zipfile.ZipFile(archive) as z:
        for path in sorted(z.namelist()):
            match = RUN_RE.fullmatch(path)
            if not match:
                continue
            app, actor, scenario = match.groups()
            label = int("GPT" in actor)
            family = re.sub(r"_V\d+$", "", actor) if label else actor
            raw = z.read(path)
            events, audit = parse_exec(io.BytesIO(raw))
            output.append({"id": digest(path.encode())[:16], "app": app, "task": f"{app}/{scenario}",
                           "source": family, "actor": actor, "y": label,
                           "raw_hash": digest(raw), "events": events, "audit": audit,
                           "raw_size": len(raw), "archive_path": path})
    assert len(output) == 109
    return output


def view_runs(runs, view):
    output, admission = [], collections.Counter()
    for run in runs:
        ev = run["events"]
        if view != "all_exec":
            ev = [x for x in ev if x["login"]]
        if view == "login_no_wrappers":
            ev = [x for x in ev if x["command"] not in WRAPPERS]
        label = "agent" if run["y"] else "human"
        admission[f"available_{label}"] += 1
        if view == "login_first50":
            if len(ev) < 50:
                admission[f"excluded_budget_{label}"] += 1
                continue
            ev = ev[:50]
        commands = tuple(x["command"] for x in ev)
        f, quality = stats(commands)
        if len(commands) < 10 or quality["eligible_pairs"] < 5:
            admission[f"excluded_evidence_{label}"] += 1
            continue
        admission[f"admitted_{label}"] += 1
        output.append({**run, "commands": commands, "features": f, "quality": quality,
                       "seq_hash": digest(json.dumps(commands).encode()),
                       "login_share": sum(x["login"] for x in run["events"]) / max(1, len(run["events"]))})
    return output, dict(admission)


def partitions(rows, panel, source_folds):
    tasks = sorted({r["task"] for r in rows})
    if panel == "task_holdout":
        keys = tasks
    elif panel == "source_holdout":
        keys = range(3)
    elif panel == "crossed_task_source_holdout":
        keys = list(itertools.product(tasks, range(3)))
    elif panel == "application_holdout":
        keys = sorted({r["app"] for r in rows})
    else:
        raise ValueError(panel)
    for key in keys:
        if panel == "task_holdout":
            test = [i for i, r in enumerate(rows) if r["task"] == key]
            train = [i for i, r in enumerate(rows) if r["task"] != key]
        elif panel == "source_holdout":
            test = [i for i, r in enumerate(rows) if source_folds[r["source"]] == key]
            train = [i for i, r in enumerate(rows) if source_folds[r["source"]] != key]
        elif panel == "crossed_task_source_holdout":
            task, fold = key
            test = [i for i, r in enumerate(rows) if r["task"] == task and source_folds[r["source"]] == fold]
            train = [i for i, r in enumerate(rows) if r["task"] != task and source_folds[r["source"]] != fold]
        else:
            test = [i for i, r in enumerate(rows) if r["app"] == key]
            train = [i for i, r in enumerate(rows) if r["app"] != key]
        if not test:
            continue
        hashes = {rows[i]["seq_hash"] for i in test}
        original_n = len(train)
        train = [i for i in train if rows[i]["seq_hash"] not in hashes]
        assert not set(train) & set(test)
        assert not {rows[i]["seq_hash"] for i in train} & hashes
        if panel in {"source_holdout", "crossed_task_source_holdout"}:
            assert not {rows[i]["source"] for i in train} & {rows[i]["source"] for i in test}
        if panel in {"task_holdout", "crossed_task_source_holdout"}:
            assert not {rows[i]["task"] for i in train} & {rows[i]["task"] for i in test}
        yield str(key), train, test, original_n - len(train)


def fit_predict(rows, train, test, method):
    clf = LogisticRegression(C=1., class_weight="balanced", solver="liblinear", max_iter=2000, random_state=SEED)
    if method == "lexical":
        xtrain = [" ".join(rows[i]["commands"]) for i in train]
        xtest = [" ".join(rows[i]["commands"]) for i in test]
        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                             token_pattern=r"(?u)\b[\w.+-]+\b"), clf)
    else:
        def vector(i):
            if method == "length":
                return [math.log1p(len(rows[i]["commands"]))]
            return [np.nan if rows[i]["features"][k] is None else rows[i]["features"][k] for k in FEATURES]
        xtrain, xtest = np.array([vector(i) for i in train]), np.array([vector(i) for i in test])
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), clf)
    model.fit(xtrain, [rows[i]["y"] for i in train])
    return model.predict_proba(xtest)[:, 1]


def metrics(y, score):
    y, score = np.asarray(y), np.asarray(score)
    pred = score >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": len(y), "human": int((y == 0).sum()), "agent": int((y == 1).sum()),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
            "human_fpr": float(fp/(tn+fp)) if tn+fp else None,
            "agent_fnr": float(fn/(fn+tp)) if fn+tp else None,
            "auroc": float(roc_auc_score(y, score)) if len(set(y)) == 2 else None,
            "uncalibrated_brier": float(brier_score_loss(y, score)),
            "confusion_matrix_human_agent": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def cluster_interval(rows, indices, y, score):
    rng = np.random.default_rng(SEED)
    groups = sorted({rows[i]["task"] for i in indices})
    positions = {g: [j for j, i in enumerate(indices) if rows[i]["task"] == g] for g in groups}
    values = []
    y, score = np.asarray(y), np.asarray(score)
    for _ in range(1000):
        sample = [i for g in rng.choice(groups, len(groups), replace=True) for i in positions[g]]
        yy, pp = y[sample], score[sample] >= .5
        if len(set(yy)) == 2:
            values.append(((pp[yy == 0] == 0).mean() + (pp[yy == 1] == 1).mean()) / 2)
    return list(map(float, np.quantile(values, [.025, .975]))) if values else None


def evaluate(rows, view, panel, source_folds, deadline):
    predictions = {method: {} for method in METHODS}
    fold_info = []
    for fold, train, test, purged in partitions(rows, panel, source_folds):
        if time.monotonic() > deadline:
            raise TimeoutError("Frozen pilot model-computation budget exceeded")
        valid = len({rows[i]["y"] for i in train}) == 2
        fold_info.append({"fold": fold, "train": len(train), "test": len(test), "purged": purged, "available": valid})
        if not valid:
            continue
        for method in METHODS:
            scores = fit_predict(rows, train, test, method)
            for index, score in zip(test, scores):
                if index in predictions[method]:
                    raise ValueError("Repeated test row in an out-of-fold panel")
                predictions[method][index] = float(score)
    results, records = [], []
    for method in METHODS:
        indices = sorted(predictions[method])
        if not indices or len({rows[i]["y"] for i in indices}) < 2:
            results.append({"view": view, "panel": panel, "method": method, "available": False})
            continue
        y = [rows[i]["y"] for i in indices]
        score = [predictions[method][i] for i in indices]
        result = {"view": view, "panel": panel, "method": method, "available": True,
                  **metrics(y, score), "task_conditional_ba_interval": cluster_interval(rows, indices, y, score),
                  "by_application": {}}
        for app in sorted({rows[i]["app"] for i in indices}):
            loc = [j for j, i in enumerate(indices) if rows[i]["app"] == app]
            if len({y[j] for j in loc}) == 2:
                result["by_application"][app] = metrics([y[j] for j in loc], [score[j] for j in loc])
        results.append(result)
        for i, p in zip(indices, score):
            records.append({"view": view, "panel": panel, "method": method, "run_id": rows[i]["id"],
                            "app": rows[i]["app"], "task": rows[i]["task"], "source_group": rows[i]["source"],
                            "label": rows[i]["y"], "agent_score_uncalibrated": p, "prediction": int(p >= .5)})
    return results, records, fold_info


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.monotonic()
    checked = checks()
    runs = load_runs()
    OUT.mkdir(parents=True, exist_ok=True)
    human = sorted({r["source"] for r in runs if not r["y"]})
    agents = sorted({r["source"] for r in runs if r["y"]})
    assert len(human) == 6 and agents == ["GPT4.1", "GPT4o", "GPT5"]
    folds = {name: i % 3 for i, name in enumerate(human)}
    folds.update({name: i for i, name in enumerate(agents)})
    seen = collections.defaultdict(list)
    raw_hashes = collections.defaultdict(list)
    for run in runs:
        raw_hashes[run["raw_hash"]].append(run["id"])
        for event in run["events"]:
            seen[event["event_hash"]].append(run["id"])
    shared = {k: sorted(set(v)) for k, v in seen.items() if len(set(v)) > 1}
    if any(len(v) > 1 for v in raw_hashes.values()):
        raise ValueError("Identical raw run logs require grouping before this protocol can run")
    shared_counts = collections.Counter(x for ids in shared.values() for x in ids)
    provenance = {"status": "exploratory_not_confirmatory", "date": "2026-11-01",
                  "archive_sha256": ARCHIVE_SHA, "script_sha256": digest(pathlib.Path(__file__).read_bytes()),
                  "protocol_sha256": digest((EXP / "program.md").read_bytes()),
                  "action_map_sha256": digest(json.dumps(ATOM_GROUP, sort_keys=True).encode()),
                  "parser_source_sha256": digest((ROOT / "attackagent/auditd.py").read_bytes()),
                  "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                  "sklearn": sklearn.__version__, "seed": SEED, "measurement_checks": checked,
                  "raw_runs": len(runs), "raw_class_counts": dict(collections.Counter(r["y"] for r in runs)),
                  "source_folds": folds, "cross_run_identical_event_fingerprints": len(shared),
                  "runs_with_shared_events": len(shared_counts), "admission": {}, "folds": {}}
    registry = []
    for r in runs:
        registry.append({"run_id": r["id"], "application": r["app"], "task": r["task"],
                         "source_group": r["source"], "actor_repeat": r["actor"], "label": r["y"],
                         "raw_sha256": r["raw_hash"], "raw_bytes": r["raw_size"],
                         "successful_launches": len(r["events"]),
                         "login_launches": sum(e["login"] for e in r["events"]),
                         "shared_execution_events": shared_counts[r["id"]],
                         "background_launches": sum(not e["login"] for e in r["events"])})
    write_csv(OUT / "run_registry.csv", registry)
    (OUT / "parser_audit.json").write_text(json.dumps({r["id"]: r["audit"] for r in runs}, indent=2) + "\n")
    all_results, all_records, feature_rows = [], [], []
    for view in VIEWS:
        rows, admission = view_runs(runs, view)
        provenance["admission"][view] = admission
        for r in rows:
            feature_rows.append({"view": view, "run_id": r["id"], "app": r["app"], "task": r["task"],
                                 "source_group": r["source"], "label": r["y"], **r["quality"], **r["features"]})
        for panel in ("task_holdout", "source_holdout", "crossed_task_source_holdout", "application_holdout"):
            result, records, info = evaluate(rows, view, panel, folds, started + 1200)
            all_results.extend(result)
            all_records.extend(records)
            provenance["folds"][f"{view}/{panel}"] = info
            for r in result:
                if r.get("available"):
                    print(view, panel, r["method"], r["n"], f"BA={r['balanced_accuracy']:.4f}", flush=True)
    provenance["elapsed_seconds"] = time.monotonic() - started
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (OUT / "metrics.json").write_text(json.dumps(all_results, indent=2) + "\n")
    write_csv(OUT / "predictions.csv", all_records)
    write_csv(OUT / "features.csv", feature_rows)
    write_csv(OUT / "metrics.csv", [{k: v for k, v in r.items() if k not in {"by_application", "task_conditional_ba_interval", "confusion_matrix_human_agent"}} for r in all_results])
    print(f"Saved fixed pilot in {time.monotonic()-started:.1f}s; no association-benefit experiment.")


if __name__ == "__main__":
    main()
