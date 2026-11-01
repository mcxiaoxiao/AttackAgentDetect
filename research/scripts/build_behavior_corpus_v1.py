"""Build an audited multi-source trajectory catalog; no performance evaluation."""
from __future__ import annotations

import collections
import ast
import csv
import gzip
import hashlib
import io
import json
import pathlib
import re
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from attackagent.traces import ATOM_GROUP, command_heads, terminal_lines  # noqa: E402

RAW = ROOT / "research/data/raw/cyber-task-horizons"
HF = ROOT / "research/data/external/dataset-review-20261101/hf"
MANIFESTS = ROOT / "research/data/manifests"
OUT = ROOT / "research/results/behavior_corpus_v1"
SITE = ROOT / "site/data"
REV = "178c0da1800f9c07311e1b8d6c61dd8104e7b4ae"
CTF = {"intercode-ctf", "cybench", "nyuctf"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def json_samples(path):
    if path.name.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            data = io.BytesIO(f.read())
        archive = zipfile.ZipFile(data)
    else:
        archive = zipfile.ZipFile(path)
    with archive:
        for member in sorted(archive.namelist()):
            if member.startswith("samples/") and member.endswith(".json"):
                with archive.open(member) as f:
                    yield member, json.load(f)


def args_object(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def shell_inputs(messages):
    """Read explicit shell tool inputs only, never infer commands from outputs."""
    output = []
    skipped = collections.Counter()
    for message in messages:
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls", []) or []
        content = message.get("content", [])
        if isinstance(content, list):
            calls = [*calls, *[x for x in content if isinstance(x, dict) and x.get("type") == "tool_use"]]
        for call in calls:
            function = call.get("function", call.get("name", ""))
            if isinstance(function, dict):
                args = args_object(function.get("arguments", {}))
                function = function.get("name", "")
            else:
                args = args_object(call.get("arguments", call.get("input", {})))
            if function in {"bash", "terminal", "execute", "execute_command", "run_command", "shell"}:
                text = args.get("cmd", args.get("command", ""))
                if isinstance(text, str) and text.strip():
                    output.append(text)
                else:
                    skipped["unrecognized_shell_argument"] += 1
            else:
                skipped["other_tool_calls"] += 1
    return output, dict(skipped)


def normalize_task(benchmark, value):
    task = str(value)
    if benchmark == "cvebench":
        task = re.sub(r"-one_day$", "", task)
    return task


def record(source, revision, locator, benchmark, task, mode, model, texts, observation, url, license, framework):
    commands = command_heads("\n".join(texts))
    uid = hashlib.sha256(f"{source}/{revision}/{locator}".encode()).hexdigest()[:20]
    namespace = f"{source}/{benchmark}/{task}"
    return {"trace_id": uid, "source_dataset": source, "source_revision": revision,
            "record_locator": locator, "source_url": url, "benchmark": benchmark,
            "task_id": task, "task_group": namespace, "mode": mode, "model": model,
            "framework": framework, "observation_format": observation, "license": license,
            "track": "ctf" if benchmark in CTF else "security" if benchmark in {"cvebench", "cybergym"} else "command_tasks",
            "relation_scope": "within-source-task-identity",
            "normalization": "known-command-heads-lossy-v1",
            "commands": commands, "command_count": len(commands), "submitted_shell_chunks": len(texts),
            "action_counts": dict(collections.Counter(ATOM_GROUP[c] for c in commands))}


def load_cth():
    manifests = [ROOT / "research/data/manifests/intercode_public_subset.json",
                 ROOT / "research/data/manifests/crossbenchmark_public_subset.json",
                 ROOT / "research/experiments/multidimensional_rerun_v1/source_manifest.json"]
    expected, model_sets, benchmarks = {}, {}, {}
    for p in manifests:
        m = json.loads(p.read_text())
        assert m["revision"] == REV
        model_sets.update(m.get("model_sets", {}))
        for b, ids in m.get("agent_evalsets", {}).items():
            for i in ids:
                model_sets[i] = m.get("model", "unknown")
                benchmarks[i] = b
        for key in ("files", "human_files", "agent_files"):
            entries = m.get(key, [])
            if not isinstance(entries, list):
                continue
            for row in entries:
                if not isinstance(row, dict):
                    continue
                expected[row["path"]] = row
                if row.get("model") and "/eval_logs/eval-set-" in row["path"]:
                    model_sets[row["path"].split("/")[2]] = row["model"]
                if row.get("benchmark") and "/eval_logs/eval-set-" in row["path"]:
                    benchmarks[row["path"].split("/")[2]] = row["benchmark"]
    ordered = sorted(expected)
    audit, provenance, selected = collections.Counter(), [], {}
    for rel in ordered:
        p = RAW / rel
        e = expected[rel]
        if not p.is_file():
            raise FileNotFoundError(rel)
        actual = sha(p)
        if actual != e["sha256"] or p.stat().st_size != e["size"]:
            raise ValueError(f"Pinned source validation failed: {rel}")
        provenance.append({"source": "cyber-task-horizons", "revision": REV, "file": rel,
                           "bytes": p.stat().st_size, "sha256": actual})
        human = "/human/" in rel
        file_samples = 0
        for member, sample in json_samples(p):
            file_samples += 1
            audit["decoded_human_samples" if human else "decoded_agent_samples"] += 1
            if human:
                benchmark = rel.split("/")[3]
                logs = sample.get("store", {}).get("HumanAgentState:logs", {})
                texts = [line for key, value in sorted(logs.items()) if key.endswith(".input") and isinstance(value, str)
                         for line in terminal_lines(value)]
                mode, model, observation = "human", "not-applicable", "script-terminal-input"
                match = re.search(r"session_\d+", rel)
                session = match.group() if match else rel
                key = (benchmark, str(sample.get("id")), session)
                priority = ("_partial" not in rel, rel)
            else:
                evalset = rel.split("/")[2]
                benchmark = benchmarks.get(evalset)
                if not benchmark:
                    benchmark = "cybench" if "cybench" in rel else "nyuctf" if "nyuctf" in rel else "intercode-ctf"
                if evalset not in model_sets:
                    raise ValueError(f"Missing source model mapping: {evalset}")
                texts, skipped = shell_inputs(sample.get("messages", []))
                audit.update(skipped)
                mode, model, observation = "agent", model_sets[evalset], "inspect-shell-tool-call"
                key = (evalset, str(sample.get("id")), sample.get("epoch", 1))
                priority = (True, rel)
            task = normalize_task(benchmark, sample.get("id", "unknown"))
            r = record("cyber-task-horizons", REV, rel + "#" + member, benchmark, task, mode, model,
                       texts, observation, "https://github.com/lyptus-research/cyber-task-horizons-data/blob/" + REV + "/" + rel,
                       "CC-BY-4.0", "human_cli" if human else "Inspect-ReAct")
            r["source_task_id"] = str(sample.get("id", "unknown"))
            if benchmark == "cvebench":
                r["relation_scope"] = "within-source-vulnerability-target"
                r["task_condition"] = "one_day" if r["source_task_id"].endswith("-one_day") else "source-not-specified"
            if key in selected:
                audit["overlapping_source_snapshots_replaced"] += 1
            if key not in selected or priority > selected[key][0]:
                selected[key] = (priority, r)
        provenance[-1]["decoded_samples"] = file_samples
    return [x[1] for x in selected.values()], provenance, dict(audit)


def load_hf():
    base = HF / "antieval-cybench-files"
    manifest = json.loads((MANIFESTS / "behavior_corpus_hf_cybench_v1.json").read_text())
    rows, provenance, audit = [], [], collections.Counter()
    for f in manifest["files"]:
        p = base / f["path"]
        if sha(p) != f["sha256"]:
            raise ValueError("HF input hash mismatch")
        provenance.append({"source": manifest["dataset"], "revision": manifest["revision"],
                           "file": f["path"], "bytes": p.stat().st_size, "sha256": f["sha256"]})
        for line_number, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            audit["decoded_records"] += 1
            meta = sample.get("metadata", {})
            texts, skipped = shell_inputs(sample.get("input", []))
            audit.update(skipped)
            model = meta.get("model", f["path"].removeprefix("cybench_").removesuffix(".jsonl"))
            r = record(manifest["dataset"], manifest["revision"], f["path"] + "#L" + str(line_number),
                       "cybench", str(sample.get("id")), "agent", model, texts, "hf-shell-tool-call",
                       f["url"], "not-stated-by-source", "source-not-specified")
            r["task_variant"] = str(meta.get("variant_name", "not-specified"))
            r["source_task_id"] = str(sample.get("id"))
            r["task_condition"] = r["task_variant"]
            # A matching challenge title is not proof of a common version/capture contract.
            r["cross_source_task_equivalence"] = "not-asserted"
            rows.append(r)
    return rows, provenance, dict(audit)


def load_agenttrace():
    base = HF / "agenttrace-files"
    manifest = json.loads((MANIFESTS / "behavior_corpus_hf_agenttrace_v1.json").read_text())
    rows, provenance, audit = [], [], collections.Counter()
    for f in manifest["files"]:
        p = base / f["local"]
        if sha(p) != f["sha256"]:
            raise ValueError("AgentTrace input hash mismatch")
        provenance.append({"source": manifest["dataset"], "revision": manifest["revision"],
                           "file": f["path"], "bytes": p.stat().st_size, "sha256": f["sha256"]})
        for line_number, line in enumerate(p.read_text().splitlines(), 1):
            sample = json.loads(line)
            audit["decoded_records"] += 1
            texts = []
            for span in sample.get("spans", []):
                if span.get("tool_name") != "bash":
                    continue
                value = span.get("tool_input")
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                    except ValueError:
                        if value.startswith("kwargs="):
                            try:
                                parsed = ast.literal_eval(value[len("kwargs="):])
                            except (ValueError, SyntaxError):
                                audit["unparsed_collector_argument"] += 1
                                parsed = ""
                        else:
                            parsed = value
                else:
                    parsed = value
                if isinstance(parsed, dict):
                    parsed = parsed.get("command", parsed.get("cmd", ""))
                if isinstance(parsed, str) and parsed.strip():
                    texts.append(parsed)
            meta = sample.get("metadata", {})
            task = f"{meta.get('fixture_version', 'unspecified')}/{meta.get('task_id', sample.get('trace_id'))}"
            r = record(manifest["dataset"], manifest["revision"], f["path"] + "#L" + str(line_number),
                       "nl2bash", task, "agent", str(sample.get("model", "unknown")), texts,
                       "instrumented-shell-tool-span", f["url"], "Apache-2.0; upstream task terms retained", "smolagents")
            r["cross_source_task_equivalence"] = "not-asserted"
            r["source_task_id"] = str(meta.get("task_id", sample.get("trace_id")))
            r["task_condition"] = str(meta.get("fixture_version", "unspecified"))
            rows.append(r)
    return rows, provenance, dict(audit)


def export_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SITE.mkdir(exist_ok=True)
    cth, prov_a, audit_a = load_cth()
    hf, prov_b, audit_b = load_hf()
    cli, prov_c, audit_c = load_agenttrace()
    decoded = cth + hf + cli
    rows = [r for r in decoded if r["command_count"]]
    excluded = [{"trace_id": r["trace_id"], "source_dataset": r["source_dataset"],
                 "benchmark": r["benchmark"], "mode": r["mode"],
                 "reason": "no-explicit-retained-common-command-heads", "record_locator": r["record_locator"]}
                for r in decoded if not r["command_count"]]
    export_csv(OUT / "excluded_records.csv", excluded, ["trace_id", "source_dataset", "benchmark", "mode", "reason", "record_locator"])
    ids = [r["trace_id"] for r in rows]
    assert len(ids) == len(set(ids))
    groups = collections.defaultdict(set)
    for r in rows:
        groups[r["task_group"]].add(r["mode"])
    for r in rows:
        r["has_human_agent_peer"] = groups[r["task_group"]] == {"human", "agent"}
        r["evaluation_role"] = "task-retrieval-and-mode-proxy" if r["source_dataset"] == "cyber-task-horizons" else "agent-task-transfer" if r["source_dataset"] == "antieval/cybench-trajectories" else "short-command-and-fixture-stress"
    rows.sort(key=lambda r: (r["source_dataset"], r["benchmark"], r["task_id"], r["mode"], r["trace_id"]))
    fields = ["trace_id", "source_dataset", "source_revision", "benchmark", "track", "task_id", "task_group",
              "mode", "model", "framework", "observation_format", "command_count", "submitted_shell_chunks",
              "has_human_agent_peer", "evaluation_role", "relation_scope", "source_task_id", "task_condition",
              "cross_source_task_equivalence", "license", "source_url", "record_locator", "normalization"]
    export_csv(OUT / "trace_registry.csv", rows, fields)
    export_csv(SITE / "behavior-corpus-v1.csv", rows, fields)
    with (OUT / "evidence.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({"trace_id": r["trace_id"], "commands": r["commands"],
                                "actions": [ATOM_GROUP[c] for c in r["commands"]],
                                "observation_format": r["observation_format"], "normalization": r["normalization"]}) + "\n")
    summary = {"version": "behavior-corpus-v1", "date": "2026-11-01", "status": "catalog_expansion_not_new_effectiveness_results",
               "traces": len(rows), "human": sum(r["mode"] == "human" for r in rows),
               "agent": sum(r["mode"] == "agent" for r in rows), "task_groups": len(groups),
               "matched_human_agent_task_groups": sum(v == {"human", "agent"} for v in groups.values()),
               "matched_traces": sum(r["has_human_agent_peer"] for r in rows),
               "ctf_matched_human_agent_task_groups": len({r["task_group"] for r in rows if r["track"] == "ctf" and r["has_human_agent_peer"]}),
               "ctf_traces": sum(r["track"] == "ctf" for r in rows),
               "supplement_traces": sum(r["track"] != "ctf" for r in rows),
               "excluded_empty_command_view": len(decoded) - len(rows),
               "exclusions_by_source": dict(collections.Counter(r["source_dataset"] for r in excluded)),
               "length_coverage": {str(n): sum(r["command_count"] >= n for r in rows) for n in [1, 2, 5, 10, 20]},
               "quality": {"cth": audit_a, "hf_cybench": audit_b, "hf_agenttrace": audit_c}, "sources": [], "benchmarks": [],
               "old_evaluation": {"traces": 97, "task_groups": 40, "pairs": 1558,
                                  "status": "historical_scores_not_recomputed_for_expanded_catalog"},
               "limits": ["Task identity is source-qualified; no cross-source identity mapping is assumed.",
                          "Known-command-head normalization is a lossy common view, not complete host execution telemetry.",
                          "Human terminal capture and Agent tool calls remain different observation mechanisms.",
                          "A broader catalog is coverage evidence, not measured generalization or improved attribution."]}
    for source in sorted({r["source_dataset"] for r in rows}):
        subset = [r for r in rows if r["source_dataset"] == source]
        summary["sources"].append({"source": source, "traces": len(subset),
                                   "human": sum(r["mode"] == "human" for r in subset),
                                   "agent": sum(r["mode"] == "agent" for r in subset),
                                   "task_groups": len({r["task_group"] for r in subset}),
                                   "models": sorted({r["model"] for r in subset if r["mode"] == "agent"})})
    for benchmark in sorted({r["benchmark"] for r in rows}):
        subset = [r for r in rows if r["benchmark"] == benchmark]
        summary["benchmarks"].append({"benchmark": benchmark, "traces": len(subset),
                                      "human": sum(r["mode"] == "human" for r in subset),
                                      "agent": sum(r["mode"] == "agent" for r in subset),
                                      "task_groups": len({r["task_group"] for r in subset})})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "provenance.json").write_text(json.dumps({"files": prov_a + prov_b + prov_c,
                                                     "builder_sha256": sha(pathlib.Path(__file__)),
                                                     "command_adapter_sha256": sha(ROOT / "attackagent/traces.py")}, indent=2) + "\n")
    public = [{k: v for k, v in r.items() if k != "commands"} | {"commands_preview": r["commands"][:24]} for r in rows]
    (SITE / "behavior-corpus-v1.json").write_text(json.dumps({"summary": summary, "traces": public}, separators=(",", ":")) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"quality", "limits"}}, indent=2))


if __name__ == "__main__":
    main()
