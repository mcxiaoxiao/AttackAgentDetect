#!/usr/bin/env python3
"""Package the current manuscript, its figures, and frozen analysis records."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "research/paper"
DEST = ROOT / "site/assets/behavioral-fingerprint-v2"
ZIP_PATH = DEST / "human-agent-fingerprint-draft-source.zip"
FIGURES = [
    "cth-five-share-matched.pdf", "cth-five-share-delta-ba.pdf",
    "cth-xgb-pair-type-deltas.pdf", "nyu-aa-paired-deltas.pdf",
    "mode-human.pdf", "mode-agent.pdf", "attackagent-owl.png",
    "fingerprint-v3-overview.png", "fingerprint-bst-dimensions.pdf",
    "exact-mixture-v3.pdf", "fingerprint-v2-matrix-example.pdf",
    "fingerprint-observed-rhythm.pdf", "fingerprint-recognition-matrix.pdf",
    "fingerprint-conditioned-association.pdf", "fingerprint-association-comparison.pdf",
]
V1_FILES = [
    "environment.json", "baseline_reproduction.json", "metrics.json", "summary.csv",
    "error_transitions.csv", "pair_predictions.csv.gz", "paired_score_cases.csv",
    "run_manifest.json", "verification.json", "original_table_snapshot.json",
]
V1_DIRS = ["cth", "ait"]
V1_EXPERIMENTS = ["README.md", "run.py", "run.sh", "summarize.py", "verify.py", "rebuild_training_pairs.py", "reproduce_baselines.py"]
V2_EXPERIMENTS = [
    "analyze_cth_five_share.py", "analyze_independent_nyu_agent.py", "verify_independent_nyu_agent.py",
    "build_unified_bst_ablation.py", "build_paper_results.py", "independent_nyu_aa_protocol.md",
]
V2_RESULTS = [
    "five_share/five_share_analysis.json", "five_share/five_share_rows.csv",
    "unified_ablation/unified_bst_ablation.json", "unified_ablation/unified_bst_ablation.csv",
    "independent_nyu_agent/analysis.json", "independent_nyu_agent/pair_manifest.csv.gz",
    "independent_nyu_agent/predictions.csv.gz", "independent_nyu_agent/training_pairs.csv.gz",
    "independent_nyu_agent/run_manifest.json", "independent_nyu_agent/verification.json",
    "independent_nyu_task_manifest.json", "paper_assets/paper_display_data.json",
]


def collect():
    entries: dict[Path, Path] = {}

    def add(src: Path, rel: Path):
        if not src.is_file():
            raise FileNotFoundError(src)
        entries[src] = rel

    add(PAPER / "human-agent-fingerprint-draft.tex", Path("research/paper/human-agent-fingerprint-draft.tex"))
    for f in ("matched-v2-main-tables.tex", "matched-v2-appendix-tables.tex"):
        add(PAPER / f, Path("research/paper") / f)
    for name in FIGURES:
        add(PAPER / "figures" / name, Path("research/paper/figures") / name)
    for name in V1_FILES:
        add(ROOT / "research/results/matched_fingerprint_v1" / name,
            Path("research/results/matched_fingerprint_v1") / name)
    for folder in V1_DIRS:
        base = ROOT / "research/results/matched_fingerprint_v1" / folder
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix != ".log":
                add(path, Path("research/results/matched_fingerprint_v1") / path.relative_to(ROOT / "research/results/matched_fingerprint_v1"))
    for name in V1_EXPERIMENTS:
        add(ROOT / "research/experiments/matched_fingerprint_v1" / name,
            Path("research/experiments/matched_fingerprint_v1") / name)
    for name in V2_EXPERIMENTS:
        add(ROOT / "research/experiments/matched_fingerprint_v2" / name,
            Path("research/experiments/matched_fingerprint_v2") / name)
    for name in V2_RESULTS:
        add(ROOT / "research/results/matched_fingerprint_v2" / name,
            Path("research/results/matched_fingerprint_v2") / name)
    for name in ("matched-fingerprint-v2.md", "fingerprint-paper-storyline.md"):
        add(ROOT / "research/reports" / name, Path("research/reports") / name)
    add(ROOT / "research/experiments/exact_mixture_matched_fingerprint_plan.md",
        Path("research/experiments/exact_mixture_matched_fingerprint_plan.md"))
    return entries


def make_readme():
    return """# Matched behavioral-fingerprint draft bundle

This bundle contains the standalone TeX manuscript and every figure/table input required to compile it. It also includes frozen pair predictions, manifests, aggregate analyses, verifiers, and the research notes needed to reproduce the reported summaries. Raw AIT logs and raw command text are not included. Upstream dataset terms remain applicable.

## Compile the manuscript

From the extracted bundle root:

```sh
cd research/paper
tectonic --reruns 2 human-agent-fingerprint-draft.tex
```

The manuscript uses 10 pt type and 0.72 in margins. The references end on page 8, and the complete PDF has 15 pages with appendices. It includes all eight figures and the owl logo from the overnight draft.

## Recompute summary tables and figures

The frozen v1 pair predictions and the v2 five-share and NYU predictions are included. From the extracted bundle root, install the versions recorded in `research/results/matched_fingerprint_v1/environment.json` and run:

```sh
.venv/bin/python research/experiments/matched_fingerprint_v2/analyze_cth_five_share.py
.venv/bin/python research/experiments/matched_fingerprint_v2/build_unified_bst_ablation.py
.venv/bin/python research/experiments/matched_fingerprint_v2/analyze_independent_nyu_agent.py
.venv/bin/python research/experiments/matched_fingerprint_v2/build_paper_results.py
```

The final command regenerates the TeX tables and figures. In the full repository it also updates the research-page results section; page layout and dependencies remain in the repository. The public-source NYU task manifest and frozen protocol are included. NYU is a one-shot confirmation set and must not be used to select a different model, threshold, representation, or subgroup.

## Evidence scope

Same-task and same-challenge labels are retrieval proxies. They do not verify shared attack activity, campaign, operator, or Agent instance. The external NYU evaluation is Agent--Agent only; independent Human--Agent confirmation remains unavailable. See `research/reports/matched-fingerprint-v2.md` for the bounded results and negative findings.
"""


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PAPER / "human-agent-fingerprint-draft.pdf", DEST / "human-agent-fingerprint-draft.pdf")
    shutil.copy2(PAPER / "human-agent-fingerprint-draft.tex", DEST / "human-agent-fingerprint-draft.tex")
    entries = collect()
    with tempfile.TemporaryDirectory(prefix="matched-fingerprint-source-") as td:
        staging = Path(td) / "bundle"
        staging.mkdir()
        for src, rel in entries.items():
            out = staging / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
        readme = staging / "README.md"
        readme.write_text(make_readme())
        checks = []
        for p in sorted(x for x in staging.rglob("*") if x.is_file()):
            checks.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(staging).as_posix()}")
        (staging / "SHA256SUMS.txt").write_text("\n".join(checks) + "\n")
        with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for p in sorted(x for x in staging.rglob("*") if x.is_file()):
                zf.write(p, p.relative_to(staging).as_posix())
        with tempfile.TemporaryDirectory(prefix="matched-fingerprint-compile-") as td2:
            extracted = Path(td2)
            with zipfile.ZipFile(ZIP_PATH) as zf:
                zf.extractall(extracted)
            compile_dir = extracted / "research/paper"
            subprocess.run(["tectonic", "--reruns", "2", "human-agent-fingerprint-draft.tex"],
                           cwd=compile_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            built = compile_dir / "human-agent-fingerprint-draft.pdf"
            if not built.is_file() or built.stat().st_size < 50_000:
                raise RuntimeError("Extracted source bundle did not compile to a plausible PDF")
    print(json.dumps({"bundle": str(ZIP_PATH.relative_to(ROOT)), "bytes": ZIP_PATH.stat().st_size,
                      "included_files": len(entries) + 2, "independently_compiled": True,
                      "pdf": str((DEST / "human-agent-fingerprint-draft.pdf").relative_to(ROOT)),
                      "pdf_sha256": hashlib.sha256((DEST / "human-agent-fingerprint-draft.pdf").read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
