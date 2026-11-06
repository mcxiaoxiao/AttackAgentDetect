#!/usr/bin/env python3
"""Build and validate the local, raw-log-free short-paper source archive."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "research/paper"
RESULTS = ROOT / "research/results"
EXP = ROOT / "research/experiments"
OUT = ROOT / "research/reproducibility/same-challenge-retrieval-20261106.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"required executable not found on PATH: {name}")
    return found


def add_file(source: Path, arcname: str, stage: Path, included: set[str]) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target = stage / arcname
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    included.add(arcname)


def add_tree_files(source_dir: Path, prefix: str, stage: Path,
                   included: set[str], exclude_names: set[str] | None = None) -> None:
    exclude_names = exclude_names or set()
    for source in sorted(source_dir.iterdir()):
        if source.is_file() and source.name not in exclude_names:
            add_file(source, f"{prefix}/{source.name}", stage, included)


def run(command: list[str], cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}")
    return result.stdout


def page_count(pdfinfo: str, path: Path) -> int:
    output = run([pdfinfo, str(path)], cwd=path.parent)
    for line in output.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"pdfinfo did not report a page count: {path}")


def compile_and_verify(root: Path, tectonic: str, pdfinfo: str,
                       verifier: Path, output: Path) -> dict:
    paper = root / "paper"
    run([tectonic, "--reruns", "2", "same-challenge-retrieval-shortpaper.tex"], cwd=paper)
    run([tectonic, "--reruns", "2", "same-challenge-retrieval-supplement.tex"], cwd=paper)
    main_pages = page_count(pdfinfo, paper / "same-challenge-retrieval-shortpaper.pdf")
    supplement_pages = page_count(pdfinfo, paper / "same-challenge-retrieval-supplement.pdf")
    if main_pages > 8:
        raise AssertionError(f"main paper exceeds the eight-page limit: {main_pages}")
    if supplement_pages < 1:
        raise AssertionError("supplement PDF is empty")
    run([sys.executable, str(verifier), "--output", str(output)], cwd=root)
    adjustment_script = root / "code/analyze_pair_support_multiplicity.py"
    adjustment_path = root / "results/pair_support/multiplicity_adjustment.json"
    if adjustment_script.exists():
        expected_adjustment = json.loads(adjustment_path.read_text(encoding="utf-8"))
        run([sys.executable, str(adjustment_script)], cwd=root)
        rebuilt = json.loads(adjustment_path.read_text(encoding="utf-8"))
        if rebuilt != expected_adjustment:
            raise AssertionError("pair-support max-T output changed during clean-stage recomputation")
    return {"main_pages": main_pages, "supplement_pages": supplement_pages,
            "recomputed_values": json.loads(output.read_text(encoding="utf-8"))}


def README() -> str:
    return """# Same-Challenge Retrieval — frozen source and result bundle

This archive contains the short paper, supplementary material, figures, frozen pair-level predictions, experiment protocols, source scripts, and point-estimate verifiers.

## Recheck the reported point estimates

From this directory, run:

```sh
python3 code/verify_shortpaper_tables.py --output results/recomputed_table_values.json
```

The table verifier uses only the Python standard library. It re-aggregates the included frozen pair predictions; it does not fit models. Its checks cover the five CTH action-bigram mixture shares, matched CTH XGBoost rates, both Agent-only XGBoost panels, the five-arm mechanism contrasts, each shuffled-seed point contrast, and the four XGBoost pair-support increments. A separate script recomputes the paired four-contrast max-T intervals from the included CTH predictions.

## Compile the PDFs

With Tectonic installed:

```sh
cd paper
tectonic --reruns 2 same-challenge-retrieval-shortpaper.tex
tectonic --reruns 2 same-challenge-retrieval-supplement.tex
```

The main paper includes references and must remain within eight pages. The supplement holds the complete measurement schema, full mixture metrics, threshold/no-match diagnostics, overlap sensitivity, mechanism controls, the exploratory pair-support analysis, source-specific tests, and CTH model-snapshot retrieval results.

## Scope of reproducibility

Pair-level files contain task labels, hashed trace IDs, scores, decisions, and thresholds; this archive excludes raw NYU transcripts, raw command strings, and raw event logs. The frozen outputs allow exact point-estimate re-aggregation without fitting. Re-running feature extraction and model fitting from source logs requires the upstream CTH/AIT source data, the pinned NYU repository revision, and the original experiment repository environment. The included `source/` scripts document those pipelines but cannot recreate excluded source-derived inputs by themselves.

The CTH label is same-task/challenge identity. It does not establish a common operator, Agent instance, attack, campaign, or real-world activity. CTH role composition is observational because collection interface and recorded role covary.
"""


def build() -> Path:
    tectonic, pdfinfo = tool("tectonic"), tool("pdfinfo")
    with tempfile.TemporaryDirectory(prefix="same-challenge-bundle-") as temp:
        stage = Path(temp) / "bundle"
        stage.mkdir()
        included: set[str] = set()

        for name in ("same-challenge-retrieval-shortpaper.tex",
                     "same-challenge-retrieval-shortpaper.pdf",
                     "same-challenge-retrieval-supplement.tex",
                     "same-challenge-retrieval-supplement.pdf"):
            add_file(PAPER / name, f"paper/{name}", stage, included)
        add_file(PAPER / "figures/mixture-ranking-shortpaper.pdf",
                 "paper/figures/mixture-ranking-shortpaper.pdf", stage, included)

        protocol_names = (
            "shortpaper_review_protocol.md",
            "independent_cth_aa_protocol.md",
            "independent_nyu_aa_protocol.md",
            "nyu_snapshot_retrieval_protocol.md",
            "cth_model_snapshot_retrieval_protocol.md",
            "pair_support_diagnostic_record.md",
        )
        for name in protocol_names:
            add_file(EXP / f"matched_fingerprint_v2/{name}", f"protocol/{name}", stage, included)

        code_files = (
            (EXP / "matched_fingerprint_v2/verify_shortpaper_tables.py", "code/verify_shortpaper_tables.py"),
            (EXP / "matched_fingerprint_v2/analyze_shortpaper_frozen.py", "code/analyze_shortpaper_frozen.py"),
            (EXP / "matched_fingerprint_v2/run_shortpaper_mechanism.py", "code/run_shortpaper_mechanism.py"),
            (EXP / "matched_fingerprint_v2/analyze_independent_cth_agent.py", "code/analyze_independent_cth_agent.py"),
            (EXP / "matched_fingerprint_v2/analyze_independent_nyu_agent.py", "code/analyze_independent_nyu_agent.py"),
            (EXP / "matched_fingerprint_v2/analyze_frozen_ranking_metrics.py", "code/analyze_frozen_ranking_metrics.py"),
            (EXP / "matched_fingerprint_v2/run_snapshot_retrieval.py", "code/run_snapshot_retrieval.py"),
            (EXP / "matched_fingerprint_v2/run_fold_internal_dimension_selection.py", "code/run_fold_internal_dimension_selection.py"),
            (EXP / "matched_fingerprint_v2/audit_window_feature_coverage.py", "code/audit_window_feature_coverage.py"),
            (EXP / "matched_fingerprint_v2/analyze_pair_support_multiplicity.py", "code/analyze_pair_support_multiplicity.py"),
            (EXP / "matched_fingerprint_v2/run_cth_model_snapshot_retrieval.py", "code/run_cth_model_snapshot_retrieval.py"),
            (EXP / "matched_fingerprint_v2/run_pair_support.py", "source/run_pair_support.py"),
            (EXP / "matched_fingerprint_v2/analyze_pair_support.py", "source/analyze_pair_support.py"),
            (EXP / "matched_fingerprint_v2/run_arguments.py", "source/run_arguments.py"),
            (EXP / "matched_fingerprint_v2/argument_profile.py", "source/argument_profile.py"),
            (EXP / "matched_fingerprint_v1/run.py", "source/matched_fingerprint_v1_run.py"),
            (EXP / "overnight_fingerprint_v1/run.py", "source/overnight_fingerprint_v1_run.py"),
            (ROOT / "attackagent/fingerprint_v2.py", "source/attackagent_fingerprint_v2.py"),
            (ROOT / "attackagent/traces.py", "source/attackagent_traces.py"),
        )
        for source, arcname in code_files:
            add_file(source, arcname, stage, included)
        add_file(ROOT / "research/reports/same-challenge-retrieval-shortpaper-review-2026-11-05.md",
                 "reports/same-challenge-retrieval-shortpaper-review-2026-11-05.md", stage, included)
        add_file(ROOT / "research/reports/fingerprint-target-and-ranker-2026-11-05.md",
                 "reports/fingerprint-target-and-ranker-2026-11-05.md", stage, included)
        add_file(ROOT / "research/reports/fingerprint-dimension-and-ranking-followup-2026-11-05.md",
                 "reports/fingerprint-dimension-and-ranking-followup-2026-11-05.md", stage, included)
        add_file(RESULTS / "association_fingerprint_v3/exact_mixture.json",
                 "results/frozen/exact_mixture.json", stage, included)
        add_file(RESULTS / "overnight_fingerprint_v1/task_folds.csv",
                 "results/frozen/task_folds.csv", stage, included)

        add_tree_files(RESULTS / "matched_fingerprint_v2/shortpaper_frozen_20261105",
                       "results/frozen", stage, included)
        add_tree_files(RESULTS / "matched_fingerprint_v2/shortpaper_mechanism_20261105",
                       "results/mechanism", stage, included)

        pair_support = RESULTS / "matched_fingerprint_v2/pair_support"
        for name in ("pair_predictions.csv.gz", "run_manifest.json", "analysis.json",
                     "multiplicity_adjustment.json"):
            add_file(pair_support / name, f"results/pair_support/{name}", stage, included)

        for result_name, alias in (
            ("cth_model_snapshot_retrieval_20261106", "cth_snapshot_conditional"),
            ("cth_model_snapshot_retrieval_direct_20261106", "cth_snapshot_direct"),
        ):
            result_dir = RESULTS / f"matched_fingerprint_v2/{result_name}"
            for name in ("analysis.json", "gallery_metrics.csv.gz", "predictions.csv.gz"):
                add_file(result_dir / name, f"results/{alias}/{name}", stage, included)

        for fold in range(3):
            for variant, alias in (("evidence", "evidence"), ("BS", "BS")):
                source = RESULTS / f"matched_fingerprint_v1/cth/fold-{fold}-xgb-{variant}.csv.gz"
                add_file(source, f"results/cth_mixed/fold-{fold}-xgb-{alias}.csv.gz", stage, included)
            audit = RESULTS / f"matched_fingerprint_v1/cth/fold-{fold}-audit.json"
            add_file(audit, f"results/cth_mixed/fold-{fold}-audit.json", stage, included)

        for source_name, archive_name in (("independent_cth_agent", "cth_aa"),
                                          ("independent_nyu_agent", "nyu_aa")):
            source_dir = RESULTS / f"matched_fingerprint_v2/{source_name}"
            for name in ("analysis.json", "run_manifest.json", "predictions.csv.gz"):
                add_file(source_dir / name, f"results/{archive_name}/{name}", stage, included)

        for fold in range(3):
            for variant in ("evidence", "BS", "BST"):
                source = RESULTS / f"matched_fingerprint_v1/ait/fold-{fold}-xgb-{variant}.csv.gz"
                add_file(source, f"results/ait/fold-{fold}-xgb-{variant}.csv.gz", stage, included)
            add_file(RESULTS / f"matched_fingerprint_v1/ait/fold-{fold}-audit.json",
                     f"results/ait/fold-{fold}-audit.json", stage, included)
        add_file(RESULTS / "matched_fingerprint_v1/ait/input_audit.json",
                 "results/ait/input_audit.json", stage, included)

        for name in ("analysis.json", "task_pair_metrics.csv.gz"):
            add_file(RESULTS / f"matched_fingerprint_v2/snapshot_retrieval_20261105/{name}",
                     f"results/snapshot/{name}", stage, included)
        for name in ("dimension_selection.json", "outer_fold_predictions.csv.gz",
                     "window_feature_coverage.json"):
            add_file(RESULTS / f"matched_fingerprint_v2/dimension_selection_20261104/{name}",
                     f"results/dimension_selection/{name}", stage, included)

        (stage / "README.md").write_text(README(), encoding="utf-8")
        included.add("README.md")

        # Compile from the clean staging directory, then reproduce the table points.
        stage_verifier = stage / "code/verify_shortpaper_tables.py"
        report_output = stage / "results/recomputed_table_values.json"
        report_output.parent.mkdir(parents=True, exist_ok=True)
        check_result = compile_and_verify(stage, tectonic, pdfinfo,
                                          stage_verifier, report_output)
        included.add("results/recomputed_table_values.json")
        (stage / "results/bundle_build_check.json").write_text(
            json.dumps({"clean_stage_compile": {"main_pages": check_result["main_pages"],
                                                 "supplement_pages": check_result["supplement_pages"]},
                        "point_estimate_checks": "passed", "raw_logs_included": False},
                       indent=2, sort_keys=True) + "\n", encoding="utf-8")
        included.add("results/bundle_build_check.json")
        # The PDFs in the clean staging directory are the compiled deliverables.

        checksum_lines = []
        for arcname in sorted(included):
            checksum_lines.append(f"{sha256(stage / arcname)}  {arcname}")
        (stage / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        included.add("SHA256SUMS.txt")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        temp_zip = OUT.with_suffix(".zip.part")
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as archive:
            for arcname in sorted(included):
                archive.write(stage / arcname, arcname)

        # Validate the actual archive after extraction, independent of its staging copy.
        with tempfile.TemporaryDirectory(prefix="same-challenge-extracted-") as extracted:
            extracted_root = Path(extracted)
            with zipfile.ZipFile(temp_zip) as archive:
                archive.extractall(extracted_root)
            for line in (extracted_root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
                expected, arcname = line.split("  ", 1)
                if sha256(extracted_root / arcname) != expected:
                    raise AssertionError(f"archive checksum mismatch: {arcname}")
            extracted_check = extracted_root / "results/recomputed_table_values_check.json"
            validate = compile_and_verify(extracted_root, tectonic, pdfinfo,
                                          extracted_root / "code/verify_shortpaper_tables.py",
                                          extracted_check)
            if validate["recomputed_values"] != check_result["recomputed_values"]:
                raise AssertionError("re-aggregated values changed after ZIP extraction")
            if validate["main_pages"] != check_result["main_pages"]:
                raise AssertionError("main PDF page count changed after extraction")
            if validate["supplement_pages"] != check_result["supplement_pages"]:
                raise AssertionError("supplement PDF page count changed after extraction")

        os.replace(temp_zip, OUT)
        with zipfile.ZipFile(OUT, "r") as archive:
            for name in ("same-challenge-retrieval-shortpaper.pdf",
                         "same-challenge-retrieval-supplement.pdf"):
                (PAPER / name).write_bytes(archive.read(f"paper/{name}"))
    return OUT


if __name__ == "__main__":
    print(build())
