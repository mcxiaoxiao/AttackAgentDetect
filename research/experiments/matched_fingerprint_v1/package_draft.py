#!/usr/bin/env python3
"""Synchronize draft PDF/TeX and package every referenced figure."""
import pathlib
import re
import shutil
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
PAPER = ROOT / "research/paper"
SITE = ROOT / "site/assets/behavioral-fingerprint-v2"
tex = PAPER / "human-agent-fingerprint-draft.tex"
pdf = PAPER / "human-agent-fingerprint-draft.pdf"
figures = list(dict.fromkeys(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex.read_text())))
assert pdf.exists()
SITE.mkdir(parents=True, exist_ok=True)
(SITE / "figures").mkdir(parents=True, exist_ok=True)
shutil.copy2(tex, SITE / tex.name)
shutil.copy2(pdf, SITE / pdf.name)
readme = (
    "Research draft. Compile human-agent-fingerprint-draft.tex with the figures directory beside it.\n"
    "The document uses an inline bibliography; no raw dataset or session is included.\n"
    "Experiment code and aggregate results are under research/experiments/matched_fingerprint_v1 "
    "and research/results/matched_fingerprint_v1 in the project.\n"
    "Owl mark: Selina Bauder / OpenMoji, CC BY-SA 4.0, unmodified. "
    "Source: https://openmoji.org/library/emoji-1F989/\n"
)
(SITE / "DRAFT-SOURCE-README.txt").write_text(readme)
with zipfile.ZipFile(SITE / "human-agent-fingerprint-draft-source.zip", "w",
                     compression=zipfile.ZIP_DEFLATED) as z:
    z.write(tex, tex.name)
    z.writestr("README.txt", readme)
    for name in figures:
        p = PAPER / "figures" / name
        assert p.is_file(), p
        dest = SITE / "figures" / name
        if dest.exists() and dest.read_bytes() != p.read_bytes() and name != "matched-mixture-v1.pdf":
            raise RuntimeError(f"Existing site figure differs; refusing to overwrite: {dest}")
        if not dest.exists() or dest.read_bytes() != p.read_bytes():
            shutil.copy2(p, dest)
        z.write(p, "figures/" + name)
print("Packaged", len(figures), "referenced assets with draft TeX and PDF")
