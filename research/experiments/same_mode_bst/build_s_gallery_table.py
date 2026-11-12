#!/usr/bin/env python3
"""Build the appendix TeX table from the frozen optimization metrics."""
from __future__ import annotations

import argparse
import json
import pathlib


ROWS = [
    ("E", "E"),
    ("base_pairwise", "Base, pair cutoff"),
    ("base", "Base, gallery guard"),
    ("fullS_pairwise", "Full S, pair cutoff"),
    ("fullS", "Full S, gallery guard"),
    ("S_scalar", "S scalar summaries"),
    ("S_revisit_reuse", "S revisit / reuse"),
    ("S_smoothed_transition", "S + smoothed transitions"),
    ("S_ordered_ngrams", "S + category n-grams"),
    ("selected_nested", "Nested selection"),
]
METRICS = ["hit1", "ba", "fmr", "fsr", "no_match_accept"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    datasets = {
        name: json.loads((args.input_dir / f"{name}_metrics.json").read_text())["results"]
        for name in ("cth", "ait")
    }
    rows: list[str] = []
    for data_index, (dataset, metrics) in enumerate(datasets.items()):
        present = [(key, label, metrics[key]) for key, label in ROWS if key in metrics]
        best = {
            metric: (max if metric in ("hit1", "ba") else min)(
                row[2][metric] for row in present
            )
            for metric in METRICS
        }
        if data_index:
            rows.append(r"\midrule")
        for row_index, (key, label, result) in enumerate(present):
            prefix = dataset.upper() if row_index == 0 else ""
            values = []
            for metric in METRICS:
                value = result[metric] * 100
                formatted = f"{value:.2f}"
                if abs(result[metric] - best[metric]) < 1e-12:
                    formatted = rf"\textbf{{{formatted}}}"
                values.append(formatted)
            rows.append(
                f"{prefix} & {label} & " + " & ".join(values) + r" \\"
            )

    table = r"""\begin{table}[!htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.06}
\caption{S-dimension and gallery-threshold follow-up on the same outer candidates. Base is E+B for CTH and E+B+T for AIT. Pair cutoff uses the original inner pairwise FMR rule; gallery guard uses inner task-macro FMR and no-match constraints. S variants append the named S features to Base. Rates are task-macro percentages; higher Hit@1/BA and lower FMR/FSR/no-match are favorable. Bold marks the best value per dataset and metric. The 2$\times$ hard-negative arms match their parent arms at reported precision and are omitted.}
\label{tab:s-gallery-followup}
\begin{tabularx}{\linewidth}{@{}p{.40in}p{1.52in}*{5}{>{\centering\arraybackslash}X}@{}}
\toprule
Data & Profile / cutoff & Hit@1 $\uparrow$ & BA $\uparrow$ & FMR $\downarrow$ & FSR $\downarrow$ & No-match $\downarrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
