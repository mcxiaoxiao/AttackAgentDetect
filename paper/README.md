# Research manuscript packets

`mixed-gallery-main.tex` is the current paper-ready packet. It includes the motivation, data source and license, task-held-out protocol, measured results, bibliography, and appendix. Its figures are copied to `figures/mixed_gallery_*.pdf`; the generating code and machine-readable results remain in `scripts/run_mixed_gallery.py` and `results/`.

`main.tex`, `experiment-results.tex`, and `appendix.tex` document an earlier exploratory protocol. They are retained for research history and must not be presented as the current result. Their figures are produced by `scripts/generate_paper_figures.py`.

Compile the current packet from the repository's LaTeX editor with `mixed-gallery-main.tex`. From a system with Tectonic, run `tectonic -X compile --keep-logs --keep-intermediates mixed-gallery-main.tex` in this directory.
