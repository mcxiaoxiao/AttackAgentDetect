# Paper sources

`mixed-gallery-main.tex` presents the execution-mode and behavior-correlation study on 177 public CTF traces. `main.tex` combines the pairwise association method, task-held-out results for 97 traces across 40 task groups, and the execution-mode analysis.

Compile either entry point from this directory with Tectonic:

~~~bash
tectonic -X compile --keep-logs --keep-intermediates mixed-gallery-main.tex
tectonic -X compile --keep-logs --keep-intermediates main.tex
~~~

The figures are in `figures/`; their generating scripts and machine-readable results are in `../scripts/` and `../results/`.
