# Short-trace extension results

The frozen four-command-window, eight-command-minimum extension expanded the Agent-only panels to 219 traces / 86 tasks in CTH and 120 traces / 27 tasks in Antieval. Command + 24-value matrix Hit@1 changes were -3.57 pp (95% paired task-cluster CI -6.82 to -0.43) and -2.89 pp (-7.50 to +1.79), respectively. MRR and AUC are included in `metrics.json`; all methods and per-query predictions are preserved.

The result does not support a matrix benefit in these short-trace panels. The four-command window captures local behavior and is a separate sensitivity result from the original two-window eight-command representation.

Reproduce with `.venv/bin/python research/experiments/short_trace_extension_v1/run.py`.
