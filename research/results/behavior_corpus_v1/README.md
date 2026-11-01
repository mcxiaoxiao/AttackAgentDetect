# Source-linked behavior corpus v1

See `research/reports/behavior-corpus-expansion-v1.md` for counts, definitions, source terms and limitations.

`evidence.jsonl` contains label-free normalized inputs; `trace_registry.csv` is an evaluator/provenance sidecar. Keep the two roles separate. `excluded_records.csv` records records without a usable common-command view. `provenance.json` pins every source file.

To rebuild from a fresh checkout, fetch the exact source files and then run the builder:

```sh
python3 research/scripts/fetch_behavior_corpus_v1.py --download
python3 research/scripts/build_behavior_corpus_v1.py
```

The full pinned input cache is 1,443,787,470 bytes across 120 files. Existing files are validated and reused. Run the fetcher without `--download` to check the cache without downloading anything, or use `--source hf` for the eight HF files only. The versioned manifests are under `research/data/manifests/` and `research/experiments/multidimensional_rerun_v1/source_manifest.json`. Every input is checked by size and SHA-256 before use.

No model evaluation has been run on the expanded corpus. Historical scores remain separate.
