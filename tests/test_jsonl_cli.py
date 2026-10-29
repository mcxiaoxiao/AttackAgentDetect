import contextlib
import io
import json
import pathlib
import tempfile
import unittest

from attackagent.jsonl_cli import main, read_jsonl


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "jsonl"


class JsonlCliSmokeTest(unittest.TestCase):
    def test_bundled_example_runs_and_scores_are_marked_uncalibrated(self):
        stdout = io.StringIO()
        args = [
            "--train", str(EXAMPLES / "train.jsonl"),
            "--queries", str(EXAMPLES / "queries.jsonl"),
            "--candidates", str(EXAMPLES / "candidates.jsonl"),
            "--top-k", "2",
        ]
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(args), 0)
        result = json.loads(stdout.getvalue())
        row = result["results"][0]
        self.assertFalse(row["query"]["score_calibrated"])
        self.assertEqual(len(row["candidate_ranking"]), 2)
        self.assertEqual(row["scope"], "same-benchmark, label-free trace ranking; task labels are not consumed at inference")

    def test_inference_ignores_script_annotation_in_input_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            query_path = pathlib.Path(directory) / "script.jsonl"
            query = {
                "schema_version": 1,
                "trace_id": "script-demo",
                "benchmark": "demo-benchmark",
                "mode": "script",
                "commands": ["pwd", "ls"],
                "actions": ["inspect", "enumerate"],
            }
            query_path.write_text(json.dumps(query) + "\n", encoding="utf-8")
            stdout = io.StringIO()
            args = [
                "--train", str(EXAMPLES / "train.jsonl"),
                "--queries", str(query_path),
                "--candidates", str(EXAMPLES / "candidates.jsonl"),
            ]
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(args), 0)
            row = json.loads(stdout.getvalue())["results"][0]
            self.assertIn(row["query"]["prediction"], {"human", "agent"})
            self.assertIsNone(row["query"]["abstention_reason"])
            self.assertTrue(row["candidate_ids_preserved"])
            self.assertTrue(row["candidate_ranking"])

    def test_training_requires_independent_activity_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "bad.jsonl"
            row = {
                "schema_version": 1,
                "trace_id": "trace",
                "benchmark": "benchmark",
                "mode": "human",
                "commands": ["pwd"],
                "actions": ["inspect"],
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "activity_id"):
                read_jsonl(path, training=True)


if __name__ == "__main__":
    unittest.main()
