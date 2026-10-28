import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("replay_t2", Path(__file__).with_name("replay_t2.py"))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ReplayTests(unittest.TestCase):
    def test_signed_duplicate_line_recall(self):
        result = replay.patch_metrics("+x\n", "+x\n+x\n-x\n")
        self.assertAlmostEqual(result["target_line_recall"], 1 / 3)
        self.assertEqual(result["target_line_precision"], 1)
        self.assertIsNone(replay.patch_metrics("+x\n", "")['target_line_recall'])

    def test_prepare_and_compare_preserve_inputs_and_failures(self):
        image = next(iter(replay.image_map()))
        history = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": [{"type": "text", "text": "full prompt\n\n"}]},
            {"role": "assistant", "content": "<think>reason</think>", "tool_calls": [
                {"function": {"name": "submit", "arguments": "{}"}}]},
        ]
        row = dict(instance_id="example_1", docker_image=image, messages=json.dumps(history),
                   target_patch="+expected\n", rollout_patch="+expected\n", problem_statement="short issue")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            source.write_text("\n".join(json.dumps(dict(row, instance_id=f"example_{i}")) for i in (1, 2)))
            prepared = root / "prepared"
            with redirect_stdout(io.StringIO()):
                replay.prepare(argparse.Namespace(source=str(source), out=str(prepared), limit=0, repo=None, per_repo=0))
            tasks = json.loads((prepared / "instances.json").read_text())
            self.assertEqual(tasks[0]["problem_statement"], "full prompt\n\n")
            self.assertEqual(tasks[0]["extra_fields"], {})
            self.assertNotIn("target_patch", tasks[0])
            output = root / "trajs" / "example_1"
            output.mkdir(parents=True)
            (output / "example_1.traj").write_text(json.dumps({
                "history": history, "info": {"exit_status": "exit_cost"}}))
            (output / "example_1.pred").write_text(json.dumps({"model_patch": "+other\n"}))
            report = root / "report"
            with redirect_stdout(io.StringIO()):
                replay.compare(argparse.Namespace(references=str(prepared / "references.jsonl"),
                                                  trajs=str(root / "trajs"), out=str(report)))
            summary = json.loads((report / "summary.json").read_text())
            self.assertEqual(summary["expected_tasks"], 2)
            self.assertEqual(summary["missing_trajectories"], 1)
            self.assertEqual(summary["qwen_submitted_rate"], 0)
            self.assertEqual(summary["qwen_nonempty_patch_rate"], .5)
            self.assertEqual(summary["qwen_target_recall_ge_0.5_rate"], 0)
            self.assertEqual(summary["initial_prompt_mismatches"], [])


if __name__ == "__main__":
    unittest.main()
