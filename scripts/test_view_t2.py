import json
from pathlib import Path
import re
import tempfile
import unittest

from view_t2 import build_reader, load_tasks, readable


class ReaderTests(unittest.TestCase):
    def reference(self, iid='example_1'):
        return {'instance_id': iid, 'messages': json.dumps([
            {'role': 'system', 'content': 'system', 'agent': 'main'},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'line one\n\nline two\n'}]},
            {'role': 'assistant', 'content': '</script><script>alert(1)</script>',
             'thought': '<think>original</think>', 'tool_calls': [
                 {'id': 'call_1', 'function': {'name': 'bash', 'arguments': json.dumps({'command': 'a\nb'})}}]},
            {'role': 'tool', 'content': 'literal \\n stays literal', 'tool_call_ids': ['call_1']}])}

    def test_exports_preserve_messages_and_escape_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.reference()
            references = root / 'references.jsonl'
            references.write_text(json.dumps(row) + '\n')
            original = json.loads(row['messages'])
            page = build_reader(references, root / 'reader.html').read_text()
            self.assertEqual(json.loads((root / 'reader/example_1.json').read_text()), original)
            embedded = re.search(r'<script id="dataset" type="application/json">(.*?)</script>', page, re.S)[1]
            self.assertNotIn('<script', embedded)
            text = (root / 'reader/example_1.txt').read_text()
            self.assertEqual(json.loads(embedded)[0]['text'], text)
            self.assertRegex(text, r'line one\n[ ]*\n[ ]*line two')
            self.assertIn('thought:', text)
            self.assertIn('literal \\\\n stays literal', text)
            self.assertNotIn('"command": "a\\nb"', text)
            self.assertEqual(json.loads(row['messages']), original)

    def test_qwen_exports_only_existing_histories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = root / 'references.jsonl'
            refs.write_text('\n'.join(json.dumps(self.reference(f'example_{i}')) for i in (1, 2)))
            traj = root / 'raw/example_1/example_1.traj'
            traj.parent.mkdir(parents=True)
            history = [{'role': 'assistant', 'content': 'qwen'}]
            traj.write_text(json.dumps({'history': history}))
            tasks = load_tasks(refs, root / 'raw')
            self.assertEqual([t['id'] for t in tasks], ['example_1', 'example_1.qwen', 'example_2'])
            self.assertEqual(tasks[1]['messages'], history)

    def test_side_by_side_pairs_by_id_and_keeps_both_histories_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = root / 'references.jsonl'
            rows = [self.reference(f'example_{i}') for i in (1, 2)]
            refs.write_text('\n'.join(json.dumps(row) for row in rows))
            history = [{'role': 'assistant', 'content': '<think>reason\nfully</think>ordinary answer',
                        'thought': 'original auxiliary field',
                        'tool_calls': [{'id': 'call_q', 'function': {'name': 'bash', 'arguments': '{"command":"pwd"}'}}]},
                       {'role': 'tool', 'content': '</script><script>untrusted</script>'}]
            traj = root / 'raw/example_2/example_2.traj'
            traj.parent.mkdir(parents=True)
            traj.write_text(json.dumps({'history': history}))
            page = build_reader(refs, root / 'reader.html', root / 'raw').read_text()
            embedded = json.loads(re.search(r'<script id="dataset" type="application/json">(.*?)</script>', page, re.S)[1])
            self.assertEqual([row['id'] for row in embedded], ['example_1', 'example_2'])
            self.assertIsNone(embedded[0]['qwen_text'])
            self.assertIsNone(embedded[0]['qwen_stats'])
            self.assertEqual(embedded[1]['qwen_stats']['turns'], 1)
            self.assertEqual(embedded[1]['qwen_stats']['tool_calls'], 1)
            self.assertIsNotNone(embedded[1]['qwen_stats']['filtered_tokens'])
            self.assertEqual(embedded[1]['text'], readable(json.loads(rows[1]['messages'])) + '\n')
            self.assertEqual(embedded[1]['qwen_text'], readable(history) + '\n')
            self.assertEqual(json.loads((root / 'reader/example_2.qwen.json').read_text()), history)
            self.assertIn('尚无这条任务的 Qwen 轨迹。', page)
            self.assertNotIn('__FIRST_QWEN__', page)
            self.assertNotIn('__TOKENIZER__', page)

    def test_final_patch_and_submission_status_are_paired_without_changing_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.reference()
            row['rollout_patch'] = 'diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-old\n+new\n'
            refs = root / 'references.jsonl'
            refs.write_text(json.dumps(row))
            path = root / 'trajs/example_1/example_1.traj'
            path.parent.mkdir(parents=True)
            history = json.loads(row['messages'])
            path.write_text(json.dumps({'history': history, 'info': {'exit_status': 'cost_limit', 'submission': 'stale'}}))
            path.with_suffix('.pred').write_text(json.dumps({'model_patch': ''}))
            page = build_reader(refs, root / 'reader.html', root / 'trajs').read_text()
            pair = json.loads(re.search(r'<script id="dataset" type="application/json">(.*?)</script>', page, re.S)[1])[0]
            self.assertEqual(pair['stats']['patch'], {'added': 1, 'deleted': 1, 'total': 2})
            self.assertEqual(pair['qwen_stats']['patch']['total'], 0)
            self.assertIsNone(pair['qwen_stats']['filtered_tokens'])
            self.assertIn('未提交', pair['qwen_stats']['filter_error'])
            self.assertEqual(json.loads((root / 'reader/example_1.qwen.json').read_text()), history)

    def test_no_global_backslash_replacement(self):
        self.assertEqual(readable('actual\nnewline'), 'actual\nnewline')
        self.assertEqual(readable('literal \\n'), 'literal \\n')

    def test_api_record_export_keeps_native_history_response_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = root / 'references.jsonl'
            refs.write_text(json.dumps(self.reference()) + '\n')
            api = root / 'api'
            api.mkdir()
            record = {'request': {'messages': [{'role': 'assistant', 'reasoning': 'original\nreason',
                                               'content': 'commentary'}]},
                      'response': {'error': {'message': 'failed'}}, 'status': 400,
                      'tokenization': {'tokens': [10, 20]}}
            (api / 'call.json').write_text(json.dumps(record))
            build_reader(refs, root / 'reader.html', api=api)
            self.assertEqual(json.loads((root / 'reader/api.call.json').read_text()), record)
            self.assertIn('commentary', (root / 'reader/api.call.txt').read_text())


if __name__ == '__main__':
    unittest.main()
