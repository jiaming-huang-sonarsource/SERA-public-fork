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
