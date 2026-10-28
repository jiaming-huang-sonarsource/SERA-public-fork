#!/usr/bin/env python3
"""Export original T2 messages as readable text and lossless formatted JSON."""

import argparse
import html
import json
from pathlib import Path
import re


def readable(value, indent=0, field=None):
    """Render every field in source order, expanding strings for reading only."""
    pad = '  ' * indent
    if isinstance(value, str) and field == 'arguments':
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            pass
        else:
            # Keep the JSON export untouched; only expand this embedded JSON in text.
            return readable(decoded, indent)
    if isinstance(value, dict):
        if not value:
            return pad + '{}'
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)) or (isinstance(item, str) and ('\n' in item or key == 'arguments')):
                lines.append(pad + key + ':')
                lines.append(readable(item, indent + 1, key))
            else:
                lines.append(pad + key + ': ' + json.dumps(item, ensure_ascii=False))
        return '\n'.join(lines)
    if isinstance(value, list):
        if not value:
            return pad + '[]'
        return '\n'.join(pad + '-\n' + readable(item, indent + 1) for item in value)
    if isinstance(value, str):
        return '\n'.join(pad + line for line in value.split('\n'))
    return pad + json.dumps(value, ensure_ascii=False)


def load_tasks(references, trajs=None, api=None):
    tasks = []
    seen = set()
    for line in Path(references).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        iid = row['instance_id']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', iid) or iid in {'.', '..'} or iid in seen:
            raise ValueError(f'Invalid or duplicate instance ID: {iid}')
        seen.add(iid)
        messages = json.loads(row['messages']) if isinstance(row['messages'], str) else row['messages']
        tasks.append({'id': iid, 'messages': messages})
        if trajs:
            path = Path(trajs) / iid / f'{iid}.traj'
            if path.exists():
                tasks.append({'id': iid + '.qwen', 'messages': json.loads(path.read_text())['history']})
    if api:
        for path in sorted(Path(api).glob('*.json')):
            # Full wire records, including failures and token IDs, remain untouched.
            tasks.append({'id': 'api.' + path.stem, 'messages': json.loads(path.read_text())})
    if not tasks:
        raise ValueError('Reference file is empty')
    return tasks


def build_reader(references, out, trajs=None, api=None):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    exports = out.with_suffix('')
    exports.mkdir(exist_ok=True)
    tasks = load_tasks(references, trajs, api)
    for task in tasks:
        messages = task.pop('messages')
        (exports / (task['id'] + '.json')).write_text(json.dumps(messages, ensure_ascii=False, indent=2) + '\n')
        task['text'] = readable(messages) + '\n'
        (exports / (task['id'] + '.txt')).write_text(task['text'])
    data = json.dumps(tasks, ensure_ascii=False).replace('<', '\\u003c')
    template = Path(__file__).with_name('t2_reader.html').read_text()
    assert template.count('__T2_DATA__') == 1
    # The first trajectory is also present without JavaScript.
    replacements = {'__T2_DATA__': data, '__FIRST_TRAJECTORY__': html.escape(tasks[0]['text'])}
    page = re.sub(r'__T2_DATA__|__FIRST_TRAJECTORY__', lambda match: replacements[match[0]], template)
    out.write_text(page)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', required=True)
    parser.add_argument('--out', required=True, help='Minimal HTML page; text/JSON files go in a sibling folder')
    parser.add_argument('--trajs', help='Optional Qwen run-batch output directory')
    parser.add_argument('--api', help='Optional directory of complete original API records')
    args = parser.parse_args()
    print(build_reader(args.references, args.out, args.trajs, args.api).resolve())


if __name__ == '__main__':
    main()
