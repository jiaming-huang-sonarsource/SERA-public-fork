#!/usr/bin/env python3
"""Export original T2 messages as readable text and lossless formatted JSON."""

import argparse
import html
import json
from pathlib import Path
import re

from t2_stats import TokenCounter, trajectory_stats


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
    combined_path = Path(trajs) / 'preds.json' if trajs else None
    combined = json.loads(combined_path.read_text()) if combined_path and combined_path.exists() else {}
    for line in Path(references).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        iid = row['instance_id']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', iid) or iid in {'.', '..'} or iid in seen:
            raise ValueError(f'Invalid or duplicate instance ID: {iid}')
        seen.add(iid)
        messages = json.loads(row['messages']) if isinstance(row['messages'], str) else row['messages']
        tasks.append({'id': iid, 'source': 'sera', 'messages': messages, 'patch': row.get('rollout_patch')})
        if trajs:
            path = Path(trajs) / iid / f'{iid}.traj'
            if path.exists():
                traj = json.loads(path.read_text())
                pred_path = path.with_suffix('.pred')
                pred = json.loads(pred_path.read_text()) if pred_path.exists() else combined.get(iid, {})
                info = traj.get('info', {})
                status = info.get('exit_status')
                tasks.append({'id': iid + '.qwen', 'source': 'qwen', 'messages': traj['history'],
                              'patch': pred.get('model_patch', info.get('submission')),
                              'submitted': str(status).lower() == 'submitted' if status is not None else None})
    if api:
        for path in sorted(Path(api).glob('*.json')):
            # Full wire records, including failures and token IDs, remain untouched.
            tasks.append({'id': 'api.' + path.stem, 'source': 'api', 'messages': json.loads(path.read_text())})
    if not tasks:
        raise ValueError('Reference file is empty')
    return tasks


def build_reader(references, out, trajs=None, api=None, tokenizer=None):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    exports = out.with_suffix('')
    exports.mkdir(exist_ok=True)
    tasks = load_tasks(references, trajs, api)
    counter = TokenCounter(tokenizer)
    for task in tasks:
        messages = task.pop('messages')
        patch, submitted = task.pop('patch', None), task.pop('submitted', None)
        task['stats'] = trajectory_stats(messages, patch, counter, submitted) if task['source'] != 'api' else None
        (exports / (task['id'] + '.json')).write_text(json.dumps(messages, ensure_ascii=False, indent=2) + '\n')
        task['text'] = readable(messages) + '\n'
        (exports / (task['id'] + '.txt')).write_text(task['text'])
    # Pair by instance ID, keeping every original field and each model's ordering.
    by_id = {task['id']: task for task in tasks}
    pairs = [dict(task, qwen_text=by_id.get(task['id'] + '.qwen', {}).get('text'),
                  qwen_stats=by_id.get(task['id'] + '.qwen', {}).get('stats'))
             for task in tasks if task['source'] != 'qwen']
    data = json.dumps(pairs, ensure_ascii=False).replace('<', '\\u003c')
    template = Path(__file__).with_name('t2_reader.html').read_text()
    assert template.count('__T2_DATA__') == 1
    # The first pair is also present without JavaScript.
    replacements = {'__T2_DATA__': data, '__FIRST_TRAJECTORY__': html.escape(pairs[0]['text']),
                    '__FIRST_QWEN__': html.escape(pairs[0]['qwen_text'] or '尚无这条任务的 Qwen 轨迹。'),
                    '__TOKENIZER__': html.escape(counter.description)}
    page = re.sub(r'__T2_DATA__|__FIRST_TRAJECTORY__|__FIRST_QWEN__|__TOKENIZER__',
                  lambda match: replacements[match[0]], template)
    out.write_text(page)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', required=True)
    parser.add_argument('--out', required=True, help='Minimal HTML page; text/JSON files go in a sibling folder')
    parser.add_argument('--trajs', help='Optional Qwen run-batch output directory')
    parser.add_argument('--api', help='Optional directory of complete original API records')
    parser.add_argument('--tokenizer', help='Local tokenizer.json for exact counts (requires tokenizers); otherwise estimates')
    args = parser.parse_args()
    print(build_reader(args.references, args.out, args.trajs, args.api, args.tokenizer).resolve())


if __name__ == '__main__':
    main()
