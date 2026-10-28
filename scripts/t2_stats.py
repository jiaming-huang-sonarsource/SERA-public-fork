"""Read-only trajectory statistics using SERA's actual training postprocessing."""

from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sera.constants import HERMES_DEFAULT_SYSTEM_PROMPT
from sera.datagen.data.postprocess.utils import (
    add_train_key, reformat_assistant_message, transform_traj_hermes,
)


class TokenCounter:
    def __init__(self, tokenizer_path=None):
        self.estimated = tokenizer_path is None
        self.description = 'UTF-8 字节数 ÷ 4 估算；不能据此严格判断 32k 边界'
        self.tokenizer = None
        if tokenizer_path:
            from tokenizers import Tokenizer
            path = Path(tokenizer_path)
            self.tokenizer = Tokenizer.from_file(str(path))
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            name = ('Qwen3-32B' if digest == 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'
                    else '自定义')
            self.description = f'{name} tokenizer，SHA256 {digest[:12]}；两侧使用同一 tokenizer'

    def __call__(self, text):
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False).ids)
        return (len(text.encode('utf-8')) + 3) // 4


def tool_text(call):
    function = call['function']
    arguments = function['arguments']
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return '<tool_call>\n' + json.dumps({'name': function['name'], 'arguments': arguments}) + '\n</tool_call>'


def text_content(message):
    content = message.get('content') or ''
    if isinstance(content, list):
        return '\n'.join(part['text'] for part in content)
    return content


def training_messages(history):
    """Use the configured Hermes/keep_only_think pipeline without mutating history."""
    history = copy.deepcopy(history)
    for message in history:
        if message['role'] != 'assistant':
            continue
        # The recorder normally embeds reasoning in content already. Support explicit
        # reasoning too, without counting the same thought twice.
        content = text_content(message)
        reasoning = message.get('reasoning', message.get('reasoning_content'))
        if reasoning is not None and '<think>' not in content:
            content = f'<think>\n{reasoning}\n</think>\n\n{content}'
        message['content'] = content
        if not message.get('tool_calls') and content != 'Exit due to cost limit':
            raise ValueError('缺少结构化 tool call，无法按 Hermes 清洗')
    with redirect_stdout(io.StringIO()):
        transformed = transform_traj_hermes({'history': history}, HERMES_DEFAULT_SYSTEM_PROMPT)
        dataset = reformat_assistant_message(add_train_key([transformed]), 'keep_only_think')
    if not dataset:
        raise ValueError('未通过 SERA think/tool_call 格式校验')
    return dataset[0]['messages']


def patch_lines(patch):
    if patch is None:
        return None
    added = deleted = 0
    in_hunk = False
    for line in patch.splitlines():
        if line.startswith('diff ') or line.startswith('@@'):
            in_hunk = line.startswith('@@')
        elif in_hunk:
            added += line.startswith('+')
            deleted += line.startswith('-')
    return {'added': added, 'deleted': deleted, 'total': added + deleted}


def trajectory_stats(history, patch, counter, submitted=None):
    assistants = [m for m in history if m['role'] == 'assistant']
    calls = [call for m in assistants for call in (m.get('tool_calls') or [])]
    results = [m for m in history if m['role'] == 'tool']
    stats = {'turns': len(assistants), 'tool_calls': len(calls), 'estimated': counter.estimated,
             'patch': patch_lines(patch), 'filtered_tokens': None, 'filter_error': None}
    try:
        stats['tool_call_mean'] = (sum(counter(tool_text(c)) for c in calls) / len(calls)) if calls else None
    except (KeyError, TypeError, ValueError):
        stats['tool_call_mean'] = None
    # Average one response message, including the Hermes wrapper; no repeated prefixes.
    stats['tool_result_mean'] = (sum(counter('<tool_response>\n' + text_content(m) + '\n</tool_response>')
                                     for m in results) / len(results)) if results else None
    try:
        if submitted is False:
            raise ValueError('未提交，SERA 默认丢弃')
        messages = training_messages(history)
        chatml = ''.join(f'<|im_start|>{m["role"]}\n{m["content"]}<|im_end|>\n' for m in messages)
        stats['filtered_tokens'] = counter(chatml)
    except (KeyError, TypeError, ValueError, AssertionError) as error:
        stats['filter_error'] = str(error) or '无法按 SERA 格式清洗'
    return stats
