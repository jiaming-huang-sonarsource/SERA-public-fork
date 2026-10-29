import copy
import unittest

from t2_stats import TokenCounter, patch_lines, training_messages, trajectory_stats


class CharacterCounter:
    estimated = False

    def __call__(self, text):
        return len(text)


def assistant(content, name='bash'):
    return {'role': 'assistant', 'content': content, 'thought': 'DO NOT COUNT AUXILIARY',
            'tool_calls': [{'function': {'name': name, 'arguments': '{"command":"pwd"}'}}]}


class StatsTests(unittest.TestCase):
    def test_real_filter_keeps_think_tools_and_masked_context_without_mutation(self):
        history = [{'role': 'system', 'content': 'raw system'},
                   {'role': 'user', 'content': [{'type': 'text', 'text': 'task'}]},
                   assistant('<think>reason</think>DROP COMMENTARY'),
                   {'role': 'tool', 'content': 'MASKED BUT STILL IN CONTEXT'},
                   assistant('wrap this when no think', 'submit')]
        original = copy.deepcopy(history)
        messages = training_messages(history)
        text = ''.join(m['content'] for m in messages)
        self.assertNotIn('DROP COMMENTARY', text)
        self.assertNotIn('DO NOT COUNT AUXILIARY', text)
        self.assertNotIn('raw system', text)
        self.assertIn('function signatures', text)
        self.assertIn('<tool_response>\nMASKED BUT STILL IN CONTEXT\n</tool_response>', text)
        self.assertIn('<think>\nwrap this when no think\n</think>', text)
        self.assertEqual([m['train'] for m in messages], [False, False, True, False, True])
        expected = ''.join(f'<|im_start|>{m["role"]}\n{m["content"]}<|im_end|>\n' for m in messages)
        stats = trajectory_stats(history, '', CharacterCounter())
        self.assertEqual(stats['filtered_tokens'], len(expected))
        self.assertEqual(stats['turns'], 2)
        self.assertEqual(history, original)

    def test_explicit_reasoning_is_retained_once(self):
        message = assistant('commentary')
        message['reasoning'] = 'explicit thought'
        content = training_messages([message])[0]['content']
        self.assertIn('explicit thought', content)
        self.assertNotIn('commentary', content)
        message['content'] = '<think>explicit thought</think>commentary'
        self.assertEqual(training_messages([message])[0]['content'].count('explicit thought'), 1)

    def test_invalid_and_unsubmitted_are_not_reported_as_zero_tokens(self):
        history = [assistant('<think>one</think><think>two</think>')]
        self.assertIsNone(trajectory_stats(history, None, CharacterCounter())['filtered_tokens'])
        self.assertIsNotNone(trajectory_stats(history, None, CharacterCounter())['filter_error'])
        self.assertIsNone(trajectory_stats([assistant('valid')], None, CharacterCounter(), False)['filtered_tokens'])
        no_calls = [{'role': 'assistant', 'content': 'plain answer'}]
        self.assertIsNone(trajectory_stats(no_calls, None, CharacterCounter())['filtered_tokens'])

    def test_tool_averages_use_calls_and_responses_not_whole_prompts(self):
        history = [assistant('reason'), {'role': 'tool', 'content': 'ok'}]
        stats = trajectory_stats(history, '', CharacterCounter())
        self.assertEqual(stats['tool_call_mean'], len('<tool_call>\n{"name": "bash", "arguments": {"command": "pwd"}}\n</tool_call>'))
        self.assertEqual(stats['tool_result_mean'], len('<tool_response>\nok\n</tool_response>'))
        self.assertIsNone(trajectory_stats([], None, CharacterCounter())['tool_call_mean'])

    def test_patch_counts_blank_lines_and_header_like_code_inside_hunks(self):
        patch = 'diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1,2 +1,3 @@\n-old\n+new\n+\n+++code\n context\n'
        self.assertEqual(patch_lines(patch), {'added': 3, 'deleted': 1, 'total': 4})
        self.assertEqual(patch_lines(''), {'added': 0, 'deleted': 0, 'total': 0})
        self.assertIsNone(patch_lines(None))

    def test_fallback_is_explicitly_estimated(self):
        counter = TokenCounter()
        self.assertTrue(counter.estimated)
        self.assertEqual(counter('abcd'), 1)
        self.assertEqual(counter(''), 0)


if __name__ == '__main__':
    unittest.main()
