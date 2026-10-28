import copy
import json
from pathlib import Path
import tempfile
import unittest

from qwen_t2_proxy import Proxy, agent_response, upstream_request


class ProxyTests(unittest.TestCase):
    def response(self):
        return {"id": "test", "choices": [{"message": {
            "role": "assistant", "reasoning": "inspect\ncarefully", "content": "I will inspect the file.",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "bash", "arguments": '{"command":"pwd"}'}}]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 123, "completion_tokens": 45}}

    def test_agent_view_preserves_commentary_tools_and_original_response(self):
        response = self.response()
        original = copy.deepcopy(response)
        viewed = agent_response(response)
        self.assertEqual(response, original)
        self.assertEqual(viewed["choices"][0]["message"]["content"],
                         "<think>inspect\ncarefully</think>I will inspect the file.")
        self.assertEqual(viewed["choices"][0]["message"]["tool_calls"], response["choices"][0]["message"]["tool_calls"])
        self.assertEqual(viewed["usage"], response["usage"])

    def test_complete_records_original_and_agent_views_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            proxy = Proxy("http://localhost:8000/v1", tmp)
            payload = {"model": "qwen", "messages": [{"role": "user", "content": "test"}]}
            proxy.request = lambda *args: (200, self.response())
            status, response = proxy.complete(payload, "Bearer never-record-this")
            self.assertEqual(status, 200)
            record = json.loads(next(Path(tmp).glob("*.json")).read_text())
            self.assertEqual(record["response"], self.response())
            self.assertEqual(record["request"], dict(payload, parallel_tool_calls=False))
            self.assertEqual(record["agent_response"], response)
            self.assertNotIn("never-record-this", json.dumps(record))

    def test_error_and_streaming_attempts_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            proxy = Proxy("http://localhost:8000/v1", tmp)
            error = {"error": {"message": "context too long"}}
            proxy.request = lambda *args: (400, error)
            self.assertEqual(proxy.complete({}), (400, error))
            self.assertEqual(proxy.complete({"stream": True})[0], 400)
            self.assertEqual(len(list(Path(tmp).glob("*.json"))), 2)

    def test_history_roundtrip_preserves_thinking_commentary_and_tools(self):
        response = self.response()
        message = agent_response(response)["choices"][0]["message"]
        # The pinned agent keeps content/tools and discards independent reasoning.
        saved = {key: message[key] for key in ("role", "content", "tool_calls")}
        payload = {"messages": [{"role": "user", "content": "literal <think>user</think>"},
                                saved, {"role": "tool", "tool_call_id": "call_1", "content": "observation"}]}
        original = copy.deepcopy(payload)
        forwarded = upstream_request(payload)
        self.assertEqual(payload, original)
        self.assertEqual(forwarded["messages"][1], response["choices"][0]["message"])
        self.assertEqual(forwarded["messages"][0], payload["messages"][0])
        self.assertEqual(forwarded["messages"][2], payload["messages"][2])

    def test_empty_thinking_and_legacy_field_roundtrip(self):
        payload = {"messages": [{"role": "assistant", "content": "<think></think>  answer\n"}]}
        self.assertEqual(upstream_request(payload)["messages"][0],
                         {"role": "assistant", "reasoning": "", "content": "  answer\n"})
        legacy = {"messages": [{"role": "assistant", "reasoning_content": "old", "content": "answer"}]}
        self.assertEqual(upstream_request(legacy)["messages"][0]["reasoning"], "old")

    def test_ambiguous_history_and_parallel_calls_are_rejected(self):
        for payload in [
            {"parallel_tool_calls": True}, {"n": 2},
            {"messages": [{"role": "assistant", "content": "<think>unfinished"}]},
            {"messages": [{"role": "assistant", "content": "<think>a</think>b", "reasoning": "different"}]},
            {"messages": [{"role": "assistant", "reasoning": "a", "reasoning_content": "b"}]},
        ]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                upstream_request(payload)

    def test_native_budget_counts_tools_thinking_and_output_reserve(self):
        with tempfile.TemporaryDirectory() as tmp:
            proxy = Proxy("http://localhost:8000/v1", tmp, context_window=131072)
            calls = []
            count = 122880
            def request(path, payload, authorization):
                calls.append((path, payload))
                if path == "/tokenize":
                    return 200, {"count": count, "max_model_len": 131072, "tokens": [1, 2]}
                return 200, self.response()
            proxy.request = request
            payload = {"model": "qwen", "max_tokens": 8192, "tools": [],
                       "chat_template_kwargs": {"preserve_thinking": True},
                       "messages": [{"role": "assistant", "content": "<think>reason</think>answer"}]}
            self.assertEqual(proxy.complete(payload)[0], 200)
            self.assertEqual(calls[0][1]['messages'][0]['reasoning'], 'reason')
            self.assertEqual(calls[0][1]['tools'], [])
            self.assertEqual(calls[0][1]['chat_template_kwargs'], payload['chat_template_kwargs'])
            self.assertTrue(calls[1][1]['return_token_ids'])
            count += 1
            calls.clear()
            self.assertEqual(proxy.complete(payload)[0], 400)
            self.assertEqual(len(calls), 1)  # Never generate over budget or truncate history.
            records = [json.loads(p.read_text()) for p in Path(tmp).glob('*.json')]
            self.assertTrue(all(r['tokenization']['tokens'] == [1, 2] for r in records))


if __name__ == "__main__":
    unittest.main()
