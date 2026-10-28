import copy
import json
from pathlib import Path
import tempfile
import unittest

from qwen_t2_proxy import Proxy, agent_response


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
            self.assertEqual(record["request"], payload)
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


if __name__ == "__main__":
    unittest.main()
