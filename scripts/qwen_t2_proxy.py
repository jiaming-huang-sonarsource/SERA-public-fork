#!/usr/bin/env python3
"""Lossless Qwen API records and a tagged-content view for unmodified SERA."""

import argparse
import copy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


def agent_response(response):
    """Keep the original response separately; only annotate the agent's view."""
    result = copy.deepcopy(response)
    for choice in result.get("choices", []):
        message = choice.get("message", {})
        reasoning = message.get("reasoning")
        if reasoning is None:
            reasoning = message.get("reasoning_content")
        if reasoning is not None:
            message["content"] = f"<think>{reasoning}</think>" + (message.get("content") or "")
    return result


def upstream_request(payload):
    """Undo only our leading think annotation on assistant messages."""
    result = copy.deepcopy(payload)
    if result.get("parallel_tool_calls") or result.get("n", 1) != 1:
        raise ValueError("SERA requires one completion and one tool call per step")
    result["parallel_tool_calls"] = False
    for message in result.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        reasoning = message.get("reasoning", message.get("reasoning_content"))
        legacy = message.get("reasoning_content")
        if reasoning is not None and legacy is not None and reasoning != legacy:
            raise ValueError("Conflicting reasoning fields")
        if isinstance(content, str) and content.startswith("<think>"):
            end = content.find("</think>", len("<think>"))
            if end < 0:
                raise ValueError("Unclosed assistant think annotation")
            tagged = content[len("<think>"):end]
            if reasoning is not None and tagged != reasoning:
                raise ValueError("Reasoning differs from the assistant think annotation")
            reasoning = tagged
            message["content"] = content[end + len("</think>"):]
        if reasoning is not None:
            message["reasoning"] = reasoning
            message.pop("reasoning_content", None)
    return result


class Proxy:
    def __init__(self, upstream, records, timeout=900):
        self.upstream = upstream.rstrip("/")
        if not self.upstream.endswith("/v1"):
            raise ValueError("Upstream must end with /v1")
        self.records = Path(records)
        self.records.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def request(self, path, payload=None, authorization=None):
        headers = {"Content-Type": "application/json"}
        if authorization:
            headers["Authorization"] = authorization
        body = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
        url = self.upstream[:-3] + path
        try:
            with urlopen(Request(url, data=body, headers=headers), timeout=self.timeout) as response:
                return response.status, json.loads(response.read() or b"{}")
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def complete(self, payload, authorization=None):
        record = {"started_at": datetime.now(timezone.utc).isoformat(), "agent_request": payload}
        started = time.monotonic()
        status, response = 502, {"error": {"message": "Upstream request failed"}}
        try:
            if payload.get("stream"):
                raise ValueError("This SERA recorder accepts non-streaming requests only")
            record["request"] = upstream_request(payload)
            status, response = self.request("/v1/chat/completions", record["request"], authorization)
            record["response"] = response
            if status == 200:
                response = agent_response(response)
            record["agent_response"] = response
        except ValueError as error:
            status, response = 400, {"error": {"message": str(error)}}
        except (URLError, TimeoutError, OSError) as error:
            record["error"] = str(error)
        finally:
            record.update(status=status, seconds=round(time.monotonic() - started, 6))
            # Each request, including failed/retried calls, gets its own complete record.
            path = self.records / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex + ".json")
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        return status, response

    def server(self, host="127.0.0.1", port=18001):
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def send_json(self, status, payload):
                body = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path not in ("/health", "/v1/models"):
                    return self.send_json(404, {"error": "Unknown route"})
                try:
                    self.send_json(*proxy.request(self.path, authorization=self.headers.get("Authorization")))
                except (URLError, TimeoutError, OSError) as error:
                    self.send_json(502, {"error": str(error)})

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    return self.send_json(404, {"error": "Unknown route"})
                try:
                    payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    if not isinstance(payload, dict):
                        raise ValueError("Expected a JSON object")
                except ValueError as error:
                    return self.send_json(400, {"error": str(error)})
                self.send_json(*proxy.complete(payload, self.headers.get("Authorization")))

            def log_message(self, *args):
                pass

        return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, help="Actual Qwen endpoint ending in /v1")
    parser.add_argument("--records", required=True, help="Directory for original API request/response records")
    parser.add_argument("--port", type=int, default=18001)
    args = parser.parse_args()
    with Proxy(args.upstream, args.records).server(port=args.port) as server:
        print(f"SERA endpoint: http://127.0.0.1:{server.server_port}/v1", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
