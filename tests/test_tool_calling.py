"""Tool calling: incremental stream parser, validation/repair loop, large
toolsets, and cross-surface conformance (OpenAI, Google, Anthropic)."""
import base64
import http.client
import json
import threading
import unittest
from unittest import mock

from gemini_web2api.config import CONFIG
from gemini_web2api.server import GeminiHandler, ThreadedServer
from gemini_web2api.tools import (
    StreamToolCallParser, generate_validated, messages_to_prompt,
    extract_openai_tool_defs, extract_google_tool_defs, _minify_schema,
    validate_tool_arguments, parse_tool_calls, parse_google_calls_as_tool_calls,
)


def _events(parser, chunks):
    events = []
    for c in chunks:
        events.extend(parser.feed(c))
    events.extend(parser.finish())
    return events


class StreamParserTests(unittest.TestCase):
    def test_mixed_text_and_tool_call(self):
        parser = StreamToolCallParser()
        events = _events(parser, [
            "Let me check.```tool_call\n",
            '{"name": "get_weather", "argu',
            'ments": {"city": "Tokyo"}}\n',
            "``` done.",
        ])
        kinds = [e[0] for e in events]
        self.assertEqual(kinds.count("tool_start"), 1)
        self.assertEqual(kinds.count("tool_args"), 1)
        self.assertEqual(kinds.count("tool_end"), 1)
        self.assertEqual("".join(e[1] for e in events if e[0] == "text"),
                         "Let me check. done.")
        args = next(e for e in events if e[0] == "tool_args")
        self.assertEqual(json.loads(args[2]), {"city": "Tokyo"})
        self.assertEqual(parser.tool_count, 1)

    def test_parallel_tool_calls(self):
        parser = StreamToolCallParser()
        block = '```tool_call\n{"name": "f", "arguments": {"a": 1}}\n```\n'
        events = _events(parser, [block, block])
        starts = [e for e in events if e[0] == "tool_start"]
        self.assertEqual(len(starts), 2)
        self.assertEqual([s[1] for s in starts], [0, 1])
        self.assertEqual(parser.tool_count, 2)

    def test_opener_split_across_chunks(self):
        parser = StreamToolCallParser()
        text = 'text```tool_call\n{"name": "x", "arguments": {}}```end'
        # Feed one char at a time: nothing may be lost or duplicated.
        events = _events(parser, [text[i:i + 3] for i in range(0, len(text), 3)])
        self.assertEqual("".join(e[1] for e in events if e[0] == "text"), "textend")
        self.assertEqual(parser.tool_count, 1)
        self.assertEqual(next(e for e in events if e[0] == "tool_start")[2], "x")

    def test_truncated_json_autoclosed(self):
        parser = StreamToolCallParser()
        events = _events(parser, ['```tool_call\n{"name": "x", "arguments": {"a": "va'])
        self.assertEqual(parser.tool_count, 1)
        args = next(e for e in events if e[0] == "tool_args")
        self.assertEqual(json.loads(args[2]), {"a": "va"})

    def test_non_block_opener_is_literal_text(self):
        parser = StreamToolCallParser()
        events = _events(parser, ["look: ```tool_call is not a block here"])
        self.assertEqual("".join(e[1] for e in events if e[0] == "text"),
                         "look: ```tool_call is not a block here")
        self.assertEqual(parser.tool_count, 0)

    def test_function_call_opener_supported(self):
        parser = StreamToolCallParser()
        events = _events(parser, ['```function_call\n{"name": "g", "args": {"k": 1}}\n```'])
        self.assertEqual(parser.tool_count, 1)
        args = json.loads(next(e for e in events if e[0] == "tool_args")[2])
        self.assertEqual(args, {"k": 1})

    def test_control_chars_in_string_repaired(self):
        parser = StreamToolCallParser()
        events = _events(parser, ['```tool_call\n{"name": "w", "arguments": {"code": "line1\nline2"}}\n```'])
        self.assertEqual(parser.tool_count, 1)
        args = json.loads(next(e for e in events if e[0] == "tool_args")[2])
        self.assertEqual(args["code"], "line1\nline2")


class SchemaAndValidationTests(unittest.TestCase):
    def test_minify_schema_strips_boilerplate(self):
        schema = {"$schema": "http://x", "title": "T", "type": "object",
                  "additionalProperties": False,
                  "properties": {"a": {"type": "string", "title": "A"}}}
        out = _minify_schema(schema)
        self.assertNotIn("$schema", out)
        self.assertNotIn("title", out)
        self.assertNotIn("additionalProperties", out)
        self.assertNotIn("title", out["properties"]["a"])
        self.assertIn("type", out["properties"]["a"])

    def test_validate_required_and_types(self):
        schema = {"type": "object",
                  "required": ["path"],
                  "properties": {"path": {"type": "string"},
                                 "count": {"type": "integer"},
                                 "flag": {"type": "boolean"}}}
        self.assertEqual(validate_tool_arguments({"path": "x", "count": 2, "flag": True}, schema), [])
        errs = validate_tool_arguments({"count": 2}, schema)
        self.assertTrue(any("path" in e for e in errs))
        errs = validate_tool_arguments({"path": "x", "count": "two"}, schema)
        self.assertTrue(any("integer" in e for e in errs))
        errs = validate_tool_arguments({"path": "x", "flag": "yes"}, schema)
        self.assertTrue(any("boolean" in e for e in errs))

    def test_validate_enum_and_nested(self):
        schema = {"type": "object", "properties": {
            "mode": {"type": "string", "enum": ["a", "b"]},
            "opts": {"type": "object", "required": ["x"],
                     "properties": {"x": {"type": "number"}}}}}
        self.assertEqual(validate_tool_arguments({"mode": "a", "opts": {"x": 1.5}}, schema), [])
        self.assertTrue(validate_tool_arguments({"mode": "z"}, schema))
        self.assertTrue(validate_tool_arguments({"opts": {}}, schema))

    def test_parse_tool_calls_parallel_shape(self):
        text = ('```tool_call\n{"name": "a", "arguments": {"x": 1}}\n```\n'
                '```tool_call\n{"name": "b", "arguments": {"y": 2}}\n```')
        clean, calls = parse_tool_calls(text)
        self.assertEqual([c["function"]["name"] for c in calls], ["a", "b"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"x": 1})
        self.assertEqual(clean, "")

    def test_parse_google_normalized_shape(self):
        text = '```function_call\n{"name": "a", "args": {"x": 1}}\n```'
        clean, calls = parse_google_calls_as_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"x": 1})


class RepairLoopTests(unittest.TestCase):
    def test_invalid_arguments_trigger_repair(self):
        tool = {"type": "function", "function": {
            "name": "read", "parameters": {"type": "object", "required": ["path"],
                                           "properties": {"path": {"type": "string"}}}}}
        defs = extract_openai_tool_defs([tool])
        responses = [
            '```tool_call\n{"name": "read", "arguments": {"wrong": 1}}\n```',
            '```tool_call\n{"name": "read", "arguments": {"path": "a.py"}}\n```',
        ]
        gen = mock.Mock(side_effect=lambda p: responses.pop(0))
        CONFIG["tool_validate_retry"] = 1
        try:
            clean, calls = generate_validated("prompt", defs, "auto", gen)
        finally:
            CONFIG.pop("tool_validate_retry", None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "a.py"})
        self.assertEqual(gen.call_count, 2)
        repair_prompt = gen.call_args_list[1].args[0]
        self.assertIn("read", repair_prompt)
        self.assertIn("path", repair_prompt)
        self.assertIn("prompt", repair_prompt)  # original context preserved

    def test_valid_arguments_no_repair(self):
        tool = {"type": "function", "function": {
            "name": "read", "parameters": {"type": "object", "required": ["path"],
                                           "properties": {"path": {"type": "string"}}}}}
        defs = extract_openai_tool_defs([tool])
        gen = mock.Mock(return_value='```tool_call\n{"name": "read", "arguments": {"path": "a"}}\n```')
        clean, calls = generate_validated("prompt", defs, "auto", gen)
        self.assertEqual(gen.call_count, 1)
        self.assertEqual(len(calls), 1)

    def test_unrepairable_calls_dropped(self):
        tool = {"type": "function", "function": {
            "name": "read", "parameters": {"type": "object", "required": ["path"],
                                           "properties": {"path": {"type": "string"}}}}}
        defs = extract_openai_tool_defs([tool])
        gen = mock.Mock(return_value='```tool_call\n{"name": "read", "arguments": {}}\n```')
        CONFIG["tool_validate_retry"] = 1
        try:
            clean, calls = generate_validated("prompt", defs, "auto", gen)
        finally:
            CONFIG.pop("tool_validate_retry", None)
        self.assertEqual(calls, [])  # broken args never reach the client

    def test_no_tools_passthrough(self):
        gen = mock.Mock(return_value="plain answer")
        clean, calls = generate_validated("prompt", [], "auto", gen)
        self.assertEqual((clean, calls), ("plain answer", []))


class LargeToolsetTests(unittest.TestCase):
    def _tools(self, n):
        return [{"type": "function", "function": {
            "name": f"tool_{i:03d}",
            "description": f"Tool number {i} " + "x" * 400,
            "parameters": {"type": "object", "properties": {"a": {"type": "string"}}},
        }} for i in range(n)]

    def test_80_tools_fit_new_budget(self):
        tools = self._tools(80)
        prompt, _ = messages_to_prompt([{"role": "user", "content": "hi"}], tools, "auto")
        for i in range(80):
            self.assertIn(f"tool_{i:03d}", prompt)

    def test_old_budget_would_have_truncated(self):
        # Sanity: 80 tools x ~460 bytes exceed the old 35k cap.
        self.assertGreater(len(json.dumps(self._tools(80))), 35000)

    def test_budget_respected_when_configured(self):
        tools = self._tools(80)
        CONFIG["tool_max_tools"] = 10
        try:
            prompt, _ = messages_to_prompt([{"role": "user", "content": "hi"}], tools, "auto")
        finally:
            CONFIG.pop("tool_max_tools", None)
        self.assertIn("tool_000", prompt)
        self.assertNotIn("tool_050", prompt)

    def test_google_defs_extracted(self):
        req = {"tools": [{"functionDeclarations": [
            {"name": "g", "description": "d",
             "parameters": {"type": "object", "title": "P",
                            "properties": {"a": {"type": "string"}}}}]}]}
        defs = extract_google_tool_defs(req)
        self.assertEqual(len(defs), 1)
        self.assertNotIn("title", defs[0]["parameters"])


class _BaseServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadedServer(("127.0.0.1", 0), GeminiHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.original_config = dict(CONFIG)
        CONFIG["api_keys"] = []
        CONFIG["log_requests"] = False

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def post_json(self, path, payload):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("POST", path, body=json.dumps(payload),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = response.read().decode()
        connection.close()
        return response.status, body


class OpenAIStreamingConformanceTests(_BaseServerTest):
    TOOLS = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "required": ["city"],
                           "properties": {"city": {"type": "string"}}},
        },
    }]

    def test_streaming_tool_call_deltas(self):
        model_output = ('Checking.```tool_call\n{"name": "get_weather", '
                        '"arguments": {"city": "Paris"}}\n```')
        with mock.patch("gemini_web2api.server.generate_stream",
                        return_value=iter(["Checking.```tool", '_call\n{"name": "get_weather", ',
                                           '"arguments": {"city": "Paris"}}\n```'])):
            status, body = self.post_json("/v1/chat/completions", {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": self.TOOLS,
                "stream": True,
            })
        self.assertEqual(status, 200)
        chunks = [json.loads(l[len("data: "):]) for l in body.splitlines() if l.startswith("data: {")]
        # role chunk first, no full-message buffering: content delta precedes tool deltas
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        deltas = [c["choices"][0]["delta"] for c in chunks[1:-1]]
        tool_deltas = [d for d in deltas if "tool_calls" in d]
        self.assertEqual(len(tool_deltas), 2)  # tool_start + tool_args
        first = tool_deltas[0]["tool_calls"][0]
        self.assertEqual(first["index"], 0)
        self.assertEqual(first["function"]["name"], "get_weather")
        self.assertTrue(first["id"].startswith("call_"))
        args = tool_deltas[1]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(json.loads(args), {"city": "Paris"})
        finish = [c for c in chunks if c["choices"] and c["choices"][0].get("finish_reason")]
        self.assertEqual(finish[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    def test_streaming_include_usage(self):
        with mock.patch("gemini_web2api.server.generate_stream", return_value=iter(["hello"])):
            status, body = self.post_json("/v1/chat/completions", {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            })
        chunks = [json.loads(l[len("data: "):]) for l in body.splitlines() if l.startswith("data: {")]
        usage_chunks = [c for c in chunks if "usage" in c]
        self.assertEqual(len(usage_chunks), 1)
        self.assertIn("prompt_tokens", usage_chunks[0]["usage"])

    def test_nonstream_validated_tool_calls(self):
        with mock.patch("gemini_web2api.server.generate",
                        return_value='```tool_call\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n```'):
            status, body = self.post_json("/v1/chat/completions", {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": self.TOOLS,
            })
        data = json.loads(body)
        msg = data["choices"][0]["message"]
        self.assertEqual(data["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "get_weather")

    def test_nonstream_rejects_invalid_tool_args(self):
        with mock.patch("gemini_web2api.server.generate",
                        return_value='```tool_call\n{"name": "get_weather", "arguments": {}}\n```'):
            status, body = self.post_json("/v1/chat/completions", {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": self.TOOLS,
                "tool_choice": "required",
            })
        data = json.loads(body)
        msg = data["choices"][0]["message"]
        self.assertNotIn("tool_calls", msg)  # broken args never reach the client


class GoogleSurfaceTests(_BaseServerTest):
    def test_nonstream_multiple_function_calls(self):
        req = {
            "contents": [{"role": "user", "parts": [{"text": "weather and time"}]}],
            "tools": [{"functionDeclarations": [
                {"name": "get_weather", "parameters": {"type": "object", "required": ["city"],
                                                       "properties": {"city": {"type": "string"}}}},
                {"name": "get_time", "parameters": {"type": "object", "properties": {}}},
            ]}],
        }
        model_output = ('```function_call\n{"name": "get_weather", "args": {"city": "Paris"}}\n```\n'
                        '```function_call\n{"name": "get_time", "args": {}}\n```')
        with mock.patch("gemini_web2api.server.generate", return_value=model_output):
            status, body = self.post_json("/v1beta/models/gemini-3.6-flash:generateContent", req)
        data = json.loads(body)
        parts = data["candidates"][0]["content"]["parts"]
        fcs = [p["functionCall"] for p in parts if "functionCall" in p]
        self.assertEqual([f["name"] for f in fcs], ["get_weather", "get_time"])
        self.assertEqual(fcs[0]["args"], {"city": "Paris"})

    def test_streaming_function_call_chunks(self):
        req = {
            "contents": [{"role": "user", "parts": [{"text": "weather"}]}],
            "tools": [{"functionDeclarations": [
                {"name": "get_weather", "parameters": {"type": "object", "required": ["city"],
                                                       "properties": {"city": {"type": "string"}}}}]}],
        }
        with mock.patch("gemini_web2api.server.generate_stream",
                        return_value=iter(['```function_call\n{"name": "get_weather", ',
                                           '"args": {"city": "Paris"}}\n```'])):
            status, body = self.post_json(
                "/v1beta/models/gemini-3.6-flash:streamGenerateContent", req)
        events = [json.loads(l[len("data: "):]) for l in body.splitlines() if l.startswith("data: {")]
        fcs = []
        for ev in events:
            for cand in ev.get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    if "functionCall" in part:
                        fcs.append(part["functionCall"])
        self.assertEqual(len(fcs), 1)
        self.assertEqual(fcs[0]["args"], {"city": "Paris"})
        self.assertEqual(events[-1]["candidates"][0]["finishReason"], "STOP")


class AnthropicSurfaceTests(_BaseServerTest):
    TOOLS = [{
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {"type": "object", "required": ["city"],
                         "properties": {"city": {"type": "string"}}},
    }]

    def test_nonstream_tool_use(self):
        with mock.patch("gemini_web2api.server.generate",
                        return_value='```tool_call\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n```'):
            status, body = self.post_json("/v1/messages", {
                "model": "gemini-3.6-flash",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": self.TOOLS,
            })
        data = json.loads(body)
        self.assertEqual(data["type"], "message")
        self.assertEqual(data["stop_reason"], "tool_use")
        tool_uses = [b for b in data["content"] if b["type"] == "tool_use"]
        self.assertEqual(len(tool_uses), 1)
        self.assertEqual(tool_uses[0]["name"], "get_weather")
        self.assertEqual(tool_uses[0]["input"], {"city": "Paris"})

    def test_nonstream_with_tool_result_roundtrip(self):
        # A follow-up turn carrying tool_result must convert cleanly.
        with mock.patch("gemini_web2api.server.generate", return_value="The weather is nice."):
            status, body = self.post_json("/v1/messages", {
                "model": "gemini-3.6-flash",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {"role": "assistant", "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                         "input": {"city": "Paris"}}]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1",
                         "content": "22C sunny"}]},
                ],
                "tools": self.TOOLS,
            })
        data = json.loads(body)
        self.assertEqual(data["content"][0]["text"], "The weather is nice.")
        self.assertEqual(data["stop_reason"], "end_turn")

    def test_streaming_blocks(self):
        with mock.patch("gemini_web2api.server.generate_stream",
                        return_value=iter(['Text first.```tool_call\n{"name": "get_weather", ',
                                           '"arguments": {"city": "Paris"}}\n```'])):
            status, body = self.post_json("/v1/messages", {
                "model": "gemini-3.6-flash",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": self.TOOLS,
                "stream": True,
            })
        events = []
        for block in body.strip().split("\n\n"):
            lines = block.splitlines()
            etype = next((l[len("event: "):] for l in lines if l.startswith("event: ")), None)
            edata = next((l[len("data: "):] for l in lines if l.startswith("data: ")), None)
            if etype and edata:
                events.append((etype, json.loads(edata)))
        types = [e[0] for e in events]
        self.assertEqual(types[0], "message_start")
        self.assertEqual(types[-1], "message_stop")
        self.assertIn("content_block_start", types)
        # input_json_delta is the delta *type* inside a content_block_delta event
        self.assertTrue(any(e[1].get("delta", {}).get("type") == "input_json_delta"
                            for e in events if e[0] == "content_block_delta"))
        starts = [e[1] for e in events if e[0] == "content_block_start"]
        self.assertEqual(starts[0]["content_block"]["type"], "text")
        self.assertEqual(starts[1]["content_block"]["type"], "tool_use")
        self.assertEqual(starts[1]["content_block"]["name"], "get_weather")
        stops = [e[1] for e in events if e[0] == "content_block_stop"]
        self.assertEqual(len(stops), 2)
        msg_delta = next(e[1] for e in events if e[0] == "message_delta")
        self.assertEqual(msg_delta["delta"]["stop_reason"], "tool_use")

    def test_tool_choice_any_maps_to_required(self):
        with mock.patch("gemini_web2api.server.generate", return_value="no tool"), \
             mock.patch("gemini_web2api.server.messages_to_prompt",
                        wraps=messages_to_prompt) as mtp:
            self.post_json("/v1/messages", {
                "model": "gemini-3.6-flash",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": self.TOOLS,
                "tool_choice": {"type": "any"},
            })
            self.assertEqual(mtp.call_args.args[2], "required")


if __name__ == "__main__":
    unittest.main()
