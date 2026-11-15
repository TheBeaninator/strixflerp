"""FLERP-REPLY and FLERP-CONSTRAIN without a GPU: serve.flerp.http built from the fake recipe's closure, the stand-in
executor wrapped with a scripted reply (tests/cpp/script_fake.c), and Qwen3.8-Flash-Next's own tokenizer, so the reply
module (runtime.flerp.reply_strixite) splits real reasoning / content / tool-call text. Needs g++ and the model's
tokenizer.json (FLERP_TEST_TOKENIZER, default the pod's copy); skipped without them."""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from flerplib.catalog import Catalog                  # noqa: E402
from flerplib.resolve import Assembly                 # noqa: E402

TOKENIZER = os.environ.get("FLERP_TEST_TOKENIZER", "/data/models/strix-infer/tok/tokenizer.json")
RECIPE = "qwen38-flash-next.flerp-serve-fake"
SCRIPT_FAKE = os.path.join(ROOT, "tests", "cpp", "script_fake.c")
END = 248046  # <|im_end|>

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "the weather",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                                                            "days": {"type": "integer"}},
                                                           "required": ["city"]}}}
CALL = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"


def build(out_dir):
    """The fake recipe's binary, compiled unit by unit with the host compiler (its CMake tree wants TheRock), the
    stand-in's source swapped for the scripted wrapper."""
    cat = Catalog(ROOT)
    asm = Assembly(cat, cat.recipe(RECIPE))
    assert asm.errors == [], asm.errors
    (_, _, ids), = asm.binaries.values()
    objs = []
    for mid in ids:
        m = cat.by_id[mid]
        if m.lang not in ("c", "c++") or not m.sources:
            continue
        incs = []
        for dep in [mid] + asm.deps(mid):
            dm = cat.by_id[dep]
            incs += [os.path.normpath(os.path.join(dm.dir, d)) for d in dm.includes]
        for src in m.sources:
            path = os.path.join(m.dir, src)
            if mid == "executor.flerp.fake_abi":
                path = SCRIPT_FAKE
            obj = os.path.join(out_dir, mid + "-" + os.path.basename(src) + ".o")
            cmd = (["gcc", "-std=c11"] if m.lang == "c" else ["g++", "-std=c++20"]) + ["-O1", "-w", "-c", path, "-o", obj]
            cmd += ["-D" + d for d in m.build.get("defines", [])] + ["-I" + i for i in dict.fromkeys(incs)]
            subprocess.run(cmd, check=True)
            objs.append(obj)
    exe = os.path.join(out_dir, "strixflerp-script-fake")
    subprocess.run(["g++", "-o", exe] + objs + ["-lpthread", "-lm"], check=True)
    return exe


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, exe, work, *args):
        self.port, self.script = free_port(), os.path.join(work, f"script-{len(os.listdir(work))}.ids")
        open(self.script, "w").close()
        self.log = open(os.path.join(work, f"server-{self.port}.log"), "w")
        self.proc = subprocess.Popen([exe, "--port", str(self.port), "--model-id", "fake", f"text.tokenizer={TOKENIZER}",
                                      "executor.vocab=248077", "executor.lo=1000", "executor.hi=30000", "executor.reply=4000",
                                      f"executor.end={END}", f"executor.script={self.script}", *args],
                                     stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(300):
            try:
                self.get("/health")
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("the server did not start: " + open(self.log.name).read())

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.log.close()

    def get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=10) as r:
            return json.loads(r.read())

    def post(self, path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode()
                status = r.status
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
        if body.get("stream"):
            chunks = [json.loads(line[6:]) for line in raw.split("\n") if line.startswith("data: ") and line != "data: [DONE]"]
            return status, chunks
        return status, json.loads(raw)

    def say(self, text, end=True):
        """The scripted reply: text's ids, then the end of turn."""
        ids = self.post("/tokenize", {"content": text})[1]["tokens"]
        with open(self.script, "w") as f:
            f.write(" ".join(str(i) for i in ids + ([END] if end else [])))
        return ids


def chat(**kw):
    body = {"messages": [{"role": "user", "content": "What is the weather in Paris?"}], "temperature": 0, "max_tokens": 2000,
            "flerp_trace": True}
    body.update(kw)
    return body


@unittest.skipUnless(shutil.which("g++") and shutil.which("gcc") and os.path.isfile(TOKENIZER), "needs g++ and the model's tokenizer.json")
class Reply(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="flerp-reply-")
        cls.exe = build(cls.dir)
        cls.srv = Server(cls.exe, cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls.srv.close()
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_health_names_the_reply_module(self):
        self.assertIn("strixite reply parser", self.srv.get("/health")["reply"])

    def test_reasoning_and_content(self):
        self.srv.say("The user wants a greeting.\n</think>\n\nHello there!")
        st, r = self.srv.post("/v1/chat/completions", chat())
        self.assertEqual(st, 200, r)
        m = r["choices"][0]["message"]
        self.assertEqual(m["reasoning_content"], "The user wants a greeting.")
        self.assertEqual(m["content"], "Hello there!")
        self.assertNotIn("tool_calls", m)
        self.assertEqual(r["choices"][0]["finish_reason"], "stop")
        self.assertGreater(r["usage"]["completion_tokens_details"]["reasoning_tokens"], 0)

    def test_tool_call(self):
        self.srv.say("I should look it up.\n</think>\n\n" + CALL)
        st, r = self.srv.post("/v1/chat/completions", chat(tools=[WEATHER]))
        self.assertEqual(st, 200, r)
        c = r["choices"][0]
        self.assertEqual(c["finish_reason"], "tool_calls")
        self.assertIsNone(c["message"]["content"])
        self.assertEqual(c["message"]["reasoning_content"], "I should look it up.")
        (call,) = c["message"]["tool_calls"]
        self.assertEqual((call["index"], call["type"], call["function"]["name"]), (0, "function", "get_weather"))
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Paris", "days": 3})   # typed from the schema
        self.assertTrue(call["id"])

    def test_streamed_tool_call_deltas(self):
        self.srv.say("I should look it up.\n</think>\n\n" + CALL)
        st, chunks = self.srv.post("/v1/chat/completions", chat(tools=[WEATHER], stream=True))
        self.assertEqual(st, 200, chunks)
        deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
        self.assertEqual(deltas[0], {"role": "assistant", "content": ""})
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        self.assertEqual(reasoning, "I should look it up.")
        calls = [tc for d in deltas for tc in d.get("tool_calls", [])]
        self.assertEqual(calls[0]["function"], {"name": "get_weather", "arguments": ""})   # the call's start: id and name
        self.assertTrue(calls[0]["id"])
        self.assertTrue(all(tc["index"] == 0 for tc in calls))
        args = "".join(tc["function"]["arguments"] for tc in calls[1:])
        self.assertEqual(json.loads(args), {"city": "Paris", "days": 3})
        self.assertGreater(len(calls), 2)                                                  # arguments in pieces
        finish = [c["choices"][0]["finish_reason"] for c in chunks if c.get("choices")]
        self.assertEqual(finish[-1], "tool_calls")

    def test_tool_call_begun_inside_the_think_block_gets_the_close(self):
        self.srv.say("I should look it up.\n" + CALL)
        st, r = self.srv.post("/v1/chat/completions", chat(tools=[WEATHER]))
        self.assertEqual(st, 200, r)
        c = r["choices"][0]
        self.assertEqual(c["finish_reason"], "tool_calls")
        self.assertEqual(json.loads(c["message"]["tool_calls"][0]["function"]["arguments"]), {"city": "Paris", "days": 3})
        self.assertEqual(r["flerp"]["reply"]["think_closed_for_call"], 1)
        self.assertIn(255, [s[0] for s in r["flerp"]["steps"]])                           # the close was fed

    def test_think_close_on_call_off_leaves_the_call_in_the_reasoning(self):
        srv = Server(self.exe, self.dir, "--think-close-on-call", "off")
        try:
            srv.say("I should look it up.\n" + CALL)
            st, r = srv.post("/v1/chat/completions", chat(tools=[WEATHER]))
            self.assertEqual(st, 200, r)
            self.assertNotIn("tool_calls", r["choices"][0]["message"])
            self.assertIn("get_weather", r["choices"][0]["message"]["reasoning_content"])
        finally:
            srv.close()

    def test_thinking_budget_feeds_the_stop(self):
        self.srv.say("one two three four five six seven eight nine ten eleven twelve.\n</think>\n\nDone.")
        st, r = self.srv.post("/v1/chat/completions", chat(max_thinking_tokens=4))
        self.assertEqual(st, 200, r)
        self.assertTrue(r["flerp"]["reply"]["thinking_budget_hit"])
        self.assertEqual(r["flerp"]["reply"]["thinking_budget"], 4)
        self.assertIn("Considering the limited time by the user", r["choices"][0]["message"]["reasoning_content"])
        self.assertTrue(r["choices"][0]["message"]["content"])

    def test_budgets_that_disagree_are_refused(self):
        st, r = self.srv.post("/v1/chat/completions", chat(max_thinking_tokens=4, reasoning={"max_tokens": 5}))
        self.assertEqual(st, 400)
        self.assertIn("thinking budgets disagree", r["error"]["message"])

    def test_think_end_guard(self):
        self.srv.say("Still thinking", end=True)                                         # the end of turn inside the think block
        st, r = self.srv.post("/v1/chat/completions", chat(max_tokens=300, max_thinking_tokens=1000))
        self.assertEqual(st, 200, r)
        self.assertGreater(r["flerp"]["reply"].get("think_end_blocked", 0), 0)
        self.assertEqual(r["choices"][0]["finish_reason"], "length")                  # never ended inside the block
        st, r = self.srv.post("/v1/chat/completions", chat(max_tokens=1200, enable_thinking=True))
        self.assertEqual(st, 200, r)                                                     # the budget's stop closes it,
        self.assertTrue(r["flerp"]["reply"]["thinking_budget_hit"])                      # then the end of turn is taken
        self.assertEqual(r["flerp"]["reply"]["thinking_budget"], 1200 - 180)
        self.assertEqual(r["choices"][0]["finish_reason"], "stop")

    def test_small_max_tokens_leaves_thinking_out(self):
        self.srv.say("Hi!")
        st, r = self.srv.post("/v1/chat/completions", chat(max_tokens=50))
        self.assertEqual(st, 200, r)
        self.assertEqual(r["flerp"]["reply"]["thinking"], "skipped")
        self.assertEqual(r["choices"][0]["message"]["content"], "Hi!")
        self.assertNotIn("reasoning_content", r["choices"][0]["message"])

    def test_stop_string_ends_the_content(self):
        self.srv.say("ok\n</think>\n\nHello STOP world")
        st, r = self.srv.post("/v1/chat/completions", chat(stop=["STOP"]))
        self.assertEqual(st, 200, r)
        self.assertEqual(r["choices"][0]["message"]["content"], "Hello")
        self.assertEqual(r["choices"][0]["finish_reason"], "stop")

    def test_tool_role_messages_render(self):
        self.srv.say("ok\n</think>\n\nIt is sunny.")
        msgs = [{"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                                                                        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]
        st, r = self.srv.post("/v1/chat/completions", chat(messages=msgs, tools=[WEATHER]))
        self.assertEqual(st, 200, r)
        self.assertEqual(r["choices"][0]["message"]["content"], "It is sunny.")

    def test_tool_choice_none_renders_no_tools(self):
        self.srv.say("Hi!")
        _, with_tools = self.srv.post("/v1/chat/completions", chat(tools=[WEATHER], max_tokens=50))
        _, none = self.srv.post("/v1/chat/completions", chat(tools=[WEATHER], tool_choice="none", max_tokens=50))
        _, without = self.srv.post("/v1/chat/completions", chat(max_tokens=50))
        self.assertEqual(none["flerp"]["prompt_token_ids"], without["flerp"]["prompt_token_ids"])
        self.assertGreater(len(with_tools["flerp"]["prompt_token_ids"]), len(without["flerp"]["prompt_token_ids"]))

    def test_refusals(self):
        st, r = self.srv.post("/v1/chat/completions", chat(tools=[WEATHER], tool_choice="required"))
        self.assertEqual((st, r["error"]["code"]), (400, "not_served"))
        st, r = self.srv.post("/v1/chat/completions", chat(tools=[{"type": "function"}]))
        self.assertEqual((st, r["error"]["param"]), (400, "tools[0].function"))
        st, r = self.srv.post("/v1/chat/completions", chat(tools=[WEATHER], response_format={"type": "json_object"}))
        self.assertEqual(st, 400)
        self.assertIn("together with tools is not supported", r["error"]["message"])


@unittest.skipUnless(shutil.which("g++") and shutil.which("gcc") and os.path.isfile(TOKENIZER), "needs g++ and the model's tokenizer.json")
class Constrain(unittest.TestCase):
    """response_format as masks the front end hands the executor (FLERP_CAP_MASKS), compiled by the reply module."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="flerp-constrain-")
        cls.exe = build(cls.dir)
        cls.srv = Server(cls.exe, cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls.srv.close()
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_a_reply_the_grammar_allows_goes_through(self):
        self.srv.say('{"city": "Paris", "days": 3}')
        st, r = self.srv.post("/v1/chat/completions", chat(enable_thinking=False, response_format={"type": "json_object"}))
        self.assertEqual(st, 200, r)
        self.assertEqual(json.loads(r["choices"][0]["message"]["content"]), {"city": "Paris", "days": 3})
        self.assertEqual(r["choices"][0]["finish_reason"], "stop")

    def test_masks_refuse_what_the_grammar_does_not_allow(self):
        self.srv.say("Hello, it is sunny.")
        st, r = self.srv.post("/v1/chat/completions", chat(enable_thinking=False, max_tokens=24, response_format={"type": "json_object"}))
        self.assertEqual(st, 200, r)
        content = r["choices"][0]["message"]["content"]
        self.assertFalse(content.startswith("Hello"), content)
        self.assertTrue(content.lstrip().startswith("{"), content)
        _, free = self.srv.post("/v1/chat/completions", chat(enable_thinking=False, max_tokens=24))
        self.assertTrue(free["choices"][0]["message"]["content"].startswith("Hello"))          # the same script, no format

    def test_schema(self):
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
        self.srv.say('{"ok": true}')
        fmt = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
        st, r = self.srv.post("/v1/chat/completions", chat(enable_thinking=False, response_format=fmt))
        self.assertEqual(st, 200, r)
        self.assertEqual(json.loads(r["choices"][0]["message"]["content"]), {"ok": True})
        self.srv.say('{"no": 1}')                                                      # a key the schema refuses
        st, r = self.srv.post("/v1/chat/completions", chat(enable_thinking=False, max_tokens=24, response_format=fmt))
        self.assertEqual(st, 200, r)
        self.assertFalse(r["choices"][0]["message"]["content"].startswith('{"no"'))

    def test_thinking_then_the_grammar(self):
        self.srv.say('Need JSON.\n</think>\n\n{"a": 1}')
        st, r = self.srv.post("/v1/chat/completions", chat(enable_thinking=True, response_format={"type": "json_object"}))
        self.assertEqual(st, 200, r)
        self.assertEqual(r["choices"][0]["message"]["reasoning_content"], "Need JSON.")
        self.assertEqual(json.loads(r["choices"][0]["message"]["content"]), {"a": 1})

    def test_drafts_the_constraint_refuses_are_not_verified(self):
        srv = Server(self.exe, self.dir, "executor.draft=step")
        try:
            body = chat(enable_thinking=False, max_tokens=40)
            _, free = srv.post("/v1/chat/completions", body)
            st, r = srv.post("/v1/chat/completions", dict(body, response_format={"type": "json_object"}))
            self.assertEqual(st, 200, r)
            self.assertGreater(free["flerp"]["drafted"], 0)
            self.assertLess(r["flerp"]["drafted"], free["flerp"]["drafted"])
        finally:
            srv.close()

    def test_an_executor_without_masks_refuses(self):
        srv = Server(self.exe, self.dir, "executor.masks=off")
        try:
            st, r = srv.post("/v1/chat/completions", chat(response_format={"type": "json_object"}))
            self.assertEqual((st, r["error"]["code"], r["error"]["param"]), (400, "executor_cannot", "response_format"))
        finally:
            srv.close()

    def test_refusals(self):
        st, r = self.srv.post("/v1/chat/completions", chat(response_format={"type": "xml"}))
        self.assertEqual((st, r["error"]["param"]), (400, "response_format"))
        st, r = self.srv.post("/v1/chat/completions", chat(response_format={"type": "json_schema"}))
        self.assertEqual((st, r["error"]["param"]), (400, "response_format.json_schema"))


if __name__ == "__main__":
    unittest.main()
