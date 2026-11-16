"""FLERP-REPLAYKIT: the recorder (tools/replay/record_proxy.py), the scrubber (tools/replay/scrub.py) and the replay
gate (gate/replay.py). Stand-in HTTP servers for the mechanics; the front end over the stand-in executor
(recipe qwen38-flash-next.flerp-serve-fake) as both servers of a replay, where its binary and tokenizer are at hand."""
import hashlib
import http.client
import http.server
import json
import os
import shutil
import socket
import socketserver
import stat
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import unittest
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "replay"))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import record_proxy                                   # noqa: E402
import replay                                         # noqa: E402
import scrub                                          # noqa: E402

SECRET = "sk-ant-" + "x" * 30
STORY = "Tell me a long story about a lighthouse keeper who counts the waves every night"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def state_bytes(tokens):
    """A stand-in cache state: 2 MiB a function of the token count."""
    block = hashlib.sha256(str(tokens).encode()).digest() * 64
    return block * (2 * 1024 * 1024 // len(block))


class Stub(http.server.BaseHTTPRequestHandler):
    """An origin-like server. `variant` changes the reply text; ids and times are random, as the origins' are."""
    protocol_version = "HTTP/1.1"
    variant = "same"
    seen = None

    def log_message(self, *a):
        pass

    def reply(self, status, body, ctype="application/json", chunked=False):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for part in (body[:7], body[7:]):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
            self.wfile.write(b"0\r\n\r\n")
        else:
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, json.dumps({"status": "ok", "busy": time.time()}).encode())
        else:
            self.reply(200, json.dumps({"object": "list", "data": [{"id": "stub"}]}).encode())

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        type(self).seen.append((self.path, dict(self.headers), raw if len(raw) < 4096 else hashlib.sha256(raw).hexdigest()))
        if self.path == "/cache/export":
            self.reply(200, state_bytes(self.headers.get("x-cache-tokens")), "application/octet-stream")
            return
        if self.path == "/cache/import":
            self.reply(200, json.dumps({"imported": hashlib.sha256(raw).hexdigest()}).encode())
            return
        req = json.loads(raw)
        text = f"{self.variant}:{hashlib.sha256(json.dumps(req['messages']).encode()).hexdigest()[:8]}:seed={req.get('seed')}"
        common = {"id": f"chatcmpl-{os.urandom(4).hex()}", "created": int(time.time() * 1000), "model": "stub"}
        if req.get("stream"):
            ev = [dict(common, choices=[{"delta": {"content": text[:5]}}]), dict(common, choices=[{"delta": {"content": text[5:]}}])]
            body = "".join(f"data: {json.dumps(e)}\n\n" for e in ev) + "data: [DONE]\n\n"
            self.reply(200, body.encode(), "text/event-stream", chunked=True)
            return
        body = dict(common, choices=[{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                    usage={"prompt_tokens": 9, "completion_tokens": 3}, timings={"ms": time.time()})
        self.reply(400 if req.get("bad") else 200, json.dumps(body).encode())


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def stub(variant="same"):
    h = type("S", (Stub,), {"variant": variant, "seen": []})
    srv = ThreadingServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, h


def proxy(upstream, log):
    srv = record_proxy.serve(f"127.0.0.1:{free_port()}", upstream, log, "strixite")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def call(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    out = r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
    c.close()
    return out


def log_lines(path, n, timeout=10):
    """The log's records once it holds n of them: a record is appended just after its response is sent."""
    t0 = time.time()
    while True:
        with open(path) as f:
            lines = [json.loads(x) for x in f if x.strip()]
        if len(lines) >= n or time.time() - t0 > timeout:
            return lines
        time.sleep(0.05)


def chat(messages, **kw):
    return json.dumps(dict({"model": "m", "messages": messages, "max_tokens": 32}, **kw)).encode()


def record_session(port):
    """The traffic of one FreeFitty-like session: two turns of a thread, a stream, a cache export and its import."""
    sysm = {"role": "system", "content": "You are the pod's careful coding agent; answer briefly."}
    t1 = [sysm, {"role": "user", "content": STORY + f" (key {SECRET})"}]
    auth = {"Authorization": "Bearer " + "t" * 40, "Content-Type": "application/json"}
    out = [call(port, "POST", "/v1/chat/completions", chat(t1), auth)]
    t2 = t1 + [{"role": "assistant", "content": "Once there was a keeper who counted waves until morning."},
               {"role": "user", "content": "Now make the ending happier and shorter than the first one."}]
    out.append(call(port, "POST", "/v1/chat/completions", chat(t2, seed=11), auth))
    out.append(call(port, "POST", "/v1/chat/completions", chat(t1, stream=True), auth))
    out.append(call(port, "GET", "/health"))
    exp = call(port, "POST", "/cache/export", b"", {"x-cache-tokens": "4096"})
    out.append(exp)
    out.append(call(port, "POST", "/cache/import", exp[2], {"Content-Type": "application/octet-stream"}))
    img = [{"role": "user", "content": [{"type": "text", "text": "what is this"},
                                        {"type": "image_url", "image_url": {"url": "https://example.org/a.png"}}]}]
    out.append(call(port, "POST", "/v1/chat/completions", chat(img), auth))
    return out


class Mechanics(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="flerp-replay-test-")
        self.log = os.path.join(self.dir, "rec", "log.jsonl")
        self.servers = []

    def tearDown(self):
        for s in self.servers:
            s.shutdown()
            s.server_close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def up(self, variant="same"):
        srv, h = stub(variant)
        self.servers.append(srv)
        return srv, h

    def recorded(self):
        srv, h = self.up()
        px = proxy(f"http://127.0.0.1:{srv.server_address[1]}", self.log)
        self.servers.append(px)
        answers = record_session(px.server_address[1])
        log_lines(self.log, len(answers))
        return srv, h, answers

    def test_recorder_forwards_unchanged_and_logs_each_exchange(self):
        srv, h, answers = self.recorded()
        port = srv.server_address[1]
        direct = call(port, "POST", "/cache/export", b"", {"x-cache-tokens": "4096"})
        self.assertEqual(answers[4][2], direct[2])                              # 2 MiB through, byte for byte
        self.assertEqual(answers[2][1].get("content-type"), "text/event-stream")
        self.assertTrue(answers[2][2].endswith(b"data: [DONE]\n\n"))
        sent = [s for s in h.seen if s[0] == "/v1/chat/completions"]
        self.assertEqual(json.loads(sent[0][2])["messages"][1]["content"], STORY + f" (key {SECRET})")   # the body unchanged
        self.assertTrue(sent[0][1].get("Authorization", "").startswith("Bearer t"))                      # forwarded ...
        lines = log_lines(self.log, 7)
        self.assertEqual(len(lines), 7)
        self.assertEqual(stat.S_IMODE(os.stat(self.log).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.dirname(self.log)).st_mode), 0o700)
        first = lines[0]
        self.assertNotIn("authorization", first["req_headers"])                                     # ... never logged
        for k in ("id", "method", "path", "t_start", "t_req_end", "t_end", "status", "resp_headers", "in_flight_at_start"):
            self.assertIn(k, first)
        self.assertEqual(first["request"]["messages"][1]["content"], STORY + f" (key {SECRET})")
        self.assertEqual(json.loads(first["response_raw"])["choices"][0]["finish_reason"], "stop")
        exp, imp = lines[4], lines[5]
        self.assertNotIn("response_raw", exp)
        self.assertEqual((exp["resp_body_bytes"], exp["resp_body_sha256"]), (len(direct[2]), hashlib.sha256(direct[2]).hexdigest()))
        self.assertNotIn("request", imp)
        self.assertEqual(imp["req_body_sha256"], exp["resp_body_sha256"])
        self.assertTrue(all(x["req_complete"] and x["resp_complete"] for x in lines))

    def test_in_flight_is_counted(self):
        srv, _ = self.up()
        px = proxy(f"http://127.0.0.1:{srv.server_address[1]}", self.log)
        self.servers.append(px)
        slow = threading.Event()

        class Slow(Stub):
            def do_POST(s):
                slow.wait(5)
                Stub.do_POST(s)
        s2 = ThreadingServer(("127.0.0.1", 0), type("S2", (Slow,), {"seen": []}))
        threading.Thread(target=s2.serve_forever, daemon=True).start()
        self.servers.append(s2)
        px2 = proxy(f"http://127.0.0.1:{s2.server_address[1]}", self.log)
        self.servers.append(px2)
        th = threading.Thread(target=call, args=(px2.server_address[1], "POST", "/v1/chat/completions", chat([{"role": "user", "content": "a"}])))
        th.start()
        time.sleep(1.0)
        t2 = threading.Thread(target=call, args=(px2.server_address[1], "POST", "/v1/chat/completions", chat([{"role": "user", "content": "b"}])))
        t2.start()
        time.sleep(1.0)
        slow.set()
        th.join(), t2.join()
        lines = sorted(log_lines(self.log, 2), key=lambda x: x["t_start"])
        self.assertEqual([x["in_flight_at_start"] for x in lines], [0, 1])

    def test_scrub_writes_a_private_corpus_and_a_textless_manifest(self):
        self.recorded()
        corpus = os.path.join(self.dir, "corpus-v0")
        man = os.path.join(self.dir, "corpus-v0.manifest.json")
        self.assertEqual(scrub.main(["--out", corpus, "--manifest", man, self.log]), 0)
        self.assertEqual(stat.S_IMODE(os.stat(corpus).st_mode), 0o700)
        files = sorted(os.listdir(os.path.join(corpus, "strixite")))
        self.assertEqual(len(files), 7)
        for fn in files:
            self.assertEqual(stat.S_IMODE(os.stat(os.path.join(corpus, "strixite", fn)).st_mode), 0o600)
        with open(os.path.join(corpus, "strixite", files[0])) as f:
            e = json.load(f)
        self.assertIn("<SCRUB:apikey>", e["request"]["messages"][1]["content"])
        self.assertNotIn(SECRET, json.dumps(e))
        with open(man) as f:
            text = f.read()
        m = json.loads(text)
        self.assertNotIn("lighthouse", text)
        self.assertNotIn("keeper", text)
        self.assertEqual(len(m["exchanges"]), 7)
        self.assertEqual([x["order"] for x in m["exchanges"]], list(range(7)))
        chats = [x for x in m["exchanges"] if x["route"] == "POST /v1/chat/completions"]
        self.assertEqual(chats[0]["conversation"], chats[1]["conversation"])          # one thread, two turns
        self.assertFalse(chats[-1]["replayable"])                                      # the remote image
        census = m["census"]["strixite"]
        self.assertEqual(census["cache route"], 2)
        self.assertEqual(census["images"], 1)
        self.assertEqual(census["stream"], 1)
        self.assertEqual(scrub.main(["--check", corpus, "--manifest", man]), 0)
        # each failure the check exists for
        os.chmod(corpus, 0o755)
        self.assertEqual(scrub.main(["--check", corpus, "--manifest", man]), 1)
        os.chmod(corpus, 0o700)
        p = os.path.join(corpus, "strixite", files[1])
        with open(p) as f:
            e = json.load(f)
        e["request"]["messages"][0]["content"] += " " + SECRET
        with open(p, "w") as f:
            json.dump(e, f)
        self.assertEqual(scrub.main(["--check", corpus, "--manifest", man]), 1)
        scrub.main(["--out", corpus, "--manifest-only", "--manifest", man])            # the manifest follows the file
        self.assertEqual(scrub.main(["--check", corpus, "--manifest", man]), 1)        # the secret is still there
        leak = dict(m, leak=STORY)
        with open(man, "w") as f:
            json.dump(leak, f)
        self.assertEqual(scrub.check(corpus, man), 1)

    def replay(self, variant_b="same", extra=()):
        self.log = os.path.join(self.dir, f"rec-{variant_b}-{len(extra)}", "log.jsonl")
        self.recorded()
        corpus = os.path.join(self.dir, f"corpus-{variant_b}-{len(extra)}")
        scrub.main(["--out", corpus, self.log])
        a, ha = self.up()
        b, hb = self.up(variant_b)
        out = os.path.join(self.dir, f"res-{variant_b}-{len(extra)}")
        rc = replay.main(["--corpus", corpus, "--a", f"http://127.0.0.1:{a.server_address[1]}", "--b", f"http://127.0.0.1:{b.server_address[1]}",
                          "--assume-fresh", "--out", out, *extra])
        with open(os.path.join(out, "results.json")) as f:
            return rc, json.load(f), ha, hb, out

    def test_replay_of_equal_servers_passes_and_injects_seeds(self):
        rc, res, ha, hb, out = self.replay()
        self.assertEqual((rc, res["verdict"]), (0, "PASS"), json.dumps(res, indent=1)[:3000])
        self.assertEqual(res["counts"]["replayed"], 5)                              # 3 chats, export, import
        self.assertEqual(sorted(s["why"] for s in res["skipped"]), ["remote image URL", "state route (not a reply)"])
        chats = [json.loads(s[2]) for s in ha.seen if s[0] == "/v1/chat/completions"]
        ids = [r["id"] for r in res["results"] if r["route"] == "POST /v1/chat/completions"]
        self.assertEqual(chats[0]["seed"], replay.seed_of(ids[0]))                  # injected, from the exchange id
        self.assertEqual(chats[1]["seed"], 11)                                      # recorded, kept
        self.assertEqual([json.loads(s[2]) for s in hb.seen if s[0] == "/v1/chat/completions"], chats)
        imports = [s for s in ha.seen if s[0] == "/cache/import"]
        self.assertEqual(imports[0][2], hashlib.sha256(state_bytes("4096")).hexdigest())   # its own export, sent back
        with open(os.path.join(out, "summary.md")) as f:
            summary = f.read()
        self.assertIn("Verdict: **PASS**", summary)
        with open(os.path.join(out, "results.json")) as f:
            self.assertNotIn("lighthouse", f.read())                                # results carry no message text

    def test_replay_finds_each_difference(self):
        rc, res, *_ = self.replay("other")
        self.assertEqual((rc, res["verdict"]), (1, "FAIL"))
        bad = [r for r in res["results"] if not r["equal"]]
        self.assertEqual(len(bad), 3)
        self.assertEqual(bad[0]["body"]["first_path"], "choices[0].message.content")
        self.assertEqual(bad[2]["body"]["compared"], "sse")
        # the volatile list is what makes equal servers equal
        rc, res, *_ = self.replay("same", ("--volatile", "created,timings"))
        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual({r["body"].get("first_path") for r in res["results"] if not r["equal"]}, {"id", "[0].id"})

    def test_fresh_start_is_required(self):
        self.recorded()
        corpus = os.path.join(self.dir, "corpus")
        scrub.main(["--out", corpus, self.log])
        a, _ = self.up()
        out = os.path.join(self.dir, "res")
        url = f"http://127.0.0.1:{a.server_address[1]}"
        self.assertEqual(replay.main(["--corpus", corpus, "--a", url, "--b", url, "--out", out]), 1)
        with open(os.path.join(out, "summary.md")) as f:
            self.assertIn("A NOT SHOWN", f.read())

    def test_body_comparisons(self):
        a = {"id": "x", "choices": [{"message": {"content": "hi", "tool_calls": [{"id": "call_1", "function": {}}]}}]}
        b = json.loads(json.dumps(a))
        b["choices"][0]["message"]["tool_calls"][0]["id"] = "call_2"
        self.assertEqual(replay.first_path(a, b), "choices[0].message.tool_calls[0].id")
        replay.drop_paths(a, ["choices.*.message.tool_calls.*.id"])
        replay.drop_paths(b, ["choices.*.message.tool_calls.*.id"])
        self.assertIsNone(replay.first_path(a, b))
        self.assertEqual(replay.first_path({"a": 1, "b": 2}, {"b": 2, "a": 1}), "$ (key order)")


def fake_front_end():
    """(binary, tokenizer, [fake args]) of the stand-in front end's gate, or None where they are not on this host."""
    with open(os.path.join(ROOT, "gate", "qwen38-flash-next.flerp-serve-fake.toml"), "rb") as f:
        cfg = tomllib.load(f)
    with open(os.path.join(ROOT, cfg["inherit"]), "rb") as f:
        tok = tomllib.load(f)["host"]["tokenizer"]
    binary = os.path.join(ROOT, "build", "qwen38-flash-next.flerp-serve-fake", "ninja", "strixflerp-qwen38-flash-next.flerp-serve-fake")
    if not (os.path.isfile(binary) and os.access(binary, os.X_OK) and os.path.isfile(tok)):
        return None
    return binary, tok, [str(a) for a in cfg["fake"]["args"]]


FAKE = fake_front_end()


@unittest.skipUnless(FAKE, "the stand-in front end's binary or tokenizer is not on this host (control-node has both)")
class FrontEndAsBothServers(unittest.TestCase):
    """Record traffic through the recorder in front of the front end over executor.flerp.fake_abi, scrub it, and replay
    it with the gate starting two fresh front ends: equal executors pass, an executor with another model fails."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="flerp-replay-fe-")
        os.chmod(cls.dir, 0o700)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def cmd(self, port, extra=()):
        binary, tok, args = FAKE
        keys = {x.split("=")[0] for x in extra}
        args = [x for x in args if x.split("=")[0] not in keys]      # an override replaces the gate's own value
        return " ".join([binary, "--host", "127.0.0.1", "--port", str(port), "--model-id", "fake", f"text.tokenizer={tok}",
                         *args, "executor.draft=step", *extra])

    def test_record_scrub_replay(self):
        port = free_port()
        log = os.path.join(self.dir, "rec", "log.jsonl")
        srv = subprocess.Popen(self.cmd(port), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for _ in range(600):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                    break
                except OSError:
                    time.sleep(0.1)
            px = proxy(f"http://127.0.0.1:{port}", log)
            p = px.server_address[1]
            sysm = {"role": "system", "content": "You are a careful assistant on the pod."}
            u1 = [sysm, {"role": "user", "content": STORY}]
            r1 = call(p, "POST", "/v1/chat/completions", chat(u1))
            self.assertEqual(r1[0], 200, r1[2][:300])
            # The second turn's assistant line is fixed text, NOT the reply just received. The first request is
            # recorded without a seed (that is the point of it), so its reply is one draw under a random seed, and a
            # corpus that embeds it is different traffic on every run -- with it, this test's FAIL set moved between
            # 4 and 5 exchanges run to run on unchanged code (review d0b17fd3d2: 7 failures in 26 runs of the Verify
            # command). Traffic a test puts exact numbers on has to be the same traffic every run; that is what a
            # recorded corpus is for. The unseeded first request still proves seed injection (see the diff below).
            u2 = u1 + [{"role": "assistant", "content": "Once there was a keeper who counted the waves until morning."},
                       {"role": "user", "content": "And then what happened to the keeper?"}]
            self.assertEqual(call(p, "POST", "/v1/chat/completions", chat(u2))[0], 200)
            self.assertEqual(call(p, "POST", "/v1/chat/completions", chat(u1, temperature=0.8, top_k=20))[0], 200)   # sampled, seed injected
            self.assertEqual(call(p, "POST", "/v1/chat/completions", chat(u1, temperature=0.8, top_k=20, seed=3))[0], 200)
            self.assertEqual(call(p, "POST", "/v1/chat/completions", chat(u1, stream=True))[0], 200)
            self.assertEqual(call(p, "GET", "/v1/models")[0], 200)
            self.assertEqual(call(p, "POST", "/v1/chat/completions", chat(u1, tools=[{"type": "function", "function": {"name": "f"}}]))[0], 400)
            log_lines(log, 7)
            px.shutdown()
            px.server_close()
        finally:
            os.killpg(srv.pid, 15)
            srv.wait(30)
        corpus = os.path.join(self.dir, "corpus")
        man = os.path.join(self.dir, "corpus.manifest.json")
        self.assertEqual(scrub.main(["--out", corpus, "--manifest", man, log]), 0)
        self.assertEqual(scrub.main(["--check", corpus, "--manifest", man]), 0)

        def run(extra_b, name):
            pa, pb = free_port(), free_port()
            out = os.path.join(self.dir, name)
            rc = replay.main(["--corpus", corpus, "--a", f"http://127.0.0.1:{pa}", "--b", f"http://127.0.0.1:{pb}",
                              "--a-kind", "flerp", "--b-kind", "flerp", "--a-cmd", self.cmd(pa), "--b-cmd", self.cmd(pb, extra_b),
                              "--ready-timeout", "120", "--out", out])
            with open(os.path.join(out, "results.json")) as f:
                return rc, json.load(f)

        rc, res = run((), "same")
        self.assertEqual((rc, res["verdict"]), (0, "PASS"), json.dumps([r for r in res["results"] if not r["equal"]], indent=1)[:3000])
        self.assertEqual(res["counts"]["replayed"], 7)
        self.assertEqual(res["counts"]["tokens_compared"], 5)        # the five chats that ran; the refusal has none
        # `tokens_compared` counts exchanges where BOTH sides produced an id list, so a reader that
        # returned nothing would show 0, not a PASS. And the lists it compared must hold real ids:
        # comparing [] to [] is equal but proves nothing (review d0b17fd3d2, finding 1's worry).
        for r in [x for x in res["results"] if isinstance(x["tokens"].get("compared"), str)]:
            self.assertTrue(r["tokens"]["a"] and r["tokens"]["b"], r)
        self.assertTrue(res["a"]["fresh"] and res["b"]["fresh"])
        rc, res = run(("executor.reply=30",), "other-model")
        self.assertEqual((rc, res["verdict"]), (1, "FAIL"))
        diff = [r for r in res["results"] if not r["equal"]]
        self.assertEqual(len(diff), 5, diff)                       # all five chats: B stops at its own token 30, the refusal is equal
        for r in diff:
            self.assertEqual((r["tokens"]["compared"], r["tokens"]["equal"]), ("token_ids", False), r)
            self.assertIn(r["tokens"]["first_difference"], (30, 31), r)   # where the shorter model's end falls
            self.assertIn("choices[0]", r["body"]["first_path"])


if __name__ == "__main__":
    unittest.main()
