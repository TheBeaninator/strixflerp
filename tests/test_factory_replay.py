"""factory_replay against a tiny fake OpenAI server (card 2587bc1599): no GPU, no pod, stdlib only.

  python3 -m unittest tests.test_factory_replay
"""
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "replay"))
import factory_replay as fr  # noqa: E402


def words(messages):
    return sum(len(str(m.get("content") or "").split()) for m in messages)


class Fake(ThreadingHTTPServer):
    """Replies are a function of the messages alone, unless leaky: then company changes the reply."""
    daemon_threads = True

    def __init__(self, leaky=False):
        super().__init__(("127.0.0.1", 0), Handler)
        self.leaky, self.lock = leaky, threading.Lock()
        self.in_flight = self.max_in_flight = 0
        self.bodies, self.seen = [], set()
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self):
        return "http://127.0.0.1:%d" % self.server_address[1]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def do_GET(self):
        raw = {"/health": b'{"status":"ok"}', "/metrics": b"# TYPE strix_slots gauge\nstrix_slots 2\n"}.get(self.path)
        self.send_response(200 if raw else 404)
        self.send_header("Content-Length", str(len(raw or b"")))
        self.end_headers()
        self.wfile.write(raw or b"")

    def do_POST(self):
        s = self.server
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/v1/chat/completions":  # pragma: no cover
            self.send_error(404)
            return
        with s.lock:
            s.in_flight += 1
            company = s.in_flight
            s.max_in_flight = max(s.max_in_flight, s.in_flight)
            s.bodies.append(body)
        time.sleep(0.03)
        msgs = body["messages"]
        keys = fr.prefix_keys(msgs)
        with s.lock:
            cached = 0
            for k in range(len(msgs), 0, -1):       # the longest prefix this server has seen before
                if keys[k - 1] in s.seen:
                    cached = words(msgs[:k])
                    break
            s.seen.update(keys)
            s.in_flight -= 1
        text = "reply " + hashlib.sha256(keys[-1].encode()).hexdigest()[:10]
        if s.leaky and company > 1:
            text += " in company"
        n = 5 + len(msgs)
        out = {"id": "chatcmpl-%d" % time.monotonic_ns(), "object": "chat.completion", "model": body["model"],
               "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": words(msgs), "completion_tokens": n, "total_tokens": words(msgs) + n,
                         "prompt_tokens_details": {"cached_tokens": cached}}}
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def exchange(i, target, messages, t, path="/v1/chat/completions", status=200, **more):
    e = {"id": "%s-%05d" % (target, i), "target": target, "method": "POST", "path": path, "t_start": t, "t_end": t + 2.5,
         "status": status, "req_complete": True, "resp_complete": True,
         "request": {"model": "recorded-model", "messages": messages, "max_tokens": 16384, "stream": True,
                     "temperature": 0.7},
         "response_raw": json.dumps({"usage": {"prompt_tokens": 100, "completion_tokens": 7,
                                               "prompt_tokens_details": {"cached_tokens": 60}}})}
    e.update(more)
    return e


def make_corpus(root):
    """Three conversations (3, 2 and 1 turns) interleaved in time, plus exchanges a load replay must leave out."""
    def convo(name, turns):
        msgs = [{"role": "system", "content": "you are thread " + name}, {"role": "user", "content": name + " first ask"}]
        out = [list(msgs)]
        for k in range(1, turns):
            msgs += [{"role": "assistant", "content": "%s recorded answer %d" % (name, k)},
                     {"role": "user", "content": "%s follow up %d with more words" % (name, k)}]
            out.append(list(msgs))
        return out
    a, b, c = convo("alpha", 3), convo("beta", 2), convo("gamma", 1)
    order = [("strixite", a[0]), ("strixite", b[0]), ("strixite", a[1]), ("pair", c[0]), ("strixite", b[1]), ("strixite", a[2])]
    ex = [exchange(i + 1, tg, m, 1000.0 + i) for i, (tg, m) in enumerate(order)]
    ex.append(exchange(20, "strixite", None, 1100.0, path="/cache/probe"))
    ex.append(exchange(21, "strixite", a[0], 1101.0, status=400))
    ex.append(exchange(22, "strixite", a[0], 1102.0, replayable=False))
    ex.append(exchange(23, "synthetic", a[0], 1103.0))
    for e in ex:
        d = os.path.join(root, e["target"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, e["id"] + ".json"), "w") as f:
            json.dump(e, f)
    return root


class FactoryReplay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.corpus = make_corpus(os.path.join(self.tmp.name, "corpus"))
        self.set = fr.freeze(self.corpus)
        self.servers = []

    def tearDown(self):
        for s in self.servers:
            s.shutdown()
            s.server_close()
        self.tmp.cleanup()

    def fake(self, **kw):
        s = Fake(**kw)
        self.servers.append(s)
        return s

    def test_freeze_chains_turns_into_streams(self):
        s = self.set
        self.assertEqual(s["exchanges"], 6)
        self.assertEqual([len(x["turns"]) for x in s["streams"]], [3, 2, 1])
        self.assertEqual([x["target"] for x in s["streams"]], ["strixite", "strixite", "pair"])
        self.assertEqual([t["messages"] for t in s["streams"][0]["turns"]], [2, 4, 6])
        self.assertEqual(s["skipped"], {"not a chat completion": 1, "status 400": 1, "not replayable": 1})
        self.assertEqual(s["streams"][0]["turns"][0]["recorded"],
                         {"wall_s": 2.5, "prompt_tokens": 100, "cached_tokens": 60, "completion_tokens": 7})
        self.assertNotIn("alpha", json.dumps(s))                 # numbers and hashes only, no message text
        self.assertEqual(fr.freeze(self.corpus)["set_sha256"], s["set_sha256"])
        self.assertEqual(fr.freeze(self.corpus, targets={"strixite"})["exchanges"], 5)
        self.assertEqual([len(x["turns"]) for x in fr.freeze(self.corpus, max_turns=2)["streams"]], [2, 1, 2, 1])

    def test_verify_catches_a_changed_exchange(self):
        self.assertEqual(fr.verify(self.set, self.corpus), [])
        t = self.set["streams"][1]["turns"][0]
        with open(os.path.join(self.corpus, t["file"]), "a") as f:
            f.write(" ")
        bad = fr.verify(self.set, self.corpus)
        self.assertEqual(len(bad), 1)
        self.assertIn(t["file"], bad[0])
        srv = self.fake()
        sp = os.path.join(self.tmp.name, "set.json")
        with open(sp, "w") as f:
            json.dump(self.set, f)
        with contextlib.redirect_stderr(io.StringIO()):
            rc = fr.main(["run", "--set", sp, "--corpus", self.corpus, "--url", srv.url, "--model", "m", "--concurrency", "2"])
        self.assertEqual(rc, 2)
        self.assertEqual(srv.bodies, [])                         # nothing was sent

    def test_run_numbers_and_identity(self):
        srv = self.fake()
        res = fr.run(self.set, self.corpus, srv.url, "dsv41", concurrency=2, reps=2, max_tokens=64, solo="all",
                     strip=["temperature_does_not_exist"], extra={"enable_thinking": False})
        self.assertEqual([p["phase"] for p in res["phases"]], ["solo-1", "rep-1", "rep-2"])
        self.assertEqual(len(srv.bodies), 18)                    # 6 exchanges x (1 solo + 2 reps)
        for b in srv.bodies:                                     # every request: the named model, greedy, capped
            self.assertEqual((b["model"], b["temperature"], b["stream"], b["max_tokens"], b["enable_thinking"]),
                             ("dsv41", 0, False, 64, False))
            self.assertIn("seed", b)
        self.assertEqual(srv.max_in_flight, 2)
        solo, rep1, rep2 = res["phases"]
        self.assertEqual((solo["max_in_flight"], rep1["max_in_flight"]), (1, 2))
        self.assertEqual((solo["turns"], solo["errors"], rep1["turns"], rep1["errors"]), (6, 0, 6, 0))
        tokens = 7 + 9 + 11 + 7 + 9 + 7                          # the fake's completion_tokens: 5 + messages
        self.assertEqual(rep1["completion_tokens"], tokens)
        self.assertAlmostEqual(rep1["committed_tokens_per_s"], tokens / rep1["wall_s"], delta=5)
        self.assertGreater(rep1["committed_tokens_per_s"], solo["committed_tokens_per_s"])   # two in flight beat one
        self.assertLessEqual(rep1["turn_p50_s"], rep1["turn_p95_s"])
        self.assertGreaterEqual(rep1["turn_p50_s"], 0.03)
        # solo starts cold: each turn prefills what its stream did not send before; the reps find every prompt cached
        self.assertGreater(solo["prefilled_tokens_per_request"], 0)
        self.assertEqual(rep1["prefilled_tokens_per_request"], 0)
        self.assertEqual(rep1["cached_tokens_per_request"], rep1["prompt_tokens_per_request"])
        self.assertEqual(solo["prefilled_tokens_per_request"] + solo["cached_tokens_per_request"],
                         solo["prompt_tokens_per_request"])
        self.assertEqual([(i["phase"], i["against"], i["compared_by"], i["streams_checked"], i["identical"], i["differs"])
                          for i in res["identity"]], [("rep-1", "solo-1", "text", 3, 3, 0), ("rep-2", "solo-1", "text", 3, 3, 0)])
        self.assertEqual((res["errors"], res["streams_not_identical"]), (0, 0))
        seen = []
        fr.run(self.set, self.corpus, srv.url, "m", concurrency=2, reps=2, solo="1",
               checkpoint=lambda r: seen.append((len(r["phases"]), r["reps_done"], len(r["identity"]))))
        self.assertEqual(seen, [(1, 0, 0), (2, 1, 1), (3, 2, 2)])   # written after every phase
        self.assertEqual(res["median_of_reps"]["committed_tokens_per_s"],
                         round((rep1["committed_tokens_per_s"] + rep2["committed_tokens_per_s"]) / 2, 3))
        text = fr.report(res)
        for want in ("committed tokens a second", "turn time p50", "p95", "prefilled tokens a request",
                     "cached tokens a request", "3 of 3 streams identical"):
            self.assertIn(want, text)
        self.assertNotIn("alpha", json.dumps(res))

    def test_a_reply_that_changes_in_company_is_caught(self):
        srv = self.fake(leaky=True)
        res = fr.run(self.set, self.corpus, srv.url, "m", concurrency=2, reps=1, solo="all")
        i = res["identity"][0]
        self.assertGreaterEqual(i["differs"], 1)
        self.assertEqual(i["identical"] + i["differs"], 3)
        self.assertEqual(res["streams_not_identical"], i["differs"])
        self.assertIn("differs at turn", fr.report(res))

    def test_solo_subset_and_two_solo_passes(self):
        srv = self.fake()
        res = fr.run(self.set, self.corpus, srv.url, "m", concurrency=3, reps=1, solo="2", solo_passes=2)
        self.assertEqual([(p["phase"], p["turns"]) for p in res["phases"]], [("solo-1", 5), ("solo-2", 5), ("rep-1", 6)])
        self.assertEqual([(i["phase"], i["against"], i["streams_checked"], i["identical"]) for i in res["identity"]],
                         [("solo-1", "solo-2", 2, 2), ("rep-1", "solo-2", 2, 2)])
        self.assertIn("identity covers the first 2 of 3 streams", fr.report(res))

    def test_first_streams_cuts_by_the_sha_of_the_first_turn(self):
        cut = fr.first_streams(self.set, 2)
        shas = sorted(x["turns"][0]["sha256"] for x in self.set["streams"])
        self.assertEqual([x["turns"][0]["sha256"] for x in cut["streams"]], shas[:2])
        self.assertEqual(cut["exchanges"], sum(len(x["turns"]) for x in cut["streams"]))
        self.assertEqual((cut["set_sha256"], cut["selection"]["of_streams"], len(self.set["streams"])),
                         (self.set["set_sha256"], 3, 3))
        srv = self.fake()
        res = fr.run(cut, self.corpus, srv.url, "m", concurrency=2, reps=1)
        self.assertEqual((res["streams"], res["phases"][-1]["turns"]), (2, cut["exchanges"]))
        self.assertIn("selection: first 2 streams by the sha256 of their first turn", fr.report(res))

    def test_token_ids_are_compared_when_the_reply_has_them(self):
        a = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "flerp": {"token_ids": [1, 2, 3]}}
        b = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "flerp": {"token_ids": [1, 2, 4]}}
        self.assertEqual(fr.signature(a)[0], "ids")
        self.assertNotEqual(fr.signature(a), fr.signature(b))
        c = {"choices": [{"message": {"content": "x", "tool_calls": [{"id": "call_1", "function": {"name": "f", "arguments": "{}"}}]}}]}
        d = {"choices": [{"message": {"content": "x", "tool_calls": [{"id": "call_2", "function": {"name": "f", "arguments": "{}"}}]}}]}
        self.assertEqual(fr.signature(c), fr.signature(d))       # a random call id is not a difference
        self.assertEqual(fr.signature(c)[0], "text")

    def test_a_dead_engine_is_an_error_not_a_number(self):
        srv = self.fake()
        url = srv.url
        srv.shutdown()
        srv.server_close()
        self.servers.clear()
        res = fr.run(self.set, self.corpus, url, "m", concurrency=2, reps=1, solo="none", timeout=5)
        self.assertEqual((res["errors"], res["phases"][0]["completion_tokens"]), (6, 0))
        self.assertEqual(res["identity"], [])

    def test_command_line_end_to_end(self):
        srv = self.fake()
        sp, out = os.path.join(self.tmp.name, "set.json"), os.path.join(self.tmp.name, "out")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(fr.main(["freeze", "--corpus", self.corpus, "--out", sp]), 0)
            self.assertEqual(fr.main(["verify", "--set", sp, "--corpus", self.corpus]), 0)
            rc = fr.main(["run", "--set", sp, "--corpus", self.corpus, "--url", srv.url + "/v1", "--model",
                          "qwen3.8-flash-next", "--concurrency", "2", "--reps", "3", "--out", out, "--quiet",
                          "--wait-ready", "5", "--expect-slots", "2"])
            with contextlib.redirect_stderr(io.StringIO()):                # the wrong slot count sends nothing
                n = len(srv.bodies)
                self.assertEqual(fr.main(["run", "--set", sp, "--corpus", self.corpus, "--url", srv.url, "--model", "m",
                                          "--concurrency", "2", "--expect-slots", "4"]), 2)
                self.assertEqual(len(srv.bodies), n)
        self.assertEqual(rc, 0)
        self.assertIn("engine reports strix_slots 2", buf.getvalue())
        printed = buf.getvalue()
        self.assertIn("frozen 6 exchanges in 3 streams", printed)
        self.assertIn("committed tokens a second", printed)
        self.assertIn("identity rep-3 against solo-1 (by text): 3 of 3 streams identical", printed)
        with open(os.path.join(out, "result.json")) as f:
            res = json.load(f)
        self.assertEqual((res["model"], res["concurrency"], res["reps"], res["exchanges"]), ("qwen3.8-flash-next", 2, 3, 6))
        self.assertEqual(len(res["turns"]["rep-3"]), 6)
        self.assertTrue(os.path.getsize(os.path.join(out, "summary.md")) > 0)
        self.assertEqual({b["model"] for b in srv.bodies}, {"qwen3.8-flash-next"})


if __name__ == "__main__":
    unittest.main()
