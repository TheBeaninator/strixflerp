"""tools/ablate: the expert-mask ablation driver (pod card 5bb471d3b9), against fake servers and a fake eval harness.

Proofs: the mask rules and the canonical id are the engine's (mask.zig: the same refusals, the same id for the same
pairs in any order, "" for an empty mask); the coarsened KL (the shared top-K support plus a remainder bucket, pod card
73bab11681) matches hand calculations, including a token the masked side does not list, never exceeds the full KL and
equals it when K is the vocabulary; positions() reads the fork server's reply (bytes, a null logprob, two ids with one
text); a fake server's top_logprobs give the hand-computed KL at the first position; against a fake OpenAI server pair that agree, every KL is 0 and agreement 1; against a masked fake
that changes some prompts' greedy token at a known position, those prompts' agreeing prefix is that position, KL is
pooled only over positions with the same context (0 before the change, > 0 at it), the others stay 0, and the summary's
mean and p95 are those of the rows; an endpoint that refuses logprobs (as the served dsv41 build does) gives KL null
with the reason and agreement from /tokenize, and --require-kl refuses; the gate driver runs the harness's run.py for
the masked engine (--engine NAME=URL, the gate's arms on that engine only) and the unmasked one with the same arm names,
marks the masked run, and its deltas equal a hand count; --base-run reuses a run; an empty mask is refused unless asked.
"""
import hashlib
import http.server
import json
import math
import os
import random
import shutil
import sys
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "tools"))
import ablate as A  # noqa: E402

VOCAB = ["t%d" % i for i in range(50)]


def logits_at(prompt_key, context):
    h = hashlib.sha256(("%s|%s" % (prompt_key, ",".join(context))).encode()).digest()
    return [h[i % 32] / 64.0 + (i % 7) * 0.01 for i in range(len(VOCAB))]


class Fake(http.server.BaseHTTPRequestHandler):
    """A greedy OpenAI chat server over a 50-token vocabulary. server.flip: {prompt key: position} where the masked
    engine boosts another token from that position on; server.refuse: logprobs are refused as fields.zig refuses them."""

    def log_message(self, *a):
        pass

    def reply(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.seen.append((self.path, body))
        if self.path == "/tokenize":
            return self.reply(200, {"count": 0, "tokens": [VOCAB.index(t) for t in body["prompt"].split()]})
        if body.get("logprobs") and self.server.refuse:
            return self.reply(400, {"error": {"message": "logprobs are not supported by this model or backend", "type": "invalid_request_error"}})
        key = json.dumps(body["messages"], sort_keys=True)
        ctx, content = [], []
        for pos in range(body["max_tokens"]):
            lg = logits_at(key, ctx)
            at = self.server.flip.get(key)
            if at is not None and pos >= at:
                lg[(pos * 13 + 5) % len(VOCAB)] += 10.0
            mx = max(lg)
            lse = mx + math.log(sum(math.exp(x - mx) for x in lg))
            lp = [x - lse for x in lg]
            best = max(range(len(lp)), key=lambda i: (lp[i], -i))
            top = sorted(range(len(lp)), key=lambda i: (-lp[i], i))[:body.get("top_logprobs") or 0]
            content.append({"token": VOCAB[best], "logprob": lp[best], "bytes": list(VOCAB[best].encode()),
                            "top_logprobs": [{"token": VOCAB[i], "logprob": lp[i]} for i in top]})
            ctx.append(VOCAB[best])
        msg = {"role": "assistant", "content": " ".join(ctx)}
        choice = {"index": 0, "message": msg, "finish_reason": "length", "logprobs": {"content": content} if body.get("logprobs") else None}
        return self.reply(200, {"choices": [choice], "usage": {"completion_tokens": len(ctx)}})


def serve(flip=None, refuse=False):
    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    s.flip, s.refuse, s.seen = flip or {}, refuse, []
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, "http://127.0.0.1:%d" % s.server_address[1]


def put(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def get(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def prompts(n=8):
    rows = []
    for i in range(n):
        dom = ["code", "review", "vision", "chat"][i % 4]
        rows.append({"id": "p%02d" % i, "domain": dom, "messages": [{"role": "user", "content": "prompt %d about %s" % (i, dom)}]})
    return rows


def key_of(p):
    return json.dumps(p["messages"], sort_keys=True)


class MaskRules(unittest.TestCase):
    def test_the_engines_rules_and_id(self):
        text = "# route_join mask: suspects with q < 0.05, method=perm-fdr\n3:17\n\n  5:1, 2 ,9  # x\r\n39:383\n3:17\n"
        m = A.parse_mask(text)
        self.assertEqual(m, {3: {17}, 5: {1, 2, 9}, 39: {383}})
        self.assertEqual(A.canonical(m), "3:17\n5:1,2,9\n39:383\n")
        self.assertEqual(A.mask_id(m), "7672079e14c6b86a")   # mask.zig's test asserts the same id
        self.assertEqual(A.mask_id(A.parse_mask("39:383\n5:9,2,1\n3:17")), "7672079e14c6b86a")
        self.assertEqual(A.mask_id(A.parse_mask("# only a header\n")), "")
        for bad, word in (("1:2\n40:1", "line 2"), ("1:384", "routed"), ("1:+2", "bad expert"), ("1:1_0", "bad expert"),
                          ("17", "not 'layer"), ("1:2,,3", "bad expert"), ("x:1", "bad layer")):
            with self.assertRaises(A.MaskError) as cm:
                A.parse_mask(bad)
            self.assertIn(word, str(cm.exception), bad)
        ok = "7:" + ",".join(str(e) for e in range(A.EXPERTS - A.TOP_K))
        self.assertEqual(len(A.parse_mask(ok)[7]), A.EXPERTS - A.TOP_K)
        with self.assertRaises(A.MaskError):
            A.parse_mask(ok + ",%d" % (A.EXPERTS - A.TOP_K))


class Stats(unittest.TestCase):
    def test_coarse_kl_by_hand(self):
        L = math.log
        p = {"a": L(0.5), "b": L(0.3)}
        self.assertEqual(A.coarse_kl(p, p), 0.0)
        q = {"a": L(0.4), "b": L(0.4)}
        self.assertAlmostEqual(A.coarse_kl(p, q), 0.5 * L(0.5 / 0.4) + 0.3 * L(0.3 / 0.4) + 0.2 * L(0.2 / 0.2), places=12)
        # only a is listed by both: the remainder is 1 - 0.6 on P's side (c and its tail), 1 - 0.5 on Q's (b and its tail)
        p2 = {"a": L(0.6), "c": L(0.3)}
        q2 = {"a": L(0.5), "b": L(0.4)}
        self.assertAlmostEqual(A.coarse_kl(p2, q2), 0.6 * L(0.6 / 0.5) + 0.4 * L(0.4 / 0.5), places=12)
        # nothing shared: one bucket each side, all the mass: 0
        self.assertEqual(A.coarse_kl({"a": L(0.9)}, {"b": L(0.9)}), 0.0)

    def test_coarse_kl_is_a_lower_bound_and_exact_over_the_whole_vocabulary(self):
        rng = random.Random(7)
        for _ in range(50):
            n = 30
            lp = []
            for _side in range(2):
                xs = [rng.gauss(0, 2) for _ in range(n)]
                mx = max(xs)
                lse = mx + math.log(sum(math.exp(x - mx) for x in xs))
                lp.append({"t%d" % i: x - lse for i, x in enumerate(xs)})
            full = sum(math.exp(lp[0][t]) * (lp[0][t] - lp[1][t]) for t in lp[0])
            top = lambda m, k: dict(sorted(m.items(), key=lambda kv: -kv[1])[:k])
            self.assertAlmostEqual(A.coarse_kl(lp[0], lp[1]), full, places=9)
            for k in (1, 5, 20):
                self.assertLessEqual(A.coarse_kl(top(lp[0], k), top(lp[1], k)), full + 1e-12)

    def test_positions_read_the_fork_servers_reply(self):
        L = math.log
        R = chr(0xFFFD)  # a partial UTF-8 token's text
        rep = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "length",
                            "logprobs": {"content": [
                                {"token": "h", "logprob": L(0.5), "bytes": [104],
                                 "top_logprobs": [{"token": "h", "logprob": L(0.5), "bytes": [104]},
                                                  {"token": R, "logprob": L(0.1), "bytes": [226, 128]},
                                                  {"token": R, "logprob": L(0.05), "bytes": [240, 159]}]},
                                {"token": "i", "logprob": None, "bytes": [105],
                                 "top_logprobs": [{"token": "j", "logprob": L(0.6), "bytes": [106]}]}]}}]}
        pos = A.positions(rep)
        self.assertEqual([t for t, _ in pos], [(104,), (105,)])               # keyed by bytes
        self.assertAlmostEqual(math.exp(pos[0][1][(226, 128)]), 0.1, places=12)  # one text, two ids: kept apart
        self.assertAlmostEqual(math.exp(pos[0][1][(240, 159)]), 0.05, places=12)
        self.assertEqual(set(pos[1][1]), {(106,)})                           # a null logprob is left out
        # one side lists both partial ids, the other one of them: still a lower bound on the full KL
        p_full = {(104,): 0.5, (226, 128): 0.3, (240, 159): 0.15, (106,): 0.05}
        q_full = {(104,): 0.5, (226, 128): 0.05, (240, 159): 0.3, (106,): 0.15}

        def reply_of(full, listed):
            return {"choices": [{"logprobs": {"content": [{"token": "h", "logprob": L(full[(104,)]), "bytes": [104],
                    "top_logprobs": [{"token": R if b[0] > 127 else chr(b[0]), "logprob": L(full[b]), "bytes": list(b)} for b in listed]}]}}]}
        bp = A.positions(reply_of(p_full, [(104,), (226, 128), (240, 159)]))[0][1]
        mp = A.positions(reply_of(q_full, [(104,), (240, 159)]))[0][1]
        full = sum(p * L(p / q_full[b]) for b, p in p_full.items())
        self.assertLessEqual(A.coarse_kl(bp, mp), full)

    def test_compare_pools_kl_only_over_the_same_context(self):
        L = math.log
        same = {"x": L(0.7), "y": L(0.2)}
        diff = {"y": L(0.7), "x": L(0.2)}
        base = [("x", same), ("x", same), ("x", same), ("x", same)]
        masked = [("x", same), ("x", same), ("y", diff), ("x", same)]
        c = A.compare(base, masked)
        self.assertEqual((c["prefix"], c["agree"], c["positions"]), (2, 3, 4))
        self.assertEqual(len(c["kl_matched"]), 3)          # positions 0, 1 and 2 (where they part)
        self.assertEqual(c["kl_matched"][:2], [0.0, 0.0])
        self.assertGreater(c["kl_matched"][2], 0.5)
        self.assertEqual(c["kl_first"], 0.0)
        ids = A.compare([1, 2, 3], [1, 2, 3, 4])
        self.assertEqual((ids["agree"], ids["positions"], ids["prefix"], ids["kl_first"]), (3, 4, 3, None))

    def test_p95_interpolates(self):
        self.assertEqual(A.p95([]), None)
        self.assertEqual(A.p95([3.0]), 3.0)
        xs = list(range(1, 21))                     # numpy.percentile(xs, 95) = 19.05
        self.assertAlmostEqual(A.p95(xs), 19.05)


class KL(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.servers = []
        self.ps = prompts()
        self.pf = os.path.join(self.d, "prompts.jsonl")
        with open(self.pf, "w") as f:
            for p in self.ps:
                f.write(json.dumps(p) + "\n")
        self.mask = os.path.join(self.d, "mask.txt")
        put(self.mask, "# test\n3:17\n5:1,2,9\n39:383\n")

    def tearDown(self):
        for s in self.servers:
            s.shutdown()
            s.server_close()
        shutil.rmtree(self.d)

    def pair(self, flip=None, refuse=False):
        b, bu = serve()
        m, mu = serve(flip, refuse)
        self.servers += [b, m]
        return b, bu, m, mu

    def run_kl(self, bu, mu, *extra):
        out = os.path.join(self.d, "out")
        rc = A.main(["kl", "--base", bu, "--masked", mu, "--prompts", self.pf, "--mask", self.mask, "--out", out, "--positions", "8"] + list(extra))
        rows = [json.loads(l) for l in get(os.path.join(out, "kl-rows.jsonl")).splitlines()]
        return rc, rows, json.loads(get(os.path.join(out, "kl-summary.json"))), out

    def test_agreeing_engines_give_zero_kl_and_full_agreement(self):
        b, bu, m, mu = self.pair()
        rc, rows, s, out = self.run_kl(bu, mu)
        self.assertEqual(rc, 0)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(r["kl_first"] == 0.0 and r["agreement"] == 1.0 and r["exact"] and r["prefix"] == 8 for r in rows))
        self.assertEqual(s["kl_status"], "ok")
        self.assertEqual((s["all"]["kl_first_mean"], s["all"]["kl_first_p95"], s["all"]["agreement_mean"]), (0.0, 0.0, 1.0))
        self.assertEqual(s["mask"]["id"], "7672079e14c6b86a")
        self.assertEqual(sorted(s["domains"]), ["chat", "code", "review", "vision"])
        path, body = m.seen[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual((body["temperature"], body["seed"], body["max_tokens"], body["logprobs"], body["top_logprobs"]), (0, 0, 8, True, 20))
        self.assertIn("mask=7672079e14c6b86a", get(os.path.join(out, "kl-summary.md")))

    def test_a_masked_engine_that_parts_at_a_known_position(self):
        flip = {key_of(self.ps[1]): 0, key_of(self.ps[2]): 3, key_of(self.ps[6]): 5}
        b, bu, m, mu = self.pair(flip=flip)
        rc, rows, s, _ = self.run_kl(bu, mu)
        by = {r["id"]: r for r in rows}
        for pid, at in (("p01", 0), ("p02", 3), ("p06", 5)):
            r = by[pid]
            self.assertEqual(r["prefix"], at, pid)
            self.assertEqual(len(r["kl_matched"]), at + 1, pid)
            self.assertTrue(all(k == 0.0 for k in r["kl_matched"][:at]), pid)
            self.assertGreater(r["kl_matched"][at], 1.0, pid)
            self.assertLess(r["agreement"], 1.0)
        for pid in ("p00", "p03", "p04", "p05", "p07"):
            self.assertEqual((by[pid]["kl_first"], by[pid]["agreement"]), (0.0, 1.0), pid)
        firsts = sorted(r["kl_first"] for r in rows)
        self.assertAlmostEqual(s["all"]["kl_first_mean"], sum(firsts) / 8)
        self.assertAlmostEqual(s["all"]["kl_first_p95"], A.p95(firsts))
        pooled = [k for r in rows for k in r["kl_matched"]]
        self.assertEqual(s["all"]["kl_matched_positions"], len(pooled))
        self.assertAlmostEqual(s["all"]["kl_matched_mean"], sum(pooled) / len(pooled))
        self.assertEqual(s["domains"]["review"]["prompts"], 2)      # p01 and p05
        self.assertGreater(s["domains"]["review"]["kl_first_mean"], 0.5)
        self.assertEqual(s["domains"]["code"]["kl_first_mean"], 0.0)  # p00, p04

    def test_top_logprobs_from_the_server_give_the_hand_computed_kl(self):
        # p01's masked engine boosts token (0 * 13 + 5) % 50 by 10 from position 0: the first position's KL is the
        # coarse KL of the two top-20 lists the servers return, computed here from the fake's own logits
        flip = {key_of(self.ps[1]): 0}
        b, bu, m, mu = self.pair(flip=flip)
        rc, rows, s, _ = self.run_kl(bu, mu)
        self.assertEqual(rc, 0)

        def top20(lg):
            mx = max(lg)
            lse = mx + math.log(sum(math.exp(x - mx) for x in lg))
            lp = [x - lse for x in lg]
            return {VOCAB[i]: lp[i] for i in sorted(range(len(lp)), key=lambda i: (-lp[i], i))[:20]}, lp

        base_lg = logits_at(key_of(self.ps[1]), [])
        masked_lg = list(base_lg)
        masked_lg[5] += 10.0
        (bt, blp), (mt, mlp) = top20(base_lg), top20(masked_lg)
        shared = [t for t in bt if t in mt]
        pr, qr = 1 - sum(math.exp(bt[t]) for t in shared), 1 - sum(math.exp(mt[t]) for t in shared)
        want = sum(math.exp(bt[t]) * (bt[t] - mt[t]) for t in shared) + pr * math.log(pr / qr)
        got = {r["id"]: r for r in rows}["p01"]["kl_first"]
        self.assertAlmostEqual(got, want, places=9)
        full = sum(math.exp(blp[i]) * (blp[i] - mlp[i]) for i in range(len(VOCAB)))
        self.assertLessEqual(got, full)
        self.assertGreater(got, 0.5)
        self.assertIn("both endpoints list", s["kl_definition"])

    def test_a_server_that_refuses_logprobs(self):
        flip = {key_of(self.ps[2]): 3}
        b, bu, m, mu = self.pair(flip=flip, refuse=True)
        rc, rows, s, _ = self.run_kl(bu, mu)
        self.assertEqual(rc, 0)
        self.assertIn("refuses logprobs", s["kl_status"])
        self.assertTrue(all(r["kl_first"] is None and r["kl_matched"] == [] for r in rows))
        self.assertEqual(s["all"]["with_kl"], 0)
        self.assertIsNone(s["all"]["kl_first_mean"])
        by = {r["id"]: r for r in rows}
        self.assertEqual(by["p02"]["prefix"], 3)
        self.assertEqual(by["p00"]["agreement"], 1.0)
        self.assertTrue(any(p == "/tokenize" for p, _ in m.seen) and any(p == "/tokenize" for p, _ in b.seen))
        out = os.path.join(self.d, "out2")
        with self.assertRaises(SystemExit):
            A.main(["kl", "--base", bu, "--masked", mu, "--prompts", self.pf, "--out", out, "--positions", "4", "--require-kl"])


FAKE_RUN = r'''
import argparse, hashlib, json, os, sys
ap = argparse.ArgumentParser()
for k in ("--gate", "--tasks", "--manifest", "--out", "--split", "--arms", "--suites", "--deadline-utc", "--limit-tasks"):
    ap.add_argument(k)
ap.add_argument("--engine", action="append", default=[])
ap.add_argument("--draws", action="append", default=[])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
json.dump(sys.argv[1:], open(os.path.join(a.out, "argv.json"), "w"))
url = a.engine[0].split("=", 1)[1]
tasks = [json.loads(l) for l in open(a.tasks)]
with open(os.path.join(a.out, "replies.jsonl"), "w") as f:
    for t in tasks:
        for arm in a.arms.split(","):
            for d in (1, 2):
                h = int(hashlib.sha256(("%s|%s|%d" % (t["id"], arm, d)).encode()).hexdigest()[:8], 16)
                ok = h % 3 != 0
                if "masked" in url and h % 5 == 0:
                    ok = not ok
                status = "error" if (t["id"] == "c-3" and d == 2 and "masked" in url) else "ok"
                f.write(json.dumps({"task": t["id"], "arm": arm, "draw": d, "status": status, "score": {"correct": ok}}) + "\n")
'''

FAKE_ANALYZE = r'''
import argparse, json, os
ap = argparse.ArgumentParser()
for k in ("--gate", "--tasks", "--manifest", "--run", "--out"):
    ap.add_argument(k)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
json.dump({"run": a.run}, open(os.path.join(a.out, "result.json"), "w"))
'''

GATE = '''
[engines.pair]
url = "http://192.0.2.50:18080"
model = "dsv41"
[engines.strixite]
url = "from-the-window"
model = "qwen"
[[arms]]
name = "pair-off"
engine = "pair"
suites = ["REV-PAPER"]
[[arms]]
name = "pair-low"
engine = "pair"
suites = ["REV-PAPER", "CODE-CARD"]
[[arms]]
name = "strixite"
engine = "strixite"
suites = ["REV-PAPER"]
'''


class Gate(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.repo = os.path.join(self.d, "repo")
        os.makedirs(os.path.join(self.repo, "tools", "eval"))
        put(os.path.join(self.repo, "tools", "eval", "run.py"), FAKE_RUN)
        put(os.path.join(self.repo, "tools", "eval", "analyze.py"), FAKE_ANALYZE)
        self.gate = os.path.join(self.d, "gate.toml")
        put(self.gate, GATE)
        self.tasks = os.path.join(self.d, "tasks.jsonl")
        with open(self.tasks, "w") as f:
            for i in range(6):
                f.write(json.dumps({"id": "r-%d" % i, "suite": "REV-PAPER", "family": "rf-%d" % (i // 2)}) + "\n")
            for i in range(4):
                f.write(json.dumps({"id": "c-%d" % i, "suite": "CODE-CARD", "family": "cf-%d" % i}) + "\n")
        self.manifest = os.path.join(self.d, "manifest.json")
        put(self.manifest, "{}")
        self.mask = os.path.join(self.d, "mask.txt")
        put(self.mask, "3:17\n5:1,2,9\n39:383\n")

    def tearDown(self):
        shutil.rmtree(self.d)

    def argv(self, out, *extra):
        return ["gate", "--eval-repo", self.repo, "--gate", self.gate, "--tasks", self.tasks, "--manifest", self.manifest,
                "--engine", "pair", "--masked-url", "http://masked:18080", "--mask", self.mask, "--out", out, "--split", "dev"] + list(extra)

    def expected(self, arm, suite):
        pairs = []
        for i in range(6 if suite == "REV-PAPER" else 4):
            t = ("r-%d" if suite == "REV-PAPER" else "c-%d") % i
            for d in (1, 2):
                if t == "c-3" and d == 2:
                    continue           # the masked row errored: not scored, not paired
                h = int(hashlib.sha256(("%s|%s|%d" % (t, arm, d)).encode()).hexdigest()[:8], 16)
                ok = h % 3 != 0
                pairs.append((int(ok), int(ok != (h % 5 == 0))))
        return pairs

    def test_masked_and_unmasked_runs_pair_and_give_the_deltas(self):
        out = os.path.join(self.d, "abl")
        self.assertEqual(A.main(self.argv(out, "--base-url", "http://base:18080", "--analyze")), 0)
        mark = json.loads(get(os.path.join(out, "masked", "ablation.json")))
        self.assertTrue(mark["masked"])
        self.assertEqual((mark["engine"], mark["url"], mark["mask"]["id"]), ("pair", "http://masked:18080", "7672079e14c6b86a"))
        self.assertFalse(json.loads(get(os.path.join(out, "base", "ablation.json")))["masked"])
        mv = json.loads(get(os.path.join(out, "masked", "argv.json")))
        bv = json.loads(get(os.path.join(out, "base", "argv.json")))
        self.assertEqual(mv[mv.index("--engine") + 1], "pair=http://masked:18080")
        self.assertEqual(bv[bv.index("--engine") + 1], "pair=http://base:18080")
        self.assertEqual(mv[mv.index("--arms") + 1], "pair-off,pair-low")   # the strixite arm is not on the engine
        self.assertEqual(bv[bv.index("--arms") + 1], mv[mv.index("--arms") + 1])
        self.assertTrue(os.path.exists(os.path.join(out, "masked", "analysis", "result.json")))
        rep = json.loads(get(os.path.join(out, "deltas.json")))
        got = {(r["suite"], r["arm"]): r for r in rep["deltas"]}
        self.assertEqual(sorted(got), [("CODE-CARD", "pair-low"), ("CODE-CARD", "pair-off"), ("REV-PAPER", "pair-low"), ("REV-PAPER", "pair-off")])
        for (suite, arm), r in got.items():
            pairs = self.expected(arm, suite)
            self.assertEqual(r["pairs"], len(pairs))
            self.assertAlmostEqual(r["acc_unmasked"], sum(b for b, _ in pairs) / len(pairs))
            self.assertAlmostEqual(r["acc_masked"], sum(m for _, m in pairs) / len(pairs))
            self.assertAlmostEqual(r["delta"], r["acc_masked"] - r["acc_unmasked"])
            self.assertEqual(r["lost"], sum(1 for p in pairs if p == (1, 0)))
            self.assertEqual(r["gained"], sum(1 for p in pairs if p == (0, 1)))
            lo, hi = r["ci95"]
            self.assertTrue(lo <= r["delta"] <= hi, (suite, arm, r))
        self.assertEqual(got[("CODE-CARD", "pair-low")]["unpaired_unmasked"], 1)   # c-3 draw 2
        md = get(os.path.join(out, "deltas.md"))
        self.assertIn("mask=7672079e14c6b86a", md)
        self.assertIn("| REV-PAPER | pair-off |", md)
        again = os.path.join(self.d, "abl2")
        A.main(self.argv(again, "--base-url", "http://base:18080"))
        self.assertEqual(json.loads(get(os.path.join(again, "deltas.json")))["deltas"], rep["deltas"])

    def test_an_earlier_unmasked_run_is_reused(self):
        first = os.path.join(self.d, "abl")
        A.main(self.argv(first, "--base-url", "http://base:18080"))
        out = os.path.join(self.d, "reuse")
        self.assertEqual(A.main(self.argv(out, "--base-run", os.path.join(first, "base"), "--arms", "pair-low")), 0)
        self.assertFalse(os.path.exists(os.path.join(out, "base")))
        rep = json.loads(get(os.path.join(out, "deltas.json")))
        self.assertEqual({r["arm"] for r in rep["deltas"] if r["pairs"]}, {"pair-low"})

    def test_refusals_and_dry_run(self):
        empty = os.path.join(self.d, "empty.txt")
        put(empty, "# route_join mask: nothing under the threshold\n")
        argv = self.argv(os.path.join(self.d, "x"), "--base-url", "http://base:18080")
        with self.assertRaises(SystemExit):
            A.main(argv[:argv.index("--mask") + 1] + [empty] + argv[argv.index("--mask") + 2:])
        with self.assertRaises(SystemExit):
            A.main(self.argv(os.path.join(self.d, "y")))          # no unmasked side named
        bad = os.path.join(self.d, "bad.txt")
        put(bad, "40:1\n")
        self.assertEqual(A.main(argv[:argv.index("--mask") + 1] + [bad] + argv[argv.index("--mask") + 2:]), 3)
        out = os.path.join(self.d, "dry")
        self.assertEqual(A.main(self.argv(out, "--base-url", "http://base:18080", "--dry-run")), 0)
        self.assertFalse(os.path.exists(os.path.join(out, "masked", "replies.jsonl")))
        log = get(os.path.join(out, "ablate.log"))
        self.assertEqual(log.count("tools/eval/run.py"), 2)
        self.assertFalse(os.path.exists(os.path.join(out, "deltas.json")))


if __name__ == "__main__":
    unittest.main()
