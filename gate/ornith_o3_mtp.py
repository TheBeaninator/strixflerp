#!/usr/bin/env python3
"""ORNITH-O3 step one (criteria: gate/ornith-o3-mtp.toml): how often the checkpoint's own MTP head drafts the right next
token. Proxy on control-node's GPU with Qwen3.6-35B-A3B (same architecture, blk.40 MTP block); the 4-node Ornith probe
is prepared in gate/ornith_p5.py (--mtp-probe) and needs owner's go. Run on control-node:

    python3 gate/ornith_o3_mtp.py prompts      # CPU: the 8 chat-rendered prompts -> ids files (the binary's tokenizer)
    python3 gate/ornith_o3_mtp.py off          # GPU: the binary without the flag (the engine's own Generate)
    python3 gate/ornith_o3_mtp.py on           # GPU: the same with --mtp-probe 3 and the unit-check dump of prompt 0
    python3 gate/ornith_o3_mtp.py staged       # GPU: --staged tcp world 2 on this host, the probe on rank 1 (recorded)
    python3 gate/ornith_o3_mtp.py llama        # GPU: /opt/llama llama-server --spec-type draft-mtp, n_max 1 and 3
    python3 gate/ornith_o3_mtp.py ref          # CPU: the numpy reference of the dumped steps
    python3 gate/ornith_o3_mtp.py judge        # CPU: criteria -> summary.json, summary.md

The GPU steps are run by gate/ornith_o3_mtp_run.sh through tools/gpu-queue.sh. Evidence: gate/results/ornith-o3-mtp/.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "gate", "ornith-o3-mtp.toml")
RES = os.path.join(ROOT, "gate", "results", "ornith-o3-mtp")
WORK = os.path.join(ROOT, "build", "o3-mtp-run")
GEN = os.path.join(ROOT, "build", "o3-mtp", "out", "flerp-qwen35moe-generate")
ROCM = os.environ.get("FLERP_ROCM_ROOT", "/data/tools/therock-10.1.0a20260823") + "/lib"
GGUF_PY = "/data/ark/src/github.com__ggml-org__llama.cpp/gguf-py"
POLITE = ["ionice", "-c3", "nice", "-n", "19"]


def cfg():
    with open(CFG, "rb") as f:
        return tomllib.load(f)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def prompts(c):
    p = c["proxy"]
    return [(f"code{i}", "code", t) for i, t in enumerate(p["code"])] + [(f"prose{i}", "prose", t) for i, t in enumerate(p["prose"])]


def render(text):
    # Qwen3.6 chat template, one user turn, thinking off (the template's enable_thinking=false form)
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def env():
    e = dict(os.environ)
    e["LD_LIBRARY_PATH"] = ROCM + (":" + e["LD_LIBRARY_PATH"] if e.get("LD_LIBRARY_PATH") else "")
    return e


def events(path):
    out = []
    for line in open(path):
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def run(label, argv, timeout=7200):
    os.makedirs(RES, exist_ok=True)
    out = os.path.join(RES, f"{label}.jsonl")
    t0 = time.time()
    with open(out, "w") as fo, open(os.path.join(WORK, f"{label}.err"), "w") as fe:
        p = subprocess.run(POLITE + argv, stdout=fo, stderr=fe, timeout=timeout, env=env())
    print(f"{utc()} {label}: rc {p.returncode} in {time.time() - t0:.0f} s -> {out}", flush=True)
    return p.returncode


def id_args():
    names = json.load(open(os.path.join(WORK, "prompts.json")))["names"]
    a = []
    for n in names:
        a += ["--ids-file", os.path.join(WORK, f"{n}.ids")]
    return a


# --- steps --------------------------------------------------------------------------------------------------------
def cmd_prompts(c, a):
    os.makedirs(WORK, exist_ok=True)
    ps = prompts(c)
    argv = [GEN, "--model", c["proxy"]["model"], "--tokenize"]
    for name, _, text in ps:
        f = os.path.join(WORK, f"{name}.txt")
        open(f, "w").write(render(text))
        argv += ["--prompt-file", f]
    p = subprocess.run(argv, capture_output=True, text=True, env=env())
    if p.returncode != 0:
        sys.exit(f"tokenize: rc {p.returncode}: {p.stderr[-400:]}")
    ev = [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{") and '"tokenized"' in x]
    if len(ev) != len(ps):
        sys.exit(f"tokenize: {len(ev)} events for {len(ps)} prompts")
    doc = {"names": [n for n, _, _ in ps], "classes": {n: k for n, k, _ in ps}, "ids": {}}
    for (name, _, _), e in zip(ps, ev):
        open(os.path.join(WORK, f"{name}.ids"), "w").write(" ".join(str(i) for i in e["prompt_ids"]))
        doc["ids"][name] = e["prompt_ids"]
    os.makedirs(RES, exist_ok=True)
    json.dump(doc, open(os.path.join(WORK, "prompts.json"), "w"), indent=1)
    json.dump(doc, open(os.path.join(RES, "prompts.json"), "w"), indent=1)
    print({n: len(v) for n, v in doc["ids"].items()})


def cmd_off(c, a):
    p = c["proxy"]
    return run("off", [GEN, "--model", p["model"], "--context", str(p["context"]), "--max-tokens", str(p["max_tokens"])] + id_args())


def cmd_on(c, a):
    p = c["proxy"]
    return run("on", [GEN, "--model", p["model"], "--context", str(p["context"]), "--max-tokens", str(p["max_tokens"]),
                      "--mtp-probe", str(p["depth"]), "--mtp-dump", os.path.join(WORK, "dump.bin"), "--mtp-dump-frontiers", "8"] + id_args())


def cmd_staged(c, a):
    """--staged tcp world 2 on this host (loads through the placed loader from the NVMe file), the probe on rank 1:
    the hook the 4-node plan uses. Recorded: ids and drafts against the `on` run."""
    p = c["proxy"]
    procs = []
    for r in (1, 0):
        argv = POLITE + [GEN, "--staged", "tcp", "--model", p["model"], "--context", str(p["context"]), "--max-tokens", str(p["max_tokens"]),
                         "--world", "2", "--rank", str(r), "--cuts", "even", "--port", "39950", "--host", "127.0.0.1", "--epoch", "71",
                         "--graph-modes", "on", "--single-reader", "--mtp-probe", str(p["depth"])] + id_args()
        out = os.path.join(RES, f"staged-r{r}.jsonl")
        procs.append(subprocess.Popen(argv, stdout=open(out, "w"), stderr=open(os.path.join(WORK, f"staged-r{r}.err"), "w"), env=env()))
        time.sleep(5)
    rcs = [pr.wait(timeout=7200) for pr in procs]
    print(f"{utc()} staged: rc {rcs}")
    return max(rcs)


def cmd_llama(c, a):
    """llama.cpp's own MTP drafting on the same 35B and prompts: draft_n / draft_n_accepted per request."""
    p = c["proxy"]
    names = json.load(open(os.path.join(WORK, "prompts.json")))
    doc = {"when": utc(), "binary": "/opt/llama/llama-server", "runs": {}}
    try:
        doc["version"] = subprocess.run(["/opt/llama/llama-server", "--version"], capture_output=True, text=True,
                                        env=dict(os.environ, LD_LIBRARY_PATH="/opt/llama")).stderr.strip().splitlines()[-2:]
    except Exception as e:  # noqa: BLE001
        doc["version"] = str(e)
    rc = 0
    for n_max in (1, 3):
        port = 39960 + n_max
        log = os.path.join(WORK, f"llama-n{n_max}.log")
        argv = POLITE + ["/opt/llama/llama-server", "-m", p["model"], "-ngl", "99", "-c", "4096", "-np", "1", "--port", str(port),
                         "--host", "127.0.0.1", "--spec-type", "draft-mtp", "--spec-draft-n-max", str(n_max), "--spec-draft-p-min", "0",
                         "--spec-draft-n-min", "0", "--no-warmup"]
        srv = subprocess.Popen(argv, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=dict(os.environ, LD_LIBRARY_PATH="/opt/llama"))
        runs = {}
        try:
            for _ in range(600):
                try:
                    if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200:
                        break
                except Exception:  # noqa: BLE001
                    pass
                if srv.poll() is not None:
                    break
                time.sleep(2)
            for name in names["names"]:
                body = {"prompt": names["ids"][name], "n_predict": p["max_tokens"], "temperature": 0.0, "top_k": 1, "cache_prompt": False,
                        "return_tokens": True, "seed": 1}
                req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=1800))
                runs[name] = {"timings": r.get("timings"), "tokens": r.get("tokens"), "stop_type": r.get("stop_type"),
                              "content_head": (r.get("content") or "")[:200]}
                t = r.get("timings") or {}
                print(f"{utc()} llama n_max {n_max} {name}: draft_n {t.get('draft_n')} accepted {t.get('draft_n_accepted')}", flush=True)
        except Exception as e:  # noqa: BLE001
            runs["error"] = repr(e)
            rc = 1
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=60)
            except subprocess.TimeoutExpired:
                srv.kill()
        doc["runs"][str(n_max)] = runs
        shutil.copy(log, os.path.join(RES, f"llama-n{n_max}.log"))
    json.dump(doc, open(os.path.join(RES, "llama.json"), "w"), indent=1)
    return rc


# --- numpy reference ----------------------------------------------------------------------------------------------
def read_dump(path):
    import numpy as np
    b = open(path, "rb").read()
    assert b[:4] == b"MTPD", "not an MTP dump"
    H, top, block = np.frombuffer(b[4:16], dtype=np.int32)
    rec = 5 * 4 + 2 * H * 4 + 2 * top * 4
    out = []
    for o in range(16, len(b), rec):
        r = b[o:o + rec]
        k, tok, pos, lg, am = np.frombuffer(r[:20], dtype=np.int32)
        h_in = np.frombuffer(r[20:20 + 4 * H], dtype=np.float32)
        h_out = np.frombuffer(r[20 + 4 * H:20 + 8 * H], dtype=np.float32)
        q = 20 + 8 * H
        out.append({"kind": int(k), "token": int(tok), "pos": int(pos), "logits": bool(lg), "argmax": int(am), "h_in": h_in, "h_out": h_out,
                    "top_ids": np.frombuffer(r[q:q + 4 * top], dtype=np.int32), "top_vals": np.frombuffer(r[q + 4 * top:q + 8 * top], dtype=np.float32)})
    return int(H), int(block), out


class Ref:
    """llama.cpp's graph_mtp for qwen35moe (gate/ornith-o3-mtp.toml [reference]) in float64, weights from gguf-py."""

    def __init__(self, path, block):
        sys.path.insert(0, GGUF_PY)
        import numpy as np
        import gguf
        self.np, self.gguf = np, gguf
        self.r = gguf.GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors}
        arch = "qwen35moe"
        f = self.r.fields
        val = lambda k: f[k].parts[f[k].data[0]][0]  # noqa: E731
        self.eps = float(val(f"{arch}.attention.layer_norm_rms_epsilon"))
        self.heads = int(val(f"{arch}.attention.head_count"))
        self.kv_heads = int(val(f"{arch}.attention.head_count_kv"))
        self.head_dim = int(val(f"{arch}.attention.key_length"))
        self.n_rot = int(val(f"{arch}.rope.dimension_count"))
        self.theta = float(val(f"{arch}.rope.freq_base"))
        self.used = int(val(f"{arch}.expert_used_count"))
        self.p = f"blk.{block}."
        self.cache = {}
        self.K, self.V = {}, {}

    def w(self, name, rows=None):
        """Dequantized float64 [out, in] (rows: a slice of the first axis, for the expert stacks)."""
        np, gguf = self.np, self.gguf
        key = (name, rows)
        if key in self.cache:
            return self.cache[key]
        t = self.t[name]
        data = t.data if rows is None else t.data[rows]
        x = gguf.quants.dequantize(np.ascontiguousarray(data), t.tensor_type).astype(np.float64)
        if name.endswith("output.weight") or name.endswith("token_embd.weight") or rows is not None or x.size < 64 << 20:
            self.cache[key] = x
        return x

    def rms(self, x, w):
        np = self.np
        return x / np.sqrt(np.mean(x * x) + self.eps) * w

    def rope(self, x, pos):
        # NEOX style on the first n_rot dims of each head (rope_multi with every section at the same text position)
        np = self.np
        half = self.n_rot // 2
        inv = self.theta ** (-np.arange(half, dtype=np.float64) * 2.0 / self.n_rot)
        ang = pos * inv
        c, s = np.cos(ang), np.sin(ang)
        y = x.copy()
        a, b = x[:, :half], x[:, half:self.n_rot]
        y[:, :half] = a * c - b * s
        y[:, half:self.n_rot] = b * c + a * s
        return y

    def embed(self, tok):
        np, gguf = self.np, self.gguf
        name = self.p + "nextn.embed_tokens.weight" if (self.p + "nextn.embed_tokens.weight") in self.t else "token_embd.weight"
        t = self.t[name]
        return gguf.quants.dequantize(np.ascontiguousarray(t.data[tok:tok + 1]), t.tensor_type).astype(np.float64)[0]

    def step(self, tok, h, pos, logits):
        np, p = self.np, self.p
        H = h.shape[0]
        x = np.concatenate([self.rms(self.embed(tok), self.w(p + "nextn.enorm.weight")), self.rms(h, self.w(p + "nextn.hnorm.weight"))])
        y = self.w(p + "nextn.eh_proj.weight") @ x
        # attention
        n = self.rms(y, self.w(p + "attn_norm.weight"))
        D = self.head_dim
        qg = (self.w(p + "attn_q.weight") @ n).reshape(self.heads, 2 * D)
        q, gate = qg[:, :D], qg[:, D:]
        k = (self.w(p + "attn_k.weight") @ n).reshape(self.kv_heads, D)
        v = (self.w(p + "attn_v.weight") @ n).reshape(self.kv_heads, D)
        q = np.stack([self.rms(q[i], self.w(p + "attn_q_norm.weight")) for i in range(self.heads)])
        k = np.stack([self.rms(k[i], self.w(p + "attn_k_norm.weight")) for i in range(self.kv_heads)])
        q, k = self.rope(q, pos), self.rope(k, pos)
        self.K[pos], self.V[pos] = k, v
        ps = list(range(pos + 1))
        Ks = np.stack([self.K[i] for i in ps])   # [T, kv, D]
        Vs = np.stack([self.V[i] for i in ps])
        g = self.heads // self.kv_heads
        ctx = np.zeros((self.heads, D))
        for hh in range(self.heads):
            kv = hh // g
            s = Ks[:, kv, :] @ q[hh] / math.sqrt(D)
            s = np.exp(s - s.max())
            s /= s.sum()
            ctx[hh] = s @ Vs[:, kv, :]
        ctx = ctx * (1.0 / (1.0 + np.exp(-gate)))
        y = y + self.w(p + "attn_output.weight") @ ctx.reshape(-1)
        # mixture of experts
        n = self.rms(y, self.w(p + "post_attention_norm.weight"))
        lg = self.w(p + "ffn_gate_inp.weight") @ n
        pr = np.exp(lg - lg.max())
        pr /= pr.sum()
        ids = np.argsort(-pr, kind="stable")[:self.used]
        wts = pr[ids] / max(pr[ids].sum(), 6.103515625e-5)
        silu = lambda z: z / (1.0 + np.exp(-z))  # noqa: E731
        out = np.zeros(H)
        for e, we in zip(ids, wts):
            ge = self.w(p + "ffn_gate_exps.weight", int(e))
            ue = self.w(p + "ffn_up_exps.weight", int(e))
            de = self.w(p + "ffn_down_exps.weight", int(e))
            out += we * (de @ (silu(ge @ n) * (ue @ n)))
        if (p + "ffn_up_shexp.weight") in self.t:
            sg = 1.0 / (1.0 + np.exp(-(self.w(p + "ffn_gate_inp_shexp.weight").reshape(-1) @ n)))
            out += sg * (self.w(p + "ffn_down_shexp.weight") @ (silu(self.w(p + "ffn_gate_shexp.weight") @ n) * (self.w(p + "ffn_up_shexp.weight") @ n)))
        y = y + out
        hn = p + "nextn.shared_head_norm.weight" if (p + "nextn.shared_head_norm.weight") in self.t else "output_norm.weight"
        h_out = self.rms(y, self.w(hn))
        res = {"h_out": h_out, "experts": [int(e) for e in ids]}
        if logits:
            head = p + "nextn.shared_head_head.weight" if (p + "nextn.shared_head_head.weight") in self.t else "output.weight"
            W = self.w(head)
            res["logits"] = W @ h_out
        return res


def cmd_ref(c, a):
    import numpy as np
    H, block, steps = read_dump(os.path.join(WORK, "dump.bin"))
    ref = Ref(c["proxy"]["model"], block)
    rows = []
    t0 = time.time()
    for i, s in enumerate(steps):
        r = ref.step(s["token"], s["h_in"].astype(np.float64), s["pos"], s["logits"])
        a_, b_ = r["h_out"], s["h_out"].astype(np.float64)
        cos = float(a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_)))
        row = {"i": i, "kind": s["kind"], "token": s["token"], "pos": s["pos"], "cos": cos,
               "rel_l2": float(np.linalg.norm(a_ - b_) / np.linalg.norm(a_)), "experts": r["experts"]}
        if s["logits"]:
            lg = r["logits"]
            o = np.argsort(-lg, kind="stable")[:2]
            row.update({"ref_argmax": int(o[0]), "ref_margin": float(lg[o[0]] - lg[o[1]]), "probe_argmax": s["argmax"],
                        "probe_top_val": float(s["top_vals"][0]), "ref_top_val": float(lg[o[0]]),
                        "probe_argmax_ref_logit": float(lg[s["argmax"]])})
        rows.append(row)
        if i % 10 == 0:
            print(f"{utc()} ref step {i}/{len(steps)} cos {cos:.6f} ({time.time() - t0:.0f} s)", flush=True)
    json.dump({"when": utc(), "block": block, "hidden": H, "steps": rows, "eps": ref.eps, "n_rot": ref.n_rot, "theta": ref.theta},
              open(os.path.join(RES, "ref.json"), "w"), indent=1)


# --- judge --------------------------------------------------------------------------------------------------------
def tally(ids, drafts, depth):
    """Per depth k: n frontiers whose x_{f+k} is known; marginal hits; chain hits (d_1..d_k all right)."""
    out = []
    for k in range(1, depth + 1):
        n = marg = chain = 0
        for j, d in enumerate(drafts):
            if j + k >= len(ids):
                break
            n += 1
            marg += d[k - 1] == ids[j + k]
            chain += all(d[i - 1] == ids[j + i] for i in range(1, k + 1))
        out.append({"k": k, "n": n, "marginal": marg, "chain": chain})
    return out


def schedule(ids, drafts, n_max):
    """llama.cpp's rounds: at frontier j draft n_max, accept the matching prefix L, next round at j + L + 1."""
    j = drafted = accepted = rounds = 0
    while j + 1 < len(ids) and j < len(drafts):
        m = min(n_max, len(ids) - 1 - j)
        L = 0
        while L < m and drafts[j][L] == ids[j + 1 + L]:
            L += 1
        drafted += m
        accepted += L
        rounds += 1
        j += L + 1
    return {"drafted": drafted, "accepted": accepted, "rounds": rounds}


def cmd_judge(c, a):
    doc = {"when": utc(), "criteria": {}, "recorded": {}}
    why = []
    pj = json.load(open(os.path.join(RES, "prompts.json")))
    names, classes = pj["names"], pj["classes"]
    depth = c["proxy"]["depth"]
    off = [e for e in events(os.path.join(RES, "off.jsonl")) if e.get("event") == "generated"]
    on_ev = events(os.path.join(RES, "on.jsonl"))
    on = [e for e in on_ev if e.get("event") == "generated"]
    probe = [e for e in on_ev if e.get("event") == "mtp_probe"]
    head = next((e for e in on_ev if e.get("event") == "mtp_head"), None)
    # 1. ids unchanged
    eq = [i < len(off) and i < len(on) and off[i]["ids"] == on[i]["ids"] for i in range(len(names))]
    eq_probe = [i < len(probe) and i < len(on) and probe[i]["ids"] == on[i]["ids"] for i in range(len(names))]
    c1 = all(eq) and len(off) == len(names) == len(on)
    doc["criteria"]["1_ids_unchanged"] = {"pass": c1, "per_prompt": dict(zip(names, eq)), "probe_saw_the_ids": dict(zip(names, eq_probe)),
                                          "tokens": {n: len(on[i]["ids"]) if i < len(on) else None for i, n in enumerate(names)}}
    if not c1:
        why.append("1: probe-on ids differ from the plain run's")
    # 2. unit check
    rp = os.path.join(RES, "ref.json")
    if os.path.exists(rp):
        ref = json.load(open(rp))["steps"]
        lg = [r for r in ref if "ref_argmax" in r]
        confident = [r for r in lg if r["ref_margin"] >= 0.5]
        agree = sum(r["ref_argmax"] == r["probe_argmax"] for r in lg)
        a_ok = all(r["cos"] >= 0.999 for r in ref)
        b_ok = all(r["ref_argmax"] == r["probe_argmax"] for r in confident) and lg and agree >= 0.9 * len(lg)
        c2 = bool(a_ok and b_ok and len(ref) > 0)
        doc["criteria"]["2_unit_check"] = {"pass": c2, "steps": len(ref), "min_cos": min(r["cos"] for r in ref), "max_rel_l2": max(r["rel_l2"] for r in ref),
                                           "logits_steps": len(lg), "argmax_agree": agree, "confident_steps": len(confident),
                                           "confident_agree": sum(r["ref_argmax"] == r["probe_argmax"] for r in confident),
                                           "kinds": {k: sum(r["kind"] == k for r in ref) for k in (0, 1, 2)}}
        if not c2:
            why.append("2: unit check against the numpy reference failed")
    else:
        doc["criteria"]["2_unit_check"] = {"pass": False, "missing": "ref.json"}
        why.append("2: no reference run")
    # recorded: acceptance per depth and class
    acc = {"code": [], "prose": [], "all": []}
    per = {}
    for e in probe:
        n = names[e["prompt"]]
        t = tally(e["ids"], e["drafts"], depth)
        per[n] = {"tally": t, "s1": schedule(e["ids"], e["drafts"], 1), "s3": schedule(e["ids"], e["drafts"], 3)}
        for cls in (classes[n], "all"):
            acc[cls].append(per[n])
    table = {}
    for cls, lst in acc.items():
        rows = []
        for k in range(depth):
            n = sum(x["tally"][k]["n"] for x in lst)
            m = sum(x["tally"][k]["marginal"] for x in lst)
            ch = sum(x["tally"][k]["chain"] for x in lst)
            prev = sum(x["tally"][k - 1]["chain"] for x in lst) if k else n
            prevn = sum(x["tally"][k - 1]["n"] for x in lst) if k else n
            rows.append({"k": k + 1, "n": n, "marginal": m / n if n else None, "chain": ch / n if n else None,
                         "conditional": (ch / n) / (prev / prevn) if n and prev and prevn else None})
        s1 = {q: sum(x["s1"][q] for x in lst) for q in ("drafted", "accepted", "rounds")}
        s3 = {q: sum(x["s3"][q] for x in lst) for q in ("drafted", "accepted", "rounds")}
        table[cls] = {"depth": rows, "expected_tokens_per_round": [1 + sum(r["chain"] or 0 for r in rows[:k]) for k in range(1, depth + 1)],
                      "sched_n1": s1, "sched_n3": s3}
    doc["recorded"]["acceptance"] = table
    doc["recorded"]["per_prompt"] = per
    doc["recorded"]["mtp_head"] = head
    doc["recorded"]["mtp_ms_per_step"] = {names[e["prompt"]]: e["mtp_ms_per_step"] for e in probe}
    # 3. llama.cpp
    lp = os.path.join(RES, "llama.json")
    if os.path.exists(lp):
        L = json.load(open(lp))
        out = {}
        for nm in ("1", "3"):
            runs = L["runs"].get(nm, {})
            dn = sum((r.get("timings") or {}).get("draft_n", 0) or 0 for k, r in runs.items() if k != "error")
            da = sum((r.get("timings") or {}).get("draft_n_accepted", 0) or 0 for k, r in runs.items() if k != "error")
            out[nm] = {"draft_n": dn, "draft_n_accepted": da, "rate": da / dn if dn else None, "error": runs.get("error")}
        mine1 = table["all"]["sched_n1"]
        mine_rate = mine1["accepted"] / mine1["drafted"] if mine1["drafted"] else None
        mine3 = table["all"]["sched_n3"]
        c3 = out["1"]["rate"] is not None and mine_rate is not None and abs(out["1"]["rate"] - mine_rate) <= 0.05
        doc["criteria"]["3_llama_cpp"] = {"pass": c3, "llama": out, "probe_sched_n1": mine_rate,
                                          "probe_sched_n3": mine3["accepted"] / mine3["drafted"] if mine3["drafted"] else None}
        if not c3:
            why.append("3: llama.cpp draft-mtp n_max 1 acceptance differs by more than 0.05")
    else:
        doc["criteria"]["3_llama_cpp"] = {"pass": False, "missing": "llama.json"}
        why.append("3: no llama.cpp run")
    # recorded: the staged hook (world 2 on this host)
    sp = os.path.join(RES, "staged-r1.jsonl")
    if os.path.exists(sp):
        sev = [e for e in events(sp) if e.get("event") == "mtp_probe"]
        r0 = [e for e in events(os.path.join(RES, "staged-r0.jsonl")) if e.get("event") == "staged"]
        st = {}
        for e in sev:
            n = names[e["prompt"]]
            i = e["prompt"]
            r0ids = r0[i]["ids"] if i < len(r0) else None
            same_ids = r0ids is not None and e["ids"][:len(r0ids)] == r0ids
            ref_d = probe[i]["drafts"] if i < len(probe) else []
            agree = sum(x == y for x, y in zip(e["drafts"], ref_d))
            st[n] = {"probe_ids_prefix_equals_rank0": same_ids, "rank0_equals_on_run": r0ids == on[i]["ids"] if r0ids is not None and i < len(on) else None,
                     "drafts_equal_on_run": f"{agree}/{min(len(e['drafts']), len(ref_d))}", "prompt_rows_seen": e["prompt_rows_seen"],
                     "prompt_tokens": e["prompt_tokens"]}
        doc["recorded"]["staged_w2"] = st
    doc["pass"] = not why
    doc["why"] = why
    json.dump(doc, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    write_md(doc, names)
    print(json.dumps({"pass": doc["pass"], "why": why}, indent=1))


def write_md(doc, names):
    f = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))  # noqa: E731
    L = ["# ORNITH-O3 step one: MTP draft acceptance, 35B proxy", "",
         f"Written {doc['when']} on control-node by gate/ornith_o3_mtp.py judge. Criteria: gate/ornith-o3-mtp.toml (committed before any GPU run). "
         f"**{'PASS' if doc['pass'] else 'FAIL'}**" + ("" if doc["pass"] else ": " + "; ".join(doc["why"])), "",
         "## Criteria", ""]
    for k, v in doc["criteria"].items():
        L.append(f"- **{k}**: {'pass' if v.get('pass') else 'FAIL'} - " + json.dumps({x: y for x, y in v.items() if x not in ('pass', 'per_prompt', 'probe_saw_the_ids', 'tokens')}))
    L += ["", "## Acceptance (Qwen3.6-35B-A3B blk.40, greedy, 256 tokens, chat, thinking off)", "",
          "| class | k | frontiers | marginal P(d_k right) | chain P(d_1..d_k right) | conditional a_k | E tokens/round at k |", "|---|---|---|---|---|---|---|"]
    for cls, t in doc["recorded"]["acceptance"].items():
        for r, e in zip(t["depth"], t["expected_tokens_per_round"]):
            L.append(f"| {cls} | {r['k']} | {r['n']} | {f(r['marginal'])} | {f(r['chain'])} | {f(r['conditional'])} | {f(e)} |")
    L += ["", "llama.cpp-schedule simulation of the probe (rounds as draft-mtp runs them): "
          + ", ".join(f"{cls}: n_max 1 {f(t['sched_n1']['accepted'] / t['sched_n1']['drafted'] if t['sched_n1']['drafted'] else None)}, "
                      f"n_max 3 {f(t['sched_n3']['accepted'] / t['sched_n3']['drafted'] if t['sched_n3']['drafted'] else None)}"
                      for cls, t in doc["recorded"]["acceptance"].items()), ""]
    if "staged_w2" in doc["recorded"]:
        L += ["## Staged hook (--staged tcp world 2 on control-node, probe on rank 1)", "", "```", json.dumps(doc["recorded"]["staged_w2"], indent=1), "```", ""]
    L += ["MTP ms a draft step (35B, recorded): " + json.dumps(doc["recorded"]["mtp_ms_per_step"]), ""]
    open(os.path.join(RES, "summary.md"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prompts", "off", "on", "staged", "llama", "ref", "judge"])
    a = ap.parse_args()
    c = cfg()
    os.makedirs(WORK, exist_ok=True)
    rc = globals()["cmd_" + a.cmd](c, a)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
