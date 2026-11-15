#!/usr/bin/env python3
"""Recipe gate for a qwen35moe recipe: gufo's Qwen 3.5 executor with executor.flerp.qwen35moe laid over it
(SPEC.md section 9, phase 5 step 2). No origin runs such a file with this executor, so the reference is another
engine: llama.cpp on the CPU, on the same prompt ids. Run on the gfx1151 build host:

    ./flerp gate recipe qwen36-35b-a3b.flerp-moe [--stages check,origin,build,cpu,llama,gpu] [--keep-going]

Stages, each writing evidence under gate/results/<recipe>/:
  check    `flerp check` has no errors, for the origin and for the whole catalogue.
  origin   every vendored file of the origin is byte-identical to `git archive <commit>` of the reference checkout
           (the adapted files are a module of their own and are diffed against the origin's here, as a record).
  build    `flerp assemble`, every origin file of the tree is the export's file, the tree builds with CMake.
  cpu      no GPU: the prompts' token ids from the engine's tokenizer on the MoE file, the sizes it read from the
           header, and the executor's CPU reference for a few tokens of the first prompt.
  llama    no GPU: llama.cpp (CPU) on the MoE file and on the dense file - greedy ids and its top log-probabilities
           at every position - and its eval callback on the trace prompts (hidden states, router choices).
  gpu      the dense model through the new binary against the dense recipe's recorded ids; the MoE model `repeats`
           times and once traced; then the comparison with llama.cpp and the verdict by [criteria] of the config.

The GPU is taken only when it is idle and no job of the pod waits for it (gate/strixite_recipe.py, Hold), and only
for the gpu stage. Builds and CPU runs are niced and raise their own OOM score.
"""
import argparse
import difflib
import json
import math
import os
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gate"))
from flerplib.catalog import Catalog  # noqa: E402
import gufo_recipe as G  # noqa: E402
from strixite_recipe import Hold, load_config, sh, sha256, tail, utc  # noqa: E402

STAGES = ("check", "origin", "build", "cpu", "llama", "gpu")
# Asked for by name (`--stages reference,fidelity`); not part of the recipe's verdict above, which stands as set.
FIDELITY_STAGES = ("reference", "fidelity")
ADAPTED_ROOT = "flerp/qwen35moe/"          # tree path of the adapted files; the rest of the path is the origin's


# --- comparisons (no I/O: tests/test_moe_gate.py) ---------------------------------------------------------------
def corner_rows(n):
    """The rows llama.cpp's eval callback prints of n: the first three and the last three."""
    return [r for r in range(n) if r < 3 or r + 3 >= n]


def parse_eval_callback(text, wanted):
    """Tensors of llama.cpp's common_debug_cb_eval output whose name matches the regex `wanted`:
    {name: {"type", "dims": [4 ints], "rows": [[floats]], "sum"}}. Rows are the printed inner rows, in order."""
    out, cur = {}, None
    head = re.compile(r"common_debug_cb_eval:\s+(\S+) = \((\w+)\)\s.*= \{(\d+), (\d+), (\d+), (\d+)\}\s*$")
    num = re.compile(r"-?(?:\d+\.\d+|inf|nan)")
    for line in text.splitlines():
        m = head.search(line)
        if m:
            cur = None
            if re.fullmatch(wanted, m.group(1)):
                cur = {"type": m.group(2), "dims": [int(m.group(i)) for i in range(3, 7)], "rows": [], "sum": None}
                out[m.group(1)] = cur
            continue
        if cur is None:
            continue
        s = line.strip()
        if s.startswith("sum = "):
            try:
                cur["sum"] = float(s[6:])
            except ValueError:
                pass
            cur = None
        elif s.startswith("[") and s.endswith("],") and len(s) > 3:
            cur["rows"].append([float(x) for x in num.findall(s)])
    return out


def callback_input_ids(text):
    """The token ids llama-eval-callback says it evaluates: one a line after "number of input tokens = N", each
    line with the logger's prefix ("0.00.840.192 I   14162")."""
    m = re.search(r"number of input tokens = (\d+)\s*\n", text)
    if not m:
        return []
    out = []
    for line in text[m.end():].splitlines()[:int(m.group(1))]:
        x = re.search(r"(?:^|\sI)\s+(\d+)\s*$", line)
        if not x:
            break
        out.append(int(x.group(1)))
    return out


def ends(values, n=3):
    """What the callback shows of a row: all of it up to 2n values, else the first n and the last n."""
    return list(values) if len(values) <= 2 * n else list(values[:n]) + list(values[-n:])


def compare_logprobs(mine_steps, llama_probs, first_diff):
    """Log-probabilities of this side against llama.cpp's top tokens, position by position, up to and including
    the first position where the generated ids differ (the inputs are the same up to there)."""
    rows = []
    last = min(len(mine_steps), len(llama_probs)) if first_diff is None else first_diff + 1
    for i in range(min(last, len(mine_steps), len(llama_probs))):
        mine = {int(t): v - mine_steps[i]["logsumexp"] for t, v in mine_steps[i]["top"]}
        top = llama_probs[i].get("top_logprobs") or []
        diffs = [abs(mine[t["id"]] - t["logprob"]) for t in top if t["id"] in mine]
        margin = top[0]["logprob"] - top[1]["logprob"] if len(top) > 1 else None
        rows.append({"position": i, "llama_token": llama_probs[i].get("id"), "token": mine_steps[i]["token"],
                     "llama_top": len(top), "compared": len(diffs),
                     "max_abs": round(max(diffs), 5) if diffs else None,
                     "mean_abs": round(sum(diffs) / len(diffs), 5) if diffs else None,
                     "llama_margin": None if margin is None else round(margin, 5),
                     "llama_top2": [t["id"] for t in top[:2]],
                     "llama_top1_logprob": round(top[0]["logprob"], 5) if top else None,
                     "this_logprob_of_llama_top1": round(mine[top[0]["id"]], 5) if top and top[0]["id"] in mine else None})
    return rows


def as_steps(llama_probs):
    """llama.cpp's per-position top log-probabilities in the shape of this side's steps."""
    return [{"token": p.get("id"), "logsumexp": 0.0, "top": [[t["id"], t["logprob"]] for t in p.get("top_logprobs", [])]}
            for p in llama_probs]


def pair_difference(a_steps, b_steps, n_top=20):
    """Two step lists (this side's shape) position by position while their tokens agree: over b's n_top best tokens
    that a kept, the absolute differences of log-probabilities. Returns {"positions", "max_abs", "mean_abs", ...}."""
    per, top1 = [], []
    for a, b in zip(a_steps, b_steps):
        la = {int(t): v - a["logsumexp"] for t, v in a["top"]}
        lb = [(int(t), v - b["logsumexp"]) for t, v in b["top"][:n_top]]
        d = [abs(la[t] - v) for t, v in lb if t in la]
        if d:
            per.append({"compared": len(d), "max_abs": max(d), "mean_abs": sum(d) / len(d)})
            if lb[0][0] in la:
                top1.append(abs(la[lb[0][0]] - lb[0][1]))
        if a["token"] != b["token"]:
            break
    if not per:
        return {"positions": 0}
    return {"positions": len(per), "max_abs": round(max(p["max_abs"] for p in per), 5),
            "mean_abs": round(sum(p["mean_abs"] for p in per) / len(per), 5),
            "top1_max_abs": round(max(top1), 5) if top1 else None,
            "first_position": {k: round(v, 5) if isinstance(v, float) else v for k, v in per[0].items()}}


def judge_prompt(ids, llama_ids, rows, crit):
    """[criteria] 3a, 3b, 3c for one prompt. Returns {"pass", "first_difference", ..., "why": [failures]}."""
    why = []
    d = G.first_difference(ids, llama_ids)
    out = {"tokens": [len(ids), len(llama_ids)], "first_difference": d,
           "agree": min(len(ids), len(llama_ids)) if d is None else d}
    if not rows:
        return dict(out, **{"pass": False, "why": ["no log-probabilities to compare"]})
    r0 = rows[0]
    out["first_position"] = {k: r0[k] for k in ("max_abs", "mean_abs", "compared", "llama_margin")}
    if r0["max_abs"] is None or r0["compared"] < min(10, r0["llama_top"]):
        why.append(f"first position: only {r0['compared']} of llama.cpp's {r0['llama_top']} top tokens are in the kept logits")
    elif r0["max_abs"] > crit["logprob_tolerance"]:
        why.append(f"first position: log-probabilities differ by up to {r0['max_abs']} nats (tolerance {crit['logprob_tolerance']})")
    if d is not None:
        r = rows[d] if d < len(rows) else None
        out["at_difference"] = None if r is None else {k: r[k] for k in ("max_abs", "llama_margin", "llama_top2", "token", "llama_token")}
        where = "the first generated token" if d == 0 else f"position {d}"
        if r is None or r["llama_margin"] is None:
            why.append(f"{where}: the ids differ and llama.cpp's margin there is not known")
        elif r["llama_margin"] >= crit["tie_margin"]:
            why.append(f"{where}: the ids differ where llama.cpp's margin is {r['llama_margin']} nats (a tie is below {crit['tie_margin']})")
        elif ids[d] not in r["llama_top2"]:
            why.append(f"{where}: this side's token {ids[d]} is not one of llama.cpp's two best {r['llama_top2']}")
    out["positions_compared"] = len(rows)
    vals = [r["max_abs"] for r in rows if r["max_abs"] is not None]
    out["max_abs_over_compared_positions"] = max(vals) if vals else None
    out["median_max_abs"] = round(statistics.median(vals), 5) if vals else None
    return dict(out, **{"pass": not why, "why": why})


def compare_hidden(mine_hidden, tensors, tokens, llama_tokens=None):
    """This side's hidden states after named blocks against llama.cpp's l_out-<block> as its callback printed it
    (four decimals). `llama_tokens`: how many of the prompt's tokens llama.cpp evaluated, when fewer (its -f drops
    a final newline): the states of the tokens both sides have are compared, the sums are not."""
    full, tokens = tokens, llama_tokens or tokens
    rows = []
    for h in mine_hidden:
        t = tensors.get(f"l_out-{h['layer']}")
        if not t:
            continue
        n = t["dims"][1]
        which = corner_rows(tokens) if n == tokens else ([tokens - 1] if n == 1 else None)
        if which is None or len(which) != len(t["rows"]):
            rows.append({"layer": h["layer"], "error": f"llama.cpp printed {len(t['rows'])} rows of {n}"})
            continue
        mine = {c["token"]: c["first3"] + c["last3"] for c in h["corners"]}
        diffs, scale = [], []
        for tok, vals in zip(which, t["rows"]):
            if tok in mine and len(vals) == 6:
                diffs += [abs(a - b) for a, b in zip(mine[tok], vals)]
                scale += [abs(b) for b in vals]
        my_sum = None if tokens != full else (h["sum_all_tokens_f32"] if n == tokens else h["last_token_sum_f32"])
        rows.append({"layer": h["layer"], "llama_tokens": n, "values_compared": len(diffs),
                     "max_abs": round(max(diffs), 5) if diffs else None,
                     "mean_abs": round(sum(diffs) / len(diffs), 5) if diffs else None,
                     "mean_abs_value": round(sum(scale) / len(scale), 5) if scale else None,
                     "sum": None if my_sum is None else [round(my_sum, 4), t["sum"]],
                     "last_token_rms": round(h["last_token_l2"] / math.sqrt(max(1, h.get("hidden", 1))), 5) if h.get("hidden") else None})
    return rows


def compare_router(mine_router, tensors, tokens, llama_tokens=None):
    """The experts this side's router chose against llama.cpp's ffn_moe_topk-<block> (both in selection order; the
    callback shows the first three and the last three of a token's experts). `llama_tokens` as in compare_hidden."""
    tokens = llama_tokens or tokens
    rows = []
    for r in mine_router:
        t = tensors.get(f"ffn_moe_topk-{r['layer']}")
        if not t:
            continue
        n = t["dims"][1]
        which = corner_rows(tokens) if n == tokens else ([tokens - 1] if n == 1 else None)
        if which is None or len(which) != len(t["rows"]):
            rows.append({"layer": r["layer"], "error": f"llama.cpp printed {len(t['rows'])} rows of {n}"})
            continue
        mine = {x["token"]: ends(x["ids"]) for x in r["tokens"]}
        same_order = same_set = seen = slots = slots_equal = 0
        for tok, vals in zip(which, t["rows"]):
            if tok not in mine:
                continue
            theirs = [int(v) for v in vals]
            seen += 1
            same_order += theirs == mine[tok]
            same_set += sorted(theirs) == sorted(mine[tok])
            slots += len(theirs)
            slots_equal += sum(a == b for a, b in zip(theirs, mine[tok]))
        rows.append({"layer": r["layer"], "llama_tokens": n, "tokens_compared": seen, "same_order": same_order,
                     "same_set": same_set, "slots": slots, "slots_equal": slots_equal})
    return rows


# --- fidelity: each engine against a full-precision reference (no I/O) -----------------------------------------
def step_dist(step):
    """{token: log-probability} of one recorded step (its kept logits and the row's log-sum-exp)."""
    return {int(t): v - step["logsumexp"] for t, v in step["top"]}


def fidelity_metrics(ref, eng, floor):
    """An engine's distribution against the reference's, both {token: log-probability} over the tokens each kept.
    KL(reference || engine) is summed over the tokens both kept; `covered` is the reference's mass on them."""
    both = [t for t in ref if t in eng]
    kl = sum(math.exp(ref[t]) * (ref[t] - eng[t]) for t in both)
    above = [t for t in ref if math.exp(ref[t]) >= floor]
    diffs = [abs(ref[t] - eng[t]) for t in above if t in eng]
    order = sorted(ref, key=lambda t: (-ref[t], t))
    return {"kl": kl, "covered": sum(math.exp(ref[t]) for t in both), "max_abs_above_floor": max(diffs) if diffs else None,
            "tokens_above_floor": len(above), "missing_above_floor": len(above) - len(diffs),
            "top1_agrees": bool(eng) and min(eng, key=lambda t: (-eng[t], t)) == order[0],
            "reference_margin": ref[order[0]] - ref[order[1]] if len(order) > 1 else None}


def judge_fidelity(rows, held, yardstick, crit):
    """`rows`: [{"prompt", "position", "sides": {side: fidelity_metrics}}] of one model. The sides named in `held`
    are held to [fidelity] of the config, with the side `yardstick` (llama.cpp) as the measure. Returns
    {"pass", "failures", "mean_kl": {side: ..}}."""
    failures, mean = [], {}
    sides = sorted({k for r in rows for k in r["sides"]})
    for side in sides:
        kls = [r["sides"][side]["kl"] for r in rows if side in r["sides"]]
        mean[side] = sum(kls) / len(kls) if kls else None
    for side in held:
        if mean.get(side) is None:
            failures.append(f"{side}: not measured")
            continue
        for r in rows:
            m, y, at = r["sides"].get(side), r["sides"].get(yardstick), f"{side}, {r['prompt']} position {r['position']}"
            if m is None or y is None:
                failures.append(f"{at}: a side is missing")
                continue
            if m["covered"] < crit["coverage"] or m["missing_above_floor"] or m["max_abs_above_floor"] is None:
                failures.append(f"{at}: compared tokens hold {m['covered']:.5f} of the reference's mass, {m['missing_above_floor']} above the floor missing")
                continue
            if m["kl"] > crit["kl_bound"]:
                failures.append(f"{at}: KL {m['kl']:.5f} nats is over the bound {crit['kl_bound']}")
            if m["kl"] > max(crit["kl_factor"] * y["kl"], crit["kl_floor"]):
                failures.append(f"{at}: KL {m['kl']:.5f} is over {crit['kl_factor']} times llama.cpp's {y['kl']:.5f} (floor {crit['kl_floor']})")
            if y["max_abs_above_floor"] is not None and m["max_abs_above_floor"] > max(crit["maxdiff_factor"] * y["max_abs_above_floor"], crit["maxdiff_floor"]):
                failures.append(f"{at}: max difference {m['max_abs_above_floor']:.4f} is over {crit['maxdiff_factor']} times llama.cpp's "
                                f"{y['max_abs_above_floor']:.4f} (floor {crit['maxdiff_floor']})")
            if not m["top1_agrees"] and (m["reference_margin"] or 0.0) >= crit["top1_margin"]:
                failures.append(f"{at}: the best token is not the reference's, whose margin is {m['reference_margin']:.3f} nats")
        if mean.get(yardstick) is not None and mean[side] > max(crit["mean_kl_factor"] * mean[yardstick], crit["mean_kl_floor"]):
            failures.append(f"{side}: mean KL {mean[side]:.5f} is over {crit['mean_kl_factor']} times llama.cpp's {mean[yardstick]:.5f} "
                            f"(floor {crit['mean_kl_floor']})")
    return {"pass": not failures, "failures": failures, "mean_kl": mean}


# --- stages -----------------------------------------------------------------------------------------------------
def stage_check(ctx):
    out = G.stage_check(ctx)
    out["pass"] = out["rc"] == 0 and out["catalogue_rc"] == 0
    return out


def stage_origin(ctx):
    out = G.stage_origin(ctx)
    export, _ = G.export_origin(ctx, os.path.join(ctx["res"], "diff-origin.txt"))
    adapted = {}
    for mid in ctx["recipe_modules"]():
        m = ctx["cat"].by_id[mid]
        for rel in m.files():
            if m.origin == "flerp" and rel.startswith(ADAPTED_ROOT + "src/"):
                theirs = os.path.join(export, rel[len(ADAPTED_ROOT):])
                if not os.path.isfile(theirs):
                    adapted[rel] = {"origin_file": None}
                    continue
                with open(theirs, errors="replace") as f, open(os.path.join(m.dir, "src", rel), errors="replace") as g:
                    a, b = f.read().splitlines(), g.read().splitlines()
                d = [ln for ln in difflib.unified_diff(a, b, lineterm="", n=0) if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
                adapted[rel] = {"origin_file": rel[len(ADAPTED_ROOT):], "origin_lines": len(a), "lines": len(b),
                                "added": sum(ln[0] == "+" for ln in d), "removed": sum(ln[0] == "-" for ln in d)}
    out["adapted"] = adapted
    out["pass"] = out["pass"] and bool(adapted) and all(v.get("origin_file") for v in adapted.values())
    return out


def write_ids(ctx, name, ids):
    d = os.path.join(ctx["scratch"], "prompts")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name + ".ids")
    with open(path, "w") as f:
        f.write(" ".join(str(i) for i in ids))
    return path


def stage_cpu(ctx):
    cfg = ctx["cfg"]
    log = os.path.join(ctx["res"], "cpu.txt")
    open(log, "w").close()
    binary = G.assembled_binary(ctx)
    ps = G.prompts(ctx)
    files = [x for _, _, path in ps for x in ("--prompt-file", path)]
    out = {"evidence": "cpu.txt, prompt-ids.json"}
    env = dict(G.NO_GPU, **G.runtime_env(ctx))
    rc, events, _ = G.generate(binary, ["--model", cfg["model"]["gguf"], "--tokenize"] + files, log, 900, env, G.polite(cfg))
    loaded = next((e for e in events if e.get("event") == "loaded"), {})
    out["model"] = {k: loaded.get(k) for k in ("architecture", "layers", "hidden", "intermediate", "heads", "kv_heads", "vocab", "mtp_layers",
                                               "expert_count", "expert_used_count", "expert_feed_forward_length",
                                               "expert_shared_feed_forward_length")}
    ids = {ps[e["prompt"]][0]: e["prompt_ids"] for e in events if e.get("event") == "tokenized"}
    with open(os.path.join(ctx["res"], "prompt-ids.json"), "w") as f:
        json.dump(ids, f)
        f.write("\n")
    out["prompt_tokens"] = {k: len(v) for k, v in ids.items()}
    out["tokenize_rc"] = rc
    ok = rc == 0 and len(ids) == len(ps) and loaded.get("architecture") == cfg["model"]["architecture"]
    dense_ids = os.path.join(ROOT, cfg["model"]["dense_results"], "prompt-ids.json")
    if os.path.isfile(dense_ids):
        with open(dense_ids) as f:
            out["prompt_ids_equal_dense_recipe"] = {k: v == ids.get(k) for k, v in json.load(f).items()}
    n, k = cfg["run"].get("cpu_tokens", 0), cfg["run"].get("cpu_prompts", 0)
    if ok and n and k:
        idfiles = [x for pname, _, _ in ps[:k] for x in ("--ids-file", write_ids(ctx, pname, ids[pname]))]
        env = dict(env, OMP_NUM_THREADS=str(cfg["run"].get("cpu_threads", 8)))
        rc3, ev3, secs = G.generate(binary, ["--model", cfg["model"]["gguf"], "--cpu", "--max-tokens", str(n)] + idfiles, log,
                                    cfg["run"].get("cpu_timeout_s", 900), env, G.polite(cfg))
        out["cpu_reference"] = {"rc": rc3, "seconds": secs, "tokens_asked": n, "timed_out": rc3 == 124,
                                "runs": {ps[e["prompt"]][0]: {"ids": e["ids"], "text": e["text"]} for e in ev3 if e.get("event") == "generated"}}
    tw = cfg["run"].get("three_way_prompt")
    if ok and tw in ids:
        env = dict(env, OMP_NUM_THREADS=str(cfg["run"].get("cpu_threads", 8)))
        rc4, ev4, secs = G.generate(binary, ["--model", cfg["model"]["gguf"], "--cpu", "--trace", "--trace-top", str(cfg["run"]["trace_top"]),
                                             "--max-tokens", str(cfg["run"].get("three_way_tokens", 3)), "--ids-file", write_ids(ctx, tw, ids[tw])],
                                    log, cfg["run"].get("cpu_timeout_s", 900), env, G.polite(cfg))
        traced = [e for e in ev4 if e.get("event") == "traced"]
        with open(os.path.join(ctx["res"], "cpu-trace.json"), "w") as f:
            json.dump({tw: traced[0]} if traced else {}, f)
            f.write("\n")
        out["cpu_trace"] = {"rc": rc4, "seconds": secs, "prompt": tw, "ids": traced[0]["ids"] if traced else None}
        out["evidence"] += ", cpu-trace.json"
    out["pass"] = ok
    return out


def llama_generate(ctx, model, ids, log, port):
    """llama-server on `model`: every prompt's greedy ids and top log-probabilities. Returns (runs, error)."""
    cfg, lc = ctx["cfg"], ctx["cfg"]["llama"]
    server = os.path.join(ROOT, cfg["host"]["llama_build"], "bin", "llama-server")
    cmd = G.polite(cfg) + [server, "-m", model, "--host", "127.0.0.1", "--port", str(port), "-ngl", "0",
                           "-t", str(lc.get("threads", 8)), "-c", str(cfg["run"]["context"]), "--parallel", "1"]
    srv, runs, error = G.LlamaServer(cmd, port, log), {}, None
    try:
        if not srv.wait_ready(lc.get("startup_timeout_s", 600)):
            return runs, "llama-server did not become ready"
        texts = {name: text for name, text, _ in G.prompts(ctx)}
        for pname, pids in ids.items():
            tok = srv.call("/tokenize", {"content": texts[pname], "add_special": False, "parse_special": True}, 120).get("tokens", [])
            t0 = time.time()
            r = srv.call("/completion", {"prompt": pids, "n_predict": cfg["run"]["max_tokens"], "temperature": 0, "top_k": 1,
                                         "repeat_penalty": 1.0, "cache_prompt": False, "return_tokens": True, "seed": 1,
                                         "n_probs": lc.get("n_probs", 20), "post_sampling_probs": False, "stream": False},
                         lc.get("request_timeout_s", 1800))
            tm = r.get("timings") or {}
            runs[pname] = {"ids": r.get("tokens", []), "text": r.get("content", ""), "seconds": round(time.time() - t0, 1),
                           "decode_tok_s": tm.get("predicted_per_second"), "prompt_tok_s": tm.get("prompt_per_second"),
                           "prompt_tokens_evaluated": r.get("tokens_evaluated"), "stop": r.get("stop_type") or r.get("stopped_eos"),
                           "tokenizer_ids_equal": tok == pids, "tokenizer_first_difference": G.first_difference(tok, pids),
                           "probs": [{"id": p.get("id"), "logprob": p.get("logprob"),
                                      "top_logprobs": [{"id": t.get("id"), "logprob": t.get("logprob")} for t in p.get("top_logprobs", [])]}
                                     for p in r.get("completion_probabilities", [])]}
    except (urllib.error.URLError, OSError, ValueError) as e:
        error = f"llama-server request failed: {e}"
    finally:
        srv.stop()
    return runs, error


def stage_llama(ctx):
    cfg, lc = ctx["cfg"], ctx["cfg"]["llama"]
    log = os.path.join(ctx["res"], "llama.txt")
    open(log, "w").close()
    out = {"evidence": "llama.txt, llama.json, llama-dense.json, llama-trace.json", "source": lc["source"]}
    out["source_head"] = subprocess.run(["git", "-C", lc["source"], "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    bdir = os.path.join(ROOT, cfg["host"]["llama_build"])
    if not os.path.isfile(os.path.join(bdir, "bin", "llama-server")):
        return dict(out, error=f"no llama-server under {cfg['host']['llama_build']}: run the dense recipe's llama stage")
    with open(os.path.join(ctx["res"], "prompt-ids.json")) as f:
        ids = json.load(f)
    runs, err = llama_generate(ctx, cfg["model"]["gguf"], ids, log, lc["port"])
    with open(os.path.join(ctx["res"], "llama.json"), "w") as f:
        json.dump(runs, f)
        f.write("\n")
    if err:
        return dict(out, error=err, log_tail=tail(log, 8))
    out["runs"] = {k: {x: v[x] for x in v if x not in ("ids", "text", "probs")} | {"tokens": len(v["ids"]), "positions_with_probs": len(v["probs"])}
                   for k, v in runs.items()}
    ok = len(runs) == len(ids) and all(v["ids"] and len(v["probs"]) == len(v["ids"]) for v in runs.values())

    # the dense model: the baseline for "how far two engines are apart" through the origin's unmodified path
    dense_ids = os.path.join(ROOT, cfg["model"]["dense_results"], "prompt-ids.json")
    if os.path.isfile(dense_ids):
        with open(dense_ids) as f:
            dids = json.load(f)
        druns, derr = llama_generate(ctx, cfg["model"]["dense_gguf"], dids, log, lc["port"])
        with open(os.path.join(ctx["res"], "llama-dense.json"), "w") as f:
            json.dump(druns, f)
            f.write("\n")
        out["dense"] = {"error": derr, "prompts": len(druns)}

    # hidden states and router choices: the eval callback copies every graph node out, so short prompts only
    tool = os.path.join(bdir, "bin", "llama-eval-callback")
    if not os.path.isfile(tool):
        rc, _ = sh(G.polite(cfg) + ["cmake", "-S", lc["source"], "-B", bdir, "-DLLAMA_BUILD_EXAMPLES=ON"], log, timeout=900)
        if rc == 0:
            rc, out["eval_callback_build_s"] = sh(G.polite(cfg) + ["ninja", "-C", bdir, f"-j{cfg['host'].get('build_jobs', 2)}",
                                                                 "llama-eval-callback"], log, timeout=lc.get("build_timeout_s", 3600))
    trace = {}
    if os.path.isfile(tool):
        for pname, _, path in G.prompts(ctx):
            if pname not in cfg["run"].get("trace_prompts", []):
                continue
            raw = os.path.join(ctx["scratch"], f"eval-callback-{pname}.txt")
            cmd = G.polite(cfg) + [tool, "-m", cfg["model"]["gguf"], "-f", path, "-ngl", "0", "-t", str(lc.get("threads", 8)),
                                   "-c", str(cfg["run"]["context"])]           # the program turns the warm-up off itself
            with open(log, "a") as f:
                f.write("$ " + " ".join(cmd) + f" > {raw} 2>&1\n")
            t0 = time.time()
            with open(raw, "wb") as f:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, **G.NO_GPU),
                                    timeout=lc.get("request_timeout_s", 1800)).returncode
            with open(raw, errors="replace") as f:
                text = f.read()
            tensors = parse_eval_callback(text, r"(l_out|ffn_moe_topk)-\d+")
            toks = callback_input_ids(text)
            trace[pname] = {"rc": rc, "seconds": round(time.time() - t0, 1), "bytes": len(text), "input_ids": toks,
                            "input_ids_equal": toks == ids.get(pname), "tensors": tensors,
                            # -f drops a final newline of the prompt file: the states of the tokens it did evaluate
                            # are the same tokens' states (the model is causal) and those are what is compared
                            "input_ids_are_a_prefix": bool(toks) and toks == ids.get(pname, [])[:len(toks)]}
            if rc != 0 or not tensors:
                trace[pname]["output_tail"] = text[-600:]
            os.remove(raw)                       # megabytes of every node; what is compared is in llama-trace.json
    with open(os.path.join(ctx["res"], "llama-trace.json"), "w") as f:
        json.dump(trace, f)
        f.write("\n")
    out["trace"] = {k: {"rc": v["rc"], "seconds": v["seconds"], "tensors": len(v["tensors"]), "input_ids_equal": v["input_ids_equal"],
                        "input_ids_are_a_prefix": v["input_ids_are_a_prefix"], "input_tokens": len(v["input_ids"])}
                    for k, v in trace.items()}
    # the callback's trace is part of what the gpu stage reports: a prompt it did not give is a failure here
    want = [p for p in cfg["run"].get("trace_prompts", []) if p in ids]
    out["pass"] = ok and all(trace.get(p, {}).get("rc") == 0 and trace[p]["tensors"] and trace[p]["input_ids_are_a_prefix"] for p in want)
    return out


def events_of(run, kind):
    return [e for e in run.get("events", []) if e.get("event") == kind]


def stage_gpu(ctx):
    cfg, name = ctx["cfg"], ctx["recipe"].name
    log = os.path.join(ctx["res"], "gpu.txt")
    open(log, "w").close()
    binary = G.assembled_binary(ctx)
    out = {"evidence": "gpu.txt, gpu-runs.json, compare.json, lease.txt"}
    if not os.path.isfile(binary):
        return dict(out, error="no binary: run the build stage first")
    out["binary_sha256"] = sha256(binary)
    ps = [p[0] for p in G.prompts(ctx)]
    with open(os.path.join(ctx["res"], "prompt-ids.json")) as f:
        ids = json.load(f)
    dres = os.path.join(ROOT, cfg["model"]["dense_results"])
    with open(os.path.join(dres, "prompt-ids.json")) as f:
        dids = json.load(f)
    with open(os.path.join(dres, "gpu-runs.json")) as f:
        dense_recorded = {ps[i]: g["ids"] for i, g in enumerate(json.load(f)["assembled"][0]["generated"])}
    run = cfg["run"]
    base = ["--context", str(run["context"]), "--max-tokens", str(run["max_tokens"])]
    moe_files = [x for p in ps for x in ("--ids-file", write_ids(ctx, p, ids[p]))]
    dense_files = [x for p in ps for x in ("--ids-file", write_ids(ctx, "dense-" + p, dids[p]))]
    trace = ["--trace", "--trace-top", str(run["trace_top"])]
    layers = ["--trace-layers", ",".join(str(x) for x in run["trace_layers"])]
    plan = [("dense", ["--model", cfg["model"]["dense_gguf"]] + base + dense_files),
            ("dense-traced", ["--model", cfg["model"]["dense_gguf"]] + base + trace + dense_files)]
    plan += [(f"moe-{i + 1}", ["--model", cfg["model"]["gguf"]] + base + moe_files) for i in range(run["repeats"])]
    plan += [("moe-traced", ["--model", cfg["model"]["gguf"]] + base + trace + layers + moe_files)]
    tw = run.get("three_way_prompt")
    if tw in ids:
        plan += [("moe-decode-fed", ["--model", cfg["model"]["gguf"], "--context", str(run["context"]), "--max-tokens",
                                     str(run.get("three_way_tokens", 3))] + trace + ["--decode-prompt", "--ids-file", write_ids(ctx, tw, ids[tw])])]
    hold = Hold(cfg, os.path.join(ctx["res"], "lease.txt"), out, wait_s=cfg["host"].get("gpu_wait_s", 1800))
    runs = {}
    try:
        hold.take(f"strixflerp gate {name}", "gpu stage")
        for label, args in plan:
            hold.between(f"strixflerp gate {name}", label)
            rc, events, secs = G.generate(binary, args, log, run["run_timeout_s"], G.runtime_env(ctx))
            runs[label] = {"rc": rc, "seconds": secs, "events": events}
    except RuntimeError as e:
        out["error"] = str(e)
        print(f"   {e}", flush=True)
    finally:
        hold.give()
    with open(os.path.join(ctx["res"], "gpu-runs.json"), "w") as f:
        json.dump(runs, f)
        f.write("\n")
    if "error" in out:
        return out
    out["runs"] = {k: {"rc": v["rc"], "seconds": v["seconds"]} for k, v in runs.items()}
    cmp = judge(ctx, runs, ps, dense_recorded)
    with open(os.path.join(ctx["res"], "compare.json"), "w") as f:
        json.dump(cmp, f, indent=1)
        f.write("\n")
    out.update({k: cmp[k] for k in ("dense_regression", "self_consistent", "llama", "dense_baseline", "three_way", "speed", "device", "verdict",
                                    "failures")})
    out["pass"] = cmp["verdict"]
    return out


# --- the fidelity stages ----------------------------------------------------------------------------------------
GPU_SIDES = {"gpu prompt pass": ([], {}), "gpu, prompt through the decode step": (["--decode-prompt"], {}),
             "gpu prompt pass, FP32 activations": ([], {"FLERP_PREFILL_FP32_ACTIVATIONS": "1"})}
HELD = ("gpu prompt pass", "gpu, prompt through the decode step")
LLAMA = "llama.cpp"


def fidelity_inputs(ctx):
    """{model: {"gguf", "ids": {prompt: ids}, "cont": {prompt: continuation}, "positions": {prompt: [k]}}}: the prompt
    ids and the continuation llama.cpp recorded on that model, which every engine is fed."""
    cfg, fc, out = ctx["cfg"], ctx["cfg"]["fidelity"], {}
    long_names = {p["name"] for p in cfg["prompt"] if p.get("long")}
    dres = os.path.join(ROOT, cfg["model"]["dense_results"])
    with open(os.path.join(dres, "prompt-ids.json")) as f:
        dense_ids = json.load(f)
    for model, gguf, ids, rec in (("moe", cfg["model"]["gguf"], load_json(ctx, "prompt-ids.json"), load_json(ctx, "llama.json")),
                                  ("dense", cfg["model"]["dense_gguf"], dense_ids, load_json(ctx, "llama-dense.json"))):
        pos = {p: [k for k in (fc["long_positions"] if p in long_names else fc["positions"]) if k < len(rec[p]["ids"])] for p in ids if p in rec}
        out[model] = {"gguf": gguf, "ids": {p: ids[p] for p in pos}, "cont": {p: rec[p]["ids"][:max(pos[p])] for p in pos}, "positions": pos}
    return out


def cpu_reference(ctx, model, m, pname, log, dump=None):
    """The CPU reference on one prompt at its positions, kept in the scratch by what it was asked (a long prompt
    takes hours). Returns {"dists": {k: [[id, logprob]]}, "seconds", "cached"}."""
    cfg, fc = ctx["cfg"], ctx["cfg"]["fidelity"]
    ask = json.dumps([m["gguf"], m["ids"][pname], m["cont"][pname], m["positions"][pname], fc["top"]])
    key = __import__("hashlib").sha256(ask.encode()).hexdigest()[:16]
    cache = os.path.join(ctx["scratch"], "reference", f"{model}-{pname}-{key}.json")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.isfile(cache) and not dump:
        with open(cache) as f:
            return dict(json.load(f), cached=True)
    env = dict(G.NO_GPU, OMP_NUM_THREADS=str(cfg["run"].get("cpu_threads", 8)), **G.runtime_env(ctx))
    if dump:
        if os.path.isfile(dump):
            os.remove(dump)
        env.update(FLERP_MOE_DUMP=dump, FLERP_MOE_DUMP_LAYERS=",".join(str(x) for x in fc["block_check_layers"]))
    args = ["--model", m["gguf"], "--context", str(cfg["run"]["context"]), "--cpu", "--trace", "--trace-top", str(fc["top"]),
            "--ids-file", write_ids(ctx, f"fid-{model}-{pname}", m["ids"][pname]),
            "--force-file", write_ids(ctx, f"fid-{model}-{pname}-cont", m["cont"][pname]),
            "--record-at", ",".join(str(k) for k in m["positions"][pname])]
    rc, events, secs = G.generate(G.assembled_binary(ctx), args, log, fc.get("cpu_timeout_s", 6 * 3600), env, G.polite(cfg))
    ev = next((e for e in events if e.get("event") == "traced"), None)
    if rc != 0 or ev is None:
        return {"error": f"exit {rc}", "seconds": secs}
    res = {"dists": {str(st["position"]): [[int(t), v - st["logsumexp"]] for t, v in st["top"]] for st in ev["steps"]},
           "seconds": secs, "prompt_tokens": ev["prompt_tokens"], "tokens_evaluated": ev["prompt_tokens"] + max(m["positions"][pname])}
    with open(cache, "w") as f:
        json.dump(res, f)
    return dict(res, cached=False)


def llama_positions(ctx, gguf, m, log, port):
    """llama-server on `gguf`: its top log-probabilities at every position of every prompt, the context fed as ids.
    Returns ({prompt: {k: [[id, logprob]]}}, error)."""
    cfg, lc, fc = ctx["cfg"], ctx["cfg"]["llama"], ctx["cfg"]["fidelity"]
    server = os.path.join(ROOT, cfg["host"]["llama_build"], "bin", "llama-server")
    cmd = G.polite(cfg) + [server, "-m", gguf, "--host", "127.0.0.1", "--port", str(port), "-ngl", "0",
                           "-t", str(lc.get("threads", 8)), "-c", str(cfg["run"]["context"]), "--parallel", "1"]
    srv, out, error = G.LlamaServer(cmd, port, log), {}, None
    try:
        if not srv.wait_ready(lc.get("startup_timeout_s", 600)):
            return out, "llama-server did not become ready"
        for pname, ids in m["ids"].items():
            out[pname] = {}
            for k in m["positions"][pname]:
                r = srv.call("/completion", {"prompt": ids + m["cont"][pname][:k], "n_predict": 1, "temperature": 0, "top_k": 1,
                                             "repeat_penalty": 1.0, "cache_prompt": True, "seed": 1, "n_probs": fc["top"],
                                             "post_sampling_probs": False, "stream": False}, lc.get("request_timeout_s", 1800))
                top = (r.get("completion_probabilities") or [{}])[0].get("top_logprobs", [])
                out[pname][str(k)] = [[t["id"], t["logprob"]] for t in top]
    except (urllib.error.URLError, OSError, ValueError) as e:
        error = f"llama-server request failed: {e}"
    finally:
        srv.stop()
    return out, error


def stage_reference(ctx):
    """No GPU: the full-precision references, what gives them standing, and llama.cpp at the same positions."""
    cfg, fc, lc = ctx["cfg"], ctx["cfg"]["fidelity"], ctx["cfg"]["llama"]
    log = os.path.join(ctx["res"], "reference.txt")
    open(log, "w").close()
    out = {"evidence": "reference.txt, fidelity-reference.json, block-check.json"}
    inputs = fidelity_inputs(ctx)
    doc = {"inputs": {k: {x: v[x] for x in ("gguf", "positions", "cont")} for k, v in inputs.items()}, "reference": {"moe": {}, "dense": {}},
           "llama": {}, "dense_f32": {}}
    long_names = {p["name"] for p in cfg["prompt"] if p.get("long")}
    bdir = os.path.join(ROOT, cfg["host"]["llama_build"])

    def save():
        with open(os.path.join(ctx["res"], "fidelity-reference.json"), "w") as f:
            json.dump(doc, f)
            f.write("\n")

    # 1. the CPU reference on the MoE model's short prompts; one of them dumps blocks for the independent check
    m = inputs["moe"]
    for p in [x for x in m["ids"] if x not in long_names]:
        dump = os.path.join(ctx["scratch"], "moe-blocks.bin") if p == fc["block_check_prompt"] else None
        doc["reference"]["moe"][p] = cpu_reference(ctx, "moe", m, p, log, dump)
        if dump and os.path.isfile(dump):
            cmd = G.polite(cfg) + [sys.executable, os.path.join(ROOT, "gate", "moe_block_check.py"), "--gguf", m["gguf"],
                                   "--gguf-py", os.path.join(lc["source"], "gguf-py"), "--dump", dump]
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            with open(log, "a") as f:
                f.write("$ " + " ".join(cmd) + "\n" + r.stderr[-2000:] + "\n")
            try:
                check = json.loads(r.stdout)
            except ValueError:
                check = {"error": (r.stderr or r.stdout)[-400:]}
            check["seconds"] = round(time.time() - t0, 1)
            with open(os.path.join(ctx["res"], "block-check.json"), "w") as f:
                json.dump(check, f, indent=1)
                f.write("\n")
            out["block_check"] = {k: v for k, v in check.items() if k != "detail"}
            os.remove(dump)
        save()

    # 2. the dense model: the CPU reference on the short prompts, and llama.cpp on an F32 conversion of the file
    d = inputs["dense"]
    for p in fc["dense_cpu_prompts"]:
        if p in d["ids"]:
            doc["reference"]["dense"][p] = cpu_reference(ctx, "dense", d, p, log)
    save()
    quantize, f32 = os.path.join(bdir, "bin", "llama-quantize"), os.path.join(ctx["scratch"], "dense-f32.gguf")
    if not os.path.isfile(quantize):
        sh(G.polite(cfg) + ["ninja", "-C", bdir, f"-j{cfg['host'].get('build_jobs', 2)}", "llama-quantize"], log, timeout=lc.get("build_timeout_s", 3600))
    if os.path.isfile(quantize):
        rc, secs = sh(G.polite(cfg) + [quantize, "--allow-requantize", d["gguf"], f32, "F32", str(lc.get("threads", 8))], log, timeout=3600)
        out["dense_f32"] = {"rc": rc, "seconds": secs, "bytes": os.path.getsize(f32) if os.path.isfile(f32) else 0}
        if rc == 0:
            doc["dense_f32"], err = llama_positions(ctx, f32, d, log, lc["port"])
            out["dense_f32"]["error"] = err
        if os.path.isfile(f32):
            os.remove(f32)                    # gigabytes; a rerun converts again in a minute
    save()

    # 3. llama.cpp on the files as shipped, at the same positions
    for model in ("dense", "moe"):
        doc["llama"][model], err = llama_positions(ctx, inputs[model]["gguf"], inputs[model], log, lc["port"])
        out[f"llama_{model}_error"] = err
        save()

    # 4. the MoE model's long prompt, last: seconds a token for more than a thousand tokens
    for p in [x for x in m["ids"] if x in long_names]:
        doc["reference"]["moe"][p] = cpu_reference(ctx, "moe", m, p, log)
        save()
    out["cpu_reference"] = {model: {p: {k: v.get(k) for k in ("seconds", "cached", "tokens_evaluated", "error")} for p, v in doc["reference"][model].items()}
                            for model in ("moe", "dense")}
    out["pass"] = (all("dists" in v for v in doc["reference"]["moe"].values()) and len(doc["reference"]["moe"]) == len(m["ids"])
                   and all(doc["llama"].get(k) for k in ("moe", "dense")) and "error" not in out.get("block_check", {"error": 1}))
    return out


def fidelity_reference(doc, model, prompt):
    """The reference distributions of a prompt and where they come from: the CPU reference, or (the dense model's
    long prompt) llama.cpp on the F32 conversion."""
    r = doc["reference"].get(model, {}).get(prompt, {})
    if "dists" in r:
        return r["dists"], "cpu reference"
    if model == "dense" and prompt in doc.get("dense_f32", {}):
        return doc["dense_f32"][prompt], "llama.cpp on the F32 conversion"
    return None, None


def stage_fidelity(ctx):
    cfg, fc, name = ctx["cfg"], ctx["cfg"]["fidelity"], ctx["recipe"].name
    log = os.path.join(ctx["res"], "fidelity.txt")
    open(log, "w").close()
    out = {"evidence": "fidelity.txt, fidelity-runs.json, fidelity.json, fidelity.md, fidelity-lease.txt"}
    doc = load_json(ctx, "fidelity-reference.json")
    if not doc:
        return dict(out, error="no reference: run the reference stage first")
    inputs = fidelity_inputs(ctx)
    binary = G.assembled_binary(ctx)
    hold = Hold(cfg, os.path.join(ctx["res"], "fidelity-lease.txt"), out, wait_s=cfg["host"].get("gpu_wait_s", 1800))
    runs = {}
    try:
        hold.take(f"strixflerp gate {name}", "fidelity stage")
        for model, m in inputs.items():
            union = sorted({k for ks in m["positions"].values() for k in ks})
            files = []
            for p in m["ids"]:
                cont = m["cont"][p] + [m["cont"][p][-1]] * max(0, max(union) - len(m["cont"][p]))   # padding is fed after the last position kept
                files += ["--ids-file", write_ids(ctx, f"fid-{model}-{p}", m["ids"][p]), "--force-file", write_ids(ctx, f"fid-{model}-{p}-gpu-cont", cont)]
            for side, (extra, env) in GPU_SIDES.items():
                hold.between(f"strixflerp gate {name}", f"{model}, {side}")
                args = ["--model", m["gguf"], "--context", str(cfg["run"]["context"]), "--trace", "--trace-top", str(fc["top"]),
                        "--record-at", ",".join(str(k) for k in union)] + extra + files
                rc, events, secs = G.generate(binary, args, log, cfg["run"]["run_timeout_s"], dict(G.runtime_env(ctx), **env))
                runs[f"{model}|{side}"] = {"rc": rc, "seconds": secs, "events": events}
    except RuntimeError as e:
        out["error"] = str(e)
    finally:
        hold.give()
    with open(os.path.join(ctx["res"], "fidelity-runs.json"), "w") as f:
        json.dump(runs, f)
        f.write("\n")
    if "error" in out:
        return out
    res = judge_fidelity_runs(ctx, doc, inputs, runs)
    with open(os.path.join(ctx["res"], "fidelity.json"), "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    write_fidelity_md(ctx, res, out)
    out.update(verdict=res["moe"]["judgement"]["pass"], failures=res["moe"]["judgement"]["failures"],
               mean_kl={k: v["judgement"]["mean_kl"] for k, v in res.items() if isinstance(v, dict) and "judgement" in v},
               runs={k: {"rc": v["rc"], "seconds": v["seconds"]} for k, v in runs.items()})
    out["pass"] = out["verdict"] and all(v["rc"] == 0 for v in runs.values())
    return out


def judge_fidelity_runs(ctx, doc, inputs, runs):
    fc, res = ctx["cfg"]["fidelity"], {}
    for model, m in inputs.items():
        names, rows, sources = list(m["ids"]), [], {}
        gpu = {}
        for side in GPU_SIDES:
            ev = events_of(runs.get(f"{model}|{side}", {}), "traced")
            gpu[side] = {names[e["prompt"]]: {str(st["position"]): step_dist(st) for st in e["steps"]} for e in ev}
        for p in names:
            ref, source = fidelity_reference(doc, model, p)
            if ref is None:
                continue
            sources[p] = source
            for k in m["positions"][p]:
                r = {int(t): v for t, v in ref[str(k)]}
                sides = {side: fidelity_metrics(r, gpu[side][p][str(k)], fc["prob_floor"]) for side in GPU_SIDES if str(k) in gpu[side].get(p, {})}
                ll = doc["llama"].get(model, {}).get(p, {}).get(str(k))
                if ll:
                    sides[LLAMA] = fidelity_metrics(r, {int(t): v for t, v in ll}, fc["prob_floor"])
                if model == "dense" and source == "cpu reference" and str(k) in doc.get("dense_f32", {}).get(p, {}):
                    sides["llama.cpp on the F32 conversion"] = fidelity_metrics(r, {int(t): v for t, v in doc["dense_f32"][p][str(k)]}, fc["prob_floor"])
                rows.append({"prompt": p, "position": k, "reference": source, "sides": sides})
        res[model] = {"rows": rows, "reference": sources, "judgement": judge_fidelity(rows, HELD, LLAMA, fc)}
    res["criteria"] = {k: v for k, v in fc.items() if not isinstance(v, (list, str))}
    return res


def write_fidelity_md(ctx, res, out):
    cfg, fc = ctx["cfg"], ctx["cfg"]["fidelity"]
    ref = ctx["results"].get("reference", {})
    j = res["moe"]["judgement"]
    L = [f"# Fidelity: {ctx['recipe'].name}", "", f"Host {socket.gethostname()}, written {utc()}. **{'PASSED' if j['pass'] else 'FAILED'}** under "
         f"`[fidelity]` of `{ctx['recipe'].gate['config']}` (the criteria were committed before these runs). The recipe gate's own verdict is in "
         "`summary.md` and is not changed by this.", "",
         "Each engine against a full-precision evaluation of the weights as shipped (K-quant weights dequantized, FP32 activations), on the same "
         f"context: the prompt ids and llama.cpp's recorded continuation. KL(reference || engine) in nats over the {fc['top']} tokens each side "
         f"kept; max: the largest absolute log-probability difference over tokens the reference gives at least {fc['prob_floor']}; top-1: the best "
         "token is the reference's."]
    if j["failures"]:
        L += ["", "Criteria that did not hold:"] + [f"- {x}" for x in j["failures"]]
    for model, title in (("moe", "The MoE model (held)"), ("dense", "The dense model through the same binary (a record)")):
        r = res[model]
        sides = [s for s in list(GPU_SIDES) + [LLAMA, "llama.cpp on the F32 conversion"] if any(s in x["sides"] for x in r["rows"])]
        L += ["", f"## {title}", "", f"`{cfg['model']['gguf' if model == 'moe' else 'dense_gguf']}`. Reference by prompt: {r['reference']}.", "",
              "| prompt | position | reference margin | " + " | ".join(f"{s}: KL / max / top-1" for s in sides) + " |", "|---|---|---|" + "---|" * len(sides)]
        for x in r["rows"]:
            cells = []
            for s in sides:
                v = x["sides"].get(s)
                cells.append("" if v is None else f"{v['kl']:.5f} / " + ("-" if v["max_abs_above_floor"] is None else f"{v['max_abs_above_floor']:.4f}")
                             + f" / {'yes' if v['top1_agrees'] else 'no'}" + ("" if not v["missing_above_floor"] else f" ({v['missing_above_floor']} missing)"))
            margin = next(iter(x["sides"].values()))["reference_margin"] if x["sides"] else None
            L.append(f"| {x['prompt']} | {x['position']} | {margin:.3f} | " + " | ".join(cells) + " |")
        L += ["", "Mean KL over these positions: " + ", ".join(f"{s} {r['judgement']['mean_kl'][s]:.5f}" for s in sides if r["judgement"]["mean_kl"].get(s) is not None) + "."]
        if model == "dense":
            L.append(f"Held to the same criteria this model would {'pass' if r['judgement']['pass'] else 'fail: ' + '; '.join(r['judgement']['failures'][:6])}.")
    bc = ref.get("block_check")
    L += ["", "## What gives the reference standing", ""]
    if bc:
        L.append(f"- `gate/moe_block_check.py` (numpy, llama.cpp's gguf-py dequantization, float64; no code shared with the executor) on {bc.get('records')} "
                 f"real block inputs of `{fc['block_check_prompt']}` in blocks {bc.get('layers')}: the same experts in the same order in {bc.get('ids_equal')}, "
                 f"the same set in {bc.get('ids_same_set')}; weights within {bc.get('weights_max_abs')}, the shared gate logit within {bc.get('gate_logit_max_abs')}, "
                 f"the block output within {bc.get('out_max_abs')} ({bc.get('out_max_abs_over_rms')} of its RMS) (`block-check.json`).")
    L.append("- On the dense model, llama.cpp run on an F32 conversion of the same file (`llama-quantize --allow-requantize ... F32`: dequantized weights, "
             "FP32 activations, none of the executor's code) is in the dense table as a side against gufo's CPU reference.")
    L += ["", "## Cost", "", f"CPU reference runs (8 threads): {ref.get('cpu_reference')}. F32 conversion: {ref.get('dense_f32')}.",
          f"GPU runs: {out.get('runs') or {k: v['seconds'] for k, v in load_json(ctx, 'fidelity-runs.json').items()}}; holds {out.get('holds')}."]
    with open(os.path.join(ctx["res"], "fidelity.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def load_json(ctx, name):
    path = os.path.join(ctx["res"], name)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def judge(ctx, runs, ps, dense_recorded):
    """Everything the gpu stage concludes, from the recorded runs and llama.cpp's recorded results."""
    cfg, crit, failures = ctx["cfg"], ctx["cfg"]["criteria"], []
    cmp = {"criteria": crit}
    by = lambda label, kind: {ps[e["prompt"]]: e for e in events_of(runs.get(label, {}), kind)}  # noqa: E731
    for label, r in runs.items():
        if r["rc"] != 0:
            failures.append(f"run {label} exited {r['rc']}")

    # 1. dense regression
    dense = by("dense", "generated")
    cmp["dense_regression"] = {p: G.agreement(dense[p]["ids"], dense_recorded[p]) if p in dense else {"missing": True} for p in ps}
    for p, a in cmp["dense_regression"].items():
        if not a.get("equal"):
            failures.append(f"dense regression, {p}: ids differ from the dense recipe's recorded run ({a})")

    # 2. the MoE runs against themselves
    reps = [by(f"moe-{i + 1}", "generated") for i in range(cfg["run"]["repeats"])]
    traced = by("moe-traced", "traced")
    sc = {}
    for p in ps:
        have = [r[p]["ids"] for r in reps if p in r]
        sc[p] = {"repeats_equal": len(have) == len(reps) and all(h == have[0] for h in have),
                 "traced_equal": p in traced and bool(have) and traced[p]["ids"] == have[0],
                 "argmax_is_token": traced.get(p, {}).get("argmax_is_token"), "tokens": len(have[0]) if have else 0}
        if not (sc[p]["repeats_equal"] and sc[p]["traced_equal"] and sc[p]["argmax_is_token"] and sc[p]["tokens"] > 0):
            failures.append(f"self-consistency, {p}: {sc[p]}")
    cmp["self_consistent"] = sc

    # 3. against llama.cpp
    llama, ltrace = load_json(ctx, "llama.json"), load_json(ctx, "llama-trace.json")
    cmp["llama"], cmp["logprob_rows"], cmp["hidden"], cmp["router"] = {}, {}, {}, {}
    for p in ps:
        if p not in traced or p not in llama:
            failures.append(f"llama.cpp, {p}: a side is missing")
            continue
        d = G.first_difference(traced[p]["ids"], llama[p]["ids"])
        rows = compare_logprobs(traced[p]["steps"], llama[p]["probs"], d)
        cmp["logprob_rows"][p] = rows
        cmp["llama"][p] = judge_prompt(traced[p]["ids"], llama[p]["ids"], rows, crit)
        failures += [f"llama.cpp, {p}: {w}" for w in cmp["llama"][p]["why"]]
        if p in ltrace:
            tokens = traced[p]["prompt_tokens"]
            hidden = [dict(h, hidden=traced[p].get("hidden_size") or ctx["results"].get("cpu", {}).get("model", {}).get("hidden"))
                      for h in traced[p]["hidden"]]
            n_llama = len(ltrace[p].get("input_ids", [])) or tokens
            cmp["hidden"][p] = compare_hidden(hidden, ltrace[p]["tensors"], tokens, n_llama)
            cmp["router"][p] = compare_router(traced[p]["router"], ltrace[p]["tensors"], tokens, n_llama)

    # the baseline: the dense model through the origin's unmodified path, the same comparison, not a criterion
    ldense, dtraced = load_json(ctx, "llama-dense.json"), by("dense-traced", "traced")
    cmp["dense_baseline"] = {}
    for p in ps:
        if p in dtraced and p in ldense:
            d = G.first_difference(dtraced[p]["ids"], ldense[p]["ids"])
            rows = compare_logprobs(dtraced[p]["steps"], ldense[p]["probs"], d)
            j = judge_prompt(dtraced[p]["ids"], ldense[p]["ids"], rows, crit)
            cmp["dense_baseline"][p] = {k: j.get(k) for k in ("tokens", "first_difference", "agree", "first_position", "at_difference",
                                                             "max_abs_over_compared_positions", "median_max_abs", "pass")}
            cmp["dense_baseline"][p]["traced_equal_plain"] = p in dense and dtraced[p]["ids"] == dense[p]["ids"]

    # three ways on one prompt: where the difference from llama.cpp comes from (a record)
    cmp["three_way"] = {}
    tw = cfg["run"].get("three_way_prompt")
    cpu = load_json(ctx, "cpu-trace.json").get(tw)
    fed = next(iter(events_of(runs.get("moe-decode-fed", {}), "traced")), None)
    if cpu and tw in traced and tw in llama:
        n = len(cpu["steps"])
        sides = {"gpu, prompt through the decode step": fed and fed["steps"], "gpu, prompt pass": traced[tw]["steps"][:n],
                 "llama.cpp": as_steps(llama[tw]["probs"])[:n]}
        cmp["three_way"] = {"prompt": tw, "positions": n, "cpu_reference_ids": cpu["ids"],
                            "ids": {k: [s["token"] for s in v][:n] if v else None for k, v in sides.items()},
                            "against_cpu_reference": {k: pair_difference(v, cpu["steps"]) if v else None for k, v in sides.items()},
                            "gpu_prompt_pass_against_llama": pair_difference(traced[tw]["steps"][:n], sides["llama.cpp"]),
                            "gpu_decode_fed_against_llama": pair_difference(fed["steps"], sides["llama.cpp"]) if fed else None}

    tps = [g["decode_tok_s"] for r in reps for g in r.values() if g.get("decode_tok_s", 0) > 0]
    first = {p: [round(r[p]["first_token_s"], 3) for r in reps if p in r] for p in ps}
    prompt_tokens = {p: len(reps[0][p]["prompt_ids"]) for p in ps if reps and p in reps[0]}
    cmp["speed"] = {"decode_tok_s_median": round(statistics.median(tps), 2) if tps else None,
                    "decode_tok_s_range": [round(min(tps), 2), round(max(tps), 2)] if tps else None,
                    "first_token_s": first, "prompt_tokens": prompt_tokens,
                    "prompt_tok_s": {p: [round(prompt_tokens[p] / s, 1) for s in first[p] if s > 0] for p in prompt_tokens},
                    "llama_decode_tok_s": {p: v.get("decode_tok_s") for p, v in llama.items()},
                    "llama_prompt_tok_s": {p: v.get("prompt_tok_s") for p, v in llama.items()},
                    "dense_decode_tok_s": [round(g["decode_tok_s"], 2) for g in dense.values()]}
    loads = {label: next(iter(events_of(r, "loaded")), {}) for label, r in runs.items()}
    cmp["device"] = {label: {k: e.get(k) for k in ("device_bytes", "weight_resident_bytes", "weight_regions", "load_s", "intermediate")}
                     for label, e in loads.items()}
    cmp["failures"], cmp["verdict"] = failures, not failures
    return cmp


# --- summary ----------------------------------------------------------------------------------------------------
def write_summary(ctx, results):
    cfg, name = ctx["cfg"], ctx["recipe"].name
    ok = all(results.get(s, {}).get("pass") for s in STAGES)
    doc = {"recipe": name, "host": socket.gethostname(), "written_utc": utc(), "stages": results, "pass": ok,
           "stages_run": [s for s in STAGES if s in results]}
    with open(os.path.join(ctx["res"], "summary.json"), "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    missing = [s for s in STAGES if s not in results]
    verdict = "PASSED" if ok else ("NOT COMPLETE: " + ", ".join(missing) + " not run" if missing and all(results[s].get("pass") for s in results)
                                   else "FAILED")
    L = [f"# Recipe gate: {name}", "", f"Host {socket.gethostname()}, written {utc()}. **{verdict}**.", "",
         f"Model: `{cfg['model']['gguf']}` (general.architecture {cfg['model']['architecture']}). Reference: llama.cpp "
         f"{results.get('llama', {}).get('source_head', '?')[:10]} on the CPU, the same prompt ids.", "",
         "| stage | result | when (UTC) | what |", "|---|---|---|---|"]
    g = results.get("gpu", {})
    for s in STAGES:
        r = results.get(s)
        if r is None:
            L.append(f"| {s} | not run | | |")
            continue
        what = {"check": lambda: f"{r.get('summary', '')}; catalogue: {r.get('catalogue_summary', '')}",
                "origin": lambda: f"{r.get('identical')}/{r.get('files')} catalogue files byte-identical to `git archive {r.get('commit')}`, "
                                  f"{r.get('problems')} problem(s); {len(r.get('adapted', {}))} adapted files",
                "build": lambda: f"{r.get('modules')} modules, {r.get('files_in_tree')} files in the tree; {r.get('origin_files_identical')}/"
                                 f"{r.get('origin_files')} origin files equal the export's; {r.get('compiled_total')} files compiled in {r.get('build_s')} s",
                "cpu": lambda: f"prompt tokens {r.get('prompt_tokens')}; model {r.get('model')}",
                "llama": lambda: r.get("error") or f"{len(r.get('runs', {}))} prompts; dense baseline {r.get('dense')}; eval callback {r.get('trace')}",
                "gpu": lambda: r.get("error") or (f"{len(r.get('failures', []))} failure(s) of [criteria]")}[s]()
        L.append(f"| {s} | {'PASS' if r.get('pass') else 'FAIL'} | {r.get('when_utc', '')} | {what} |")
    o, b, c = results.get("origin", {}), results.get("build", {}), results.get("cpu", {})
    if o.get("adapted"):
        L += ["", "## The adapted files", "", "Copies of the origin's files with the MoE path added, in their own module; the origin's stay as they are.", "",
              "| file | origin lines | lines added | lines removed |", "|---|---|---|---|"]
        L += [f"| `{k}` | {v.get('origin_lines')} | {v.get('added')} | {v.get('removed')} |" for k, v in sorted(o["adapted"].items())]
    if b.get("compiled"):
        L += ["", "## What was assembled", "",
              f"{b['files_in_tree']} files of {b['modules']} modules; {len(b.get('files_written_here', []))} are of origin `flerp`, the other "
              f"{b['origin_files']} are the origin's. {b['compiled_total']} files compiled:", "", "| module | files compiled |", "|---|---|"]
        L += [f"| `{m}` | {n} |" for m, n in b["compiled"].items()]
        L += ["", f"Binary: sha256 `{b['binary']['sha256'][:16]}...`, {b['binary']['bytes']} bytes."]
    if g.get("failures") is not None:
        crit = cfg["criteria"]
        L += ["", "## Verdict of the gpu stage", "",
              ("Every criterion held." if not g["failures"] else "Criteria that did not hold:")] + [f"- {x}" for x in g["failures"]]
        L += ["", "## 1. The dense model through this binary (regression)", "",
              "`" + cfg["model"]["dense_gguf"] + "`, the dense recipe's prompt ids, against its recorded GPU run (`"
              + cfg["model"]["dense_results"] + "/gpu-runs.json`, assembled, repeat 1).", "", "| prompt | tokens | generated ids |", "|---|---|---|"]
        for p, a in g.get("dense_regression", {}).items():
            L.append(f"| {p} | {a.get('lengths')} | {'equal' if a.get('equal') else 'DIFFER at ' + str(a.get('first_difference'))} |")
        L += ["", "## 2. The MoE model against itself", "", "| prompt | tokens | repeats equal | traced run equal | argmax of the copied logits is the token |", "|---|---|---|---|---|"]
        for p, a in g.get("self_consistent", {}).items():
            L.append(f"| {p} | {a['tokens']} | {a['repeats_equal']} | {a['traced_equal']} | {a['argmax_is_token']} |")
        L += ["", "## 3. Against llama.cpp on the same prompt ids", "",
              f"Log-probabilities are compared over the tokens of llama.cpp's top {cfg['llama'].get('n_probs', 20)} that are among this side's "
              f"{cfg['run']['trace_top']} largest logits, in nats. Tolerance at the first generated position {crit['logprob_tolerance']}; a tie is a "
              f"llama.cpp margin below {crit['tie_margin']}.", "",
              "| prompt | tokens this / llama | ids agree for | first position: max / mean abs difference (tokens) | at the first difference: llama margin, its two best, this side's token | max over compared positions (median) | result |",
              "|---|---|---|---|---|---|---|"]
        for p, a in g.get("llama", {}).items():
            fp, ad = a.get("first_position", {}), a.get("at_difference")
            L.append(f"| {p} | {a['tokens'][0]} / {a['tokens'][1]} | {a['agree']}{'' if a['first_difference'] is not None else ' (all)'} | "
                     f"{fp.get('max_abs')} / {fp.get('mean_abs')} ({fp.get('compared')}) | "
                     + ("none" if a["first_difference"] is None else f"{ad and ad.get('llama_margin')}, {ad and ad.get('llama_top2')}, {ad and ad.get('token')}")
                     + f" | {a.get('max_abs_over_compared_positions')} ({a.get('median_max_abs')}) | {'pass' if a['pass'] else 'FAIL'} |")
        if g.get("dense_baseline"):
            L += ["", "The same comparison on the dense model, through the origin's unmodified feed forward (the baseline; not a criterion):", "",
                  "| prompt | ids agree for | first position: max / mean abs difference | at the first difference: llama margin | max over compared positions (median) |",
                  "|---|---|---|---|---|"]
            for p, a in g["dense_baseline"].items():
                fp, ad = a.get("first_position") or {}, a.get("at_difference")
                L.append(f"| {p} | {a['agree']}{'' if a['first_difference'] is not None else ' (all)'} | {fp.get('max_abs')} / {fp.get('mean_abs')} | "
                         f"{'none' if a['first_difference'] is None else (ad and ad.get('llama_margin'))} | {a.get('max_abs_over_compared_positions')} ({a.get('median_max_abs')}) |")
        cmp = load_json(ctx, "compare.json")
        if cmp.get("hidden"):
            L += ["", "Hidden states after a block (llama.cpp's `l_out-<block>` as its eval callback prints it: four decimals, the first and last "
                  "three values of the first and last three tokens it evaluated; the tokens both sides show are compared, and the sums where it "
                  "evaluated the whole prompt):", "",
                  "| prompt | block | values compared | max abs difference | mean abs difference | mean abs value | sum this / llama |", "|---|---|---|---|---|---|---|"]
            for p, rows in cmp["hidden"].items():
                L += [f"| {p} | {r['layer']} | {r.get('values_compared', r.get('error'))} | {r.get('max_abs')} | {r.get('mean_abs')} | {r.get('mean_abs_value')} | {r.get('sum')} |" for r in rows]
        if cmp.get("router"):
            L += ["", "Experts chosen (llama.cpp's `ffn_moe_topk-<block>`: the first and last three of a token's experts in selection order, "
                  "for the first and last three tokens):", "",
                  "| prompt | block | tokens compared | same experts in the same order | same experts | slots equal |", "|---|---|---|---|---|---|"]
            for p, rows in cmp["router"].items():
                L += [f"| {p} | {r['layer']} | {r.get('tokens_compared', r.get('error'))} | {r.get('same_order')} | {r.get('same_set')} | {r.get('slots_equal')} / {r.get('slots')} |" for r in rows]
        tw = g.get("three_way") or {}
        if tw:
            L += ["", f"Three ways on `{tw['prompt']}`, the first {tw['positions']} generated positions, against the executor's CPU reference "
                  "(FP32 activations; its top 20 tokens a position): where activations are quantized. The decode step computes in FP32 like the "
                  "CPU reference; the origin's prompt chunks quantize activations for the attention and DeltaNet projections, and llama.cpp's "
                  "CPU kernels quantize them for every K-quant projection.", "",
                  "| against the CPU reference | tokens | positions compared | max abs difference | mean abs difference | largest on the best token |", "|---|---|---|---|---|---|"]
            for k, v in tw["against_cpu_reference"].items():
                if v:
                    L.append(f"| {k} | {tw['ids'][k]} | {v.get('positions')} | {v.get('max_abs')} | {v.get('mean_abs')} | {v.get('top1_max_abs')} |")
            L += ["", f"CPU reference tokens {tw['cpu_reference_ids']}. This executor against llama.cpp over the same positions (llama.cpp's top 20): "
                  f"prompt pass {tw.get('gpu_prompt_pass_against_llama')}; prompt through the decode step {tw.get('gpu_decode_fed_against_llama')}."]
        sp = g.get("speed", {})
        L += ["", "## Speed and memory (a record, nothing was tuned)", "",
              f"Decode: median {sp.get('decode_tok_s_median')} tok/s over prompts and repeats, range {sp.get('decode_tok_s_range')}; llama.cpp on "
              f"{cfg['llama'].get('threads', 8)} CPU threads: {sp.get('llama_decode_tok_s')}.",
              f"Prompt pass (time to the first token, s): {sp.get('first_token_s')} for {sp.get('prompt_tokens')} prompt tokens, so "
              f"{sp.get('prompt_tok_s')} tok/s; llama.cpp: {sp.get('llama_prompt_tok_s')} tok/s.",
              f"The dense model through this binary: {sp.get('dense_decode_tok_s')} tok/s.",
              f"Load time and device memory the executor reports: {g.get('device')}.",
              f"GPU holds: {g.get('holds')}; waits: {g.get('yields')}."]
    if c.get("cpu_reference"):
        cr = c["cpu_reference"]
        L += ["", "## Without the GPU: the executor's CPU reference", "",
              f"{cr.get('tokens_asked')} tokens asked on {cfg['run'].get('cpu_threads', 8)} threads, exit {cr.get('rc')}, {cr.get('seconds')} s"
              + (" (timed out)" if cr.get("timed_out") else "") + ". " + "; ".join(f"{k}: ids {v['ids']}, text {v['text']!r}" for k, v in cr.get("runs", {}).items()) + "."]
        lj = load_json(ctx, "llama.json")
        runs = load_json(ctx, "gpu-runs.json")
        for k, v in cr.get("runs", {}).items():
            gp = next((e["ids"] for e in events_of(runs.get("moe-1", {}), "generated") if e.get("source", "").endswith(k + ".ids")), None)
            L.append(f"{k}: llama.cpp's first {len(v['ids'])} ids {lj.get(k, {}).get('ids', [])[:len(v['ids'])]}; the GPU run's "
                     f"{gp[:len(v['ids'])] if gp else 'not run'}.")
    fid = results.get("fidelity")
    if fid:
        L += ["", "## Fidelity stages (separate from the verdict above)", "",
              f"`--stages reference,fidelity`: each engine against a full-precision reference, criteria in `[fidelity]`. "
              f"{'Passed' if fid.get('pass') else 'Failed'} on {fid.get('when_utc')}; see `fidelity.md`."]
    L += ["", "## Commands", "", "```", "tools/pod-sync.sh",
          f"tools/pod-run.sh './flerp gate recipe {name} --stages check,origin,build,cpu,llama'",
          f"tools/pod-run.sh 'echo \"moe-gpu | ./flerp gate recipe {name} --stages gpu\" > build/jobs-moe.txt; "
          "nohup setsid tools/gpu-queue.sh build/jobs-moe.txt > /dev/null 2>&1 &'",
          f"tools/pod-pull.sh gate/results/{name}", "```", "",
          "Every command a stage ran is the `$` line above its output in the stage's evidence file."]
    with open(os.path.join(ctx["res"], "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--keep-going", action="store_true", help="run later stages after a failed one")
    ap.add_argument("--summarize", action="store_true", help="rewrite summary.md from the recorded summary.json; runs nothing")
    ap.add_argument("--rejudge", action="store_true", help="recompute the gpu stage's comparison from the recorded runs; runs nothing")
    a = ap.parse_args()
    cat = Catalog(ROOT)
    recipe = cat.recipe(a.recipe)
    if recipe is None:
        sys.exit(f"no recipe {a.recipe}")
    cfg = load_config(os.path.join(ROOT, recipe.gate["config"]))
    res = os.path.join(ROOT, "gate", "results", recipe.name)
    os.makedirs(res, exist_ok=True)

    def recipe_modules():
        from flerplib.resolve import Assembly
        return Assembly(cat, recipe).modules

    ctx = {"cat": cat, "recipe": recipe, "cfg": cfg, "res": res, "scratch": os.path.join(ROOT, cfg["host"]["scratch"]),
           "recipe_modules": recipe_modules}
    os.makedirs(ctx["scratch"], exist_ok=True)
    results, prior = {}, os.path.join(res, "summary.json")
    if os.path.isfile(prior):                 # a partial run keeps the other stages' results: each says when it ran
        with open(prior) as f:
            results = json.load(f).get("stages", {})
    ctx["results"] = results
    if a.summarize:
        return 0 if write_summary(ctx, results) else 1
    if a.rejudge:
        runs, ps = load_json(ctx, "gpu-runs.json"), [p[0] for p in G.prompts(ctx)]
        with open(os.path.join(ROOT, cfg["model"]["dense_results"], "gpu-runs.json")) as f:
            dense_recorded = {ps[i]: g["ids"] for i, g in enumerate(json.load(f)["assembled"][0]["generated"])}
        cmp = judge(ctx, runs, ps, dense_recorded)
        with open(os.path.join(res, "compare.json"), "w") as f:
            json.dump(cmp, f, indent=1)
            f.write("\n")
        results.setdefault("gpu", {}).update({k: cmp[k] for k in ("dense_regression", "self_consistent", "llama", "dense_baseline", "three_way",
                                                                 "speed", "device", "verdict", "failures")})
        results["gpu"]["pass"] = cmp["verdict"]
        return 0 if write_summary(ctx, results) else 1
    fns = {"check": stage_check, "origin": stage_origin, "build": G.stage_build, "cpu": stage_cpu, "llama": stage_llama, "gpu": stage_gpu,
           "reference": stage_reference, "fidelity": stage_fidelity}
    want = a.stages.split(",")
    for s in STAGES + FIDELITY_STAGES:
        if s not in want:
            continue
        print(f"== {s}", flush=True)
        results[s] = fns[s](ctx)
        results[s]["when_utc"] = utc()
        print(f"   {'PASS' if results[s].get('pass') else 'FAIL'}", flush=True)
        write_summary(ctx, results)
        if not results[s].get("pass") and not a.keep_going:
            break
    return 0 if write_summary(ctx, results) and all(results.get(s, {}).get("pass") for s in want) else 1


if __name__ == "__main__":
    sys.exit(main())
