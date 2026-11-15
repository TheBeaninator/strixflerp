#!/usr/bin/env python3
"""Recipe gate for a gufo-origin recipe built by the origin's CMake files (SPEC.md section 9, phase 5 step 1).
Run on the gfx1151 build host:

    ./flerp gate recipe qwen35-decider-2b.gufo [--stages check,origin,build,reference,cpu,llama,gpu] [--keep-going]

Stages, each writing evidence under gate/results/<recipe>/:
  check      `flerp check --origin gufo` has no errors (the whole catalogue's result is recorded too).
  origin     every vendored file of the origin is byte-identical to `git archive <commit>` of the reference checkout,
             and the checkout is covered.
  build      `flerp assemble`, every origin file of the assembled tree is the export's file at the same path, then
             the tree builds with the origin's CMake files; the files the build compiled are listed by module.
  reference  the same program built from the export of the origin (the whole checkout, with the recipe's build
             file copied in): no build of this commit exists on the host and the origin's root does not configure
             there. The two builds must compile the same source list.
  cpu        no GPU: the prompts' token ids from the engine's tokenizer; the executor asked to open a qwen35moe
             file (what it answers is recorded, not judged); gufo's own CPU reference on the first prompts.
  llama      no GPU: llama.cpp built for the CPU from the ark's source, the same prompt ids, greedy. Recorded.
  gpu        the reference binary and the assembled one generate every prompt greedily on the GPU, one process at a
             time, `repeats` times. Pass: the prompt ids and the generated ids are equal, prompt for prompt and
             repeat for repeat, and decode speed is within the configured tolerance. Then, as a record and not a
             criterion: where the tokens of llama.cpp and of gufo's CPU reference first leave the GPU's.

The GPU is taken only when it is idle and no job of the pod waits for it (gate/strixite_recipe.py, Hold), and only
for the gpu stage. Builds and CPU runs are niced and raise their own OOM score.
"""
import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gate"))
from flerplib import origin as origin_mod  # noqa: E402
from flerplib.catalog import Catalog  # noqa: E402
from strixite_recipe import Hold, die_with_parent, load_config, long_text, sh, sha256, tail, utc  # noqa: E402

STAGES = ("check", "origin", "build", "reference", "cpu", "llama", "gpu")
NO_GPU = {"HIP_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1"}      # for the runs that must not touch it


def polite(cfg):
    """Command prefix for builds and CPU runs: lowest priority, and the first process the kernel kills when the
    host runs out of memory."""
    pre = ["nice", "-n", "19"]
    adj = cfg["host"].get("oom_score_adj")
    if adj is not None and shutil.which("choom"):
        pre = ["choom", "-n", str(adj), "--"] + pre
    return pre


def first_difference(a, b):
    """Index of the first position where two id lists differ; None when one is a prefix of the other."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def agreement(a, b):
    d = first_difference(a, b)
    return {"first_difference": d, "compared": min(len(a), len(b)) if d is None else d, "lengths": [len(a), len(b)],
            "equal": d is None and len(a) == len(b)}


def prompts(ctx):
    """[(name, text)] and the files they are written to (scratch/prompts/<name>.txt)."""
    cfg, out = ctx["cfg"], []
    d = os.path.join(ctx["scratch"], "prompts")
    os.makedirs(d, exist_ok=True)
    for p in cfg["prompt"]:
        text = (long_text(cfg["run"]["long_prompt_sentences"]) if p.get("long") else "") + p["text"]
        path = os.path.join(d, p["name"] + ".txt")
        with open(path, "w") as f:
            f.write(text)
        out.append((p["name"], text, path))
    return out


def runtime_env(ctx):
    """The ROCm libraries the binaries load (hipBLAS, hipBLASLt, rocBLAS, the HIP runtime, libomp): the build tree's
    binaries carry no run path to them, as the origin's do not (its Nix package wraps `gufo` for this)."""
    rocm = ctx["cat"].toolchains[ctx["recipe"].toolchain]["rocm_root"]
    path = f"{rocm}/lib:{rocm}/lib/llvm/lib"
    return {"LD_LIBRARY_PATH": path + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")}


def generate(binary, args, log, timeout, env=None, prefix=()):
    """Runs the generator; returns (rc, [json objects it printed], seconds). stderr goes to `log`."""
    cmd = list(prefix) + [binary] + args
    t0 = time.time()
    with open(log, "ab") as f:
        f.write(("$ " + " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted((env or {}).items())) + " "
                 + " ".join(shlex.quote(c) for c in cmd) + "\n").encode())
        f.flush()
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=f, timeout=timeout, preexec_fn=die_with_parent,
                               env=dict(os.environ, **(env or {})))
            rc, text = p.returncode, p.stdout.decode(errors="replace")
        except subprocess.TimeoutExpired as e:
            rc, text = 124, (e.stdout or b"").decode(errors="replace")
            f.write(b"\nTIMEOUT\n")
        f.write(text.encode())
    events = []
    for line in text.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    return rc, events, round(time.time() - t0, 1)


def export_origin(ctx, log):
    """A clean export of the origin at ORIGIN.toml's commit (the pod tree is a shared checkout)."""
    h = ctx["cfg"]["host"]
    commit = ctx["cat"].origin_data(h["origin"])["commit"]
    export = os.path.join(ctx["scratch"], f"origin-{commit}")
    if not os.path.isfile(os.path.join(export, ".flerp-export")):
        sh(f"rm -rf {shlex.quote(export)} && mkdir -p {shlex.quote(export)} && "
           f"git -C {shlex.quote(h['reference_checkout'])} archive {commit} | tar -x -C {shlex.quote(export)} && "
           f"touch {shlex.quote(export)}/.flerp-export", log)
    return export, commit


def openssl_options(ctx):
    h = ctx["cfg"]["host"]
    if not h.get("openssl_headers"):
        return []
    shim = os.path.join(ctx["scratch"], "openssl-include")
    os.makedirs(shim, exist_ok=True)
    link = os.path.join(shim, "openssl")
    if os.path.islink(link):
        os.remove(link)
    os.symlink(h["openssl_headers"], link)
    return [f"-DFLERP_OPENSSL_INCLUDE_DIR={shim}", f"-DFLERP_OPENSSL_CRYPTO_LIBRARY={h['openssl_crypto']}"]


def assembled_binary(ctx):
    """The recipe's one program, where build.sh links it."""
    return os.path.join(ROOT, "build", ctx["recipe"].name, next(iter(ctx["recipe"].binaries)))


def compiled_sources(build_dir, tree):
    """Tree-relative source files a CMake build directory compiled (compile_commands.json)."""
    with open(os.path.join(build_dir, "compile_commands.json")) as f:
        return sorted(os.path.relpath(os.path.realpath(e["file"]), os.path.realpath(tree)) for e in json.load(f))


# --- stages ---------------------------------------------------------------------------------------------------
def stage_check(ctx):
    log = os.path.join(ctx["res"], "check.txt")
    open(log, "w").close()
    rc, _ = sh([sys.executable, "flerp", "check", "--origin", ctx["cfg"]["host"]["origin"], "--warnings"], log)
    scoped = tail(log, 1)[0]
    full = os.path.join(ctx["res"], "check-catalogue.txt")
    open(full, "w").close()
    rc_all, _ = sh([sys.executable, "flerp", "check"], full)
    return {"pass": rc == 0, "rc": rc, "summary": scoped, "catalogue_rc": rc_all, "catalogue_summary": tail(full, 1)[0],
            "evidence": "check.txt, check-catalogue.txt"}


def stage_origin(ctx):
    h = ctx["cfg"]["host"]
    log = os.path.join(ctx["res"], "diff-origin.txt")
    open(log, "w").close()
    shutil.rmtree(os.path.join(ctx["scratch"], "origin-" + ctx["cat"].origin_data(h["origin"])["commit"]), ignore_errors=True)
    export, commit = export_origin(ctx, log)
    head = subprocess.run(["git", "-C", h["reference_checkout"], "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = origin_mod.diff_origin(ctx["cat"], h["origin"], export)
    res["problems"] = [p for p in res["problems"] if p[1] != ".flerp-export"]
    with open(os.path.join(ctx["res"], "diff-origin.json"), "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    with open(log, "a") as f:
        for mid, path, why in res["problems"]:
            f.write(f"DIFF {mid} {path}: {why}\n")
        f.write(f"diff-origin origin {h['origin']} vs git archive {commit} of {h['reference_checkout']}: {res['modules']} "
                f"module(s), {res['identical']}/{res['files']} files byte-identical, {len(res['problems'])} problem(s), "
                f"{res['unharvested']} origin files listed as unharvested\n")
    return {"pass": not res["problems"] and res["files"] > 0 and head.startswith(commit), "checkout": h["reference_checkout"],
            "commit": commit, "head": head, "modules": res["modules"], "files": res["files"], "identical": res["identical"],
            "problems": len(res["problems"]), "unharvested": res["unharvested"], "evidence": "diff-origin.txt"}


def stage_build(ctx):
    cfg, name = ctx["cfg"], ctx["recipe"].name
    log = os.path.join(ctx["res"], "build.txt")
    open(log, "w").close()
    tree_root = os.path.join(ROOT, "build", name)
    out = {"evidence": "build.txt, assembly.json, compiled-files.json", "tree": os.path.relpath(tree_root, ROOT)}
    rc, _ = sh([sys.executable, "flerp", "assemble", name], log)
    if rc != 0:
        return dict(out, rc=rc)
    with open(os.path.join(tree_root, "assembly.json")) as f:
        asm = json.load(f)
    shutil.copyfile(os.path.join(tree_root, "assembly.json"), os.path.join(ctx["res"], "assembly.json"))
    # the assembled tree against the export of the origin, file by file
    export, commit = export_origin(ctx, log)
    own = [p for p, r in asm["files"].items() if not r["module"].split(".")[1] == cfg["host"]["origin"]]
    theirs = [p for p in asm["files"] if p not in own]
    differ = [p for p in theirs if not os.path.isfile(os.path.join(export, p))
              or sha256(os.path.join(export, p)) != asm["files"][p]["sha256"]]
    out.update(files_in_tree=len(asm["files"]), origin_files=len(theirs), origin_files_identical=len(theirs) - len(differ),
               origin_files_differ=differ[:20], files_written_here=sorted(own), modules=len(asm["modules"]), commit=commit)
    cmd = polite(cfg) + [os.path.join(tree_root, "build.sh"), f"-j{cfg['host'].get('build_jobs', 2)}"] + openssl_options(ctx)
    rc, out["build_s"] = sh(cmd, log, timeout=7200)
    out["rc"] = rc
    binary = assembled_binary(ctx)
    if rc == 0 and os.path.isfile(binary):
        out["binary"] = {"path": os.path.relpath(binary, ROOT), "sha256": sha256(binary), "bytes": os.path.getsize(binary)}
        compiled = compiled_sources(os.path.join(tree_root, "out"), os.path.join(tree_root, "tree"))
        by_module = {}
        for p in compiled:
            by_module.setdefault(asm["files"].get(p, {}).get("module", "?"), []).append(p)
        with open(os.path.join(ctx["res"], "compiled-files.json"), "w") as f:
            json.dump(by_module, f, indent=1, sort_keys=True)
            f.write("\n")
        out["compiled"] = {m: len(v) for m, v in sorted(by_module.items())}
        out["compiled_total"] = len(compiled)
        out["modules_not_compiled"] = sorted(m for m in asm["modules"] if m not in by_module)
    out["pass"] = rc == 0 and "binary" in out and not differ and "?" not in out.get("compiled", {"?": 1})
    return out


def stage_reference(ctx):
    cfg, name = ctx["cfg"], ctx["recipe"].name
    log = os.path.join(ctx["res"], "reference-build.txt")
    open(log, "w").close()
    tree_root = os.path.join(ROOT, "build", name)
    with open(os.path.join(tree_root, "assembly.json")) as f:
        asm = json.load(f)
    export, commit = export_origin(ctx, log)
    out = {"evidence": "reference-build.txt", "export": os.path.relpath(export, ROOT), "commit": commit}
    copied = []
    for p, r in asm["files"].items():                 # the recipe's own files: the build file and the program
        if r["module"].split(".")[1] != cfg["host"]["origin"]:
            dst = os.path.join(export, p)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(tree_root, "tree", p), dst)
            copied.append(p)
    out["copied_into_export"] = sorted(copied)
    rocm = ctx["cat"].toolchains[ctx["recipe"].toolchain]["rocm_root"]
    options = [o.replace("${FLERP_ROCM_ROOT}", rocm) for o in asm["configure_options"]] + openssl_options(ctx)
    bdir = os.path.join(ctx["scratch"], "origin-build")
    rc, _ = sh(polite(cfg) + ["cmake", "-G", "Ninja", "-S", os.path.join(export, asm["source_dir"]), "-B", bdir] + options, log, timeout=600)
    if rc == 0:
        rc, out["build_s"] = sh(polite(cfg) + ["ninja", "-C", bdir, f"-j{cfg['host'].get('build_jobs', 2)}"] + asm["targets"], log, timeout=7200)
    out["rc"] = rc
    installed = asm["binaries"][next(iter(ctx["recipe"].binaries))]["installed"]
    binary = os.path.join(bdir, installed)
    if rc == 0 and os.path.isfile(binary):
        out["binary"] = {"path": os.path.relpath(binary, ROOT), "sha256": sha256(binary), "bytes": os.path.getsize(binary)}
        ref = compiled_sources(bdir, export)
        mine = compiled_sources(os.path.join(tree_root, "out"), os.path.join(tree_root, "tree"))
        out["compiled_total"] = len(ref)
        out["same_source_list"] = ref == mine
        out["source_list_differences"] = sorted(set(ref) ^ set(mine))[:20]
    out["pass"] = rc == 0 and "binary" in out and out.get("same_source_list", False)
    return out


def stage_cpu(ctx):
    cfg, name = ctx["cfg"], ctx["recipe"].name
    log = os.path.join(ctx["res"], "cpu.txt")
    open(log, "w").close()
    binary = assembled_binary(ctx)
    ps = prompts(ctx)
    files = [x for _, _, path in ps for x in ("--prompt-file", path)]
    out = {"evidence": "cpu.txt, prompt-ids.json"}
    rc, events, _ = generate(binary, ["--model", cfg["model"]["gguf"], "--tokenize"] + files, log, 300, dict(NO_GPU, **runtime_env(ctx)), polite(cfg))
    toks = [e for e in events if e.get("event") == "tokenized"]
    loaded = next((e for e in events if e.get("event") == "loaded"), {})
    out["model"] = {k: loaded.get(k) for k in ("architecture", "layers", "hidden", "intermediate", "heads", "kv_heads", "vocab", "mtp_layers")}
    ids = {ps[e["prompt"]][0]: e["prompt_ids"] for e in toks}
    with open(os.path.join(ctx["res"], "prompt-ids.json"), "w") as f:
        json.dump(ids, f)
        f.write("\n")
    out["prompt_tokens"] = {k: len(v) for k, v in ids.items()}
    out["tokenize_rc"] = rc
    ok = rc == 0 and len(ids) == len(ps) and loaded.get("architecture") == cfg["model"]["architecture"]

    probe = cfg["model"].get("moe_probe_gguf")
    if probe:
        plog = os.path.join(ctx["res"], "moe-probe.txt")
        open(plog, "w").close()
        res = {}
        for label, extra, env in (("cpu-reference", ["--cpu", "--tokenize"], NO_GPU),):
            rc2, ev2, _ = generate(binary, ["--model", probe] + extra + files[:2], plog, 600, dict(env, **runtime_env(ctx)), polite(cfg))
            res[label] = {"rc": rc2, "opened": any(e.get("event") == "loaded" for e in ev2),
                          "message": next((ln for ln in reversed(tail(plog, 6)) if ln.startswith("error:")), "")}
        out["moe_probe"] = {"gguf": probe, "evidence": "moe-probe.txt", **res}

    n, k = cfg["run"].get("cpu_tokens", 0), cfg["run"].get("cpu_prompts", 0)
    if ok and n and k:
        d = os.path.join(ctx["scratch"], "prompts")
        idfiles = []
        for pname, _, _ in ps[:k]:
            path = os.path.join(d, pname + ".ids")
            with open(path, "w") as f:
                f.write(" ".join(str(i) for i in ids[pname]))
            idfiles += ["--ids-file", path]
        env = dict(NO_GPU, OMP_NUM_THREADS=str(cfg["run"].get("cpu_threads", 8)), **runtime_env(ctx))
        rc3, ev3, secs = generate(binary, ["--model", cfg["model"]["gguf"], "--cpu", "--max-tokens", str(n)] + idfiles, log,
                                  cfg["run"].get("cpu_timeout_s", 900), env, polite(cfg))
        gen = [e for e in ev3 if e.get("event") == "generated"]
        out["cpu_reference"] = {"rc": rc3, "seconds": secs, "tokens_asked": n, "timed_out": rc3 == 124,
                                "runs": {ps[e["prompt"]][0]: {"ids": e["ids"], "text": e["text"], "decode_tok_s": e["decode_tok_s"]} for e in gen}}
    out["pass"] = ok
    return out


class LlamaServer:
    def __init__(self, cmd, port, log):
        self.port, self.log = port, open(log, "ab")
        self.log.write(("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n").encode())
        self.log.flush()
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, preexec_fn=die_with_parent,
                                     env=dict(os.environ, **NO_GPU))

    def call(self, path, body, timeout):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def wait_ready(self, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False
            try:
                if self.call("/health", None, 5).get("status") == "ok":
                    return True
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(2)
        return False

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


def stage_llama(ctx):
    cfg, lc = ctx["cfg"], ctx["cfg"]["llama"]
    log = os.path.join(ctx["res"], "llama.txt")
    open(log, "w").close()
    out = {"evidence": "llama.txt, llama.json", "source": lc["source"]}
    out["source_head"] = subprocess.run(["git", "-C", lc["source"], "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(os.path.join(ctx["res"], "prompt-ids.json")) as f:
        ids = json.load(f)
    bdir = os.path.join(ctx["scratch"], "llama-build")
    server = os.path.join(bdir, "bin", "llama-server")
    if not os.path.isfile(server):
        rc, _ = sh(polite(cfg) + ["cmake", "-G", "Ninja", "-S", lc["source"], "-B", bdir, "-DCMAKE_BUILD_TYPE=Release",
                                  "-DBUILD_SHARED_LIBS=OFF", "-DGGML_NATIVE=ON", "-DLLAMA_OPENSSL=OFF", "-DLLAMA_BUILD_UI=OFF",
                                  "-DLLAMA_USE_PREBUILT_UI=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF"], log, timeout=600)
        if rc == 0:
            rc, out["build_s"] = sh(polite(cfg) + ["ninja", "-C", bdir, f"-j{cfg['host'].get('build_jobs', 2)}", "llama-server"],
                                    log, timeout=lc.get("build_timeout_s", 3600))
        if rc != 0 or not os.path.isfile(server):
            return dict(out, rc=rc, error="llama.cpp did not build")
    out["binary_sha256"] = sha256(server)
    cmd = polite(cfg) + [server, "-m", cfg["model"]["gguf"], "--host", "127.0.0.1", "--port", str(lc["port"]), "-ngl", "0",
                         "-t", str(lc.get("threads", 8)), "-c", str(cfg["run"]["context"]), "--parallel", "1"]
    srv = LlamaServer(cmd, lc["port"], log)
    runs = {}
    try:
        if not srv.wait_ready(lc.get("startup_timeout_s", 600)):
            return dict(out, error="llama-server did not become ready", log_tail=tail(log, 8))
        for pname, text, _ in prompts(ctx):
            tok = srv.call("/tokenize", {"content": text, "add_special": False, "parse_special": True}, 120).get("tokens", [])
            t0 = time.time()
            r = srv.call("/completion", {"prompt": ids[pname], "n_predict": cfg["run"]["max_tokens"], "temperature": 0, "top_k": 1,
                                         "repeat_penalty": 1.0, "cache_prompt": False, "return_tokens": True, "seed": 1,
                                         "stream": False}, lc.get("request_timeout_s", 1800))
            runs[pname] = {"ids": r.get("tokens", []), "text": r.get("content", ""), "seconds": round(time.time() - t0, 1),
                           "decode_tok_s": (r.get("timings") or {}).get("predicted_per_second"),
                           "prompt_tokens_evaluated": r.get("tokens_evaluated"),
                           "stop": r.get("stop_type") or r.get("stopped_eos"),
                           "tokenizer_ids_equal_gufo": tok == ids[pname],
                           "tokenizer_first_difference": first_difference(tok, ids[pname]), "tokenizer_lengths": [len(tok), len(ids[pname])]}
    except (urllib.error.URLError, OSError, ValueError) as e:
        out["error"] = f"llama-server request failed: {e}"
    finally:
        srv.stop()
    with open(os.path.join(ctx["res"], "llama.json"), "w") as f:
        json.dump(runs, f, indent=1)
        f.write("\n")
    out["runs"] = {k: {x: v[x] for x in v if x not in ("ids", "text")} | {"tokens": len(v["ids"])} for k, v in runs.items()}
    out["pass"] = "error" not in out and len(runs) == len(ids) and all(v["ids"] for v in runs.values())
    return out


def stage_gpu(ctx):
    cfg, name = ctx["cfg"], ctx["recipe"].name
    log = os.path.join(ctx["res"], "gpu.txt")
    open(log, "w").close()
    sides = {"reference": os.path.join(ctx["scratch"], "origin-build", "flerp-gufo-qwen-generate"),
             "assembled": assembled_binary(ctx)}
    out = {"evidence": "gpu.txt, gpu-runs.json, lease.txt", "binaries": {k: {"sha256": sha256(v)} for k, v in sides.items() if os.path.isfile(v)}}
    if len(out["binaries"]) != 2:
        return dict(out, error="a binary is missing: run the build and reference stages first")
    ps = prompts(ctx)
    args = ["--model", cfg["model"]["gguf"], "--context", str(cfg["run"]["context"]), "--max-tokens", str(cfg["run"]["max_tokens"])]
    args += [x for _, _, path in ps for x in ("--prompt-file", path)]
    hold = Hold(cfg, os.path.join(ctx["res"], "lease.txt"), out, wait_s=cfg["host"].get("gpu_wait_s", 1800))
    runs = {k: [] for k in sides}
    try:
        hold.take(f"strixflerp gate {name}", "gpu stage")
        for rep in range(cfg["run"]["repeats"]):
            for side, binary in sides.items():
                hold.between(f"strixflerp gate {name}", f"{side} repeat {rep + 1}")
                rc, events, secs = generate(binary, args, log, cfg["run"]["run_timeout_s"], runtime_env(ctx))
                runs[side].append({"rc": rc, "seconds": secs, "loaded": next((e for e in events if e.get("event") == "loaded"), None),
                                   "generated": [e for e in events if e.get("event") == "generated"]})
    except RuntimeError as e:
        out["error"] = str(e)
        print(f"   {e}", flush=True)
    finally:
        hold.give()
    with open(os.path.join(ctx["res"], "gpu-runs.json"), "w") as f:
        json.dump(runs, f, indent=1)
        f.write("\n")
    if "error" in out:
        return out
    rows, ok = [], True
    for rep in range(cfg["run"]["repeats"]):
        for i, (pname, _, _) in enumerate(ps):
            r, a = (runs[s][rep]["generated"][i] if i < len(runs[s][rep]["generated"]) else None for s in ("reference", "assembled"))
            if r is None or a is None:
                rows.append({"prompt": pname, "repeat": rep + 1, "missing": True})
                ok = False
                continue
            row = {"prompt": pname, "repeat": rep + 1, "prompt_tokens": len(r["prompt_ids"]), "prompt_ids_equal": r["prompt_ids"] == a["prompt_ids"],
                   "tokens": [len(r["ids"]), len(a["ids"])], "ids": agreement(r["ids"], a["ids"]),
                   "decode_tok_s": [round(r["decode_tok_s"], 2), round(a["decode_tok_s"], 2)],
                   "first_token_s": [round(r["first_token_s"], 3), round(a["first_token_s"], 3)]}
            ok = ok and row["prompt_ids_equal"] and row["ids"]["equal"] and len(r["ids"]) > 0
            rows.append(row)
    out["comparisons"] = rows
    out["token_identity"] = ok and all(x["rc"] == 0 for s in runs for x in runs[s])
    # the reference against itself across repeats: what "identical" can mean on this engine
    out["self_consistent"] = {s: all(runs[s][0]["generated"][i]["ids"] == runs[s][rep]["generated"][i]["ids"]
                                     for rep in range(1, cfg["run"]["repeats"]) for i in range(len(ps))
                                     if i < len(runs[s][rep]["generated"]) and i < len(runs[s][0]["generated"])) for s in sides}
    tps = {s: [g["decode_tok_s"] for x in runs[s] for g in x["generated"] if g["decode_tok_s"] > 0] for s in sides}
    if all(tps.values()):
        med = {s: statistics.median(v) for s, v in tps.items()}
        tol = cfg["run"]["tps_tolerance"]
        inside = min(tps["reference"]) <= med["assembled"] <= max(tps["reference"])
        out["speed"] = {"median_decode_tok_s": {s: round(m, 2) for s, m in med.items()},
                        "reference_range": [round(min(tps["reference"]), 2), round(max(tps["reference"]), 2)],
                        "relative_difference": round(med["assembled"] / med["reference"] - 1, 4), "tolerance": tol,
                        "pass": abs(med["assembled"] / med["reference"] - 1) <= tol or inside}
    out["device"] = {s: {k: (runs[s][0]["loaded"] or {}).get(k) for k in ("device_bytes", "weight_resident_bytes", "weight_regions", "load_s")} for s in sides}
    # other engines on the same prompt ids: a record
    others = {}
    first = {ps[i][0]: g for i, g in enumerate(runs["assembled"][0]["generated"])}
    lj = os.path.join(ctx["res"], "llama.json")
    if os.path.isfile(lj):
        with open(lj) as f:
            others["llama.cpp (CPU)"] = {k: agreement(first[k]["ids"], v["ids"]) for k, v in json.load(f).items() if k in first}
    cpu = ctx["results"].get("cpu", {}).get("cpu_reference", {}).get("runs", {})
    if cpu:
        others["gufo CPU reference"] = {k: agreement(first[k]["ids"], v["ids"]) for k, v in cpu.items() if k in first}
    out["other_engines"] = others
    out["pass"] = out["token_identity"] and out.get("speed", {}).get("pass", False)
    return out


# --- summary --------------------------------------------------------------------------------------------------
def write_summary(ctx, results):
    cfg, name = ctx["cfg"], ctx["recipe"].name
    ok = all(results.get(s, {}).get("pass") for s in STAGES)
    doc = {"recipe": name, "host": socket.gethostname(), "written_utc": utc(), "stages": results, "pass": ok,
           "stages_run": [s for s in STAGES if s in results]}
    with open(os.path.join(ctx["res"], "summary.json"), "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    L = [f"# Recipe gate: {name}", ""]
    missing = [s for s in STAGES if s not in results]
    verdict = "PASSED" if ok else ("NOT COMPLETE: " + ", ".join(missing) + " not run" if missing and all(results[s].get("pass") for s in results)
                                   else "FAILED")
    L += [f"Host {socket.gethostname()}, written {utc()}. **{verdict}**.", "",
          f"Model: `{cfg['model']['gguf']}` (general.architecture {cfg['model']['architecture']}). Reference: gufo at "
          f"{results.get('origin', {}).get('commit', '?')}, built from a git export of `{cfg['host']['reference_checkout']}`.", "",
          "| stage | result | when (UTC) | what |", "|---|---|---|---|"]
    for s in STAGES:
        r = results.get(s)
        if r is None:
            L.append(f"| {s} | not run | | |")
            continue
        what = {"check": lambda: f"{r.get('summary', '')}; catalogue: {r.get('catalogue_summary', '')}",
                "origin": lambda: f"{r.get('identical')}/{r.get('files')} catalogue files byte-identical to `git archive {r.get('commit')}`, "
                                  f"{r.get('problems')} problem(s), {r.get('unharvested')} origin files listed as unharvested",
                "build": lambda: f"{r.get('modules')} modules, {r.get('files_in_tree')} files in the tree; {r.get('origin_files_identical')}/"
                                 f"{r.get('origin_files')} origin files equal the export's; {r.get('compiled_total')} files compiled in {r.get('build_s')} s",
                "reference": lambda: f"{r.get('compiled_total')} files compiled in {r.get('build_s')} s; same source list as the assembled build: "
                                     f"{r.get('same_source_list')}",
                "cpu": lambda: f"prompt tokens {r.get('prompt_tokens')}; model {r.get('model')}",
                "llama": lambda: r.get("error") or f"llama.cpp {r.get('source_head', '')[:10]}, {len(r.get('runs', {}))} prompts",
                "gpu": lambda: r.get("error") or f"token identity {r.get('token_identity')}; speed {r.get('speed', {}).get('median_decode_tok_s')}"}[s]()
        L.append(f"| {s} | {'PASS' if r.get('pass') else 'FAIL'} | {r.get('when_utc', '')} | {what} |")
    b, ref = results.get("build", {}), results.get("reference", {})
    if b.get("compiled"):
        L += ["", "## What was assembled", "",
              f"`./flerp assemble {name}` wrote {b['files_in_tree']} files of {b['modules']} modules to `{b['tree']}/tree`; "
              f"{len(b.get('files_written_here', []))} are written here ({', '.join('`' + p + '`' for p in b.get('files_written_here', []))}), "
              f"the other {b['origin_files']} are the origin's. The origin's CMake files compiled {b['compiled_total']} of them:", "",
              "| module | files compiled |", "|---|---|"]
        L += [f"| `{m}` | {n} |" for m, n in b["compiled"].items()]
        L += ["", "In the tree and not compiled (headers only, or code the Qwen 3.5 program does not build): "
              + ", ".join(f"`{m}`" for m in b.get("modules_not_compiled", [])) + ".", "",
              f"Assembled binary: sha256 `{b['binary']['sha256'][:16]}...`, {b['binary']['bytes']} bytes."]
        if ref.get("binary"):
            L.append(f"Reference binary: sha256 `{ref['binary']['sha256'][:16]}...`, {ref['binary']['bytes']} bytes (the build trees' paths "
                     "are in the debug information, so the bytes are not expected to be equal).")
    c = results.get("cpu", {})
    if c.get("moe_probe"):
        L += ["", "## The executor and a qwen35moe file", "", f"`{c['moe_probe']['gguf']}`:", ""]
        L += [f"- {k}: exit {v['rc']}, opened: {v['opened']}. `{v['message']}`" for k, v in c["moe_probe"].items() if isinstance(v, dict)]
    g = results.get("gpu", {})
    if g.get("comparisons"):
        L += ["", "## Tokens on the GPU: reference build against assembled build", "",
              "| prompt | repeat | prompt tokens | prompt ids equal | tokens ref / asm | generated ids | decode tok/s ref / asm | first token s ref / asm |",
              "|---|---|---|---|---|---|---|---|"]
        for r in g["comparisons"]:
            if r.get("missing"):
                L.append(f"| {r['prompt']} | {r['repeat']} | missing | | | | | |")
                continue
            ids = "equal" if r["ids"]["equal"] else (f"differ at {r['ids']['first_difference']}" if r["ids"]["first_difference"] is not None else "lengths differ")
            L.append(f"| {r['prompt']} | {r['repeat']} | {r['prompt_tokens']} | {r['prompt_ids_equal']} | {r['tokens'][0]} / {r['tokens'][1]} | {ids} | "
                     f"{r['decode_tok_s'][0]} / {r['decode_tok_s'][1]} | {r['first_token_s'][0]} / {r['first_token_s'][1]} |")
        sp = g.get("speed", {})
        L += ["", f"Each side equal to itself across repeats: {g.get('self_consistent')}.",
              f"Decode speed, median over prompts and repeats: {sp.get('median_decode_tok_s')} tok/s, relative difference "
              f"{sp.get('relative_difference')}, tolerance {sp.get('tolerance')}, reference range {sp.get('reference_range')}: "
              f"{'pass' if sp.get('pass') else 'FAIL'}. One process a run, {cfg['run']['max_tokens']} tokens a prompt: a record, not a benchmark.",
              f"Device memory the executor reports: {g.get('device')}.",
              f"GPU holds: {g.get('holds')}; waits: {g.get('yields')}."]
        if g.get("other_engines"):
            L += ["", "## Other engines on the same prompt ids (a record, not a criterion)", "",
                  "First position where the greedy tokens leave the assembled GPU run's (repeat 1).", "",
                  "| engine | prompt | tokens compared | first difference |", "|---|---|---|---|"]
            for eng, per in g["other_engines"].items():
                for pname, a in per.items():
                    L.append(f"| {eng} | {pname} | {min(a['lengths'])} | {'none' if a['first_difference'] is None else a['first_difference']} |")
    ll = results.get("llama", {})
    cpu_runs = c.get("cpu_reference", {}).get("runs", {})
    lj = os.path.join(ctx["res"], "llama.json")
    if cpu_runs or c.get("cpu_reference"):
        cr = c["cpu_reference"]
        L += ["", "## Without the GPU: gufo's CPU reference (from the assembled build)", "",
              f"{cr.get('tokens_asked')} tokens asked on {cfg['run'].get('cpu_threads', 8)} threads, exit {cr.get('rc')}, {cr.get('seconds')} s"
              + (" (timed out)" if cr.get("timed_out") else "") + ". "
              + "; ".join(f"{k}: {len(v['ids'])} tokens at {round(v['decode_tok_s'], 2)} tok/s, text {v['text']!r}" for k, v in cpu_runs.items()) + "."]
        if cpu_runs and os.path.isfile(lj):
            with open(lj) as f:
                lruns = json.load(f)
            cmp = {k: agreement(v["ids"], lruns[k]["ids"]) for k, v in cpu_runs.items() if k in lruns}
            L.append("Against llama.cpp on the same prompt ids: " + "; ".join(
                f"{k}: {min(a['lengths'])} tokens compared, first difference {'none' if a['first_difference'] is None else a['first_difference']}"
                for k, a in cmp.items()) + ".")
    if ll.get("runs"):
        L += ["", "llama.cpp's tokenizer on the prompt texts against gufo's: "
              + ", ".join(f"{k}: {'equal' if v['tokenizer_ids_equal_gufo'] else 'differs at ' + str(v['tokenizer_first_difference'])}" for k, v in ll["runs"].items())
              + ". Its CPU decode speed: " + ", ".join(f"{k} {v['decode_tok_s'] and round(v['decode_tok_s'], 1)} tok/s" for k, v in ll["runs"].items()) + "."]
    L += ["", "## Commands", "", "```", "tools/pod-sync.sh",
          f"tools/pod-run.sh './flerp gate recipe {name} --stages check,origin,build,reference,cpu,llama'",
          f"tools/pod-run.sh 'echo \"gufo-gpu | ./flerp gate recipe {name} --stages gpu\" > build/jobs-gufo.txt; "
          "nohup setsid tools/gpu-queue.sh build/jobs-gufo.txt > /dev/null 2>&1 &'",
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
    a = ap.parse_args()
    cat = Catalog(ROOT)
    recipe = cat.recipe(a.recipe)
    if recipe is None:
        sys.exit(f"no recipe {a.recipe}")
    cfg = load_config(os.path.join(ROOT, recipe.gate["config"]))
    res = os.path.join(ROOT, "gate", "results", recipe.name)
    os.makedirs(res, exist_ok=True)
    ctx = {"cat": cat, "recipe": recipe, "cfg": cfg, "res": res, "scratch": os.path.join(ROOT, cfg["host"]["scratch"])}
    os.makedirs(ctx["scratch"], exist_ok=True)
    results, prior = {}, os.path.join(res, "summary.json")
    if os.path.isfile(prior):                 # a partial run keeps the other stages' results: each says when it ran
        with open(prior) as f:
            results = json.load(f).get("stages", {})
    ctx["results"] = results
    if a.summarize:
        return 0 if write_summary(ctx, results) else 1
    fns = {"check": stage_check, "origin": stage_origin, "build": stage_build, "reference": stage_reference, "cpu": stage_cpu,
           "llama": stage_llama, "gpu": stage_gpu}
    want = a.stages.split(",")
    for s in STAGES:
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
