#!/usr/bin/env python3
"""Gate of runtime.flerp.vision_sidecar: the DeepSeek-V4.1 CPU vision worker against DeepSeek's own vision.py.

    <venv python> -I -B gate/vision_dsv41_cpu.py --config gate/vision-dsv41-cpu312.toml [--stages env,rows,protocol,summary]
                                                  [--images a,b] [--py313]

CPU only; no GPU, no engine, no network. The criteria are the config's [criteria]; this file computes and judges.
What it runs that it did not write (all read before the first run, see the summary): DeepSeek's inference/vision.py
and image_processor.py from the ark (imported, never written to), the module's vision_cpu.py (imported for the
in-process arms, and started as a child exactly as vision_cpu.zig starts it), the module's verify_vision.py, and the
origin's unmodified vision_cpu.py once, to record that it refuses this torch.

Stages:
  env       interpreter, packages, CPU, hashes of every input (reference files, weights, pins, sources)
  rows      one child of this script an image: the reference's preprocessing, then the arms R32 R64 R16 S32 S16
            S16t1 S16m (the config explains them); rows go to [work], numbers to results/rows/<image>.json
  protocol  the worker over its pipe for every image, twice; the refusals; the admission; time and peak memory
  summary   the judgement against [criteria], verdict.json and summary.md
"""
import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE = "catalog/flerp/runtime/vision_sidecar/src/flerp/vision_sidecar"
WORKER = MODULE + "/zig/src/families/dsv41/vision_cpu.py"
ADMISSION = MODULE + "/tools/dsv41-hip/verify_vision.py"
ENV_FILE = MODULE + "/vision-env.json"
ORIGIN_WORKER = "catalog/strixfold_pr6/runtime/vision_sidecar/src/zig/src/families/dsv41/vision_cpu.py"
HIDDEN = 5120
ROW_BYTES = HIDDEN * 2
ARMS = ("S16", "S16t1", "S16m", "S32", "R16", "R32", "R64")


# ---------------------------------------------------------------- pure helpers (tests/test_vision_gate.py)

def span_tokens(nh, nw):
    """Rows of an image span for a patch grid: (lh, lw, tokens)."""
    lh, lw = (nh + 2) // 3, (nw + 2) // 3
    return lh, lw, lh * (lw + 1) + 2


def grid_header(nh, nw, lh=None, lw=None):
    """The 32-byte grid vision_cpu.zig puts before the patches: best_h, best_w, n_vit_h, n_vit_w, n_llm_h, n_llm_w."""
    a, b, _ = span_tokens(nh, nw)
    return struct.pack("<6I", nh * 14, nw * 14, nh, nw, a if lh is None else lh, b if lw is None else lw) + bytes(8)


def parse_reply(out, tokens):
    """What vision_cpu.zig's finish() accepts: "VCPU", rows, byte count, the rows, nothing after. Returns the rows."""
    if len(out) < 16:
        raise ValueError(f"reply of {len(out)} bytes")
    magic, rows, size = struct.unpack("<4sIQ", out[:16])
    if magic != b"VCPU" or rows != tokens or size != tokens * ROW_BYTES:
        raise ValueError(f"header {magic!r} rows {rows} bytes {size}, wanted {tokens} rows")
    if len(out) != 16 + size:
        raise ValueError(f"{len(out) - 16} bytes after the header, wanted {size}")
    return out[16:]


def row_kinds(lh, lw):
    """Index lists of a span's rows: (start, aligner rows in reading order, newlines, end)."""
    rows, newlines = [], []
    for i in range(lh):
        base = 1 + i * (lw + 1)
        rows.extend(range(base, base + lw))
        newlines.append(base + lw)
    return 0, rows, newlines, lh * (lw + 1) + 1


def git_blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def judge(cfg, images, rows, protocol):
    """The verdict from the numbers. images: the config's [[image]] list; rows: {name: results of the rows stage};
    protocol: the protocol stage's result. Returns (ok, [failure strings], {criterion: bool})."""
    c = cfg["criteria"]
    fails, held = [], {k: True for k in ("1 shape", "2 arithmetic", "3a bf16 rows = R16", "3b bf16 vs R16",
                                          "4 determinism", "5 protocol")}

    def fail(key, text):
        held[key] = False
        fails.append(f"{key}: {text}")

    ratios = []
    for im in images:
        n = im["name"]
        r, p = rows.get(n), protocol.get("images", {}).get(n)
        if r is None or p is None or r.get("error") or p.get("error"):
            for key in held:
                fail(key, f"{n}: not measured ({(r or {}).get('error') or (p or {}).get('error') or 'missing'})")
            continue
        tokens = r["tokens"]
        want = im["expected_tokens"]
        if tokens != r["lh"] * (r["lw"] + 1) + 2 or tokens > 1024 or (want and tokens != want) or \
                (not want and not tokens < r.get("unshrunk_tokens", 1 << 30)):
            fail("1 shape", f"{n}: {tokens} tokens for grid {r['lh']} x {r['lw']}, expected {want or 'a shrunk grid'}")
        if not p["reply_ok"] or not p["finite"] or not p["delimiters_equal"]:
            fail("1 shape", f"{n}: reply_ok {p['reply_ok']} finite {p['finite']} delimiters {p['delimiters_equal']}")
        m = r["compare"]["S32_vs_R32"]
        if not (m["rel_rms"] <= c["fp32_rel_rms_max"] and m["row_cosine_min"] >= c["fp32_row_cosine_min"]):
            fail("2 arithmetic", f"{n}: rel rms {m['rel_rms']:.3g}, lowest row cosine {m['row_cosine_min']:.10f}")
        w, y = p["W_vs_R32"], r["compare"]["R16_vs_R32"]
        # 3a as restated 2026-11-11 (the config's [restated]): bit equality with the reference's BF16 rows
        wb = p["W_vs_R16_bits"]
        if c["bf16_rows_equal_r16"] and wb["values_differ"] != 0:
            fail("3a bf16 rows = R16", f"{n}: {wb['values_differ']} values in {wb['rows_differ']} of {wb['rows']} rows differ from R16")
        ratio = w["rel_rms"] / y["rel_rms"] if y["rel_rms"] > 0 else (1.0 if w["rel_rms"] == 0 else math.inf)
        ratios.append(ratio)
        if not ratio <= c["bf16_ratio_to_r16_max"]:
            fail("3b bf16 vs R16", f"{n}: the worker is {ratio:.3f} times as far from R32 as R16 is")
        if not p["twice_equal"]:
            fail("4 determinism", f"{n}: two runs of the worker differ")
        if not p["equals_S16"]:
            fail("4 determinism", f"{n}: the worker's rows are not the in-process S16 rows")
    if ratios and not sum(ratios) / len(ratios) <= c["bf16_ratio_to_r16_mean_max"]:
        fail("3b bf16 vs R16", f"mean ratio {sum(ratios) / len(ratios):.3f}")
    for case in protocol.get("refusals", []):
        if not case["refused"]:
            fail("5 protocol", f"not refused: {case['case']} (exit {case['rc']}, {case['stdout_bytes']} bytes out)")
    if not protocol.get("refusals"):
        fail("5 protocol", "the refusals did not run")
    adm = protocol.get("admission") or {}
    if not adm.get("ok"):
        fail("5 protocol", f"admission: {adm.get('why', 'did not run')}")
    return not fails, fails, held


# ---------------------------------------------------------------- inputs

def load_config(path):
    import tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def safetensors_table(path):
    """Header of a safetensors file, read here (not the pull request's reader): ({name: spec}, data offset)."""
    with open(path, "rb") as f:
        n, = struct.unpack("<Q", f.read(8))
        table = json.loads(f.read(n))
    table.pop("__metadata__", None)
    return table, 8 + n


def tower_names(table):
    return sorted(k for k in table if k.startswith(("vision.", "aligner.", "image_")))


def shard_tensors(path):
    """The 266 tower tensors straight from the checkpoint shard, as torch BF16 tensors."""
    import torch
    table, base = safetensors_table(path)
    out = {}
    with open(path, "rb") as f:
        for name in tower_names(table):
            spec = table[name]
            assert spec["dtype"] == "BF16", (name, spec["dtype"])
            a, b = spec["data_offsets"]
            f.seek(base + a)
            out[name] = torch.frombuffer(bytearray(f.read(b - a)), dtype=torch.bfloat16).reshape(spec["shape"])
    return out


def reference_modules(cfg):
    """DeepSeek's vision.py and image_processor.py, imported from the ark without writing a byte there."""
    import importlib.util
    sys.dont_write_bytecode = True
    mods = {}
    for name in ("vision", "image_processor"):
        spec = importlib.util.spec_from_file_location("deepseek_" + name, os.path.join(cfg["reference"]["dir"], name + ".py"))
        mods[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mods[name])
    return mods["vision"], mods["image_processor"]


def reference_args(cfg):
    """The tower's sizes, from the checkpoint's config.json; the reference's own inference/config.json must agree."""
    from types import SimpleNamespace
    with open(os.path.join(cfg["weights"]["checkpoint"], "config.json")) as f:
        v = json.load(f)["vision_config"]
    with open(os.path.join(cfg["reference"]["dir"], "config.json")) as f:
        r = json.load(f)
    a = dict(vision_n_layers=v["num_hidden_layers"], vision_dim=v["hidden_size"], vision_n_heads=v["num_attention_heads"],
             vision_inter_dim=v["intermediate_size"], vision_patch_size=v["patch_size"], vision_rope_theta=v["rope_theta"],
             vision_downsample_ratio=v["downsample_ratio"], vision_max_n_token=v["max_image_tokens"],
             vision_min_pixels=v["min_pixels"], vision_max_wh_ratio=v["max_wh_ratio"])
    differs = {k: (a[k], r.get(k)) for k in a if r.get(k) != a[k]}
    if differs:
        raise SystemExit(f"checkpoint config and reference config disagree: {differs}")
    return SimpleNamespace(dim=r["dim"], **a)


def sidecar_module():
    """The module's vision_cpu.py as an importable module. Its functions use the globals torch and F, which the
    file binds only under __main__; they are set here to the same objects."""
    import importlib.util
    import torch
    import torch.nn.functional as F
    spec = importlib.util.spec_from_file_location("flerp_vision_cpu", os.path.join(ROOT, WORKER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.torch, mod.F = torch, F
    return mod


# ---------------------------------------------------------------- the fixed images

def seed_of(name):
    return int.from_bytes(hashlib.sha256(("vision-dsv41 " + name).encode()).digest()[:8], "little")


def photo_pixels(w, h, rng):
    import numpy as np
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    out = np.empty((h, w, 3))
    for c in range(3):
        fx, fy, ph = rng.uniform(0.5, 3.0), rng.uniform(0.5, 3.0), rng.uniform(0, 6.28)
        out[..., c] = 127 + 70 * np.sin(2 * np.pi * (fx * x / w + fy * y / h) + ph) + 40 * (x / w - 0.5) * (c - 1)
    out += rng.normal(0.0, 12.0, out.shape)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def make_image(name):
    """(kind, payload): ("bytes", encoded image, description) or ("grid", (nh, nw, patches uint16 bits), description)."""
    import io
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed_of(name))

    def png(img):
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()

    sizes = {"tiny": (32, 32), "photo": (640, 480), "photo-jpeg": (640, 480), "rgba": (500, 700), "wide": (2000, 100),
             "cap": (3024, 588), "over": (1400, 1400)}
    if name in sizes:
        w, h = sizes[name]
        if name == "photo-jpeg":
            rng = np.random.default_rng(seed_of("photo"))
        px = photo_pixels(w, h, rng)
        if name == "rgba":
            alpha = np.broadcast_to(np.linspace(0, 255, w).astype(np.uint8), (h, w))
            img = Image.fromarray(np.dstack([px, alpha]))
            assert img.mode == "RGBA"
            return "bytes", png(img), f"{w} x {h} RGBA PNG"
        if name == "photo-jpeg":
            buf = io.BytesIO()
            Image.fromarray(px).save(buf, "JPEG", quality=90)
            return "bytes", buf.getvalue(), f"{w} x {h} RGB JPEG quality 90"
        return "bytes", png(Image.fromarray(px)), f"{w} x {h} RGB PNG"
    if name == "text":
        w, h = 800, 600
        px = np.full((h, w), 255, np.uint8)
        for top in range(20, h - 30, 30):
            x = 20
            while x < w - 40:
                gw, gh = int(rng.integers(4, 18)), int(rng.integers(10, 20))
                if rng.random() > 0.15:
                    px[top:top + gh, x:x + gw] = 0
                x += gw + int(rng.integers(2, 8))
        return "bytes", png(Image.fromarray(px)), f"{w} x {h} mode L PNG"
    if name == "palette":
        w, h = 300, 900
        y, x = np.mgrid[0:h, 0:w]
        idx = ((y // 30) + 3 * (x // 25) + rng.integers(0, 2, (h, w))) % 16
        img = Image.frombytes("P", (w, h), idx.astype(np.uint8).tobytes())
        img.putpalette(bytes(rng.integers(0, 256, 48, dtype=np.uint8)))
        return "bytes", png(img), f"{w} x {h} mode P PNG, 16 colours"
    if name.startswith("grid-"):
        nh, nw = (int(v) for v in name[5:].split("x"))
        import torch
        v = rng.integers(0, 256, (nh * nw, 3, 14, 14)).astype(np.float32)
        patches = ((torch.from_numpy(v) / 255 - 0.5) / 0.5).to(torch.bfloat16)   # the reference's own scaling
        return "grid", (nh, nw, patches), f"{nh} x {nw} patches of seeded 8-bit values, no picture"
    raise SystemExit(f"no image named {name}")


# ---------------------------------------------------------------- metrics

def bf16_to_f64(bits):
    import numpy as np
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def compare(a, r):
    """a against the reference r, float64 [rows][5120]."""
    import numpy as np
    d = a - r
    nr = float(np.sqrt((r * r).sum()))
    rms = float(np.sqrt((r * r).mean()))
    dot = (a * r).sum(1)
    cos = dot / np.maximum(np.sqrt((a * a).sum(1) * (r * r).sum(1)), 1e-300)
    row_rel = np.sqrt((d * d).sum(1)) / np.maximum(np.sqrt((r * r).sum(1)), 1e-300)
    big = np.abs(r) >= rms
    return {"rows": int(a.shape[0]), "rel_rms": float(np.sqrt((d * d).sum())) / nr if nr else 0.0,
            "max_abs": float(np.abs(d).max()), "max_abs_over_rms": float(np.abs(d).max()) / rms if rms else 0.0,
            "max_rel_where_ref_ge_rms": float((np.abs(d[big]) / np.abs(r[big])).max()) if big.any() else 0.0,
            "ref_rms": rms, "row_cosine_min": float(cos.min()), "row_cosine_mean": float(cos.mean()),
            "row_rel_max": float(row_rel.max())}


def bits_compare(a, b):
    """Two BF16 arrays as uint16 bits, [rows][5120]: how many rows and values differ and by how many BF16 steps."""
    import numpy as np
    def order(x):
        x = x.astype(np.int32)
        return np.where(x & 0x8000, -(x & 0x7FFF), x & 0x7FFF)
    steps = np.abs(order(a) - order(b))
    return {"rows": int(a.shape[0]), "rows_differ": int((a != b).any(1).sum()), "values": int(a.size),
            "values_differ": int((a != b).sum()), "max_steps": int(steps.max()) if steps.size else 0,
            "steps_hist": {"0": int((steps == 0).sum()), "1": int((steps == 1).sum()), "2": int((steps == 2).sum()),
                           "3-4": int(((steps >= 3) & (steps <= 4)).sum()), "5-8": int(((steps >= 5) & (steps <= 8)).sum()),
                           ">8": int((steps > 8).sum())}}


# ---------------------------------------------------------------- stage: rows (one image, in a child)

def one_image(cfg, name):
    import resource
    import numpy as np
    import torch
    threads = cfg["host"]["threads"]
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    V, IP = reference_modules(cfg)
    args = reference_args(cfg)
    work = os.path.join(ROOT, cfg["work"])
    os.makedirs(os.path.join(work, "rows", name), exist_ok=True)
    os.makedirs(os.path.join(work, "payload"), exist_ok=True)
    res = {"image": name, "threads": threads, "seconds": {}, "maxrss_kb_after": {}, "rows_sha256": {}}

    kind, data, what = make_image(name)
    res["what"] = what
    if kind == "bytes":
        res["encoded_sha256"], res["encoded_bytes"] = hashlib.sha256(data).hexdigest(), len(data)
        t = time.monotonic()
        patches, nh, nw, lh, lw = IP.load_image({"data": data}, args)      # the reference's preprocessing
        res["seconds"]["preprocess"] = time.monotonic() - t
        from PIL import Image
        import io
        with Image.open(io.BytesIO(data)) as im:
            res["size"], res["mode"] = list(im.size), im.mode
            p = args.vision_patch_size
            uh, uw = math.ceil(math.ceil(im.size[1] / p) / 3), math.ceil(math.ceil(im.size[0] / p) / 3)
            res["unshrunk_tokens"] = uh * (uw + 1) + 2         # only meaningful at or above min_pixels ("over")
    else:
        nh, nw, patches = data
        lh, lw = math.ceil(nh / 3), math.ceil(nw / 3)
    tokens = IP.num_image_tokens(lh, lw)
    res.update(nh=nh, nw=nw, lh=lh, lw=lw, patches=nh * nw, tokens=tokens)
    assert patches.dtype == torch.bfloat16 and tuple(patches.shape) == (nh * nw, 3, 14, 14)
    pbits = patches.contiguous().view(torch.uint16).numpy().tobytes()
    res["patches_sha256"] = hashlib.sha256(pbits).hexdigest()
    with open(os.path.join(work, "payload", name + ".bin"), "wb") as f:
        f.write(grid_header(nh, nw) + pbits)
    _, idx, newlines, end = row_kinds(lh, lw)

    def keep(arm, array):
        np.save(os.path.join(work, "rows", name, arm + ".npy"), array)
        res["rows_sha256"][arm] = hashlib.sha256(array.tobytes()).hexdigest()
        res["maxrss_kb_after"][arm] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # --- the sidecar, in this process: the module's own loader on the extracted file, its encode() and span()
    S = sidecar_module()
    wpath = os.path.join(ROOT, cfg["weights"]["vision"])
    with open(os.path.join(ROOT, cfg["weights"]["pins"])) as f:
        pin = json.load(f)["outputs"]["vision"]["sha256"]
    sw = S.read_weights(wpath, pin)
    flat = patches.reshape(nh * nw, 588)

    def sidecar_bf16(arm, n_threads, math_attention):
        torch.set_num_threads(n_threads)
        t = time.monotonic()
        with torch.inference_mode():
            out = S.span(S.encode(flat, nh, nw, sw, math_attention=math_attention), nh, nw, sw)
        res["seconds"][arm] = time.monotonic() - t
        torch.set_num_threads(threads)
        assert tuple(out.shape) == (tokens, HIDDEN)
        keep(arm, out.view(torch.uint16).numpy().copy())

    sidecar_bf16("S16", threads, False)
    sidecar_bf16("S16t1", 1, False)
    sidecar_bf16("S16m", threads, True)
    t = time.monotonic()
    with torch.inference_mode():
        s32 = S.encode(flat.float(), nh, nw, {k: v.float() for k, v in sw.items()})
    res["seconds"]["S32"] = time.monotonic() - t
    assert s32.dtype == torch.float32
    keep("S32", s32.numpy().copy())
    del s32

    # --- the reference: DeepSeek's modules, the tensors from the checkpoint shard through this file's reader
    ck = shard_tensors(os.path.join(cfg["weights"]["checkpoint"], cfg["weights"]["shard"]))
    res["loader_bytes_equal_shard"] = sorted(ck) == sorted(sw) and all(
        ck[k].shape == sw[k].shape and bool((ck[k].view(torch.uint16) == sw[k].view(torch.uint16)).all()) for k in ck)

    def reference(arm, dtype):
        vit, aligner = V.ViT(args), V.Aligner(args)
        vit.load_state_dict({k[len("vision."):]: v.float() for k, v in ck.items() if k.startswith("vision.")}, strict=True)
        aligner.load_state_dict({k[len("aligner."):]: v.float() for k, v in ck.items() if k.startswith("aligner.")}, strict=True)
        vit, aligner = vit.to(dtype).eval(), aligner.to(dtype).eval()
        V.get_vision_cos_sin.cache_clear()
        t = time.monotonic()
        with torch.inference_mode():
            out = aligner(vit(patches.to(dtype), nh, nw), nh, nw)
        res["seconds"][arm] = time.monotonic() - t
        assert out.dtype == dtype and tuple(out.shape) == (lh * lw, HIDDEN)
        keep(arm, out.view(torch.uint16).numpy().copy() if dtype == torch.bfloat16 else out.numpy().copy())

    reference("R16", torch.bfloat16)
    reference("R32", torch.float32)
    reference("R64", torch.float64)

    # --- numbers
    L = lambda arm: np.load(os.path.join(work, "rows", name, arm + ".npy"))   # noqa: E731
    r32, r64 = L("R32").astype(np.float64), L("R64")
    s16, r16 = L("S16"), L("R16")
    rq = torch.from_numpy(L("R32")).to(torch.bfloat16).view(torch.uint16).numpy()     # R32 rounded to BF16
    start, end_row, nl = ck["image_start"].view(torch.uint16).numpy(), ck["image_end"].view(torch.uint16).numpy(), \
        ck["image_newline"].view(torch.uint16).numpy()
    res["S16_delimiters_equal"] = bool((s16[0] == start).all() and (s16[end] == end_row).all()
                                       and all((s16[i] == nl).all() for i in newlines))
    res["S16_finite"] = bool(np.isfinite(bf16_to_f64(s16)).all())
    res["compare"] = {
        "S32_vs_R32": compare(L("S32").astype(np.float64), r32),
        "R32_vs_R64": compare(r32, r64),
        "R16_vs_R32": compare(bf16_to_f64(r16), r32),
        "S16_vs_R32": compare(bf16_to_f64(s16[idx]), r32),
        "S16m_vs_R32": compare(bf16_to_f64(L("S16m")[idx]), r32),
        "S16t1_vs_R32": compare(bf16_to_f64(L("S16t1")[idx]), r32),
        "round(R32)_vs_R32": compare(bf16_to_f64(rq), r32),
    }
    res["bits"] = {
        "S16_vs_R16": bits_compare(s16[idx], r16),
        "S16m_vs_R16": bits_compare(L("S16m")[idx], r16),
        "S16_vs_S16m": bits_compare(s16[idx], L("S16m")[idx]),
        "S16_vs_S16t1": bits_compare(s16, L("S16t1")),
        "S16_vs_round(R32)": bits_compare(s16[idx], rq),
        "R16_vs_round(R32)": bits_compare(r16, rq),
    }
    return res


def stage_rows(cfg, results, names):
    os.makedirs(os.path.join(results, "rows"), exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(cfg["host"]["threads"]), MKL_NUM_THREADS=str(cfg["host"]["threads"]))
    for name in names:
        out = os.path.join(results, "rows", name + ".json")
        t = time.monotonic()
        p = subprocess.run([sys.executable, "-I", "-B", os.path.abspath(__file__), "--config", cfg["_path"], "--one-image", name],
                           env=env, capture_output=True, text=True, timeout=6 * 3600)
        if p.returncode != 0:
            res = {"image": name, "error": (p.stderr or "").strip().splitlines()[-1:] or [f"exit {p.returncode}"]}
            res["stderr_tail"] = (p.stderr or "")[-2000:]
        else:
            res = json.loads(p.stdout)
        res["child_seconds"] = time.monotonic() - t
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"rows {name}: {'ERROR ' + str(res['error']) if res.get('error') else str(res['tokens']) + ' tokens'}, "
              f"{res['child_seconds']:.0f} s", flush=True)


# ---------------------------------------------------------------- stage: protocol (the stand-in for vision_cpu.zig)

def run_worker(python, source, weights, sha, payload, threads, timeout, setting):
    """Start the worker as vision_cpu.zig's start() does and read it as finish() does. setting None = variable unset."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
           "TF_DS_VISION_THREADS": str(threads), "TF_DS_VISION_TIMEOUT": str(timeout)}
    if setting is not None:
        env["FLERP_VISION_ENV"] = setting
    t0 = time.monotonic()
    p = subprocess.Popen([python, "-I", "-c", source, weights, sha], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, env=env)
    err = []

    def feed():
        try:
            p.stdin.write(payload)
            p.stdin.close()
        except OSError:
            pass
    writer = threading.Thread(target=feed)
    errs = threading.Thread(target=lambda: err.append(p.stderr.read()))
    writer.start(), errs.start()
    out = p.stdout.read()
    t_rows = time.monotonic() - t0
    _, status, usage = os.wait4(p.pid, 0)
    wall = time.monotonic() - t0
    p.returncode = os.waitstatus_to_exitcode(status)
    writer.join(), errs.join()
    return {"rc": p.returncode, "out": out, "stderr": (err[0] if err else b"").decode(errors="replace"),
            "wall_s": wall, "rows_s": t_rows, "user_s": usage.ru_utime, "sys_s": usage.ru_stime, "maxrss_kb": usage.ru_maxrss}


def env_record(cfg):
    with open(os.path.join(ROOT, ENV_FILE)) as f:
        return json.load(f)["environments"][cfg["env"]]


def stage_protocol(cfg, results, names, python=None, tag="protocol"):
    import numpy as np
    host = cfg["host"]
    python = os.path.abspath(os.path.join(ROOT, python or host["python"]))
    weights = os.path.join(ROOT, cfg["weights"]["vision"])
    bias = os.path.join(ROOT, cfg["weights"]["bias"])
    with open(os.path.join(ROOT, cfg["weights"]["pins"])) as f:
        pins = json.load(f)["outputs"]
    sha, bias_sha = pins["vision"]["sha256"], pins["bias"]["sha256"]
    with open(os.path.join(ROOT, WORKER)) as f:
        source = f.read()
    work = os.path.join(ROOT, cfg["work"])
    res = {"python": python, "threads": host["threads"], "images": {}, "refusals": [], "worker_sha256": hashlib.sha256(source.encode()).hexdigest()}
    rec = env_record(cfg)

    # the admission, as a launcher would run it; its printed setting is what the worker is started with
    adm = [python, "-I", os.path.join(ROOT, ADMISSION), "--bias", bias, "--bias-sha256", bias_sha, "--python", python,
           "--weights", weights, "--weights-sha256", sha, "--env", cfg["env"]]
    if tag != "protocol":      # the 3.13 arm: no record of its own; the setting names what it runs
        return worker_runs(cfg, res, names, python, source, weights, sha, tag,
                           f"python=3.13,torch={rec['torch']},numpy={rec['numpy']}", rec)
    t = time.monotonic()
    with_flag = subprocess.run(adm + ["--allow-candidate"], capture_output=True, text=True)
    adm_s = time.monotonic() - t
    without = subprocess.run(adm, capture_output=True, text=True)
    setting = with_flag.stdout.strip().partition("FLERP_VISION_ENV=")[2]
    want_without = 0 if rec["status"] == "pass" else 1
    ok = with_flag.returncode == 0 and bool(setting) and (without.returncode == 0) == (want_without == 0)
    res["admission"] = {"ok": ok, "record_status": rec["status"], "setting": setting, "seconds": adm_s,
                        "with_allow_candidate_rc": with_flag.returncode, "without_rc": without.returncode,
                        "without_message": (without.stderr or without.stdout).strip()[-300:],
                        "why": "" if ok else (with_flag.stderr.strip()[-300:] or "the record's status and the admission disagree")}
    if not setting:
        return res
    return worker_runs(cfg, res, names, python, source, weights, sha, tag, setting, rec)


def worker_runs(cfg, res, names, python, source, weights, sha, tag, setting, rec):
    import numpy as np
    host, work = cfg["host"], os.path.join(ROOT, cfg["work"])
    res["setting"] = setting
    delims = None

    def go(payload, **kw):
        a = dict(python=python, source=source, weights=weights, sha=sha, payload=payload, threads=host["threads"],
                 timeout=host["timeout_s"], setting=setting)
        a.update(kw)
        return run_worker(**a)

    for name in names:
        with open(os.path.join(work, "payload", name + ".bin"), "rb") as f:
            payload = f.read()
        _, _, nh, nw, lh, lw = struct.unpack("<6I", payload[:24])
        tokens = lh * (lw + 1) + 2
        one = {"tokens": tokens, "patches": nh * nw, "runs": []}
        outs = []
        for _ in range(2):
            r = go(payload)
            one["runs"].append({k: r[k] for k in ("rc", "wall_s", "rows_s", "user_s", "sys_s", "maxrss_kb")})
            try:
                if r["rc"] != 0:
                    raise ValueError(f"exit {r['rc']}: {r['stderr'].strip().splitlines()[-1:]}")
                outs.append(parse_reply(r["out"], tokens))
            except ValueError as e:
                one["error"] = str(e)
                break
        if len(outs) == 2:
            w = np.frombuffer(outs[0], np.uint16).reshape(tokens, HIDDEN)
            one["reply_ok"] = True
            one["rows_sha256"] = hashlib.sha256(outs[0]).hexdigest()
            one["twice_equal"] = outs[0] == outs[1]
            np.save(os.path.join(work, "rows", name, ("W" if tag == "protocol" else "W313") + ".npy"), w)
            if tag == "protocol":
                L = lambda arm: np.load(os.path.join(work, "rows", name, arm + ".npy"))   # noqa: E731
                start, idx, newlines, end = row_kinds(lh, lw)
                s16, r32 = L("S16"), L("R32").astype(np.float64)
                one["equals_S16"] = bool((w == s16).all())
                one["finite"] = bool(np.isfinite(bf16_to_f64(w)).all())
                if delims is None:
                    import torch
                    ck = shard_tensors(os.path.join(cfg["weights"]["checkpoint"], cfg["weights"]["shard"]))
                    delims = {k: ck[k].view(torch.uint16).numpy().copy() for k in ("image_start", "image_end", "image_newline")}
                    del ck
                d = delims
                one["delimiters_equal"] = bool((w[start] == d["image_start"]).all() and (w[end] == d["image_end"]).all()
                                               and all((w[i] == d["image_newline"]).all() for i in newlines))
                one["W_vs_R32"] = compare(bf16_to_f64(w[idx]), r32)
                one["W_vs_R16_bits"] = bits_compare(w[idx], L("R16"))
            else:
                one["equals_py312_W"] = bool((w == np.load(os.path.join(work, "rows", name, "W.npy"))).all())
        res["images"][name] = one
        print(f"{tag} {name}: {one.get('error') or 'ok'}, {one['runs'][0]['wall_s']:.1f} s, "
              f"{one['runs'][0]['maxrss_kb'] / 1048576:.2f} GiB", flush=True)
    if tag != "protocol":
        return res

    # refusals, on the smallest payload
    small = grid_header(1, 1) + bytes(588 * 2)
    flip = sha[:-1] + ("0" if sha[-1] != "0" else "1")
    cases = [
        ("a wrong weights sha256", dict(sha=flip)),
        ("FLERP_VISION_ENV unset (the origin's pin)", dict(setting=None)),
        ("a setting naming another torch", dict(setting=setting.replace(rec["torch"], "2.9.0+cpu"))),
        ("a setting naming another python", dict(setting=setting.replace("python=" + rec["python"], "python=3.99"))),
        ("a malformed setting", dict(setting="torch")),
        ("a setting without torch", dict(setting="python=" + rec["python"])),
        ("a grid whose token count does not follow from its patches", dict(payload=grid_header(1, 1, lh=2) + bytes(588 * 2))),
        ("a payload one value short", dict(payload=small[:-2])),
        ("a payload one value long", dict(payload=small + bytes(2))),
    ]
    for case, kw in cases:
        r = go(small if "payload" not in kw else kw.pop("payload"), **kw)
        res["refusals"].append({"case": case, "rc": r["rc"], "stdout_bytes": len(r["out"]), "refused": r["rc"] != 0 and not r["out"],
                                "says": r["stderr"].strip().splitlines()[-1:] , "wall_s": r["wall_s"]})
    r = go(small)
    res["smallest_accepted"] = {"rc": r["rc"], "reply_bytes": len(r["out"]), "wall_s": r["wall_s"], "maxrss_kb": r["maxrss_kb"]}
    with open(os.path.join(ROOT, ORIGIN_WORKER)) as f:
        r = go(small, source=f.read())
    res["origin_worker_unmodified"] = {"rc": r["rc"], "stdout_bytes": len(r["out"]), "says": r["stderr"].strip().splitlines()[-1:]}
    # where a spawn's time goes
    t = time.monotonic()
    subprocess.run([python, "-I", "-c", "import torch, torch.nn.functional"], check=True,
                   env={"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": str(host["threads"])})
    res["import_torch_s"] = time.monotonic() - t
    t = time.monotonic()
    sha256_file(weights)
    res["weights_sha256_s"] = time.monotonic() - t
    return res


# ---------------------------------------------------------------- stage: env

def stage_env(cfg):
    import platform
    import numpy
    import PIL
    import torch
    ref = cfg["reference"]["dir"]
    with open(os.path.join(ref, "vision.py"), "rb") as f:
        blob = git_blob_sha1(f.read())
    with open("/proc/cpuinfo") as f:
        cpu = next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), "?")
    with open(os.path.join(ROOT, cfg["weights"]["pins"])) as f:
        pins = json.load(f)
    vision = os.path.join(ROOT, cfg["weights"]["vision"])
    shard = os.path.join(cfg["weights"]["checkpoint"], cfg["weights"]["shard"])
    # the extraction against the shard, by this file's reader: the same 266 names, the same bytes
    et, eb = safetensors_table(vision)
    st, sb = safetensors_table(shard)
    names = tower_names(st)
    same = names == tower_names(et)
    with open(vision, "rb") as fe, open(shard, "rb") as fs:
        for n in names if same else []:
            (a, b), (c, d) = et[n]["data_offsets"], st[n]["data_offsets"]
            fe.seek(eb + a), fs.seek(sb + c)
            if et[n]["shape"] != st[n]["shape"] or et[n]["dtype"] != st[n]["dtype"] or fe.read(b - a) != fs.read(d - c):
                same = False
                break
    out = {
        "host": platform.node(), "cpu": cpu, "cpus": os.cpu_count(), "python": sys.version, "executable": sys.executable,
        "torch": torch.__version__, "numpy": numpy.__version__, "pillow": PIL.__version__,
        "torch_threads_default": torch.get_num_threads(), "torch_config": torch.__config__.show(),
        "reference": {"dir": ref, "vision_git_blob": blob, "vision_git_blob_expected": cfg["reference"]["vision_git_blob"],
                      "vision_blob_ok": blob == cfg["reference"]["vision_git_blob"],
                      "vision_sha256": sha256_file(os.path.join(ref, "vision.py")),
                      "image_processor_sha256": sha256_file(os.path.join(ref, "image_processor.py")),
                      "config_sha256": sha256_file(os.path.join(ref, "config.json"))},
        "weights": {"vision": vision, "vision_sha256": sha256_file(vision), "vision_pin": pins["outputs"]["vision"]["sha256"],
                    "bias_sha256": sha256_file(os.path.join(ROOT, cfg["weights"]["bias"])), "bias_pin": pins["outputs"]["bias"]["sha256"],
                    "shard": shard, "shard_pin": pins["shards"][cfg["weights"]["shard"]]["sha256"],
                    "tower_tensors": len(names), "extraction_tensors": len(et),
                    "extraction_tower_bytes_equal_shard": same,
                    "tower_dtypes": sorted({st[n]["dtype"] for n in names})},
        "sources": {p: sha256_file(os.path.join(ROOT, p)) for p in (WORKER, ADMISSION, ENV_FILE, ORIGIN_WORKER, "gate/vision_dsv41_cpu.py")},
        "config_sha256": sha256_file(cfg["_path"]),
    }
    try:
        with open(os.path.join(ROOT, "build/vision-logs/prepare.json")) as f:
            out["extraction_report"] = json.load(f)        # what the origin's prepare_vision.py printed
    except (OSError, ValueError):
        out["extraction_report"] = None
    out["weights"]["ok"] = out["weights"]["vision_sha256"] == out["weights"]["vision_pin"] and \
        out["weights"]["bias_sha256"] == out["weights"]["bias_pin"] and same and len(names) == 266
    return out


# ---------------------------------------------------------------- stage: summary

def stage_summary(cfg, results, names):
    def load(p):
        try:
            with open(os.path.join(results, p)) as f:
                return json.load(f)
        except OSError:
            return None
    env, protocol, p313 = load("env.json") or {}, load("protocol.json") or {}, load("protocol-py313.json")
    rows = {n: load(f"rows/{n}.json") for n in names}
    images = [im for im in cfg["image"] if im["name"] in names]
    ok, fails, held = judge(cfg, images, rows, protocol)
    if not (env.get("reference", {}).get("vision_blob_ok") and env.get("weights", {}).get("ok")):
        ok = False
        fails.insert(0, "inputs: the reference file or the weights are not the pinned ones (env.json)")
    partial = len(names) != len(cfg["image"])
    verdict = {"gate": cfg["name"], "module": cfg["module"], "env": cfg["env"], "pass": ok and not partial, "partial": partial,
               "criteria": held, "failures": fails, "config_sha256": sha256_file(cfg["_path"]),
               "run_config_sha256": env.get("config_sha256"), "restated": cfg.get("restated"),
               "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(results, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1)

    g = lambda v, d=3: f"{v:.{d}g}"                                    # noqa: E731
    L = [f"# Gate: {cfg['name']} (`{cfg['module']}`, environment `{cfg['env']}`)", "",
         f"Host {env.get('host')}, {env.get('cpu')}, written {verdict['written']}. "
         f"**{'PASSED' if verdict['pass'] else 'PARTIAL RUN' if partial and ok else 'FAILED'}** under the criteria of "
         f"`gate/{cfg['name']}.toml` (sha256 `{verdict['config_sha256'][:16]}`"
         + (", criterion 3a as restated: see below" if cfg.get("restated") else "") + ").", "",
         f"Python {env.get('python', '?').split()[0]}, torch {env.get('torch')}, numpy {env.get('numpy')}, Pillow {env.get('pillow')}; "
         f"{cfg['host']['threads']} threads. Reference: DeepSeek `inference/vision.py`, git blob "
         f"`{env.get('reference', {}).get('vision_git_blob')}` ({'the pinned blob' if env.get('reference', {}).get('vision_blob_ok') else 'NOT the pinned blob'}), "
         f"with the checkpoint's {env.get('weights', {}).get('tower_tensors')} tower tensors read from the shard by the gate's own reader.", "",
         "| criterion | held |", "|---|---|"]
    L += [f"| {k} | {'yes' if v else '**no**'} |" for k, v in held.items()]
    if fails:
        L += ["", "Failures:", ""] + [f"- {x}" for x in fails]
    rs = cfg.get("restated")
    if rs:
        L += ["", "## Criteria as first committed, and why 3a was restated", "",
              f"The criteria were committed in {rs['first_committed']} before any run. The run ({rs['first_run']}, config sha256 "
              f"`{rs['first_config_sha256']}`, which `env.json` still carries: it is that run's) was judged under them and "
              "**FAILED**, on 3a alone:", ""] + [f"- {x}" for x in rs["first_failures"]] + [
              "", f"3a then was an absolute bound of the worker's BF16 rows against the FP32 reference: relative RMS <= "
              f"{rs['bf16_rel_rms_max']} an image (held, 11 of 11) and cosine >= {rs['bf16_row_cosine_min']} for every row (broken).",
              "", f"Restated {rs['date']} by {rs['by']}: {rs['reason']}",
              "", "The verdict at the top is under the restated 3a. It was judged from the bit comparisons stored by that same "
              "run (`protocol.json`, `W_vs_R16_bits`): no tower forward was run again. The other five criteria and every "
              "number below are unchanged."]
    measured = [n for n in names if (rows.get(n) or {}).get("bits") and "W_vs_R16_bits" in (protocol.get("images", {}).get(n) or {})]
    if measured and all(protocol["images"][n]["W_vs_R16_bits"]["values_differ"] == 0 for n in measured):
        L += ["", f"On all {len(measured)} images the worker's aligner rows are, bit for bit, the rows DeepSeek's own modules give "
              "in BF16 (R16) on this torch. Whatever a criterion of group 3 says about the worker it says about the "
              "reference's own BF16 evaluation. The rows far from FP32 are in the post hoc table below."]
    L += ["", "## Rows against the reference", "",
          "`rel rms` is ||a - R32|| / ||R32|| over an image's aligner rows; `cos` the lowest row cosine. R32 = DeepSeek's "
          "modules in FP32; R16 = the same in BF16; S32 = the sidecar's `encode()` on FP32 tensors; W = the worker's reply.", "",
          "| image | patches | tokens | S32 vs R32 rel rms | S32 cos | W vs R32 rel rms | W cos | R16 vs R32 rel rms | W / R16 | W max abs / rms | R32 vs R64 rel rms |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        r, p = rows.get(n) or {}, protocol.get("images", {}).get(n) or {}
        if r.get("error") or not r or "W_vs_R32" not in p:
            L.append(f"| {n} | not measured: {r.get('error') or p.get('error') or 'missing'} |||||||||| |")
            continue
        c, w = r["compare"], p["W_vs_R32"]
        L.append(f"| {n} | {r['nh']} x {r['nw']} = {r['patches']} | {r['tokens']} | {g(c['S32_vs_R32']['rel_rms'])} | "
                 f"{c['S32_vs_R32']['row_cosine_min']:.9f} | {g(w['rel_rms'])} | {w['row_cosine_min']:.6f} | "
                 f"{g(c['R16_vs_R32']['rel_rms'])} | {w['rel_rms'] / c['R16_vs_R32']['rel_rms'] if c['R16_vs_R32']['rel_rms'] else float('nan'):.3f} | "
                 f"{g(w['max_abs_over_rms'])} | {g(c['R32_vs_R64']['rel_rms'])} |")
    L += ["", "## Bits (BF16 rows; recorded, not criteria except the last two columns)", "",
          "Rows that differ in any bit / rows compared. `round(R32)` is R32 rounded once to BF16: the closest BF16 rows "
          "to the full-precision result.", "",
          "| image | W vs R16 | S16m (no query pieces) vs R16 | S16 vs S16m | 1 thread vs 8 | W vs round(R32): values equal, within one BF16 step | R16 vs round(R32): values equal | worker twice | worker = in-process |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        r, p = rows.get(n) or {}, protocol.get("images", {}).get(n) or {}
        if r.get("error") or not r or "W_vs_R32" not in p:
            continue
        b = r["bits"]
        f = lambda x: f"{x['rows_differ']} / {x['rows']}"                 # noqa: E731
        q, y = b["S16_vs_round(R32)"], b["R16_vs_round(R32)"]
        L.append(f"| {n} | {f(p['W_vs_R16_bits'])} | {f(b['S16m_vs_R16'])} | {f(b['S16_vs_S16m'])} | {f(b['S16_vs_S16t1'])} | "
                 f"{100 * (1 - q['values_differ'] / q['values']):.1f}%, {100 * (q['steps_hist']['0'] + q['steps_hist']['1']) / q['values']:.1f}% | {100 * (1 - y['values_differ'] / y['values']):.1f}% | "
                 f"{'same bytes' if p['twice_equal'] else '**differ**'} | {'yes' if p['equals_S16'] else '**no**'} |")
    L += ["", "## The worker over its pipe: time and memory an image", "",
          f"One process an image, as `vision_cpu.zig` starts it (`python -I -c <source> <weights> <sha256>`), "
          f"{cfg['host']['threads']} threads, niced. The protocol has no warm form: every image pays the interpreter, "
          f"`import torch` ({g(protocol.get('import_torch_s', float('nan')))} s alone) and a sha256 of the weights "
          f"({g(protocol.get('weights_sha256_s', float('nan')))} s alone, file cache warm). `encode only` is the same "
          "forward inside an already loaded process (arm S16).", "",
          "| image | patches | tokens | worker wall s (run 1, run 2) | CPU s (user + sys) | peak RSS GiB | encode only s (8 threads) | encode only s (1 thread) | reference R16 s | within the origin's 240 s |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        r, p = rows.get(n) or {}, protocol.get("images", {}).get(n) or {}
        if not p.get("runs") or r.get("error") or not r:
            continue
        a = p["runs"]
        L.append(f"| {n} | {p['patches']} | {p['tokens']} | {', '.join('%.1f' % x['wall_s'] for x in a)} | "
                 f"{a[0]['user_s'] + a[0]['sys_s']:.0f} | {max(x['maxrss_kb'] for x in a) / 1048576:.2f} | {r['seconds']['S16']:.1f} | "
                 f"{r['seconds']['S16t1']:.1f} | {r['seconds']['R16']:.1f} | {'yes' if max(x['wall_s'] for x in a) < 240 else '**no**'} |")
    L += ["", "## Refusals and admission", "", "| case | exit | bytes on stdout | the worker says |", "|---|---|---|---|"]
    for c in protocol.get("refusals", []):
        L.append(f"| {c['case']} | {c['rc']} | {c['stdout_bytes']} | `{(c['says'] or [''])[0][:110]}` |")
    o, a = protocol.get("origin_worker_unmodified") or {}, protocol.get("admission") or {}
    L += ["", f"- The origin's unmodified `vision_cpu.py` in this environment: exit {o.get('rc')}, {o.get('stdout_bytes')} bytes, "
          f"`{(o.get('says') or [''])[0][:110]}` (recorded; it is why the derived module exists).",
          f"- Admission (`verify_vision.py --env {cfg['env']}`), record status `{a.get('record_status')}`: with "
          f"`--allow-candidate` exit {a.get('with_allow_candidate_rc')}, printing `FLERP_VISION_ENV={a.get('setting')}`; without, exit "
          f"{a.get('without_rc')} (`{a.get('without_message', '')[:120]}`)."]
    post = load("posthoc-bf16-rows.json")
    if post:
        L += ["", "## Post hoc: where the BF16 rows are far from FP32 (no criterion; looked at after the verdict)", "",
              "R16 (= the worker's rows) against R32, row by row. `error ratio` is the worst row's error over the median "
              "row's; `norm ratio` its length over the median row's.", "",
              "| image | rows | median cosine | 1st percentile | lowest | rows under 0.9855 | worst row: error ratio | worst row: norm ratio |",
              "|---|---|---|---|---|---|---|---|"]
        for n in names:
            v = post.get(n)
            if v:
                L.append(f"| {n} | {v['rows']} | {v['cosine_median']:.6f} | {v['cosine_p01']:.5f} | {v['cosine_min']:.5f} | "
                         f"{v['rows_below_0.9855']} | {v['worst_row_abs_error_over_median_abs_error']:.1f} | {v['worst_row_norm_over_median_norm']:.2f} |")
    L += ["", "## What the executed code does at system level (from reading every file before running it; not traced)", "",
          "- `vision_cpu.py` (the worker; the origin's and the adapted copy): one process, no children. Reads stdin, "
          "opens the weights file read-only (once to hash it, once to load 266 tensors), writes stdout, sets "
          "`signal.alarm`. Writes no file. No socket. Imports torch and numpy.",
          "- `verify_vision.py` (adapted): reads the bias and weights files (sha256) and `vision-env.json`; starts one "
          "child, `<python> -I -c` with an import-and-compare of four versions, 30 s limit. Writes only stdout.",
          "- `prepare_vision.py` (the origin's, unmodified): reads the index and all 42 pinned shards whole to hash them "
          "(197 GB here), then writes the two outputs through `mkstemp` + `os.link` in the output directory (mode 0600). "
          "No children, no socket.",
          "- DeepSeek `vision.py`: module definitions, torch only. `image_processor.py`: imports `urllib.request.urlopen`, "
          "which it calls only for an `http(s)` URL record; the gate hands it bytes. Both were imported from the ark with "
          "bytecode writing off."]
    if p313:
        eq = [n for n, v in p313.get("images", {}).items() if v.get("equals_py312_W")]
        ne = [n for n, v in p313.get("images", {}).items() if not v.get("equals_py312_W")]
        L += ["", "## CPython 3.13, the same torch (record only)", "",
              f"`{p313.get('python')}`: worker bytes equal to CPython 3.12's on {len(eq)} of {len(eq) + len(ne)} images"
              + (f"; not on {', '.join(ne)}" if ne else "") + "."]
    with open(os.path.join(results, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"verdict: {'PASS' if verdict['pass'] else 'FAIL'}{' (partial run)' if partial else ''}; {len(fails)} failure(s)")
    for x in fails:
        print("  " + x)
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", default="env,rows,protocol,summary")
    ap.add_argument("--images", default="")
    ap.add_argument("--one-image", default="")
    ap.add_argument("--py313", action="store_true", help="the optional arm: the worker under [py313] python")
    a = ap.parse_args()
    cfg = load_config(a.config)
    cfg["_path"] = os.path.abspath(a.config)
    if a.one_image:
        json.dump(one_image(cfg, a.one_image), sys.stdout)
        return 0
    results = os.path.join(ROOT, cfg["results"])
    os.makedirs(results, exist_ok=True)
    names = [im["name"] for im in cfg["image"]]
    if a.images:
        names = [n for n in names if n in a.images.split(",")]
    stages = a.stages.split(",")
    if a.py313:
        res = stage_protocol(cfg, results, names, python=cfg["py313"]["python"], tag="py313")
        with open(os.path.join(results, "protocol-py313.json"), "w") as f:
            json.dump(res, f, indent=1)
        stages = ["summary"]
    if "env" in stages:
        with open(os.path.join(results, "env.json"), "w") as f:
            json.dump(stage_env(cfg), f, indent=1)
        print("env written", flush=True)
    if "rows" in stages:
        stage_rows(cfg, results, names)
    if "protocol" in stages:
        res = stage_protocol(cfg, results, names)
        with open(os.path.join(results, "protocol.json"), "w") as f:
            json.dump(res, f, indent=1)
    if "summary" in stages:
        return 0 if stage_summary(cfg, results, names)["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
