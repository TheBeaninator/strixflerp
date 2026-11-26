"""k-dec2 SPEED-FUSE-1 (b7269694f8): the whole hc chain fused (hc_chain._hc_chain) against the served hc_dots decode
chain (_hc_post_only, _hc_finish_s role 1, _hc_mix_part of the stored streams, _hc_sinkhorn), bit for bit, then timed
back to back as HIP graphs. Weights and the row-invariance frame as in test_hc_fuse.py, but POSTED: the streams h and
the gathered partials G [world, R, D] fp32 are inputs (synthetic, world 2), as are the previous site's post / comb.
Rows 1, 3, 5, 6; the counters must leave every launch back at zero."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "tfp/tri/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import triton
import tri_kernels as K
import hc_fuse as F
import hc_chain as C

dev = "cuda"
D, NB, HC = 5120, 40, 4
WIDE = HC * D
WORLD = int(os.environ.get("WORLD", "2"))
eps, hc_eps, iters = 1e-20, 1e-6, 20
man = {}
for line in open(Path.home() / "tfp/exl3/tensors/manifest_full.tsv"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 6:
        man[p[0]] = p


def tensor(name, dtype):
    p = man[name]
    with open(p[1], "rb") as f:
        f.seek(int(p[2]))
        b = f.read(int(p[3]))
    t = torch.frombuffer(bytearray(b), dtype={"F32": torch.float32, "BF16": torch.bfloat16}[p[4]]).clone()
    return t.to(dev)


def site(l, kind):
    pre = f"layers.{l}."
    return {"fn": tensor(pre + f"hc_{kind}_fn", None).contiguous(), "base": tensor(pre + f"hc_{kind}_base", None),
            "scale": tensor(pre + f"hc_{kind}_scale", None),
            "nw": tensor(pre + ("attn_norm.weight" if kind == "attn" else "ffn_norm.weight"), None)}


def chain(s, h, g, post, comb, pre_in, h_out, out, pre_out, part, R):
    """the served hc_dots decode chain, in hcPre2's order and wiring (the mixes of the stored streams)."""
    rb = F.hc_rb(R)
    K._hc_post_only[(R, NB)](h, h_out, g, R * D, post, comb, WIDE=WIDE, D=D, KB=WIDE // NB, WORLD=WORLD,
                             PDL=False, num_warps=8)
    K._hc_finish_s[(R, 1)](h_out, part, s["base"], s["scale"], pre_in, s["nw"], out, pre_out, post, comb,
                           eps, hc_eps, out, out, out, out, out, out, out, out, D=D, NB=NB, ITERS=iters,
                           BLOCK=1024, NROT=0, RSPLIT=1, num_warps=8, SINK=False, PDL=False)
    K._hc_mix_part[(triton.cdiv(R, rb), NB, 3)](h_out, h_out, h_out, 0, post, comb, s["fn"], part, R, WIDE=WIDE,
                                                 D=D, NB=NB, SUB=128, MB=8, WORLD=1, POSTED=False, RB=rb,
                                                 num_warps=2, PDL=False)
    K._hc_sinkhorn[(R,)](part, s["base"], s["scale"], pre_out, post, comb, eps, hc_eps, D=D, NB=NB, ITERS=iters,
                         num_warps=8)


def fused(s, h, g, post, comb, pre_in, h_out, out, pre_out, part, cnt, R):
    rb = F.hc_rb(R)
    C._hc_chain[(triton.cdiv(R, rb), NB + 1, 3)](h, h_out, g, R * D, post, comb, s["fn"], part, cnt, s["base"],
                                                  s["scale"], pre_in, s["nw"], out, pre_out, eps, hc_eps, R,
                                                  WIDE=WIDE, D=D, NB=NB, SUB=128, MB=8, RB=rb, ITERS=iters,
                                                  BLOCK=1024, WORLD=WORLD, num_warps=8)


def bufs(R, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    h = (torch.randn(R, HC, D, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    gathered = (torch.randn(WORLD, R, D, generator=g) * 0.25).to(dev)
    pre_in = (torch.rand(R, HC, generator=g) * 0.8 + 0.1).to(dev)
    post = (torch.rand(R, HC, generator=g) * 2.0).to(dev)
    comb = torch.softmax(torch.rand(R, HC, HC, generator=g) * 4 - 2, dim=2).to(dev)  # the previous site's, row-wise
    return h, gathered, pre_in, post, comb


def outs(R):
    return (torch.full((R, HC, D), 7, dtype=torch.bfloat16, device=dev),   # h_out
            torch.full((R, D), 7, dtype=torch.bfloat16, device=dev), torch.full((R, HC), 7.0, device=dev),
            torch.full((R, HC), 7.0, device=dev), torch.full((R, HC * HC), 7.0, device=dev),
            torch.zeros(R * NB * 32, device=dev))


res = {"bits": []}
sites = {"L20.attn": site(20, "attn"), "L39.ffn": site(39, "ffn"), "L1.attn": site(1, "attn")}
cnt = torch.zeros(64, dtype=torch.int32, device=dev)
ok_all = True
for name, s in sites.items():
    for R in (1, 3, 5, 6):
        for seed in range(3):
            h, gathered, pre_in, post0, comb0 = bufs(R, 1000 * R + seed)
            a = outs(R)
            b = outs(R)
            pa, ca = post0.clone(), comb0.clone()
            pb, cb = post0.clone(), comb0.clone()
            chain(s, h, gathered, pa, ca, pre_in, a[0], a[1], a[2], a[5], R)
            fused(s, h, gathered, pb, cb, pre_in, b[0], b[1], b[2], b[5], cnt, R)
            torch.cuda.synchronize()
            names = ("h_out", "out", "pre_out", "post", "comb", "part")
            diffs = {}
            for nm, u, v in zip(names, (a[0], a[1], a[2], pa, ca, a[5]), (b[0], b[1], b[2], pb, cb, b[5])):
                iv = torch.int16 if u.dtype == torch.bfloat16 else torch.int32
                if nm in ("part",):
                    u = u.view(R, NB, 32)[:, :, :25].contiguous()
                    v = v.view(R, NB, 32)[:, :, :25].contiguous()
                diffs[nm] = int((u.contiguous().view(iv) != v.contiguous().view(iv)).sum())
            ok = all(d == 0 for d in diffs.values()) and int(cnt.sum()) == 0
            ok_all &= ok
            res["bits"].append({"site": name, "rows": R, "seed": seed, "diffs": diffs, "cnt_left": int(cnt.sum()),
                                "ok": ok})
            print(json.dumps(res["bits"][-1]), flush=True)
print(json.dumps({"bits_ok": ok_all}))
sys.exit(0 if ok_all else 1)
