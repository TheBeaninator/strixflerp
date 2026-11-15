"""format.flerp.convert_qwen4exp_p16 / tool.flerp.convert_qwen4exp_p16 on a small synthetic Qwen3.8-Flash-Next-shaped
BF16 checkpoint (two decoder layers, GDN + full attention with PLE, globals, MTP), no GPU.

References, all built here from the same checkpoint:
  OLD    the ORIGIN converter (format.strixite.convert_qwen4exp, v4) at the layout without p16
  PANEL  tool.flerp.convert_strixw_panel applied to OLD
  NEW    the raw path at the p16 layout (v5)
Proofs: NEW's tensors equal PANEL's component for component (bytes and hash, from the index; data compared too);
the raw path at the layout without p16 equals OLD tensor for tensor; --check-against OLD and PANEL (all tensors)
pass; --plan's file size equals the written file; refusals."""
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import strixw_header as S  # noqa: E402

TOOL_BUILD = os.path.join(ROOT, "catalog", "flerp", "tool", "convert_qwen4exp_p16", "src", "flerp",
                          "convert_qwen4exp_p16", "build.sh")
PANEL_BUILD = os.path.join(ROOT, "catalog", "flerp", "tool", "convert_strixw_panel", "src", "flerp",
                           "convert_strixw_panel", "build.sh")
V4_BUILD = os.path.join(ROOT, "catalog", "flerp", "tool", "strixw_patch", "src", "flerp", "strixw_patch", "build.sh")

FLEET = ("exp_gu=q4g128,exp_down=q4g32,sh_gu=q8g64,sh_down=q8g32,gdn_in=q4g128,gdn_out=q8g64,attn_qkv=q8g64,"
         "attn_o=q8g64,hc_down=q8g64,hc_up=q8g64,ple_kv=q8g64,lm_head=q8g64,mtp_fc=q8g64,embed=bf16")
P16 = ["exp_gu", "exp_down", "sh_gu", "sh_down", "gdn_in", "gdn_out", "attn_qkv", "attn_o", "ple_kv", "lm_head",
       "mtp_fc"]


def with_p16(layout, classes):
    return ",".join(p + "p16" if p.split("=")[0] in classes else p for p in layout.split(","))


def bf16(x):
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    u += 0x7FFF + ((u >> 16) & 1)
    return u >> 16


def write_safetensors(path, tensors):
    hdr, blobs, at = {}, [], 0
    for name, (dt, shape, vals) in tensors.items():
        if dt == "BF16":
            b = struct.pack("<%dH" % len(vals), *[bf16(v) for v in vals])
        else:
            b = struct.pack("<%dq" % len(vals), *vals)
        hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [at, at + len(b)]}
        blobs.append(b)
        at += len(b)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def checkpoint(d):
    """H=256, 4 experts of 64 (gate|up 128 rows), HC r=64, every Q tensor's rows a multiple of 16."""
    rnd = random.Random(11)
    t = {}

    def w(name, *shape, s=0.05):
        n = 1
        for x in shape:
            n *= x
        t[name] = ("BF16", list(shape), [rnd.gauss(0, s) for _ in range(n)])

    H, E, I, R = 256, 4, 64, 64

    def hc(p, inject):
        w(p + "input_mix_weight_down.weight", R, H)
        w(p + "input_mix_weight_up.weight", H, R)
        w(p + "hc_norm.weight", H, s=0.1)
        if inject:
            w(p + "block_inject_weight.weight", 32, H)

    def layer(P, full, ple):
        if full:
            a = P + "self_attn."
            w(a + "q_proj.weight", 64, H), w(a + "k_proj.weight", 32, H), w(a + "v_proj.weight", 32, H)
            w(a + "indexer.index_qk_proj.weight", 32, H), w(a + "o_proj.weight", H, 128)
            for n in ("q_norm", "k_norm", "indexer.q_layernorm", "indexer.k_layernorm"):
                w(a + n + ".weight", 32)
        else:
            a = P + "linear_attn."
            w(a + "in_proj_qkv.weight", 96, H), w(a + "in_proj_z.weight", 32, H)
            w(a + "in_proj_b.weight", 16, H), w(a + "in_proj_a.weight", 16, H), w(a + "out_proj.weight", H, 128)
            w(a + "A_log", 4), w(a + "dt_bias", 4), w(a + "conv1d.weight", 8, 1, 4), w(a + "norm.weight", 32)
        m = P + "mlp."
        w(m + "experts.gate_up_proj", E, 2 * I, H, s=0.2), w(m + "experts.down_proj", E, H, I, s=0.01)
        w(m + "shared_expert.gate_proj.weight", I, H), w(m + "shared_expert.up_proj.weight", I, H)
        w(m + "shared_expert.down_proj.weight", H, I)
        w(m + "gate.weight", E, H), w(m + "shared_expert_gate.weight", 1, H)
        hc(P + "attn_hyper_connection.", True)
        hc(P + "mlp_hyper_connection.", True)
        if ple:
            w(P + "ple.key_proj.weight", 32, H), w(P + "ple.value_proj.weight", 32, H)
            w(P + "ple.conv1d.weight", 8, 1, 4)
            for n in ("norm_conv", "norm_key", "norm_query"):
                w(P + "ple." + n + ".weight", 32)
            t[P + "ple.ple_embedding.layer_multipliers"] = ("I64", [3], [3, 5, 7])
            t[P + "ple.ple_embedding.ngram_heads_offsets"] = ("I64", [4], [0, 10, 20, 30])
            t[P + "ple.ple_embedding.ngram_heads_vocab_sizes"] = ("I64", [4], [10, 10, 10, 10])

    lm = "model.language_model."
    w(lm + "embed_tokens.weight", 128, H)
    hc(lm + "hyper_connection_mixer.", False)
    w("lm_head.weight", 128, H)
    layer(lm + "layers.0.", False, False)
    layer(lm + "layers.1.", True, True)
    w("mtp.fc_embedding.weight", H, H), w("mtp.fc_hidden.weight", H, H, s=0.01)
    w("mtp.pre_fc_norm_embedding.weight", H), w("mtp.pre_fc_norm_hidden.weight", H)
    hc("mtp.hyper_connection_mixer.", False)
    layer("mtp.layers.0.", True, False)
    items = list(t.items())
    half = len(items) // 2  # two shards, like a real checkpoint
    write_safetensors(os.path.join(d, "model-00001-of-00002.safetensors"), dict(items[:half]))
    write_safetensors(os.path.join(d, "model-00002-of-00002.safetensors"), dict(items[half:]))


def data(path, comp):
    with open(path, "rb") as f:
        f.seek(comp["offset"])
        return f.read(comp["bytes"])


@unittest.skipUnless(shutil.which("g++"), "no g++ on this host")
class ConvertQwen4expP16Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.tool, cls.panel, cls.ref = (os.path.join(cls.d, n) for n in ("tool", "panel", "ref"))
        subprocess.run([TOOL_BUILD, cls.tool], check=True)
        subprocess.run([PANEL_BUILD, cls.panel], check=True)
        subprocess.run([V4_BUILD, cls.ref, os.path.join(ROOT, "tests", "cpp", "convert_qwen4exp_p16_ref.cpp")],
                       check=True)
        cls.ckpt = os.path.join(cls.d, "ckpt")
        os.mkdir(cls.ckpt)
        checkpoint(cls.ckpt)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d)

    def run_tool(self, *args):
        return subprocess.run([self.tool, "--src", self.ckpt, "--git", "selftest", "--threads", "2", *args],
                              capture_output=True, text=True, timeout=300)

    def origin(self, name, layout):
        out = os.path.join(self.d, name)
        subprocess.run([self.ref, self.ckpt, out, layout], check=True, capture_output=True)
        return os.path.join(out, "weights.strixw")

    def ours(self, name, layout=None):
        out = os.path.join(self.d, name)
        p = self.run_tool("--out", out, *(["--layout", layout] if layout else []))
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return os.path.join(out, "weights.strixw")

    def panelize(self, old, classes):
        out = old + ".p16"
        p = subprocess.run([self.panel, "--old", old, "--out", out, "--classes", ",".join(classes)],
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return out

    def assert_same_tensors(self, a, b, expect_versions, b_layout=None):
        A, B = S.read_strixw(a), S.read_strixw(b)
        self.assertEqual((A["version"], B["version"]), expect_versions)
        self.assertEqual(A["meta"]["layout"], b_layout(B["meta"]["layout"]) if b_layout else B["meta"]["layout"])
        self.assertEqual(A["meta"]["source_fingerprint"], B["meta"]["source_fingerprint"])
        self.assertEqual([t["name"] for t in A["tensors"]], [t["name"] for t in B["tensors"]])
        for ta, tb in zip(A["tensors"], B["tensors"]):
            for k in ("encoding", "shape", "group_size", "experts", "parts"):
                self.assertEqual(ta[k], tb[k], (ta["name"], k))
            self.assertEqual([(c["role"], c["bytes"], c["hash"]) for c in ta["components"]],
                             [(c["role"], c["bytes"], c["hash"]) for c in tb["components"]], ta["name"])
            for ca, cb in zip(ta["components"], tb["components"]):
                self.assertEqual(data(a, ca), data(b, cb), ta["name"])
        return A

    def test_without_p16_equals_origin(self):
        old = self.origin("o_fleet", "default=q8g64," + FLEET)
        mine = self.ours("m_fleet", "default=q8g64," + FLEET)
        self.assert_same_tensors(old, mine, (4, 5))

    def test_p16_equals_panelized_origin(self):
        for tag, layout, classes in [
            ("split", FLEET, P16),
            # shared expert stacked as #E (exp == sh formats), and hc_down in panels too
            ("stacked", FLEET.replace("sh_gu=q8g64", "sh_gu=q4g128").replace("sh_down=q8g32", "sh_down=q4g32"),
             [c for c in P16 if c not in ("sh_gu", "sh_down")] + ["hc_down"]),
        ]:
            with self.subTest(tag):
                old = self.origin("o_" + tag, "default=q8g64," + layout)
                panel = self.panelize(old, classes)
                # sh_* share the experts' (now p16) format, so they stay stacked in the raw path too
                p16 = classes + (["sh_gu", "sh_down"] if tag == "stacked" else [])
                new = self.ours("n_" + tag, "default=q8g64," + with_p16(layout, p16))
                # stacked: the panel tool labels sh_* by the stack's old format (no own tensors); ours says p16
                unlabel = (lambda s: s.replace("sh_gu=q4g128p16", "sh_gu=q4g128").replace("sh_down=q4g32p16",
                                                                                        "sh_down=q4g32"))
                A = self.assert_same_tensors(panel, new, (5, 5), unlabel if tag == "stacked" else None)
                encs = {t["encoding"] for t in A["tensors"]}
                self.assertTrue({"q4-panel16", "q8-panel16"} <= encs, encs)
                for ref in (old, panel):
                    p = self.run_tool("--layout", "default=q8g64," + with_p16(layout, p16), "--check-against", ref,
                                      "--tensors", "all")
                    self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                    self.assertIn("differ", p.stderr)
                    self.assertIn(" 0 differ", p.stderr)
                    self.assertNotIn("FAIL", p.stderr)

    def test_default_layout_is_fleet_with_p16(self):
        p = self.run_tool("--plan", "--plan-scratch", self.d)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("layout: " + with_p16(FLEET, P16), p.stderr)
        new = self.ours("n_default")
        size = os.path.getsize(new)
        self.assertIn("file bytes %d " % size, p.stderr)
        self.assertEqual(S.read_strixw(new)["meta"]["layout"], with_p16(FLEET, P16))
        self.assertEqual([f for f in os.listdir(self.d) if f.startswith("convert_qwen4exp_p16_plan")], [])

    def test_check_detects_a_difference(self):
        old = self.origin("o_g64", "default=q8g64," + FLEET.replace("mtp_fc=q8g64", "mtp_fc=q8g32"))
        p = self.run_tool("--check-against", old, "--tensors", "mtp.fc_hidden.weight,lm_head.weight")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("FAIL mtp.fc_hidden.weight", p.stderr)
        self.assertIn("OK   lm_head.weight", p.stderr)

    def test_refusals(self):
        for lay in ["default=q8g64,hc_up=q8g64p16", "default=q8g64,embed=q8g64p16", "default=q8g64,gdn_in=q5g64p16",
                    "default=q8g64p16", "default=q8g64,gdn_in=q4g64x16", "default=q8g64,exp_gu=q8g64p16"]:
            out = os.path.join(self.d, "never")
            p = self.run_tool("--out", out, "--layout", lay)
            self.assertNotEqual(p.returncode, 0, lay)
            self.assertFalse(os.path.exists(os.path.join(out, "weights.strixw")), lay)


if __name__ == "__main__":
    unittest.main()
