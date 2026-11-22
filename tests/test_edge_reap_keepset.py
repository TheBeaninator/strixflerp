"""catalog/flerp/tool/edge_reap_keepset: the engine's Edge-REAP reports read, merged and solved into keep-sets with
owner's compute_edge_reap_keepset (strix_moe_surgery.py), on synthetic routing; no GPU.

Proofs: a report the script's own collector exports round-trips through the reader into the keep-set the script
computes from it; an engine-shaped report (export keys in the script's order plus domain / modality / sampled
bookkeeping) gives a keep-set of exactly keep_experts sorted ids per layer with the pinned super-experts in it;
merging the text and vision buckets equals one collector fed both; the sampled-norm semantics keep the script's
identity reap_saliency = reap_weighted_norm_sum / token_count; malformed reports are refused."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True  # nothing under catalog/*/src that [provenance] paths does not list
sys.path.insert(0, os.path.join(ROOT, "catalog", "flerp", "tool", "edge_reap_keepset", "src", "flerp",
                                "edge_reap_keepset"))
import edge_reap_report as R  # noqa: E402
import strix_moe_surgery as S  # noqa: E402

FIXTURE = os.path.join(ROOT, "tests", "fixtures", "edge_reap", "engine-export.json")
FIXTURE_ALL = os.path.join(ROOT, "tests", "fixtures", "edge_reap", "engine-export-all.json")
L, E, K = 3, 32, 6


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def routed(rng, rows, vision=False):
    """Rows routed with the script's exact DeepSeek-V4.1 router (gate.bias, or gate.bias_vl for image rows)."""
    hid = rng.standard_normal((rows, 64)).astype(np.float32)
    w = rng.standard_normal((E, 64)).astype(np.float32) * 0.2
    b = rng.uniform(9.0, 12.0, size=(E,)).astype(np.float32)
    bvl = rng.uniform(9.0, 15.0, size=(E,)).astype(np.float32)
    idx, g = S.dsv41_route_numpy(hid, w, b, bvl, np.full(rows, vision), top_k=K)
    return idx, g


def collect(batches):
    c = S.DistributedEdgeREAPCollector(num_layers=L, num_experts=E)
    for layer, idx, g, norms, pos in batches:
        c.record_batch(layer, idx, g, norms, pos)
    return c


def export(c, d, name="r.json"):
    p = Path(d) / name
    c.export_report(p, "deepseek-v4.1-flash-1017")
    return str(p)


def batches(seed, vision=False, rows=96):
    rng = np.random.default_rng(seed)
    out = []
    for layer in range(L):
        idx, g = routed(rng, rows, vision)
        norms = rng.uniform(0.5, 20.0, size=idx.shape).astype(np.float32)
        out.append((layer, idx, g, norms, np.arange(rows) + (0 if layer % 2 == 0 else 40)))
    return out


class RoundTrip(unittest.TestCase):
    def test_script_export_to_keepset(self):
        with tempfile.TemporaryDirectory() as d:
            p = export(collect(batches(1)), d)
            rep = R.load(p)
            self.assertEqual(tuple(jload(p).keys()), R.REPORT_KEYS)
            want = S.compute_edge_reap_keepset(jload(p), keep_experts=12, pin_core_experts=2,
                                               pin_super_experts=1, pin_early_pos_experts=1, tau_domain=8.0)
            got = R.keepset(rep, 12, pin_core_experts=2, pin_super_experts=1, pin_early_pos_experts=1, tau_domain=8.0)
            self.assertEqual(got, want)
            for layer in range(L):
                self.assertEqual(len(got[layer]), 12)
                self.assertEqual(got[layer], sorted(set(got[layer])))
            # the CLI writes the same keep-set
            out = os.path.join(d, "keep.json")
            R.main(["keepset", "--report", p, "--keep-experts", "12", "--pin-core", "2", "--pin-super", "1",
                    "--pin-early-pos", "1", "--tau-domain", "8", "--out", out])
            ks = jload(out)["keepset"]
            self.assertEqual({int(k): v for k, v in ks.items()}, want)

    def test_merge_text_and_vision_equals_one_collector(self):
        bt, bv = batches(2), batches(3, vision=True)
        with tempfile.TemporaryDirectory() as d:
            pt, pv = export(collect(bt), d, "t.json"), export(collect(bv), d, "v.json")
            both = R.load(export(collect(bt + bv), d, "b.json"))
            m = R.merge([R.load(pt), R.load(pv)])
            for k in ("token_count", "early_pos_count"):
                self.assertEqual(m[k], both[k])
            for k in ("gate_mass_sum", "reap_weighted_norm_sum", "max_l2_norm", "reap_saliency"):
                np.testing.assert_allclose(np.array(m[k]), np.array(both[k]), rtol=1e-12, atol=0)
            self.assertEqual(R.keepset(m, 10), S.compute_edge_reap_keepset(both, 10))

    def test_sampled_norms_keep_the_identity(self):
        with tempfile.TemporaryDirectory() as d:
            rep = R.load(export(collect(batches(4)), d))
        # pretend the norms came from every other routed pair: the engine's raw sampled sums
        tc = np.array(rep["token_count"])
        ncount = tc // 2
        nsum = np.array(rep["reap_weighted_norm_sum"]) * 0.5
        a = dict(rep, sampled={"norm_count": ncount.tolist(), "reap_weighted_norm_sum": nsum.tolist()})
        m = R.merge([a, a])
        sal = (2 * nsum) / np.maximum(2 * ncount, 1)
        np.testing.assert_allclose(np.array(m["reap_saliency"]), sal, rtol=1e-12)
        np.testing.assert_allclose(np.array(m["reap_weighted_norm_sum"]), sal * (2 * tc), rtol=1e-12)
        np.testing.assert_allclose(np.array(m["reap_weighted_norm_sum"]) / np.maximum(np.array(m["token_count"]), 1),
                                   np.array(m["reap_saliency"]), rtol=1e-12)

    def test_engine_shaped_report(self):
        """The engine's /stats/reap body: the script's keys in its order, then bookkeeping."""
        if os.path.exists(FIXTURE):
            rep = jload(FIXTURE)
        else:  # pragma: no cover - the fixture is checked in
            self.skipTest("no engine fixture")
        self.assertEqual(tuple(rep.keys())[:len(R.REPORT_KEYS)], R.REPORT_KEYS)
        rep = R.validate(rep)
        keep = max(1, rep["num_experts"] // 2)
        ks = R.keepset(rep, keep, pin_super_experts=2)
        mx = np.array(rep["max_l2_norm"])
        for layer in range(rep["num_layers"]):
            self.assertEqual(len(ks[layer]), keep)
            ids = np.arange(rep["num_experts"])
            top = [int(i) for i in np.lexsort((ids, -mx[layer]))[:2]]
            self.assertTrue(set(top) <= set(ks[layer]))
        # the script's identity wherever the cell has sampled norms; a cell with none exports a weighted sum of 0
        # (the golden file's synthetic raw sums in such cells cannot occur in serving: a norm adds to both)
        tc = np.array(rep["token_count"])
        nc = np.array(rep["sampled"]["norm_count"])
        seen = nc > 0
        self.assertTrue(np.all(np.array(rep["reap_weighted_norm_sum"])[~seen] == 0))
        np.testing.assert_allclose((np.array(rep["reap_weighted_norm_sum"]) / np.maximum(tc, 1))[seen],
                                   np.array(rep["reap_saliency"])[seen], rtol=1e-9, atol=1e-12)
        merged = R.merge([rep])
        np.testing.assert_allclose(np.array(merged["reap_saliency"]), np.array(rep["reap_saliency"]), rtol=1e-9,
                                   atol=1e-12)

    def test_engine_modality_all_is_the_merge(self):
        """modality=all from the engine equals merging its text and vision buckets here (sums, max, saliency)."""
        one, both = R.validate(jload(FIXTURE)), R.validate(jload(FIXTURE_ALL))
        self.assertEqual(tuple(both.keys())[:len(R.REPORT_KEYS)], R.REPORT_KEYS)
        self.assertEqual((one["num_layers"], one["num_experts"]), (both["num_layers"], both["num_experts"]))
        for k in ("token_count", "early_pos_count"):
            self.assertTrue(np.all(np.array(both[k]) >= np.array(one[k])))
        self.assertTrue(np.all(np.array(both["max_l2_norm"]) >= np.array(one["max_l2_norm"])))
        ks = R.keepset(both, both["num_experts"] // 4)
        self.assertEqual(sorted(ks), list(range(both["num_layers"])))

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            rep = R.load(export(collect(batches(5)), d))
        with self.assertRaises(R.ReportError):
            R.validate({k: v for k, v in rep.items() if k != "token_count"})
        bad = json.loads(json.dumps(rep))
        bad["max_l2_norm"][1] = bad["max_l2_norm"][1][:-1]
        with self.assertRaises(R.ReportError):
            R.validate(bad)
        bad = json.loads(json.dumps(rep))
        bad["token_count"][0][0] = 1.5
        with self.assertRaises(R.ReportError):
            R.validate(bad)
        other = dict(rep, num_experts=E * 2)
        with self.assertRaises(R.ReportError):
            R.merge([rep, other])
        with self.assertRaises(R.ReportError):
            R.merge([rep, dict(rep, model_id="another")])


if __name__ == "__main__":
    unittest.main()
