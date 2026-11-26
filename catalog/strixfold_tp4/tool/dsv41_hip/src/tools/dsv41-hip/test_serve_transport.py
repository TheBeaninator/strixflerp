#!/usr/bin/env python3
"""serve_lib.sh's transport checks (serve_transport) and the collective settings a rank's environment gets (serve_env).

    python3 -m unittest tools/dsv41-hip/test_serve_transport.py
"""
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


def resolve(rank=0, **keys):
    """serve.conf with keys set over it, through serve_transport and serve_env as the given rank: (exit code, stderr,
    the environment)."""
    sets = "; ".join(f"{k}={shlex.quote(v)}" for k, v in keys.items())
    script = (f"source {shlex.quote(str(HERE / 'serve_lib.sh'))}; source {shlex.quote(str(HERE / 'serve.conf'))}; {sets}; "
              f'RANK={rank}; serve_transport; serve_env; printf "%s\\n" "${{SERVE_ENV[@]}}"')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    env = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
    return r.returncode, r.stderr, env


RDMA = dict(TRANSPORT="rdma", COLL_PROXY_STRIPE_KIB="", RDMA_DEV0="rocep195s0", RDMA_DEV1="rocep194s0", RDMA_GID="3")


class Transport(unittest.TestCase):
    def test_tcp_keeps_its_kind_and_its_stripe(self):
        rc, err, env = resolve(TRANSPORT="tcp", COLL_KIND="proxy", TP_STRIPE="198.51.100.51", COLL_STRIPE_MIB="4")
        self.assertEqual(rc, 0, err)
        self.assertEqual(env["TF_COLL_KIND"], "proxy")
        self.assertEqual(env["TF_TP_STRIPE"], "198.51.100.51")
        self.assertNotIn("TF_RDMA_GID_INDEX", env)

    def test_rdma_without_a_stripe(self):
        rc, err, env = resolve(TP_STRIPE="", COLL_WIRE_THREADS="", **RDMA)
        self.assertEqual(rc, 0, err)
        self.assertEqual(env["TF_COLL_KIND"], "rdma")
        self.assertNotIn("TF_TP_STRIPE", env)
        self.assertNotIn("TF_COLL_WIRE_THREADS", env)
        self.assertEqual(env["TF_RDMA_DEVICES"], "rocep195s0,rocep194s0")

    def test_rdma_at_world_4_names_each_ranks_own_device(self):
        # a two-name list covers ranks 0 and 1 only: rank 2 refused it at load (716f35d40a run-2003)
        devs = dict(RDMA_DEV2="rocep9s0", RDMA_DEV3="rocep8s0")
        for rank, want in enumerate(["rocep195s0", "rocep194s0", "rocep9s0", "rocep8s0"]):
            rc, err, env = resolve(rank, WORLD="4", TP_STRIPE="", COLL_WIRE_THREADS="", **RDMA, **devs)
            self.assertEqual(rc, 0, err)
            self.assertEqual(env["TF_RDMA_DEVICES"], want)
            self.assertEqual(env["TF_DS_COLL"], "tcp")

    def test_rdma_takes_tcp_hosts_stripe_and_its_thread_a_link(self):
        # a gather larger than a slot goes through tcp_host under rdma too, so its settings reach the rank
        rc, err, env = resolve(TP_STRIPE="198.51.100.51", COLL_STRIPE_MIB="4", COLL_WIRE_THREADS="2", **RDMA)
        self.assertEqual(rc, 0, err)
        self.assertEqual(env["TF_COLL_KIND"], "rdma")
        self.assertEqual(env["TF_TP_STRIPE"], "198.51.100.51")
        self.assertEqual(env["TF_COLL_STRIPE_MIB"], "4")
        self.assertEqual(env["TF_COLL_WIRE_THREADS"], "2")
        self.assertEqual(env["TF_RDMA_GID_INDEX"], "3")
        self.assertNotIn("TF_COLL_PROXY_STRIPE_KIB", env)  # the slot gathers' stripe thread is tcp_proxy's only

    def test_rdma_refuses_the_slot_stripe_thread(self):
        rc, err, _ = resolve(TP_STRIPE="198.51.100.51", **dict(RDMA, COLL_PROXY_STRIPE_KIB="64"))
        self.assertNotEqual(rc, 0)
        self.assertIn("COLL_PROXY_STRIPE_KIB", err)

    def test_two_threads_need_a_second_link_under_either_transport(self):
        for keys in (dict(TRANSPORT="tcp", COLL_KIND="proxy"), RDMA):
            rc, err, _ = resolve(TP_STRIPE="", COLL_WIRE_THREADS="2", **keys)
            self.assertNotEqual(rc, 0)
            self.assertIn("COLL_WIRE_THREADS=2 needs TP_STRIPE", err)

    def test_rdma_devices_come_together(self):
        rc, err, _ = resolve(TP_STRIPE="", **dict(RDMA, RDMA_DEV1=""))
        self.assertNotEqual(rc, 0)
        self.assertIn("RDMA_DEV0 and RDMA_DEV1 come together", err)

    def test_another_transport_is_refused(self):
        rc, err, _ = resolve(TRANSPORT="usb")
        self.assertNotEqual(rc, 0)
        self.assertIn("TRANSPORT must be tcp, rdma or rccl", err)



def resolve_lanes(**keys):
    """serve.conf with keys set over it, through serve_lanes_cmd: (exit code, stderr, rank 1's argv)."""
    sets = "; ".join(f"{k}={shlex.quote(v)}" for k, v in keys.items())
    script = (f"source {shlex.quote(str(HERE / 'serve_lib.sh'))}; source {shlex.quote(str(HERE / 'serve.conf'))}; {sets}; "
              'RANK=1; BIN=/b; serve_lanes_cmd; printf "%s\\n" "${SERVE_CMD[@]}"')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return r.returncode, r.stderr, r.stdout.splitlines()


class ImageTokens(unittest.TestCase):
    def test_empty_is_the_engines_4096_and_no_flag_or_env(self):
        rc, err, env = resolve(MODALITIES="text_image", VISION_DIR="/v")
        self.assertEqual(rc, 0, err)
        self.assertNotIn("TF_DS_IMAGE_TOKENS", env)
        rc, err, argv = resolve_lanes(MODALITIES="text_image")
        self.assertEqual(rc, 0, err)
        self.assertNotIn("--image-tokens", argv)

    def test_a_value_reaches_both_ranks(self):
        rc, err, env = resolve(MODALITIES="text_image", VISION_DIR="/v", IMAGE_TOKENS="32768")
        self.assertEqual(rc, 0, err)
        self.assertEqual(env["TF_DS_IMAGE_TOKENS"], "32768")
        rc, err, argv = resolve_lanes(MODALITIES="text_image", IMAGE_TOKENS="32768")
        self.assertEqual(rc, 0, err)
        self.assertEqual(argv[argv.index("--image-tokens") + 1], "32768")

    def test_text_only_ignores_it(self):
        rc, err, env = resolve(MODALITIES="text", IMAGE_TOKENS="32768")
        self.assertEqual(rc, 0, err)
        self.assertNotIn("TF_DS_IMAGE_TOKENS", env)
        rc, err, argv = resolve_lanes(MODALITIES="text", IMAGE_TOKENS="32768")
        self.assertNotIn("--image-tokens", argv)

    def test_the_check_refuses_a_bad_count(self):
        lib = (HERE / "serve_lib.sh").read_text()
        check = next(l for l in lib.splitlines() if 'IMAGE_TOKENS =~' in l).strip().rstrip("|").strip()
        for value, ok in (("1", True), ("4096", True), ("65536", True), ("0", False), ("65537", False), ("32k", False), ("-5", False), ("0100", False)):
            rc = subprocess.run(["bash", "-c", f"IMAGE_TOKENS={shlex.quote(value)}; {check} || exit 7"]).returncode
            self.assertEqual(rc == 0, ok, value)


class EncodeCache(unittest.TestCase):
    """ENCODE_CACHE_MIB, the picture cache (pod card d6a577e1a1): rank 0's env only, and only with text_image."""

    def test_empty_sets_no_env(self):
        rc, err, env = resolve(MODALITIES="text_image", VISION_DIR="/v")
        self.assertEqual(rc, 0, err)
        self.assertNotIn("TF_DS_ENCODE_CACHE_MIB", env)

    def test_a_value_reaches_rank_0(self):
        for value in ("0", "512", "2048"):
            rc, err, env = resolve(MODALITIES="text_image", VISION_DIR="/v", ENCODE_CACHE_MIB=value)
            self.assertEqual(rc, 0, err)
            self.assertEqual(env["TF_DS_ENCODE_CACHE_MIB"], value)

    def test_text_only_ignores_it(self):
        rc, err, env = resolve(MODALITIES="text", ENCODE_CACHE_MIB="512")
        self.assertEqual(rc, 0, err)
        self.assertNotIn("TF_DS_ENCODE_CACHE_MIB", env)

    def test_the_check_refuses_a_bad_size(self):
        lib = (HERE / "serve_lib.sh").read_text()
        check = next(l for l in lib.splitlines() if 'ENCODE_CACHE_MIB =~' in l).strip().rstrip("|").strip()
        for value, ok in (("0", True), ("1", True), ("512", True), ("65536", True), ("65537", False), ("99999", False), ("512M", False), ("-1", False), ("0100", False)):
            rc = subprocess.run(["bash", "-c", f"ENCODE_CACHE_MIB={shlex.quote(value)}; {check} || exit 7"]).returncode
            self.assertEqual(rc == 0, ok, value)


class PillowPath(unittest.TestCase):
    """TF_PIL_PATH: the deploy's pylib/ (a Pillow for the server's python3 -I child), rank 0 with pictures, when the deploy has one."""

    def test_the_deploys_pylib_reaches_rank_0_only_with_pictures(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "pylib" / "PIL").mkdir(parents=True)
            rc, err, env = resolve(MODALITIES="text_image", VISION_DIR="/v", DEPLOY_DIR=d)
            self.assertEqual(rc, 0, err)
            self.assertEqual(env["TF_PIL_PATH"], f"{d}/pylib")
            rc, err, env = resolve(1, MODALITIES="text_image", VISION_DIR="/v", DEPLOY_DIR=d)
            self.assertNotIn("TF_PIL_PATH", env)
            rc, err, env = resolve(MODALITIES="text", DEPLOY_DIR=d)
            self.assertNotIn("TF_PIL_PATH", env)

    def test_a_deploy_without_pylib_sets_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            rc, err, env = resolve(MODALITIES="text_image", VISION_DIR="/v", DEPLOY_DIR=d)
            self.assertEqual(rc, 0, err)
            self.assertNotIn("TF_PIL_PATH", env)


def resolve_policy(drop_policy=False, **keys):
    """serve.conf with keys set over it (drop_policy: the conf's LOOP_POLICY removed, an old configuration), through
    serve_loop_policy and rank 0's serve_cmd: (exit code, stderr, rank 0's argv)."""
    sets = "; ".join(f"{k}={shlex.quote(v)}" for k, v in keys.items())
    unset = "unset LOOP_POLICY; " if drop_policy else ""
    script = (f"source {shlex.quote(str(HERE / 'serve_lib.sh'))}; source {shlex.quote(str(HERE / 'serve.conf'))}; {unset}{sets + '; ' if sets else ''}"
              'RANK=0; BIN=/b; serve_loop_policy; serve_cmd; printf "%s\\n" "${SERVE_CMD[@]}"')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return r.returncode, r.stderr, r.stdout.splitlines()


class LoopPolicy(unittest.TestCase):
    """LOOP_POLICY (pod card f64f606edd): rank 0's --loop-policy; LOOP_GUARD is its old name for close."""

    def test_serve_conf_default_is_stop(self):
        rc, err, argv = resolve_policy()
        self.assertEqual(rc, 0, err)
        self.assertEqual(argv[argv.index("--loop-policy") + 1], "stop")
        self.assertNotIn("--loop-guard", argv)

    def test_count_and_stop_pass_the_flag(self):
        for policy in ("stop", "count"):
            rc, err, argv = resolve_policy(LOOP_POLICY=policy)
            self.assertEqual(rc, 0, err)
            self.assertEqual(argv[argv.index("--loop-policy") + 1], policy)

    def test_close_is_the_old_loop_guard_argv(self):
        rc, err, argv = resolve_policy(LOOP_POLICY="close")
        self.assertEqual(rc, 0, err)
        self.assertIn("--loop-guard", argv)
        self.assertNotIn("--loop-policy", argv)

    def test_an_old_conf_keeps_loop_guard(self):
        for guard, flag in (("1", True), ("0", False)):
            rc, err, argv = resolve_policy(drop_policy=True, LOOP_GUARD=guard)
            self.assertEqual(rc, 0, err)
            self.assertEqual("--loop-guard" in argv, flag, guard)
            self.assertNotIn("--loop-policy", argv)
        rc, err, argv = resolve_policy(drop_policy=True)  # neither set: close, as LOOP_GUARD's default was
        self.assertIn("--loop-guard", argv)

    def test_off_passes_nothing(self):
        rc, err, argv = resolve_policy(LOOP_POLICY="off")
        self.assertEqual(rc, 0, err)
        self.assertNotIn("--loop-guard", argv)
        self.assertNotIn("--loop-policy", argv)

    def test_a_bad_policy_is_refused(self):
        rc, err, _ = resolve_policy(LOOP_POLICY="guard")
        self.assertNotEqual(rc, 0)
        self.assertIn("LOOP_POLICY must be", err)



class RouteRecords(unittest.TestCase):
    """ROUTE-RECORDS (28153e18f3): rank 0 only, nothing exported while off."""

    def test_off_exports_nothing(self):
        for rank in (0, 1):
            rc, err, env = resolve(rank, ROUTE_RECORDS="0")
            self.assertEqual(rc, 0, err)
            self.assertFalse([k for k in env if k.startswith("TF_DS_ROUTE_RECORDS")])

    def test_on_reaches_rank_0_only(self):
        rc, err, env = resolve(0, ROUTE_RECORDS="1", ROUTE_RECORDS_DIR="/tmp/rr", ROUTE_RECORDS_TOPK="8", ROUTE_RECORDS_MAX_MB="64")
        self.assertEqual(rc, 0, err)
        self.assertEqual(env["TF_DS_ROUTE_RECORDS"], "1")
        self.assertEqual(env["TF_DS_ROUTE_RECORDS_DIR"], "/tmp/rr")
        self.assertEqual(env["TF_DS_ROUTE_RECORDS_TOPK"], "8")
        self.assertEqual(env["TF_DS_ROUTE_RECORDS_MAX_MB"], "64")
        rc, err, env = resolve(1, ROUTE_RECORDS="1", ROUTE_RECORDS_DIR="/tmp/rr")
        self.assertEqual(rc, 0, err)
        self.assertFalse([k for k in env if k.startswith("TF_DS_ROUTE_RECORDS")])

class Acts(unittest.TestCase):
    """SERVE-ACTS (4f9ba05838): rank 0 only, nothing exported while ACTS_DIR is empty."""

    def test_off_exports_nothing(self):
        for rank in (0, 1):
            rc, err, env = resolve(rank, ACTS_DIR="", ACTS_REQUEST_MB="64")
            self.assertEqual(rc, 0, err)
            self.assertFalse([k for k in env if k.startswith("TF_DS_ACTS")])

    def test_on_reaches_rank_0_only(self):
        rc, err, env = resolve(0, ACTS_DIR="/tmp/acts", ACTS_ROWS_DIR="/tmp/rows", ACTS_REQUEST_MB="64",
                               ACTS_TOTAL_MB="4096", ACTS_QUEUE_MB="256", ACTS_DUMP_ROWS="384")
        self.assertEqual(rc, 0, err)
        self.assertEqual(env["TF_DS_ACTS_DIR"], "/tmp/acts")
        self.assertEqual(env["TF_DS_ACTS_ROWS_DIR"], "/tmp/rows")
        self.assertEqual(env["TF_DS_ACTS_REQUEST_MB"], "64")
        self.assertEqual(env["TF_DS_ACTS_TOTAL_MB"], "4096")
        self.assertEqual(env["TF_DS_ACTS_QUEUE_MB"], "256")
        self.assertEqual(env["TF_DS_ACTS_DUMP_ROWS"], "384")
        rc, err, env = resolve(0, ACTS_DIR="/tmp/acts")
        self.assertEqual(rc, 0, err)
        self.assertEqual(env["TF_DS_ACTS_DIR"], "/tmp/acts")
        self.assertFalse([k for k in env if k.startswith("TF_DS_ACTS_") and k != "TF_DS_ACTS_DIR"])
        rc, err, env = resolve(1, ACTS_DIR="/tmp/acts", ACTS_ROWS_DIR="/tmp/rows")
        self.assertEqual(rc, 0, err)
        self.assertFalse([k for k in env if k.startswith("TF_DS_ACTS")])

class MaskExperts(unittest.TestCase):
    """EXPERT-MASK (5bb471d3b9): every rank gets the mask file; empty or unset exports nothing (the base launch)."""

    def test_empty_exports_nothing(self):
        for rank in (0, 1):
            rc, err, env = resolve(rank, MASK_EXPERTS="")
            self.assertEqual(rc, 0, err)
            self.assertNotIn("TF_DS_MASK_EXPERTS", env)

    def test_the_conf_default_is_empty(self):
        for conf in ("serve.conf", "tp4.conf"):
            text = (HERE / conf).read_text()
            self.assertIn("\nMASK_EXPERTS=\n", text, conf)

    def test_set_reaches_every_rank(self):
        for rank in (0, 1):
            rc, err, env = resolve(rank, MASK_EXPERTS="/tmp/mask-a.txt")
            self.assertEqual(rc, 0, err)
            self.assertEqual(env["TF_DS_MASK_EXPERTS"], "/tmp/mask-a.txt")


if __name__ == "__main__":
    unittest.main()
