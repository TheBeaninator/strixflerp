"""What gate/pair_stall_diag.py and gate/pair_stall_sampler.py decide without a pod: the /proc parsers, the conf
copy's appended lines, a counter's share of a frame, and the leads' verdicts from synthetic samples."""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gate"))
import pair_stall_diag as D  # noqa: E402
import pair_stall_sampler as P  # noqa: E402

MOUNTSTATS = """device /dev/nvme0n1p1 mounted on /cache with fstype xfs
device 10.10.0.31:/data/models mounted on /data/models with fstype nfs4 statvers=1.1
\topts:\tro,vers=4.2,rsize=1048576
\tage:\t372000
\tbytes:\t46800000000 0 0 0 984700000000 0 240408000 0
\tRPC iostats version: 1.1  p/v: 100003/4 (nfs)
\txprt:\ttcp 0 0 1 0 0 4000 4000 0 5000 0 2 100 100
\tper-op statistics
\t        NULL: 1 1 0 44 24 0 0 0 0
\t        READ: 950000 950001 0 190000000 984800000000 1200 4500000 4600000 0
\t       WRITE: 0 0 0 0 0 0 0 0 0
device 10.20.0.31:/data/models mounted on /data/models-rail with fstype nfs statvers=1.1
\tbytes:\t369000000000 0 1000 0 369000001000 0 90000000 0
\tper-op statistics
\t        READ: 360000 360000 0 72000000 369100000000 800 700000 720000
"""
PSI = "some avg10=1.50 avg60=0.40 avg300=0.10 total=21208904\nfull avg10=0.75 avg60=0.20 avg300=0.05 total=21072628\n"
NETDEV = """Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 100 1 0 0 0 0 0 0 100 1 0 0 0 0 0 0
  eno1: 1078000 900 0 0 0 0 0 0 55000 800 0 0 0 0 0 0
enp194s0np0: 12850000 9000 0 0 0 0 0 0 12000000 8000 0 0 0 0 0 0
"""
PID_STAT = ("1467674 (tf (dsv41) lanes) S 1 1467674 1467674 0 -1 4194560 5000 0 321 0 900 400 0 0 20 0 40 0 123456 "
            "1000000 274000 18446744073709551615 1 1 0 0 0 0 0 0 0 0 0 0 17 3 0 0 77 0 0 0 0 0 0 0 0 0 0")
CONF = """# a comment with WRAP0=not this
PROMPT_CACHE_GIB=4               # kept prompt states
WRAP0='/usr/bin/env LD_LIBRARY_PATH=/data/models/tools/therock-10.2/lib'
WRAP1='/usr/bin/env LD_LIBRARY_PATH=/data/models/tools/therock-10.2/lib'
DEPLOY_DIR='$HOME/tfp/deploy/f64b5ff2e28b'   # the build
ENGRAM_AIO=0                     # rank-local
TP_STRIPE=
LEARN_DIR=/cache/tfp-learned     # shared prompt states
BENCH=1
BENCH=0
"""
HDR = {"kind": "hdr", "ifaces": {"eno1": "10.10.0.54", "enp194s0np0": "10.20.0.54"}, "mlocked": True,
       "mounts": {"/data/models": "10.10.0.31:/data/models", "/data/models-rail": "10.20.0.31:/data/models", "/mnt/nv1": "10.10.0.31:/mnt/nv1"}}


def synth(shape):
    """Samples every 2 s for 80 s and frames: 10 normal (1.4 s) in 0-20 s, 6 stalled (10 s) in 20-80 s. `shape`:
    {counter: (rate a second in normal time, rate a second in stalled time)}."""
    samples, acc = [], {k: 0.0 for k in shape}
    for i in range(41):
        t = 1000.0 + 2 * i
        samples.append({"kind": "s", "t": t, "c": dict(acc), "g": {"mem:MemAvailable": 13 * 1048576, "load1": 2.0 if t < 1020 else 40.0}})
        for k, (n, s) in shape.items():
            acc[k] += 2 * (n if t < 1020 else s)
    events = [{"kind": "ev", "ev": "frame", "frame": "fill_chunk", "rank": 1, "step": i, "ms": 1400, "t": 1000.0 + 2 * i + 1.4} for i in range(10)]
    events += [{"kind": "ev", "ev": "frame", "frame": "fill_chunk", "rank": 1, "step": 10 + i, "ms": 10000, "t": 1030.0 + 10 * i} for i in range(6)]
    return samples, events


BASE = {"net:enp194s0np0:rx_bytes": (400e6, 60e6), "net:enp194s0np0:tx_bytes": (400e6, 60e6), "net:eno1:rx_bytes": (2e3, 2e3), "net:eno1:tx_bytes": (2e3, 2e3),
        "nfs:/data/models:server_read_bytes": (0, 0), "nfs:/data/models:read_exe_ms": (0, 0), "nfs:/data/models:read_ops": (0, 0),
        "nfs:/data/models-rail:server_read_bytes": (20e6, 20e6), "nfs:/data/models-rail:read_exe_ms": (20, 20), "nfs:/data/models-rail:read_rtt_ms": (15, 15),
        "nfs:/data/models-rail:read_ops": (20, 20), "nfs:/mnt/nv1:server_read_bytes": (0, 0),
        "blk:loop0:read_sectors": (0, 0), "blk:loop0:read_ms": (0, 0), "pid:majflt": (0, 0), "vm:workingset_refault_file": (0, 0),
        "vm:pgscan_direct": (0, 0), "vm:pgscan_kswapd": (0, 0), "vm:compact_stall": (0, 0), "psi:memory:some_us": (0, 0), "psi:memory:full_us": (0, 0),
        "psi:io:some_us": (0, 0)}


def verdicts(over, cache=()):
    samples, events = synth(dict(BASE, **over))
    frames = D.frames_of(events)
    by = {c: [f for f in frames if f["cls"] == c] for c in ("stalled", "normal")}
    m = {c: D.class_metrics(samples, HDR, by[c]) for c in by}
    leads = D.judge(m["stalled"], m["normal"], D.class_metrics(samples, HDR, frames), cache, frames)
    return {k: v[0] for k, v in leads.items()}, m


class Parsers(unittest.TestCase):
    def test_mountstats(self):
        m = P.parse_mountstats(MOUNTSTATS)
        self.assertEqual(sorted(m), ["/data/models", "/data/models-rail"])
        self.assertEqual(m["/data/models"]["device"], "10.10.0.31:/data/models")
        self.assertEqual(m["/data/models"]["normal_read_bytes"], 46800000000)
        self.assertEqual(m["/data/models"]["server_read_bytes"], 984700000000)
        self.assertEqual((m["/data/models"]["read_ops"], m["/data/models"]["read_queue_ms"], m["/data/models"]["read_rtt_ms"],
                          m["/data/models"]["read_exe_ms"]), (950000, 1200, 4500000, 4600000))
        self.assertEqual(m["/data/models-rail"]["read_exe_ms"], 720000)        # a READ line without the errors column
        self.assertEqual(m["/data/models-rail"]["direct_read_bytes"], 1000)

    def test_psi_netdev_meminfo_vmstat(self):
        self.assertEqual(P.parse_psi(PSI), {"some_avg10": 1.5, "some_us": 21208904, "full_avg10": 0.75, "full_us": 21072628})
        n = P.parse_netdev(NETDEV)
        self.assertEqual(sorted(n), ["eno1", "enp194s0np0"])
        self.assertEqual(n["enp194s0np0"], {"rx_bytes": 12850000, "tx_bytes": 12000000})
        self.assertEqual(P.parse_meminfo("MemTotal: 130494228 kB\nMemAvailable:   13904532 kB\nActive(file):  1000 kB\n"),
                         {"MemAvailable": 13904532, "Active(file)": 1000})
        self.assertEqual(P.parse_vmstat("pgfault 10\npgmajfault 3\nnr_free_pages 9\nworkingset_refault_file 7\n"),
                         {"pgfault": 10, "pgmajfault": 3, "workingset_refault_file": 7})
        self.assertEqual(P.parse_block_stat("  137498        0  4098130 1020827584        0        0"),
                         {"read_ios": 137498, "read_sectors": 4098130, "read_ms": 1020827584})

    def test_pid_files(self):
        s = P.parse_pid_stat(PID_STAT)                      # a command name with spaces and parentheses
        self.assertEqual((s["state"], s["minflt"], s["majflt"], s["utime"], s["stime"], s["blkio_ticks"]), ("S", 5000, 321, 900, 400, 77))
        self.assertEqual(P.parse_pid_io("rchar: 10\nwchar: 2\nsyscr: 3\nread_bytes: 4096\n"), {"rchar": 10, "syscr": 3, "read_bytes": 4096})
        self.assertEqual(P.parse_pid_status("Name:\tx\nRssAnon:\t  725580 kB\nRssFile:\t  176492 kB\n"), {"RssAnon": 725580, "RssFile": 176492})

    def test_log_lines(self):
        e = P.parse_log_line("warning: lanes: step 308 rank 1 spent 19764 ms on its fill_chunk frame")
        self.assertEqual(e, {"ev": "frame", "step": 308, "rank": 1, "ms": 19764, "frame": "fill_chunk"})
        e = P.parse_log_line("info: prompt cache: 20337 tokens, resumed at 4096, kept 1; 20 states, 127 MiB and 0 MiB spare of 4096 MiB "
                             "(hits 26, misses 2, evicted 0, refused 0, failed 1)")
        self.assertEqual((e["ev"], e["states"], e["mib"], e["cap_mib"], e["evicted"], e["failed"]), ("cache", 20, 127, 4096, 0, 1))
        self.assertEqual(P.parse_log_line("info: lanes: step 7 rank 1 prompt pass: 68902 tokens, slot 0")["tokens"], 68902)
        self.assertEqual(P.parse_log_line("warning: collectives: rank 1 tcp_host exchange 71971 failed after 1 ms")["ev"], "log")
        self.assertIsNone(P.parse_log_line("[tensorfold] done chatcmpl-1 prompt=16"))

    def test_flat_sample(self):
        c, g = P.flat_sample({"psi.memory": PSI, "mountstats": MOUNTSTATS, "netdev": NETDEV, "loadavg": "41.70 37.50 20.70 3/900 1\n",
                              "meminfo": "MemAvailable: 100 kB\n", "vmstat": "pgmajfault 3\n", "pid.stat": PID_STAT,
                              "stat": "cpu  1 2 3 4 55 6\nprocs_blocked 12\n", "blk.loop0": "5 0 80 9"})
        self.assertEqual(c["nfs:/data/models:read_exe_ms"], 4600000)
        self.assertEqual(c["psi:memory:some_us"], 21208904)
        self.assertEqual(c["net:eno1:rx_bytes"], 1078000)
        self.assertEqual((c["pid:majflt"], c["vm:pgmajfault"], c["blk:loop0:read_sectors"], c["cpu:iowait_ticks"]), (321, 3, 80, 55))
        self.assertEqual((g["load1"], g["runnable"], g["procs_blocked"], g["mem:MemAvailable"], g["psi:memory:some_avg10"]), (41.7, 3, 12, 100, 1.5))
        self.assertNotIn("nfs:/data/models:device", c)
        json.dumps({"c": c, "g": g})


class Conf(unittest.TestCase):
    def test_values(self):
        v = D.conf_values(CONF)
        self.assertEqual(v["PROMPT_CACHE_GIB"], "4")
        self.assertEqual(v["WRAP0"], "/usr/bin/env LD_LIBRARY_PATH=/data/models/tools/therock-10.2/lib")
        self.assertEqual(v["DEPLOY_DIR"], "$HOME/tfp/deploy/f64b5ff2e28b")
        self.assertEqual((v["TP_STRIPE"], v["LEARN_DIR"], v["BENCH"]), ("", "/cache/tfp-learned", "0"))
        self.assertEqual(D.lib_dir_of(v["WRAP1"]), "/data/models/tools/therock-10.2/lib")

    def test_arms(self):
        for arm in ("S", "L", "E"):
            text = D.conf_text(CONF, arm)
            self.assertTrue(text.startswith(CONF))                          # the serving conf's bytes, then lines
            v = D.conf_values(text)
            self.assertEqual((v["BENCH"], v["DRAFTS"], v["PROMPT_CACHE_GIB"]), ("0", "1", "4"))
            self.assertEqual(v["LEARN_DIR"], f"/cache/strixflerp-stall/learned-{arm}")
            self.assertNotIn("tfp-learned", text[len(CONF):])
            self.assertEqual(v["RUN_ROOT"], "$HOME/tfp/strixflerp-stall/run")
            self.assertEqual(v["ENGRAM_AIO"], "1" if arm == "E" else "0")
            want = "/home/pod/tfp/strixflerp-stall/therock/lib:/data/models/tools/therock-10.2/lib" if arm == "L" else "/data/models/tools/therock-10.2/lib"
            self.assertEqual((v["WRAP0"], v["WRAP1"]), ("/usr/bin/env LD_LIBRARY_PATH=" + want,) * 2)

    def test_origins(self):
        o = D.origins(["/data/models/tools/therock-10.2/lib/libamdhip64.so", "/data/models-rail/tfp-strixfold/engram-1017/a.safetensors",
                       "/home/pod/tfp/strixflerp-stall/therock/lib/x.so", "/usr/lib/x86_64-linux-gnu/libc.so.6", "/cache/x"])
        self.assertEqual(o, {"mgmt-nfs": 1, "rail-nfs": 1, "scratch": 1, "root-or-ram": 1, "cache": 1})
        self.assertEqual(D.target_of("10.10.0.31:/data/models /data/models nfs4"), "/data/models")
        self.assertIsNone(D.target_of(""))
        m = [("overlay", "/", "overlay", "rw"), ("10.10.0.31:/data/models", "/data/models", "nfs4", "ro"), ("10.20.0.31:/data/models", "/data/models-rail", "nfs", "ro")]
        self.assertEqual(D.mount_of("/data/models-rail/x", m)[1], "/data/models-rail")
        self.assertEqual(D.mount_of("/data/models/tools", m)[1], "/data/models")
        self.assertEqual(D.mount_of("/home/pod", m)[1], "/")


class Staging(unittest.TestCase):
    def test_the_library_copy_keeps_the_tree_the_links_and_kpack(self):
        import subprocess

        class Local:
            said = []

            def on(self, ip, cmd, timeout=120, quiet=False):
                p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
                return p.returncode, (p.stdout + p.stderr).strip()

            def say(self, text):
                self.said.append(text)

        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "therock-10.2")
            os.makedirs(os.path.join(src, "lib", "llvm", "lib"))
            os.makedirs(os.path.join(src, ".kpack"))
            for rel, n in (("lib/libamdhip64.so.7.17", 3000), ("lib/llvm/lib/libLLVM.so.24", 2000), ("lib/libunused.so.1.0", 9000), (".kpack/blas.kpack", 500)):
                with open(os.path.join(src, rel), "wb") as f:
                    f.write(b"x" * n)
            os.symlink("libamdhip64.so.7.17", os.path.join(src, "lib", "libamdhip64.so.7"))
            os.symlink("libunused.so.1.0", os.path.join(src, "lib", "libunused.so.1"))
            keep, D.S = D.S, os.path.join(d, "scratch")
            try:
                facts = {"lib_real": os.path.join(src, "lib"), "libs": {"libamdhip64.so.7.17": 3000, "llvm/lib/libLLVM.so.24": 2000}}
                self.assertIn("MemAvailable", D.stage_libs(Local(), dict(facts, libs={"libamdhip64.so.7.17": 1}), 10 ** 6) or "")
                self.assertIsNone(D.stage_libs(Local(), facts, 0))                             # a second copy starts clean
                t = os.path.join(D.S, "therock")
                self.assertEqual(os.path.getsize(os.path.join(t, "lib", "llvm", "lib", "libLLVM.so.24")), 2000)
                self.assertEqual(os.readlink(os.path.join(t, "lib", "libamdhip64.so.7")), "libamdhip64.so.7.17")
                self.assertTrue(os.path.islink(os.path.join(t, "lib", "libunused.so.1")))      # dangling: the loader goes on
                self.assertFalse(os.path.exists(os.path.join(t, "lib", "libunused.so.1.0")))   # to the original directory
                self.assertEqual(os.path.getsize(os.path.join(t, ".kpack", "blas.kpack")), 500)
            finally:
                D.S = keep


class Attribution(unittest.TestCase):
    def test_delta_shares_by_overlap(self):
        s = [{"t": 0.0, "c": {"x": 0}, "g": {}}, {"t": 2.0, "c": {"x": 100}, "g": {}}, {"t": 4.0, "c": {"x": 100}, "g": {}}, {"t": 6.0, "c": {"x": 40}, "g": {}}]
        self.assertAlmostEqual(D.delta(s, "x", 0.0, 2.0), 100)
        self.assertAlmostEqual(D.delta(s, "x", 1.0, 3.0), 50)               # half of the first interval, none of the second
        self.assertAlmostEqual(D.delta(s, "x", 4.0, 6.0), 0)                # a counter that fell (a new process) counts 0
        self.assertIsNone(D.delta(s, "x", 10.0, 12.0))
        self.assertIsNone(D.delta(s, "y", 0.0, 2.0))

    def test_frames_and_gauges(self):
        fr = D.frames_of([{"ev": "frame", "frame": "fill_chunk", "ms": 1400, "t": 10.0, "step": 1}, {"ev": "frame", "frame": "fill_chunk", "ms": 3500, "t": 20.0},
                          {"ev": "frame", "frame": "fill_chunk", "ms": 9000, "t": 40.0}, {"ev": "frame", "frame": "round", "ms": 9000, "t": 50.0}, {"ev": "cache", "t": 1}])
        self.assertEqual([f["cls"] for f in fr], ["normal", "slow", "stalled"])
        self.assertAlmostEqual(fr[2]["a"], 31.0)
        s = [{"t": 1.0, "c": {}, "g": {"k": 1}}, {"t": 3.0, "c": {}, "g": {"k": 2}}, {"t": 5.0, "c": {}, "g": {"k": 3}}]
        self.assertEqual((D.gauge_at(s, "k", 0.5), D.gauge_at(s, "k", 3.5), D.gauge_at(s, "k", 99)), (1, 2, 3))
        self.assertEqual(D.roles(HDR), {"mgmt_if": "eno1", "rail_if": "enp194s0np0", "mgmt_mounts": ["/data/models", "/mnt/nv1"], "rail_mounts": ["/data/models-rail"]})

    def test_metrics_of_a_class(self):
        v, m = verdicts({"nfs:/data/models:server_read_bytes": (0, 30e6), "pid:majflt": (0, 40), "psi:memory:some_us": (0, 250000)})
        self.assertEqual((m["normal"]["frames"], m["stalled"]["frames"]), (10, 6))
        self.assertAlmostEqual(m["stalled"]["lib_mb_s"], 30.0, places=1)
        self.assertAlmostEqual(m["stalled"]["pid_majflt_s"], 40.0, places=1)
        self.assertAlmostEqual(m["stalled"]["psi_memory_some"], 0.25, places=2)
        self.assertAlmostEqual(m["normal"]["lib_mb_s"], 0.0, places=3)
        self.assertAlmostEqual(m["normal"]["rail_nic_mb_s"], 800.0, places=0)
        self.assertAlmostEqual(m["normal"]["eng_exe_ms_op"], 1.0, places=2)
        self.assertAlmostEqual(m["stalled"]["mem_available_gib"], 13.0, places=2)
        self.assertEqual(m["stalled"]["load1"], 40.0)


class Leads(unittest.TestCase):
    def test_quiet_stall_rules_the_nfs_and_pressure_leads_out(self):
        v, _ = verdicts({"nfs:/data/models-rail:server_read_bytes": (20e6, 0), "nfs:/data/models-rail:read_exe_ms": (20, 0), "nfs:/data/models-rail:read_ops": (20, 0)})
        self.assertEqual((v["L1"], v["L2"], v["L3"], v["L5"]), ("ruled out", "ruled out", "ruled out", "ruled out"))
        self.assertEqual(v["L4"], "not decided")                            # no prompt cache lines

    def test_l1_library_pages_reread(self):
        v, _ = verdicts({"nfs:/data/models:server_read_bytes": (0, 30e6), "nfs:/data/models:read_exe_ms": (0, 900), "nfs:/data/models:read_ops": (0, 30),
                         "pid:majflt": (0, 40), "psi:memory:some_us": (0, 400000), "net:eno1:rx_bytes": (2e3, 30e6)})
        self.assertEqual(v["L1"], "supported")
        self.assertEqual(v["L3"], "not decided")                            # pressure, but with NFS reads in flight
        self.assertEqual(v["L5"], "ruled out")                              # the management NIC's bytes are the NFS reads

    def test_l1_root_image(self):
        v, _ = verdicts({"blk:loop0:read_sectors": (0, 20000), "blk:loop0:read_ms": (0, 600), "vm:workingset_refault_file": (0, 900)})
        self.assertEqual(v["L1"], "supported")

    def test_l1_needs_faults(self):
        v, _ = verdicts({"nfs:/data/models:server_read_bytes": (0, 30e6), "nfs:/data/models:read_exe_ms": (0, 900)})
        self.assertEqual(v["L1"], "not decided")

    def test_l2_engram_reads_slow(self):
        v, _ = verdicts({"nfs:/data/models-rail:read_exe_ms": (20, 2000), "nfs:/data/models-rail:read_rtt_ms": (15, 1900)})
        self.assertEqual((v["L2"], v["L1"]), ("supported", "ruled out"))
        v, _ = verdicts({"nfs:/data/models-rail:read_exe_ms": (400, 400)})  # busy, but no more than in normal frames
        self.assertEqual(v["L2"], "not decided")

    def test_l3_pressure_alone(self):
        v, _ = verdicts({"psi:memory:some_us": (0, 300000), "vm:pgscan_direct": (0, 5000), "nfs:/data/models-rail:read_exe_ms": (20, 20)})
        self.assertEqual((v["L3"], v["L1"]), ("supported", "ruled out"))

    def test_l4_eviction(self):
        cache = [{"ev": "cache", "t": 1010.0, "states": 5, "mib": 31, "cap_mib": 4096, "evicted": 0},
                 {"ev": "cache", "t": 1015.0, "states": 600, "mib": 4090, "cap_mib": 4096, "evicted": 3}]
        self.assertEqual(verdicts({}, cache)[0]["L4"], "supported")
        cache[1]["evicted"] = 0
        self.assertEqual(verdicts({}, cache)[0]["L4"], "ruled out")
        cache[1].update(evicted=2, t=1050.0)                                # the first eviction after the first stalled frame
        self.assertEqual(verdicts({}, cache)[0]["L4"], "ruled out")

    def test_l5_wrong_nic(self):
        v, _ = verdicts({"net:eno1:rx_bytes": (140e6, 20e6), "net:eno1:tx_bytes": (140e6, 20e6), "net:enp194s0np0:rx_bytes": (1e3, 1e3),
                         "net:enp194s0np0:tx_bytes": (1e3, 1e3), "nfs:/data/models-rail:server_read_bytes": (0, 0)})
        self.assertEqual(v["L5"], "supported")

    def test_few_stalled_frames_decide_nothing(self):
        samples, events = synth(BASE)
        frames = D.frames_of(events[:11])                                   # one stalled frame
        st = [f for f in frames if f["cls"] == "stalled"]
        leads = D.judge(D.class_metrics(samples, HDR, st), None, D.class_metrics(samples, HDR, frames), (), frames)
        self.assertEqual([leads[k][0] for k in ("L1", "L2", "L3")], ["not decided"] * 3)

    def test_contrast(self):
        st, no = {"cls": "stalled"}, {"cls": "normal"}
        c = D.contrast({"S": [no] * 10 + [st] * 5, "L": [no] * 20, "E": [no] * 9 + [st] * 4})
        self.assertIn("supports L1", c["L"])
        self.assertIn("against L2", c["E"])
        self.assertIn("says nothing", D.contrast({"S": [no] * 10, "L": [no] * 20})["L"])
        self.assertIn("fewer frames", D.contrast({"S": [no] * 10 + [st] * 5, "L": [no] * 5})["L"])
        self.assertEqual(D.contrast({"L": [no]}), {})

    def test_analyze_a_directory(self):
        samples, events = synth(dict(BASE, **{"nfs:/data/models-rail:read_exe_ms": (20, 2000)}))
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "S-rank1.samples.jsonl"), "w") as f:
                f.write("\n".join(json.dumps(x) for x in [HDR] + samples + events) + '\n{"kind": "s", "t": 99')   # a cut last line
            with open(os.path.join(d, "S-rank0.samples.jsonl"), "w") as f:
                f.write("\n".join(json.dumps(x) for x in [HDR] + samples) + "\n")
            an = D.analyze(d)
            self.assertEqual((an["arms"]["S"]["frames"], an["arms"]["S"]["stalled"], an["arms"]["S"]["normal"]), (16, 6, 10))
            self.assertEqual(an["arms"]["S"]["leads"]["L2"][0], "supported")
            self.assertEqual(an["arms"]["S"]["sampler"]["rank1_longest_gap_s"], 2.0)
            text = D.render(an)
            self.assertIn("L2: SUPPORTED", text)
            self.assertIn("Engram mount READ busy", text)
            json.dumps(an)
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("no <arm>-rank1.samples.jsonl", D.render(D.analyze(d)))


if __name__ == "__main__":
    unittest.main()
