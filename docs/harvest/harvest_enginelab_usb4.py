#!/usr/bin/env python3
"""Harvest the Strix Halo USB4 driver (hyukudan/strix-halo-usb4-driver at d5de947cb405) into catalog/enginelab_usb4.
Run: python3 docs/harvest/harvest_enginelab_usb4.py               (harvest_enginelab.py runs it after the engine lab)
     python3 docs/harvest/harvest_enginelab_usb4.py --patch-gate  (on control-node: parse every patch, check the series
                                                                   files, compile the two C tools; writes
                                                                   gate/results/compile/enginelab_usb4.patches.json)

Card ORIGIN-ENGINELAB bc74e1a17d. A Linux kernel patch series for the native AMD USB4 host controllers of two Strix Halo
machines: USB4STREAM character devices (/dev/tbstreamN, configfs, batched submission, NAPI-style RX completion), the
experimental DMA-BUF GPU stream UAPI (linux/strix_gpu_stream.h) that moves data iGPU to iGPU, and AMD NHI lifecycle,
teardown and recovery fixes. It applies to Linux 72d3fcf802c4 (7.3-rc5): 49 stable patches (patches/series), three
experimental ones and one upstream backport. It is GPL-2.0, so it is its own origin beside the Apache-2.0 engine lab
(catalog/enginelab, whose runtime.enginelab.direct_transport / direct_gpu_stream / dist_gpu_direct are the userspace
half). The patches are split into transport modules by what they change (STABLE below); tests/ and docs/ are not
harvested.

Reference checkout: control-node /data/ark-inbox/src/github.com__hyukudan__strix-halo-usb4-driver#d5de947cb405d30db65f810c9248d825c1bcedc6
(a tree without git objects); laptop ~/Code/strixflerp-upstream-supplement/enginelab-usb4 (same files, tarball through
control-node get). ENGINELAB_USB4_CHECKOUT names another copy. The tree digest pins the bytes."""
import hashlib, json, os, re, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run

CHECKOUT = os.environ.get('ENGINELAB_USB4_CHECKOUT', os.path.join(harvestlib.SUPPLEMENT, 'enginelab-usb4'))
COMMIT = 'd5de947cb405d30db65f810c9248d825c1bcedc6'
TREE_SHA256 = 'bd0145b8c7a0305530e9f453d8a8d8855f88f60b4ca4d8d9ca4525e3f8eea036'
SKIP = ('.git', 'build')
AUTHORS = ['hyukudan', 'the Linux kernel authors (thunderbolt driver, Intel USB4STREAM; notices kept in the files)']
GPL = 'GPL-2.0-only'
GPL_UAPI = 'GPL-2.0-only AND GPL-2.0-only WITH Linux-syscall-note'
NOT_RUN = 'Nothing was applied, built or run against a kernel on the pod: there is no Linux 72d3fcf8 tree on the island and no USB4 link between pod nodes; the patch gate parses every patch (git apply --numstat) and checks the series files.'


def tree_files(root):
    out = []
    for d, dirs, names in os.walk(root):
        if d == root:
            dirs[:] = [x for x in dirs if x not in SKIP]
            names = [n for n in names if n not in SKIP]
        out += [os.path.relpath(os.path.join(d, n), root) for n in names]
    return sorted(out)


def tree_digest(root, files):
    h = hashlib.sha256()
    for f in files:
        h.update(f'{harvestlib.sha256(os.path.join(root, f))}  {f}\n'.encode())
    return h.hexdigest()


class TreeOrigin(Origin):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.roots = [CHECKOUT]
        head = os.path.join(CHECKOUT, '.git', 'HEAD')
        if os.path.isfile(head) and open(head).read().strip() != COMMIT:
            raise SystemExit(f'{head} names {open(head).read().strip()}, the harvest pins {COMMIT}')
        self.digest = tree_digest(CHECKOUT, self.files())
        if TREE_SHA256 and self.digest != TREE_SHA256:
            raise SystemExit(f'{CHECKOUT}: tree digest {self.digest}, the harvest pins {TREE_SHA256}')

    def files(self):
        if self._files is None:
            self._files = tree_files(CHECKOUT)
        return self._files


O = TreeOrigin('enginelab_usb4', 'enginelab-usb4', 'https://github.com/hyukudan/strix-halo-usb4-driver', COMMIT, 'main', GPL,
               AUTHORS, '')

PT = 'patches/'


def pp(*nums):
    """Stable patches by number (patches/NNNN-*.patch)."""
    out = []
    for n in nums:
        hits = [f for f in O.files() if re.fullmatch(rf'patches/{n:04d}-[^/]+\.patch', f)]
        if len(hits) != 1:
            raise SystemExit(f'patch {n:04d}: {hits}')
        out += hits
    return out


# name -> (patch numbers, summary, why, provided interface, api, regime)
STABLE = {
 'usb4stream_base': ((1,), 'patch 0001, the USB4STREAM source as the series starts it: configfs-created /dev/tbstreamN character devices for host memory over the AMD USB4 NHI, the DMA-BUF GPU stream extension and its UAPI header linux/strix_gpu_stream.h, built on Intel\'s USB4STREAM code; with the two MSI-X fixes adapted from CiruStrixLink (the original mail is in provenance/reference-patches); plus patches/series, patches/audit-series, the provenance manifests and the kernel\'s LICENSES/ texts (GPL-2.0, Linux-syscall-note) that COPYING points to',
   'the base every other patch modifies: a host-to-host and iGPU-to-iGPU byte stream over a USB4 cable, the transport the engine lab\'s two-node path runs on (catalog/ciru_link carries a different USB4 stream series for Linux 7.2.2)',
   'transport.usb4-stream/-/-.enginelab', '/dev/tbstreamN (configfs), linux/strix_gpu_stream.h ioctls', 'two Strix Halo hosts on USB4 40 Gb/s, Linux 72d3fcf8 (7.3-rc5)'),
 'gpu_queue': ((3, 4, 5, 16, 31, 36, 41, 43), 'the GPU stream queue: batched GPU ring submission (0003), fence and stop lifetime (0004), GPU queues, fences and session recovery (0005), a kernel-validated terminal frame (0016), non-blocking submit that sleeps on neither the mutex (0031) nor the DMA lock (0036), opt-in polled completion (0041) and a TX progress stall detector (0043)',
   'the DMA-BUF path with bounded queues, sync_file fences and fail-closed recovery: what makes GPU-to-GPU DMA safe to leave running under an inference server',
   'transport.usb4-gpu-queue/-/-', 'strix_gpu_stream UAPI submit / fence / poll', 'GPU-direct messages; the routing per message class picks them by size'),
 'dma_recovery': ((6, 7, 8, 9, 13, 24), 'DMA lifecycle and recovery: DMA progress, idle reset and quarantine reaping (0006), external dependency errors (0007), detached quarantine and notification lifecycle (0008), domain retirement and cooperative controller recovery (0009), bounded domain release and DMA revocation (0013), and a DMA revoke that proves relaxed ordering is off (0024)',
   'a failed link must never let the controller DMA into memory that was handed back; this is the quarantine-then-reap shape',
   'transport.usb4-recovery/-/-', 'kernel-internal (thunderbolt domain / NHI)', 'controller faults, cable pulls, module unload'),
 'stream_rx_latency': ((10, 11, 15, 23, 30, 42, 45), 'stream RX completion and latency: NAPI-style hybrid RX completion (0010), per-stream NAPI and IRQ steering (0011), spinning outside the device mutex with IRQ restore (0015), RX counters before posting (0023), a bounded spin_us parameter (0030), opt-in CPU latency QoS (0042) and opt-in contiguous DMA buffers (0045)',
   'the small-message latency side: 8.35 us round trip for 64 B with interrupts on (README); the decode exchange is many small messages a token',
   'transport.usb4-stream-latency/-/-', 'module parameters and per-stream controls', 'per-layer decode exchanges'),
 'stream_lifecycle': ((14, 17, 19, 21, 22, 25, 26, 27, 28, 37), 'stream open / close / IO correctness: open error path and frame arrays (0014), detach RAM rings and domain / XDomain teardown (0017), a bounded wait for the peer on open (0019), device state under lock (0021), resume failure fails open files (0022), a damaged RX frame fails the read (0025), a bounded wait for a TX slot on close (0026), partial IO accounting (0027), close does not flush RX (0028) and a damaged close is an error (0037)',
   'fail-closed semantics a userspace transport can rely on: an error is reported, never silently dropped',
   'transport.usb4-stream-lifecycle/-/-', 'open / read / write / close of /dev/tbstreamN', 'stream setup and teardown'),
 'nhi_teardown': ((12, 18, 20, 29, 38, 44), 'AMD NHI teardown: CM unlinks tunnels on stop (0012), AMD interface reset when idle (0018), NHI shutdown releases ring interrupts (0020), teardown reset pending until idle (0029) and surviving requests, contention and suspend (0038), and opt-in per-CPU high-priority completion work (0044)',
   'lifecycle fixes for the AMD USB4 host controller found on real hardware',
   'transport.usb4-nhi/-/-', 'kernel-internal (NHI)', 'driver stop, suspend, reset'),
 'link_training': ((2, 32, 33, 34, 35, 39, 49), 'link and XDomain: credit accounting and link diagnostics (0002), kernel-doc for the stream / tbnet / CM ops members (0032), the XDomain lane-bonding result checked (0033), the host router reset result checked (0034), opt-in Gen3 retrain (0035) serialized and exclusive (0039), and the XDomain property chunk length (0049)',
   'getting and keeping a bonded 40 Gb/s link between two hosts, with the diagnostics to see why not',
   'transport.usb4-link/-/-', 'debugfs / sysfs link diagnostics, retrain parameter', 'link bring-up'),
 'host_completion': ((40, 46, 47, 48), 'host completion path: completion path timing (0040), host completion latency tracepoints (0046), host RX-only polled completion (0047) and opt-in host descriptor batching (0048)',
   'the host-path throughput side: about 37 Gb/s a direction over two cables with four streams, about 34 Gb/s inside the engine for large pipelined transfers (README)',
   'transport.usb4-host-completion/-/-', 'tracepoints and opt-in parameters', 'bulk and prefill activations'),
}

mods = []
for name, (nums, summary, why, iface, api, regime) in STABLE.items():
    paths = pp(*nums)
    if name == 'usb4stream_base':
        paths += [PT + 'series', PT + 'audit-series', 'provenance/', 'LICENSES/']
    note = ('Interface tag: none (no catalogue interface is a USB4 stream device; the nearest are transport.verbs and the collective.* providers over RoCE / TCP). '
            f'Stable patches {", ".join(f"{n:04d}" for n in nums)} of patches/series; apply the whole series in order to Linux 72d3fcf8. ' + NOT_RUN)
    mods.append(M(f'runtime.enginelab_usb4.{name}', summary, paths, lang='patch', toolchain='', compile='n/a',
                  reason='Linux kernel patch: not compiled here (no Linux 72d3fcf8 tree on the island); python3 docs/harvest/harvest_enginelab_usb4.py --patch-gate parses it (gate/results/compile/enginelab_usb4.patches.json)',
                  provides=[P(iface, api, 'bytes only', regime)], spdx=GPL_UAPI if name in ('usb4stream_base', 'gpu_queue') else GPL,
                  why=why, notes=note, includes=[]))

declared = {p for m in mods for p in O.expand(m.paths)}
stable = [f for f in O.files() if re.fullmatch(r'patches/\d{4}-[^/]+\.patch', f)]
if sorted(set(stable) - declared):
    raise SystemExit('stable patches not in STABLE: ' + ' '.join(sorted(set(stable) - declared)))

mods += [
 M('runtime.enginelab_usb4.experimental', 'the three experimental patches on top of the stable series (patches/experimental: 0049 host bounded RX poll window, 0050 host paced RX descriptor polls, 0051 host TX memory-path experiments) and their series file',
   ['patches/experimental/'], lang='patch', toolchain='', compile='n/a', spdx=GPL, includes=[],
   reason='Linux kernel patch, experimental: parsed by the patch gate only', why='only 0051 was measured on hardware, in one window, and the stable module was restored afterwards (README): kept as the record of what was tried on the host RX / TX path',
   provides=[P('transport.usb4-host-completion/-/-.experimental', 'module parameters of the experimental host RX poll window, paced polls and TX memory paths', 'bytes only', 'not hardware-qualified')],
   notes='Interface tag: none. Must not be treated as hardware-qualified (README). ' + NOT_RUN),
 M('runtime.enginelab_usb4.kernel_backports', 'an unmodified upstream Linux fix applied after the series: drm/amdkfd dma_buf reference leak in get_dmabuf_info (e6895021a328), with its README',
   ['patches/kernel-backports/'], lang='patch', toolchain='', compile='n/a', spdx=GPL, includes=[],
   reason='Linux kernel patch: parsed by the patch gate only', why='the leak sits on the DMA-BUF import path the GPU stream uses',
   provides=[P('transport.usb4-gpu-queue/-/-.backport', 'AMDKFD_IOC_GET_DMABUF_INFO error path', 'n/a', 'the pinned 7.3-rc5 build')],
   notes='Interface tag: none. Upstream commit, original authorship kept. ' + NOT_RUN),
 M('runtime.enginelab_usb4.gpu_stream_uapi', 'the experimental GPU stream UAPI header overlay/include/uapi/linux/strix_gpu_stream.h (as captured after patch 0001)',
   ['overlay/include/'], lang='c', toolchain='', compile='header-only', spdx='GPL-2.0-only WITH Linux-syscall-note', includes=['src/overlay/include'],
   reason='kernel UAPI header; the engine lab\'s include/strix/linux_direct_gpu_stream_uapi.hpp is the userspace binding', why='the ioctl contract between the kernel GPU stream and userspace transports',
   provides=[P('transport.usb4-gpu-stream-uapi/-/-', 'struct and ioctl definitions of linux/strix_gpu_stream.h', 'n/a', 'userspace on the patched kernel')],
   notes='Interface tag: none. overlay/ is an immutable capture of pristine source + 0001, not the current driver on its own (README). ' + NOT_RUN),
 M('runtime.enginelab_usb4.gpu_stream_driver', 'the DMA-BUF GPU stream driver as captured after 0001: overlay/drivers/thunderbolt/gpu_stream.c, gpu_stream.h and gpu_stream_logic.h',
   ['overlay/drivers/thunderbolt/gpu_stream.c', 'overlay/drivers/thunderbolt/gpu_stream.h', 'overlay/drivers/thunderbolt/gpu_stream_logic.h'],
   lang='c', toolchain='', compile='n/a', spdx=GPL, includes=[], sources=['overlay/drivers/thunderbolt/gpu_stream.c'],
   reason='Linux kernel source: needs the kernel tree to build', why='the readable form of the GPU-direct path (the patches are the authority)',
   provides=[P('transport.usb4-gpu-stream/-/-.driver', 'kernel-internal GPU stream (DMA-BUF import, rings, fences)', 'bytes only', 'patched kernel')],
   notes='Interface tag: none. Capture only: apply the patches for the current driver. ' + NOT_RUN),
 M('runtime.enginelab_usb4.nhi_overlay', 'the rest of the 0001 capture: overlay/drivers/thunderbolt (domain.c, nhi.c / nhi.h / nhi_regs.h, pci.c, stream.c, tunnel.c, xdomain.c, Kconfig), the pristine base/include/linux/thunderbolt.h and the kernel config fragment config/usb4-gpu.fragment',
   ['overlay/drivers/', 'base/', 'config/'], exclude=['overlay/drivers/thunderbolt/gpu_stream.*', 'overlay/drivers/thunderbolt/gpu_stream_logic.h'],
   lang='c', toolchain='', compile='n/a', spdx=GPL, includes=[], sources=[],
   reason='Linux kernel source: needs the kernel tree to build', why='the stream device and NHI as the series starts them; the config fragment names the options a test kernel needs',
   provides=[P('transport.usb4-stream/-/-.overlay', 'kernel-internal thunderbolt stream / NHI', 'bytes only', 'patched kernel')],
   notes='Interface tag: none. Capture only. ' + NOT_RUN),
 M('tool.enginelab_usb4.diag', 'diagnostics and build helpers: usb4_diagnose.py (controller / ring topology and budget), usb4_linkstat.py, validate_native.py and update_validation_manifest.py, check_kconfig.py, check_build_log.py, apply-series.sh and prepare_overlay.py',
   ['tools/usb4_diagnose.py', 'tools/usb4_linkstat.py', 'tools/validate_native.py', 'tools/update_validation_manifest.py', 'tools/check_kconfig.py', 'tools/check_build_log.py', 'tools/apply-series.sh', 'tools/prepare_overlay.py'],
   lang='python', toolchain='', compile='n/a', spdx=GPL, includes=[], reason='python and shell: not compiled (read, never run here)',
   why='inspect the link before creating streams (README: usb4_diagnose.py --require-budget)', provides=[P('tool.usb4-diagnose/-/-', 'python3 tools/usb4_diagnose.py [--require-budget]', 'n/a', 'a host on the patched kernel')],
   notes='Interface tag: none. ' + NOT_RUN),
 M('tool.enginelab_usb4.bench', 'stream benchmarks: pingpong.c (round-trip latency), stream_mem_bench.c (throughput) and tools/lab/run_stream_bench.py',
   ['tools/pingpong.c', 'tools/stream_mem_bench.c', 'tools/lab/run_stream_bench.py'], lang='c', toolchain='', compile='n/a', spdx=GPL, includes=[],
   sources=['tools/pingpong.c', 'tools/stream_mem_bench.c'], reason='host C tools; the patch gate compiles both with the host cc (gate/results/compile/enginelab_usb4.patches.json)',
   why='the measurement protocol behind the README numbers (docs/BENCHMARKS.md, not harvested)', provides=[P('tool.usb4-stream-bench/-/-', 'pingpong / stream_mem_bench on /dev/tbstreamN', 'n/a', 'two hosts')],
   notes='Interface tag: none. ' + NOT_RUN),
 M('tool.enginelab_usb4.lab', 'the lab setup: usb4stream_describe.py and its udev rule and refresh service, setup-usb4stream-lab, node_check.sh, the lab service and its reviewed-kernel drop-in, and the dedicated-stream configuration',
   ['tools/lab/'], exclude=['tools/lab/run_stream_bench.py'], lang='python', toolchain='', compile='n/a', spdx=GPL, includes=[],
   reason='python, shell and systemd units: not compiled, not installed', why='how the two lab nodes are brought up the same way each boot (describe service, udev rule, dedicated-stream config); a reference for a pod USB4 pair, not installed here',
   provides=[P('tool.usb4-lab-setup/-/-', 'setup-usb4stream-lab, usb4stream_describe.py --logs --summary', 'n/a', 'the two lab hosts')],
   notes='Interface tag: none. ' + NOT_RUN),
]

ORIGIN_TOML = '''
# Strix Halo USB4 driver: a research Linux kernel patch series for the native AMD USB4 host controllers of two Strix
# Halo machines (USB4STREAM /dev/tbstreamN, the experimental DMA-BUF GPU stream UAPI, AMD NHI lifecycle / recovery
# fixes). Companion of origin enginelab (card ORIGIN-ENGINELAB bc74e1a17d). Not upstream, tested on one pair.
name      = "enginelab_usb4"
repo      = "https://github.com/hyukudan/strix-halo-usb4-driver"
commit    = "d5de947cb405d30db65f810c9248d825c1bcedc6"
branch    = "main"
version   = "49 stable patches (patches/series, 0001-0049) + 3 experimental (patches/experimental) + 1 upstream backport, for Linux 72d3fcf802c45d00b300f25b848a93c3a2bd7c7e (7.3-rc5); README: about 37 Gb/s a direction over two USB4 40 Gb/s cables with four streams, about 34 Gb/s inside the engine for large pipelined transfers, 8.35 us round trip for 64 B, MiniMax decode exchange 2.91 -> 1.89 ms a token"
reference_checkout = "control-node:/data/ark-inbox/src/github.com__hyukudan__strix-halo-usb4-driver#d5de947cb405d30db65f810c9248d825c1bcedc6 (ark copy, .git without objects); laptop: ~/Code/strixflerp-upstream-supplement/enginelab-usb4 (the same files, by tarball through control-node get); docs/harvest/harvest_enginelab_usb4.py pins the tree digest"

[licence]
spdx      = "GPL-2.0-only"
files     = ["COPYING"]                                # copied verbatim next to this file; LICENSES/ (GPL-2.0, Linux-syscall-note) is in runtime.enginelab_usb4.usb4stream_base
third_party = "ATTRIBUTIONS.md: copied / modified Linux files keep their copyright and SPDX notices; the kernel extension is GPL-2.0-only, the experimental UAPI GPL-2.0 WITH Linux-syscall-note. The GPU stream extension originated in hyukudan/strix-halo-engine-lab; USB4STREAM builds on Intel's code. Two MSI-X fixes in 0001 were adapted from CiruStrixLink's Linux 7.2.2 series (Jonathan Yates, 'thunderbolt: Flush posted MSI-X interrupt clears', original mail in provenance/reference-patches; and 'Clear stale MSI-X status before ring arm', linking Benjamin S. Berman's upstream submission). patches/kernel-backports holds an unmodified upstream drm/amdkfd fix."

[toolchain]
host      = "a Linux kernel tree at 72d3fcf8 and its build toolchain; python3 for the tools"
device    = "none"
build     = "tools/apply-series.sh applies patches/series to the pinned tree; build with a unique local version and config/usb4-gpu.fragment (README 'Build and install'); not done on the island"
'''

LEFT_OUT = {'tests/': 'C harnesses and python tests of the patched code', 'docs/, README.md, BUILD.md, CHANGELOG.md, ATTRIBUTIONS.md': 'documentation',
            '.github/, .gitignore, .gitattributes': 'repository plumbing', 'COPYING': 'copied beside ORIGIN.toml'}


def patch_gate():
    """Parse every harvested patch with git apply --numstat (no tree needed), check each series file names patches that
    exist, compile the two C tools with the host cc. Writes gate/results/compile/enginelab_usb4.patches.json."""
    t0 = time.time()
    res = {'origin': 'enginelab_usb4', 'commit': COMMIT, 'patches': [], 'series': [], 'c_tools': []}
    for f in sorted(x for x in O.files() if x.endswith('.patch')):
        p = subprocess.run(['git', 'apply', '--numstat', os.path.join(CHECKOUT, f)], capture_output=True, text=True)
        files = [l.split('\t')[2] for l in p.stdout.splitlines() if l.count('\t') >= 2]
        res['patches'].append({'patch': f, 'rc': p.returncode, 'files': len(files), 'touches': sorted(set(files))[:20], 'stderr': p.stderr[-500:]})
    for s in ('patches/series', 'patches/experimental/series', 'patches/audit-series'):
        base = os.path.dirname(s)
        names = [l.strip() for l in open(os.path.join(CHECKOUT, s)) if l.strip() and not l.startswith('#')]
        miss = [n for n in names if not os.path.isfile(os.path.join(CHECKOUT, base, n))]
        res['series'].append({'series': s, 'entries': len(names), 'missing': miss})
    cc = os.environ.get('CC', 'cc')
    for c in ('tools/pingpong.c', 'tools/stream_mem_bench.c'):
        out = os.path.join('/tmp', f'enginelab_usb4_{os.getpid()}_{os.path.basename(c)}.o')
        p = subprocess.run([cc, '-O2', '-Wall', '-I', os.path.join(CHECKOUT, 'overlay/include/uapi'), '-c', os.path.join(CHECKOUT, c), '-o', out], capture_output=True, text=True)
        res['c_tools'].append({'source': c, 'cmd': f'{cc} -O2 -Wall -I overlay/include/uapi -c {c}', 'rc': p.returncode, 'stderr': p.stderr[-800:]})
        if os.path.exists(out):
            os.remove(out)
    ok = all(x['rc'] == 0 for x in res['patches']) and all(not x['missing'] for x in res['series'])
    res['status'] = 'pass' if ok else 'fail'
    res['summary'] = (f"{sum(1 for x in res['patches'] if x['rc'] == 0)} of {len(res['patches'])} patches parse; series: " +
                      ', '.join(f"{x['series']} {x['entries']} entries, {len(x['missing'])} missing" for x in res['series']) +
                      '; C tools: ' + ', '.join(f"{x['source']} rc={x['rc']}" for x in res['c_tools']))
    res['seconds'] = round(time.time() - t0, 2)
    path = os.path.join(harvestlib.REPO, 'gate', 'results', 'compile', 'enginelab_usb4.patches.json')
    json.dump(res, open(path, 'w'), indent=1)
    print('patch gate:', res['status'] + ':', res['summary'])
    return 0 if ok else 1


if __name__ == '__main__':
    if '--patch-gate' in sys.argv:
        sys.exit(patch_gate())
    # run() prunes directories of catalog/enginelab_usb4 that are not modules, so the licence copies make their own
    _copy = harvestlib.shutil.copyfile
    def _copy_mkdir(src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        return _copy(src, dst)
    harvestlib.shutil.copyfile = _copy_mkdir
    run(O, mods, ORIGIN_TOML, LEFT_OUT, ['COPYING'])
    print(f'enginelab_usb4 tree digest {O.digest}')
