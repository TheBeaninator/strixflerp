#!/usr/bin/env python3
"""Harvest CiruStrixLink (ciru-ai/CiruStrixLink at 1c7f1ae, v0.3.3) into catalog/ciru_link.
Run: python3 docs/harvest/harvest_ciru_link.py              (rebuild catalog/ciru_link)
     python3 docs/harvest/harvest_ciru_link.py --go-gate    (control-node CPU: go vet, compile the tests, build the binary)
     python3 docs/harvest/harvest_ciru_link.py --patch-gate (check the 16 patches against SHA256SUMS and parse each)

Card ORIGIN-CIRU-LINK 682cd99ba2. CiruStrixLink is the open transport under Ciru's two-host GLM5.3-Flash-CIRU-STRIX-IU4
package (TP2 over USB4): a static Go binary that qualifies both hosts, configures USB4NET (thunderbolt-net), arms the
patched thunderbolt_stream NHI endpoints, emits runtime transport environments, runs a dependency-free TCP link test and
serves an operator console; plus a 16-patch Linux 7.2.2 series (USB4STREAM zero-copy slot mode, HopID ownership, ring
fixes and import of GPU-owned DMA-BUF frame pools) that the direct GPU path needs.

Scope: every Go package and every kernel patch, one module a family (FAMILIES below). There is no GPU code in this tree:
the HIP bridge that packs and reduces the BF16 [8,4096] partials (nhi-m8-bf16/vllm_nhi_allreduce.cu, MIT) ships inside the
jcbtc GLM runtime package on Hugging Face, which is not in the ark and is not harvested; ORIGIN.toml records its design and
published numbers. The console's fonts and images, docs, experiments, CI and repository plumbing are listed as unharvested.

The reference checkout is the ark copy on control-node
(/data/ark-inbox/src/github.com__ciru-ai__CiruStrixLink#1c7f1ae89298c5be2ac4f916b589c77a10c903d8, a tree with a shallow
.git); the laptop holds the same files, moved by a tarball through control-node get, in
~/Code/strixflerp-upstream-supplement/ciru_link. CIRU_LINK_CHECKOUT names another copy. The tree digest pins the bytes."""
import datetime, glob, hashlib, json, os, re, shutil, subprocess, sys, tempfile, tomllib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run

CHECKOUT = os.environ.get('CIRU_LINK_CHECKOUT', os.path.join(harvestlib.SUPPLEMENT, 'ciru_link'))
COMMIT = '1c7f1ae89298c5be2ac4f916b589c77a10c903d8'
# sha256 over the lines "<sha256 of the file>  <origin-relative path>\n" of every file, sorted by path (.git left out)
TREE_SHA256 = '980c879db2f0bd5c39481fcc509a61943077279063cafd935f7a4158a0450d41'
SKIP = ('.git', 'build', 'dist')
GO_MODULE = 'github.com/ciru-ai/CiruStrixLink'


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
    """The origin's file set is the checkout's files (an ark tree: no git objects to ask), pinned by digest."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.roots = [CHECKOUT]
        head = os.path.join(CHECKOUT, '.git', 'HEAD')
        if os.path.isfile(head) and open(head).read().strip() != COMMIT:
            raise SystemExit(f'{head} names {open(head).read().strip()}, the harvest pins {COMMIT}')
        got = tree_digest(CHECKOUT, self.files())
        if got != TREE_SHA256:
            raise SystemExit(f'{CHECKOUT}: tree digest {got}, the harvest pins {TREE_SHA256} (CiruStrixLink {COMMIT[:12]})')

    def files(self):
        if self._files is None:
            self._files = tree_files(CHECKOUT)
        return self._files


O = TreeOrigin('ciru_link', 'ciru_link', 'https://github.com/ciru-ai/CiruStrixLink', COMMIT, 'main (v0.3.3 line)',
               'Apache-2.0', ['Ciru (ciru-ai)'], '')
KD = 'kernel/linux-7.2.2/'
# patch authors, from each patch's From: line (names as the patches give them)
JY = ['Jonathan Yates (jyatesdotdev/strix-rdma 06c1836)']
JR = ['Jordan Yates (jyatesdotdev/strix-rdma 06c1836)']
CIRU16 = ['Ciru (ciru-ai; the patch names "Codex USB4STREAM Analysis" as author)']
GPL = 'GPL-2.0-only'
GPL_UAPI = 'GPL-2.0-only AND GPL-2.0-only WITH Linux-syscall-note'
NOT_RUN = 'Nothing was run on the pod beyond the compile and patch gates; no USB4 link exists between pod nodes.'


def patches(*nums):
    return [p for p in O.files() if p.startswith(KD) and p.endswith('.patch') and int(os.path.basename(p)[:4]) in nums]


# family -> module spec. tag: the strixflerp interface class the family matches, or None. requires between Go packages
# come from their import lines (go_requires); patch families require the series base.
FAMILIES = [
 dict(id='runtime.ciru_link.tbstream_zerocopy', tag=None, lang='patch', spdx=GPL_UAPI, authors=JY,
  paths=patches(1, 2, 3, 4, 5, 6, 7) + [KD + 'README.md', KD + 'SHA256SUMS', KD + 'COPYING', KD + 'LICENSES/'],
  summary='the base of the Linux 7.2.2 USB4STREAM patch series: NHI rings that suppress per-frame completions (0001), the thunderbolt_stream zero-copy slot mode with fixed TX / RX frame pools mapped into the client and its uapi header include/uapi/linux/thunderbolt-stream.h (0002), exclusive zero-copy ownership (0003), flagged RX reposting (0004), an interrupt at RX message boundaries (0005), TX DMA ownership synchronisation (0006) and hardened mode transitions (0007); with the series README, SHA256SUMS and the kernel COPYING / LICENSES',
  why='this is the kernel half of a two-sided message transport over a USB4 cable between two Strix Halos: fixed rings, message-boundary completion and explicit reposting are the shape any USB4 tensor exchange needs, the catalogue\'s other transports being RoCE verbs and TCP',
  provides=[P('transport.usb4-stream-zerocopy/-/-', '/dev/tbstreamN with the TBSTREAM_ZC_* ioctls of include/uapi/linux/thunderbolt-stream.h (slot pools, submit, completion events, flagged RX repost)', 'bytes only; frames of up to 4 KiB, message boundaries carried by completion flags', 'two hosts on one USB4 link; one exclusive owner a device; ring 4095 frames, HopID 9/9 on the qualified profile')],
  notes='Interface tag: none (no strixflerp interface is a USB4 stream device; the nearest are transport.verbs/-/- and collective.allgather-tcp/-/f32, which run over RoCE and TCP). Patches 1-15 are jyatesdotdev/strix-rdma 06c1836 rebased on Linux v7.2.2 (README.md). The uapi header carries GPL-2.0 WITH Linux-syscall-note, the rest GPL-2.0. Apply in number order to a fresh v7.2.2 tree; SHA256SUMS covers all 16 patches, which are split over five modules here. ' + NOT_RUN),
 dict(id='runtime.ciru_link.tbstream_hopid', tag=None, lang='patch', spdx=GPL, authors=JY, paths=patches(8, 9),
  summary='HopID ownership between thunderbolt-net and thunderbolt_stream: the network driver records which receive HopID it owns (0008), and the stream driver attaches its HopIDs in one transaction with rollback on failure (0009)',
  why='USB4NET keeps HopID 8 for control and portable traffic while the stream device takes 9/9 on the same cable, so the two never steal each other\'s path; the same split (a control link beside a data path) is what a pod pair over USB4 would run',
  provides=[P('transport.usb4-hopid-ownership/-/-', 'thunderbolt-net receive HopID tracking (drivers/net/thunderbolt/main.c); transactional HopID attach in drivers/thunderbolt/stream.c', 'n/a (control path)', 'USB4NET HopID 8 and stream HopID 9/9 on the qualified profile')],
  requires=['module:runtime.ciru_link.tbstream_zerocopy'],
  notes='Interface tag: none. Applies after patches 0001-0007. ' + NOT_RUN),
 dict(id='runtime.ciru_link.tbstream_dmabuf', tag=None, lang='patch', spdx=GPL_UAPI, authors=JY + JR, paths=patches(13, 14),
  summary='import of GPU-owned DMA-BUFs into thunderbolt_stream: a DMA-BUF import probe with the scatter-gather helpers in drivers/thunderbolt/stream-sg.h (0013) and imported DMA-BUF frame pools that the NHI rings transmit from and receive into (0014; TBSTREAM_ZC_IMPORT / TBSTREAM_ZC_ENABLE)',
  why='this is the step that takes the CPU out of the payload path: dedicated uncached HIP allocations exported with hipMemGetHandleForAddressRange(..., hipMemRangeHandleTypeDmaBufFd) become the USB4 controller\'s TX / RX pools, so GPU kernels pack and consume tensors in place. The shape would go under a flerp collective provider for two hosts joined by USB4 (an allreduce-p2p beside the RoCE ones)',
  provides=[P('transport.usb4-dmabuf-import/-/-', 'TBSTREAM_ZC_IMPORT (DMA-BUF fd per pool), TBSTREAM_ZC_ENABLE and the import probe ioctl (include/uapi/linux/thunderbolt-stream.h)', 'bytes only; pools must be dedicated, uncached device allocations', 'per device one TX and one RX pool of whole 4-KiB frames')],
  requires=['module:runtime.ciru_link.tbstream_zerocopy'],
  notes='Interface tag: none (no catalogue interface imports a DMA-BUF into a link; the gpu side that fills these pools is the HIP bridge in the jcbtc GLM runtime package, not harvested, ORIGIN.toml [bridge]). stream-sg.h is GPL-2.0, the uapi additions GPL-2.0 WITH Linux-syscall-note. Applies after patches 0001-0012. ' + NOT_RUN),
 dict(id='runtime.ciru_link.tbstream_ring_fixes', tag=None, lang='patch', spdx=GPL, authors=JY + JR + CIRU16, paths=patches(11, 12, 15, 16),
  summary='NHI ring reliability fixes: flush posted MSI-X interrupt clears (0011), prime the RX ring before enabling the DMA paths (0012), skip CLOSE toward an already-closed peer (0015), and Ciru\'s patch 16, which clears and flushes stale non-auto-clear MSI-X status before a reused ring is armed and reads the enable register back',
  why='patch 16 fixes a reproduced missing-first-completion failure on a reused ring; every fix here is about the first completion after an arm or the last one before a close, the moments a two-host transport loses messages',
  provides=[P('transport.usb4-ring-arm/-/-', 'drivers/thunderbolt/nhi.c ring arm / MSI-X handling; drivers/thunderbolt/stream.c RX priming and close', 'n/a (control path)', 'reused rings on the AMD USB4 NHI of Strix Halo')],
  requires=['module:runtime.ciru_link.tbstream_zerocopy'],
  notes='Interface tag: none. In the series these interleave with 0013-0014 by number; apply all 16 in order. ' + NOT_RUN),
 dict(id='tool.ciru_link.tbstream_diag', tag=None, lang='patch', spdx=GPL_UAPI, authors=JY, paths=patches(10),
  summary='zero-copy progress diagnostics for thunderbolt_stream: ring and slot progress counters in nhi.c and stream.c with their uapi query (0010, the largest patch of the series)',
  why='a stalled two-host exchange is only debuggable when both ends report where each ring stands; this is the instrumentation the series was qualified with',
  provides=[P('transport.usb4-stream-progress/-/-', 'the progress-diagnostic ioctl and counters added to include/uapi/linux/thunderbolt-stream.h and include/linux/thunderbolt.h', 'counters only', 'diagnostics, any ring')],
  requires=['module:runtime.ciru_link.tbstream_zerocopy'],
  notes='Interface tag: none. Applies after patches 0001-0009. ' + NOT_RUN),
 dict(id='runtime.ciru_link.nhi_endpoint', tag=None, lang='go',
  paths=['internal/transport/'],
  summary='the NHI endpoint lifecycle in Go (internal/transport): Inspect discovers thunderbolt-net, the stream service and device index, HopIDs, ring 4095 and throttle 8192 ns, open device holders and the lease; Reconcile turns two host reports into one pair verdict (arm_allowed, nhi_ready, lease_available); BuildEndpointPlan / ApplyEndpoint prepare and clean up an endpoint under a file lock without killing holders; GenerateEnvironment emits the portable (NCCL_SOCKET_IFNAME==thunderbolt0, GLOO_SOCKET_IFNAME) or NHI runtime environment',
  why='the two-sided arming order (inspect both, reconcile, prepare both, inspect again, then launch) is the part of a pair transport that keeps one half-prepared host from hanging the other; flerp\'s own two-node paths could take the same pair-verdict shape',
  provides=[P('transport.usb4-nhi-endpoint/-/-', 'transport.Inspect, Reconcile, BuildEndpointPlan, ApplyEndpoint, LoadReport (internal/transport/transport.go, endpoint.go)', 'n/a (control path); JSON reports, schema_version 1', 'two hosts, one USB4 link'),
            P('transport.collective-env-usb4/-/-', 'transport.GenerateEnvironment, Environment.DotEnv, ParseDotEnv (internal/transport/env.go)', 'n/a', 'RCCL / Gloo socket binding (portable) or the NHI adapter variables')],
  notes='Interface tag: none. Runtime needs the patched thunderbolt_stream (runtime.ciru_link.tbstream_*); not a build dependency. ' + NOT_RUN),
 dict(id='runtime.ciru_link.usb4net', tag=None, lang='go', paths=['internal/link/'],
  summary='USB4NET discovery and setup in Go (internal/link): finds thunderbolt-net interfaces and Thunderbolt paths, loads the driver, plans complementary /30 addresses for roles a and b, MTU and route, applies or rolls back the plan through ip / NetworkManager commands, and verifies the path MTU to the peer',
  why='the portable baseline every NHI arm starts from (carrier, exact peer route, source address); the pod links its nodes over 100 GbE, so this is the shape for a USB4 side link, not a replacement',
  provides=[P('transport.usb4net-setup/-/-', 'link.Discover, ProbeLocal, SelectInterface, RouteTo, RoleAddresses, BuildSetupPlan, BuildRollbackPlan, Apply, VerifyPathMTU (internal/link/link.go)', 'n/a (control path)', 'Linux thunderbolt-net, point-to-point /30')],
  notes='Interface tag: none. ' + NOT_RUN),
 dict(id='tool.ciru_link.linktest', tag=None, lang='go', paths=['internal/bench/', 'internal/quality/'],
  summary='a dependency-free TCP link test (internal/bench: token-authenticated protocol on port 55321, upload / download throughput over N streams, RTT percentiles, payload integrity and reconnect checks) and the policy that turns its numbers into a conservative runtime class with heartbeat, timeout, reconnect and in-flight limits (internal/quality)',
  why='a link test with an integrity check and a declared policy output is what a two-node bring-up needs before any collective runs; nothing in the catalogue tests a link this way',
  provides=[P('verify.link-test-usb4/-/-', 'bench.Serve, bench.Run, Config, Result (internal/bench/bench.go)', 'throughput in Gb/s, RTT in ms; integrity by payload check', 'any TCP path; the origin runs it over USB4NET'),
            P('verify.link-quality-policy/-/-', 'quality.Classify(Metrics) Policy (internal/quality/quality.go)', 'thresholds on throughput, p99 RTT, reconnects and integrity', 'transport-neutral')],
  notes='Interface tag: none. ' + NOT_RUN),
 dict(id='tool.ciru_link.prereq', tag=None, lang='go', paths=['internal/prereq/', 'internal/install/'],
  summary='host prerequisites and install plans in Go: a versioned, UI-safe report of the kernel, modules, tools (ip, ethtool, ping, NetworkManager) and permissions the transport needs (internal/prereq), and allowlisted install plans by package manager that never change kernels, reboot or edit declarative OS state (internal/install)',
  why='a prerequisites report with stable status words is the bring-up check a pod node pair would run before arming a link',
  provides=[P('verify.host-prereq-usb4/-/-', 'prereq.Check (internal/prereq/prereq.go)', 'n/a', 'Linux x86-64 hosts (Ubuntu / Debian, Fedora / RHEL, Arch, NixOS)'),
            P('tool.install-plan-usb4/-/-', 'install.Build, install.Apply (internal/install/install.go)', 'n/a', 'allowlisted package installs only')],
  notes='Interface tag: none. ' + NOT_RUN),
 dict(id='serve.ciru_link.console', tag=None, lang='go', paths=['internal/ui/'], exclude=['internal/ui/static/assets/**'],
  summary='the embedded operator console and peer agent (internal/ui): a loopback HTTP console on 7749 with a static page (index.html, app.js, styles.css) for transport state, bounded link tests, model PP/TG and draft acceptance read from one frontend\'s /v1/models and /metrics; a USB4-bound agent on 7748 behind a peer token; and the optional Launch control that starts the packaged GLM TP2 pair rank 0 then rank 1, checks context readiness and rolls back on failure',
  why='a two-host serving pair is one service with two halves; the paired load / unload with readiness checks and rollback is the shape a pod TP2 keeper needs, and the metrics reading avoids double-counting mirrored ranks',
  provides=[P('serve.operator-console-usb4/-/-', 'ui.NewConsole, Console.ServeHTTP, ConsoleConfig (internal/ui/console.go, ui.go)', 'n/a', 'loopback 127.0.0.1:7749'),
            P('serve.peer-agent-usb4/-/-', 'ui.NewAgent, Agent.ServeHTTP, AgentConfig (internal/ui/agent.go)', 'n/a', 'binds the USB4 address only, TCP 7748, token-protected'),
            P('serve.tp2-pair-control/-/-', 'ui.RunModelNode and the Launch handlers (internal/ui/launch.go, model.go, status_helper.go)', 'n/a', 'the packaged GLM5.3-Flash-CIRU-STRIX-IU4 pair, fixed complementary ranks, scoped root-owned helpers')],
  notes='Interface tag: none (serve.gufo.http and flerp\'s servers are model frontends, not pair consoles). The console\'s static/assets (IBM Plex fonts, Ciru emblem, logo and favicon) are not harvested: the fonts\' OFL licence text is not in the tree and the images are Ciru branding; //go:embed static still compiles without them. ' + NOT_RUN),
 dict(id='tool.ciru_link.cli', tag=None, lang='go', paths=['cmd/', 'go.mod', 'Makefile'],
  summary='the ciru-strixlink command (cmd/ciru-strixlink/main.go) with go.mod and the Makefile: version, prerequisites, setup / rollback, doctor, serve / test, transport inspect / reconcile / endpoint / env, agent and ui, built as one static linux/amd64 binary (CGO_ENABLED=0)',
  why='the entry point that composes every internal package; go.mod (no dependencies, go 1.22) and the Makefile are what make the module set build on its own',
  provides=[P('tool.cli-ciru-strixlink/-/-', 'main (cmd/ciru-strixlink/main.go); make build | linux-amd64 | test | vet', 'n/a', 'Linux x86-64')],
  notes='Interface tag: none. ' + NOT_RUN),
]


def go_requires():
    """{module id: ['module:<id>', ...]}: the Go modules whose packages a module's .go files import."""
    pkg_owner = {}
    for f in FAMILIES:
        if f['lang'] == 'go':
            for p in O.expand(f['paths'], f.get('exclude', ())):
                if p.endswith('.go'):
                    pkg_owner[os.path.dirname(p)] = f['id']
    out = {}
    imp = re.compile(r'"' + re.escape(GO_MODULE) + r'/([^"]+)"')
    for f in FAMILIES:
        if f['lang'] != 'go':
            continue
        need = set()
        for p in O.expand(f['paths'], f.get('exclude', ())):
            if p.endswith('.go'):
                for pkg in imp.findall(open(O.find(p)).read()):
                    owner = pkg_owner.get(pkg)
                    if owner and owner != f['id']:
                        need.add('module:' + owner)
        out[f['id']] = sorted(need)
    return out


GATE_GO = 'go: built by the Go gate, not gate/compile/run.py (python3 docs/harvest/harvest_ciru_link.py --go-gate on control-node CPU: go vet, the tests compiled without running, the static binary built)'
GATE_PATCH = 'Linux kernel patch: not compiled here (no Linux 7.2.2 tree on the island); python3 docs/harvest/harvest_ciru_link.py --patch-gate checks every patch against SHA256SUMS and parses it with git apply --numstat (gate/results/compile/ciru_link.patches.json)'


def mods():
    req = go_requires()
    out = []
    for f in FAMILIES:
        go = f['lang'] == 'go'
        out.append(M(f['id'], f['summary'], f['paths'], provides=f['provides'],
                     requires=list(f.get('requires', [])) + req.get(f['id'], []), lang=f['lang'], sources=(), includes=[],
                     toolchain='', authors=f.get('authors'), spdx=f.get('spdx'), notes=f['notes'], why=f['why'],
                     compile='n/a', reason=GATE_GO if go else GATE_PATCH, exclude=f.get('exclude', ())))
    return out


ORIGIN_TOML = '''
# CiruStrixLink: Ciru's USB4 transport and two-host operator console for AMD Strix Halo, the open part of the
# GLM5.3-Flash-CIRU-STRIX-IU4 TP2 deployment: a static Go binary plus a 16-patch Linux 7.2.2 USB4STREAM series whose
# DMA-BUF import lets the USB4 NHI rings move GPU-owned buffers (card ORIGIN-CIRU-LINK 682cd99ba2).
name      = "ciru_link"
repo      = "https://github.com/ciru-ai/CiruStrixLink"
commit    = "1c7f1ae89298c5be2ac4f916b589c77a10c903d8"
branch    = "main"
version   = "v0.3.3 line (CHANGELOG.md tops at 0.3.3; docs/architecture.md names the release commit 220060a58b8164ed0676da0cda60b48759989553; the ark pins main at 1c7f1ae, docs/releases/v0.3.3.md)"
reference_checkout = "control-node:/data/ark-inbox/src/github.com__ciru-ai__CiruStrixLink#1c7f1ae89298c5be2ac4f916b589c77a10c903d8 (ark copy: the tree at the commit with a shallow .git); laptop: ~/Code/strixflerp-upstream-supplement/ciru_link (the same 111 files, by tarball through control-node get); docs/harvest/harvest_ciru_link.py pins the tree digest"

[licence]
spdx      = "Apache-2.0"
files     = ["LICENSE"]                                  # copied verbatim next to this file
third_party = "LICENSE: Apache-2.0, the Go utility. kernel/linux-7.2.2/*.patch are Linux-kernel derivative work under GPL-2.0 (the uapi header thunderbolt-stream.h GPL-2.0 WITH Linux-syscall-note), patches 1-15 from jyatesdotdev/strix-rdma 06c1836 with their authors kept, patch 16 by Ciru; COPYING and LICENSES/ ride in runtime.ciru_link.tbstream_zerocopy. internal/ui/static/assets (IBM Plex fonts, SIL OFL-1.1, no licence text in the tree; Ciru emblem, logo, favicon) is not harvested. The HIP bridge (MIT) and the GLM runtime are in the jcbtc Hugging Face package, not in this tree."

[toolchain]
host      = "Go 1.22 (go.mod; no module dependencies), static linux/amd64 with CGO_ENABLED=0 -trimpath (Makefile linux-amd64); the gate builds with control-node's /opt/go (go1.27.1) and GOTOOLCHAIN=local"
device    = "none: no GPU code in this tree. The direct GPU path needs the patched kernel (Linux v7.2.2 + the 16 patches) and the HIP bridge from the GLM runtime package"
build     = "make build | linux-amd64 | test | vet (Makefile); kernel: apply kernel/linux-7.2.2/0001-0016 in order to a fresh v7.2.2 tree (kernel/linux-7.2.2/README.md)"

[gate]
go        = "python3 docs/harvest/harvest_ciru_link.py --go-gate on control-node CPU: the six Go modules assembled from catalog/ciru_link into one tree, go vet ./..., go test -run '^$' ./... (tests compiled, none run) and the static binary built (gate/results/compile/<module>.json for the six Go modules, binary sha256 in each)"
patches   = "python3 docs/harvest/harvest_ciru_link.py --patch-gate: 16 of 16 patches match kernel/linux-7.2.2/SHA256SUMS and parse with git apply --numstat (gate/results/compile/ciru_link.patches.json); not applied or compiled: no Linux 7.2.2 tree on the island"
hip       = "n/a: the origin holds no HIP source; the bridge was not harvested"

[bridge]                                                 # recorded, not harvested
source    = "nhi-m8-bf16/vllm_nhi_allreduce.cu inside GLM5.3-Flash-CIRU-STRIX-IU4-kernels-source-v0.1.0-rc1.tar.gz (huggingface.co/jcbtc/GLM5.3-Flash-CIRU-STRIX-IU4 at a1ab828d8bd5, runtime/packages); MIT SPDX notice per docs/architecture.md; not in the ark"
design    = "a small C ABI over HIP device pointers and the caller's stream (no PyTorch): TX / RX pools from hipExtMallocWithFlags(..., hipDeviceMallocUncached), exported with hipMemGetHandleForAddressRange(..., hipMemRangeHandleTypeDmaBufFd) and imported with TBSTREAM_ZC_IMPORT, enabled with TBSTREAM_ZC_ENABLE. Exact M8 message: BF16 [8,4096] = 65,536 bytes as 16 4-KiB frames plus one 32-byte footer frame (protocol / dtype / rank, epoch, sequence, payload length; commit word at byte 28 written last with a system-scope release store, polled with an acquire load). Both ranks compute BF16(float(local) + float(peer)). Circular addressing modulo the whole 4,095-frame pool. Synchronous call: pack, wait, submit TX, GPU footer wait / validate / sum, reap events, repost RX. Other shapes and collectives stay on RCCL"
numbers   = "docs/performance.md and README.md, two Ryzen AI Max+ 395 hosts, Linux 7.2.2, ROCm 10: 64-KiB BF16 [8,4096] all-reduce median 109.897-110.807 us direct NHI vs 340.904-342.745 us RCCL (16 warmups, 64 measured calls a rank, 80 of 80 exchanges clean, exact sums; bit-for-bit over 102 calls / 3.34 M elements); single-token activation + credit 45.505 us; T128 activation + credit 1,042.402-1,043.402 us I8G32 vs 1,642.803 us BF16 (experimental codec); 6-MiB BF16 [768,4096] all-reduce 4,516-4,521 us vs RCCL 4,943-4,961 us; TCP echo p50 / p95 / p99 130.975 / 192.473 / 249.991 us. Model effect: the synchronous M8 bridge gave no serving gain; one-deep deferred retirement (not in the published archive) gave +4.05% synthetic all-accept (44.17 -> 45.96 tok/s) and +8.76% natural structured output (21.23 -> 23.09 tok/s) at equal accepted drafts; PP768 prompt 472.241 -> 477.314 tok/s (+1.07%)"

[runtime]                                                # the GLM package this transport serves: proprietary, recorded only
name      = "jcbtc/GLM5.3-Flash-CIRU-STRIX-IU4 (Hugging Face): GLM-5.3-Flash on two 128-GiB Strix Halos, vLLM V2 with a GLM hybrid loader, gfx1151 IU4 kernels, DFlash2 k=5 speculative decoding, TP2 PP1, 64K / 128K / 256K context profiles (6 / 12 / 8 GiB KV a rank), prefix caching off"
licence   = "the runtime package is not open-licensed as a whole and is not harvested; the parent GLM-5.3-Flash is MIT, the wtdcode AWQ derivative the weights come from declares no licence (redistribution pending per the model doc); the DFlash2 draft (incoai/GLM-5.3-Flash-DFlash2 bf582e4e) is CC BY-NC-ND 4.0"
numbers   = "experiments/dflash-256k-20260903/RESULTS.md, 65,680-token prompt, 256 generated, 262,272-token profile, direct NHI: target only 9.38 tok/s; DFlash2 k=3 12.16 tok/s (39.27% acceptance); k=5 15.12 tok/s (37.53%), repeats 12.93 and 14.56 tok/s; TTFT about 175-178 s in every mode (first-use Triton compile of the DeepGEMM paged MQA logits stage); the doc calls it a low-acceptance stress result, not a model-speed baseline"
'''

LEFT_OUT = {
    'internal/ui/static/assets/': 'IBM Plex fonts (SIL OFL-1.1, its licence text is not in the tree) and Ciru branding images; the console compiles without them',
    'docs/, README.md, CHANGELOG.md, SECURITY.md': 'documentation; read for the module records and ORIGIN.toml [bridge] / [runtime]',
    'docs/benchmarks/': 'the M8 raw measurement records and their provenance, quoted in ORIGIN.toml [bridge] numbers',
    'experiments/': 'the DFlash 256K protocol and results of the GLM runtime, quoted in ORIGIN.toml [runtime]',
    'package.json, scripts/dev-preview.js': 'the console\'s JavaScript check and preview server (development only)',
    '.github/, .gitignore, .gitattributes': 'CI, release workflow and repository plumbing',
    'LICENSE': 'copied beside ORIGIN.toml',
}


def interfaces_md():
    """catalog/ciru_link/MODULES.md: each module with its files, lines, interface tag and the catalogue's nearest kin."""
    bt = '\x60'
    kin = {}
    cat = os.path.join(harvestlib.REPO, 'catalog')
    for origin in sorted(os.listdir(cat)):
        if origin == O.name or not os.path.isdir(os.path.join(cat, origin)):
            continue
        for d, _, fs in os.walk(os.path.join(cat, origin)):
            if 'module.toml' in fs:
                t = tomllib.load(open(os.path.join(d, 'module.toml'), 'rb'))
                for p in t.get('provides', []):
                    fam = p.get('interface', '').split('.')[0]
                    if fam in ('transport', 'collective'):
                        kin.setdefault(fam, set()).add(t['id'])
    o = ['# CiruStrixLink modules by strixflerp interface', '',
         'One row for each module of CiruStrixLink at 1c7f1ae (v0.3.3 line). Card ORIGIN-CIRU-LINK 682cd99ba2. Written by',
         bt + 'docs/harvest/harvest_ciru_link.py' + bt + ' from its ' + bt + 'FAMILIES' + bt + ' table.', '',
         '- **tag**: the strixflerp interface class the module matches, or none. Every module here is none: the catalogue has no',
         '  USB4 transport, and its collectives (' + bt + 'collective.*' + bt + ') and transports (' + bt + 'transport.*' + bt + ') run over RoCE verbs or TCP.',
         '  The module\'s ' + bt + '[[provides]]' + bt + ' names what it would be called.',
         '- The GPU side of the direct path (the HIP bridge that packs, validates and sums the BF16 partials) is not in this',
         '  tree; ORIGIN.toml ' + bt + '[bridge]' + bt + ' records its design and numbers.', '',
         '| module | lang | files | lines | provides | tag |', '|---|---|---:|---:|---|---|']
    tot_f = tot_l = 0
    for f in FAMILIES:
        files = O.expand(f['paths'], f.get('exclude', ()))
        lines = sum(len(open(O.find(p), errors='replace').read().splitlines()) for p in files)
        tot_f += len(files)
        tot_l += lines
        o.append('| %s%s%s | %s | %d | %d | %s | %s |' % (bt, f['id'], bt, f['lang'], len(files), lines,
                 ', '.join(bt + p['interface'] + bt for p in f['provides']), f['tag'] or 'none'))
    o += ['', '%d modules, %d files, %d lines.' % (len(FAMILIES), tot_f, tot_l), '',
          '## Nearest kin in the catalogue (same family, different link)', '']
    for fam in sorted(kin):
        o.append('- %s%s.*%s: %s' % (bt, fam, bt, ', '.join(bt + m + bt for m in sorted(kin[fam]))))
    o += ['', '## Where the shapes would go', '',
          '- The DMA-BUF import (' + bt + 'runtime.ciru_link.tbstream_dmabuf' + bt + ') with the [bridge] message layout is the shape of a',
          '  two-host ' + bt + 'collective.allreduce-p2p' + bt + ' provider over USB4: GPU-owned pools, a footer with a release-stored commit word, exact BF16 sums.',
          '- The pair verdict of ' + bt + 'runtime.ciru_link.nhi_endpoint' + bt + ' (inspect both, reconcile, prepare both, inspect again) fits any',
          '  two-node bring-up; the paired load / unload with rollback in ' + bt + 'serve.ciru_link.console' + bt + ' fits a TP2 keeper.',
          '', '## Not done here', '',
          'Nothing was run: no USB4 link joins pod nodes. The Go modules compile on control-node (' + bt + '--go-gate' + bt + '); the patches',
          'are checked against SHA256SUMS and parsed (' + bt + '--patch-gate' + bt + ') but not applied: no Linux 7.2.2 tree is on the island.']
    open(os.path.join(harvestlib.REPO, 'catalog', O.name, 'MODULES.md'), 'w').write('\n'.join(o) + '\n')


def sh(cmd, cwd, env):
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-3000:]


def go_gate():
    """Assemble the Go modules of catalog/ciru_link into one tree and vet / compile them with the local Go. CPU only."""
    gobin = os.environ.get('GO', shutil.which('go') or '/opt/go/bin/go')
    root = os.path.join(harvestlib.REPO, 'catalog', O.name)
    gomods = [f['id'] for f in FAMILIES if f['lang'] == 'go']
    tmp = tempfile.mkdtemp(prefix='ciru-link-go-')
    try:
        tree = os.path.join(tmp, 'src')
        for mid in gomods:
            kind, _, name = mid.split('.', 2)
            src = os.path.join(root, kind, name, 'src')
            shutil.copytree(src, tree, dirs_exist_ok=True)
        env = dict(os.environ, GOTOOLCHAIN='local', GOPROXY='off', GOFLAGS='-mod=mod', GOTELEMETRY='off',
                   GOCACHE=os.path.join(tmp, 'cache'), GOPATH=os.path.join(tmp, 'gopath'), CGO_ENABLED='0',
                   GOOS='linux', GOARCH='amd64')
        rc_v, ver = sh([gobin, 'version'], tree, env)
        steps = []
        bin_path = os.path.join(tmp, 'ciru-strixlink-linux-amd64')
        for label, cmd in (('vet', [gobin, 'vet', './...']),
                           ('test-compile', [gobin, 'test', '-count=1', '-run', '^$', './...']),
                           ('build', [gobin, 'build', '-trimpath', '-ldflags', '-s -w -X main.version=0.3.3', '-o', bin_path, './cmd/ciru-strixlink'])):
            t0 = datetime.datetime.now()
            rc, out = sh(cmd, tree, env)
            steps.append(dict(step=label, cmd=' '.join(cmd[1:]), rc=rc, seconds=round((datetime.datetime.now() - t0).total_seconds(), 2), tail=out[-1500:]))
        ok = all(s['rc'] == 0 for s in steps)
        binsha = harvestlib.sha256(bin_path) if os.path.isfile(bin_path) else None
        base = dict(gate='compile-go', arch='x86_64 (host code)', host=os.uname().nodename,
                    date=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), toolchain=ver.strip(),
                    tree='catalog/ciru_link Go modules ' + ', '.join(gomods) + ' assembled into one tree',
                    env='GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=mod CGO_ENABLED=0 GOOS=linux GOARCH=amd64',
                    steps=steps, binary_sha256=binsha, binary_bytes=os.path.getsize(bin_path) if binsha else None,
                    status='pass' if ok else 'fail')
        if not ok:
            base['reason'] = next(s['step'] + ': ' + s['tail'][-300:] for s in steps if s['rc'] != 0)
        outdir = os.path.join(harvestlib.REPO, 'gate', 'results', 'compile')
        os.makedirs(outdir, exist_ok=True)
        for mid in gomods:
            json.dump(dict(module=mid, **base), open(os.path.join(outdir, mid + '.json'), 'w'), indent=1)
            open(os.path.join(outdir, mid + '.json'), 'a').write('\n')
        print(f"go gate: {base['status']}: " + ', '.join(f"{s['step']} rc={s['rc']} {s['seconds']}s" for s in steps) +
              f"; {len(gomods)} modules; binary sha256 {binsha}; {ver.strip()}")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def patch_gate():
    """Every patch of the catalogue copy against SHA256SUMS, and git apply --numstat (parse only, no tree needed)."""
    root = os.path.join(harvestlib.REPO, 'catalog', O.name)
    found = {os.path.basename(p): p for p in glob.glob(os.path.join(root, '*', '*', 'src', KD, '*.patch'))}
    sums = glob.glob(os.path.join(root, '*', '*', 'src', KD, 'SHA256SUMS'))
    want = dict(reversed(l.split()) for l in open(sums[0]).read().splitlines() if l.strip())
    rows = []
    for name in sorted(set(want) | set(found)):
        got = harvestlib.sha256(found[name]) if name in found else None
        rc, out = sh(['git', 'apply', '--numstat', found[name]], root, os.environ) if name in found else (None, 'missing')
        rows.append(dict(patch=name, module=os.path.relpath(found[name], root).split('/src/')[0] if name in found else None,
                         sha256_ok=got == want.get(name), numstat_rc=rc, numstat=out.strip().splitlines()))
    ok = all(r['sha256_ok'] and r['numstat_rc'] == 0 for r in rows) and len(rows) == 16
    res = dict(gate='patch-check', origin=O.name, host=os.uname().nodename,
               date=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
               what='sha256 against kernel/linux-7.2.2/SHA256SUMS and git apply --numstat (parse only); not applied: no Linux 7.2.2 tree',
               patches=len(rows), passed=sum(1 for r in rows if r['sha256_ok'] and r['numstat_rc'] == 0),
               status='pass' if ok else 'fail', results=rows)
    out = os.path.join(harvestlib.REPO, 'gate', 'results', 'compile', 'ciru_link.patches.json')
    json.dump(res, open(out, 'w'), indent=1)
    open(out, 'a').write('\n')
    print(f"patch gate: {res['status']}: {res['passed']} of {res['patches']} patches match SHA256SUMS and parse")
    return 0 if ok else 1


if __name__ == '__main__':
    if '--go-gate' in sys.argv:
        sys.exit(go_gate())
    if '--patch-gate' in sys.argv:
        sys.exit(patch_gate())
    run(O, mods(), ORIGIN_TOML, LEFT_OUT, ['LICENSE'])
    interfaces_md()
