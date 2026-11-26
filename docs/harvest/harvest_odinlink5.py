#!/usr/bin/env python3
"""Harvest OdinLink-Five (Geramy/OdinLink-Five at 0315225) into catalog/odinlink5.
Run: python3 docs/harvest/harvest_odinlink5.py           (rebuild catalog/odinlink5)
     python3 docs/harvest/harvest_odinlink5.py --c-gate  (control-node CPU: compile every userspace C unit with gcc, the
                                                          way the origin's CMake does, from the catalogue copy)

Card ORIGIN-VLLM-STRIX d2a7b91dec (the transport under vllm-strix-halo, catalog/vllm_strix). OdinLink-Five is a
Thunderbolt 5 / USB4 host-to-host transport for Linux: a kernel driver (odl_tb5.ko) that registers a protocol handler
on the stock thunderbolt core, logs into the peer over XDomain and moves 4-KiB frames on NHI DMA rings, exposing
/dev/odl_tb5_N with a stream API; a userspace library (libodl_tb5); RCCL and NCCL net plugins; a libibverbs provider;
optional nvCOMP compression; a GLib daemon with D-Bus, sync and FUSE; a GTK tray; a test CLI; an Apple-Silicon backend
and a macOS kext for Linux -> Mac memory; a Python Thunderbolt-IP tensor bridge.

Scope: every source family, one module each (FAMILIES below); docs, screenshots, CI and container plumbing are listed
as unharvested. Licence: no licence file in the tree. README.md "License": kernel driver GPL v2, all userspace MIT;
the files' SPDX lines say MIT for most of the driver too, GPL-2.0 for odl_tb5_debugfs.c / loopback / service_apple;
rccl/official/nccl/*.h are NVIDIA's Apache-2.0 plugin headers. Each module's spdx follows its files.

The reference checkout is the ark copy on control-node (/data/ark-inbox/src/github.com__Geramy__OdinLink-Five, an
unpinned ark row: a shallow clone at main 03152258f555, 2026-11-04, "Merge pull request #37 from
davidcanar/multi-link-source-xdomain"); the laptop holds the same files, moved by a tarball through control-node get, in
~/Code/strixflerp-upstream-supplement/odinlink5. ODINLINK5_CHECKOUT names another copy. vllm-strix-halo builds against
4534f58 + its odinlink-local.patch; 4534f58 is not in the ark copy (shallow), so this harvest takes the ark's HEAD."""
import datetime, hashlib, json, os, shutil, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvestlib
from harvestlib import Origin, M, P, run

CHECKOUT = os.environ.get('ODINLINK5_CHECKOUT', os.path.join(harvestlib.SUPPLEMENT, 'odinlink5'))
COMMIT = '03152258f55507aa3f62aeb6a42fd09084dd8f23'
TREE_SHA256 = '6ff37b950ace04ee09454d662043d0eedb8beed667384c58392f376639dca02d'
SKIP = ('.git',)


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
    """The ark copy's files, pinned by the tree digest (its .git/HEAD is a branch ref; packed-refs names the commit)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.roots = [CHECKOUT]
        pr = os.path.join(CHECKOUT, '.git', 'packed-refs')
        if os.path.isfile(pr) and COMMIT not in open(pr).read():
            raise SystemExit(f'{pr} does not name {COMMIT}')
        got = tree_digest(CHECKOUT, self.files())
        if got != TREE_SHA256 and os.environ.get('ODINLINK5_PRINT_DIGEST'):
            print(got); sys.exit(0)
        if got != TREE_SHA256:
            raise SystemExit(f'{CHECKOUT}: tree digest {got}, the harvest pins {TREE_SHA256} (OdinLink-Five {COMMIT[:12]})')

    def files(self):
        if self._files is None:
            self._files = tree_files(CHECKOUT)
        return self._files


O = TreeOrigin('odinlink5', 'odinlink5', 'https://github.com/Geramy/OdinLink-Five', COMMIT, 'main',
               'MIT', ['Geramy (OdinLink-Five) and contributors'], '')
NOT_RUN = 'Nothing was run: no Thunderbolt / USB4 link joins pod nodes, no driver was loaded (card d2a7b91dec).'
GPL = 'GPL-2.0-only'
NV = 'Apache-2.0'


def F(id, paths, summary, why, provides=(), tag=None, requires=(), lang='c', spdx=None, notes='', exclude=(),
      cgate=None):
    """cgate: list of (source, [extra flags]) the C gate compiles, or None (not compiled, reason in notes).
    includes: the module's include roots (src/<origin path>), as the origin's CMake target exports them."""
    return dict(id=id, paths=list(paths), summary=summary, why=why, provides=list(provides), tag=tag,
                requires=list(requires), lang=lang, spdx=spdx, notes=notes, exclude=list(exclude), cgate=cgate,
                includes=INCLUDES.get(id, []))


INCLUDES = {
    'runtime.odinlink5.lib': ['src/lib/include'],
    'runtime.odinlink5.rccl_net_plugin': ['src/third_party/rccl', 'src/rccl/official'],
    'runtime.odinlink5.nccl_net_plugin': ['src/third_party/nccl'],
    'runtime.odinlink5.verbs_provider': ['src/verbs/src'],
    'runtime.odinlink5.compress': ['src/compress/include'],
    'runtime.odinlink5.driver_core': ['src/driver/uapi', 'src/driver'],
    'runtime.odinlink5.mac_rdma': ['src/mac/include', 'src/mac/kext'],
    'tool.odinlink5.cli': ['src/cli/src'],
    'tool.odinlink5.daemon': ['src/daemon/src'],
    'tool.odinlink5.tray': ['src/tray/src'],
}


LIBINC = ['-I', 'lib/include']
FAMILIES = [
 F('runtime.odinlink5.lib', ['lib/'],
   'libodl_tb5, the userspace API over /dev/odl_tb5_N: open / close / peer state, the stream API (stream_open by filter id or kernel-assigned id 20-255, stream_send / recv / wait_tx / wait_rx, dmabuf send / recv), the legacy double-buffer API (mmap\'d TX / RX buffers, send / recv / swap) and completions, with the ioctl, types and RCCL / NCCL shared-memory stats headers',
   'every consumer above the driver (the RCCL and NCCL plugins, odl_ar2, odl_mq, the CLI and the daemon) speaks this API; streams give ordered, message-framed, multiplexed channels over one cable, the property odl_ar2 relies on for slot reuse',
   provides=[P('transport.odl-stream/-/-', 'odl_tb5_open, odl_tb5_stream_open / send / recv / wait_tx / wait_rx / send_dmabuf / recv_dmabuf, odl_tb5_get_peer (lib/include/odl_tb5/odl_tb5.h)', 'bytes; messages fragmented into 4024-byte frames, frag-index checked, delivered whole and in order', 'one point-to-point link a device; streams 1-255')],
   tag=None, requires=['module:runtime.odinlink5.driver_core'],
   cgate=[(s, LIBINC + ['-I', 'driver/uapi', '-fPIC']) for s in ('lib/src/odl_tb5_dev.c', 'lib/src/odl_tb5_xfer.c', 'lib/src/odl_tb5_peer.c', 'lib/src/odl_tb5_completion.c', 'lib/src/odl_tb5_stream.c')],
   notes='Interface tag: none (the catalogue\'s transports are RoCE verbs and TCP stage relays). The CMake target adds driver/uapi as a private include, hence the requires on runtime.odinlink5.driver_core. ' + NOT_RUN),
 F('runtime.odinlink5.rccl_net_plugin', ['rccl/', 'third_party/rccl/'],
   'the RCCL net plugin (librccl_net_odl_tb5.so, symlinked librccl-net.so for RCCL\'s probe): ncclNet_v7 over the stream API, NCCL_PTR_HOST only so RCCL stages GPU <-> host itself, one shared ref-counted device handle a process with each connection a stream multiplexed over the link, blocking stream transfers moved to background threads so isend / irecv / test keep RCCL\'s non-blocking contract; plus odl_tb5_net_v7.c, an alternative against NVIDIA\'s real v7 headers (rccl/official) that the build does not compile',
   'RCCL silently falls back to TCP sockets (~4x slower) when no net plugin loads, and it ignores LD_PRELOAD verbs interposers (docs/GPU.md); this plugin is how vllm-strix-halo\'s prefill collectives ride the cable, and the shape any RCCL-over-USB4 pod pair would load',
   provides=[P('collective.net-plugin-rccl/-/-.odl', 'ncclNet_v7 (NCCL_NET_PLUGIN=odl_tb5 / librccl-net.so): init, devices, getProperties, listen, connect, accept, regMr, isend, irecv, test, close*', 'bytes; host pointers', 'RCCL over one Thunderbolt / USB4 link')],
   tag=None, requires=['module:runtime.odinlink5.lib'], spdx='MIT AND Apache-2.0',
   cgate=[('rccl/src/odl_tb5_plugin.c', LIBINC + ['-I', 'third_party/rccl', '-fPIC']), ('rccl/src/odl_tb5_net_v7.c', LIBINC + ['-I', 'rccl/official', '-fPIC'])],
   notes='Interface tag: none (collective.* providers in the catalogue are collectives, not RCCL transports). rccl/official/nccl/*.h are NVIDIA\'s plugin headers (Apache-2.0, SPDX in each); the plugin sources carry no SPDX line and fall under the README\'s MIT for userspace. odl_tb5_net_v7.c says it replaces odl_tb5_plugin.c, but rccl/CMakeLists.txt builds only odl_tb5_plugin.c; both are compiled by the gate. ' + NOT_RUN),
 F('runtime.odinlink5.nccl_net_plugin', ['nccl/', 'third_party/nccl/'],
   'the NCCL net plugin (libnccl-net-ODL_TB5.so) for CUDA hosts: the same stream transport with up to 16 devices, 64 registered regions and 16 comms, GPU memory exported as DMA-BUF through a dlopen\'d cuMemGetHandleForAddressRange for the dmabuf send / recv path, optional nvCOMP compression of large payloads and shared-memory stats',
   'it is the GPU-direct variant of the RCCL plugin: on hardware where the GPU can export DMA-BUFs the copy through host memory goes away. gfx1151 has no GPUDirect here, but the DMA-BUF seam is the one CiruStrixLink\'s patched kernel uses on Strix Halo',
   provides=[P('collective.net-plugin-nccl/-/-.odl-dmabuf', 'ncclNet (NCCL_NET_PLUGIN=ODL_TB5)', 'bytes; host or DMA-BUF exported device memory; optional compression', 'NCCL over Thunderbolt')],
   tag=None, requires=['module:runtime.odinlink5.lib', 'module:runtime.odinlink5.compress'],
   cgate=[('nccl/src/odl_tb5_nccl_plugin.c', LIBINC + ['-I', 'third_party/nccl', '-I', 'compress/include', '-fPIC'])],
   notes='Interface tag: none. ' + NOT_RUN),
 F('runtime.odinlink5.verbs_provider', ['verbs/'],
   'a libibverbs provider over OdinLink: devices, protection domains, memory regions, completion queues, queue pairs and post / poll ops mapped onto the stream API, usable two ways, as an LD_PRELOAD interposer that wraps ibv_open_device for /dev/odl_tb5_* and passes real InfiniBand / RoCE through, or as an rdma-core provider plugin (libodl_tb5-rdmav34.so) so ibv_devinfo lists odl_tb5; with a mock and loopback / dmabuf tests and the technical manual',
   'verbs is the API MPI, NCCL\'s IB transport and ggml-rpc RDMA already speak; a provider makes the cable look like an RDMA NIC to them without code changes, the way strixflerp\'s RoCE rail is reached today (transport.verbs)',
   provides=[P('transport.verbs/-/-.odl', 'ibv_* through libodl_tb5_verbs.so (LD_PRELOAD) or the rdma-core provider plugin', 'bytes; two-sided semantics underneath', 'one Thunderbolt / USB4 link')],
   tag='transport.verbs', requires=['module:runtime.odinlink5.lib'],
   cgate=[(s, LIBINC + ['-I', 'verbs/src', '-fPIC']) for s in ('verbs/src/odl_tb5_verbs_device.c', 'verbs/src/odl_tb5_verbs_pd.c', 'verbs/src/odl_tb5_verbs_mr.c', 'verbs/src/odl_tb5_verbs_cq.c', 'verbs/src/odl_tb5_verbs_qp.c', 'verbs/src/odl_tb5_verbs_ops.c', 'verbs/src/odl_tb5_verbs_discovery.c', 'verbs/src/odl_tb5_verbs_main.c', 'verbs/src/odl_tb5_provider_plugin.c')],
   notes='Interface tag: transport.verbs (an ibverbs device, like the catalogue\'s RoCE provider). Needs libibverbs headers to compile. ' + NOT_RUN),
 F('runtime.odinlink5.compress', ['compress/'],
   'optional link compression: a wire header [odl_compress_header | compressed payload | padding], an nvCOMP backend (gdeflate / lz4 / snappy on the GPU) and a stub when nvCOMP is absent, selected by ODL_COMPRESS, ODL_COMPRESS_ALGO, ODL_COMPRESS_THRESHOLD (256 KiB) and ODL_COMPRESS_LEVEL, with a bench',
   'a 40-80 Gb/s cable is slower than GPU memory, so compressing large collectives on the GPU can win; the stub keeps the plugin linkable where no compressor exists (AMD hosts)',
   provides=[P('transport.payload-compress/-/-', 'odl_compress_* (compress/include/odl_tb5/odl_compress.h)', 'lossless', 'payloads >= threshold; both peers must match')],
   tag=None, cgate=[(s, ['-I', 'compress/include', '-fPIC']) for s in ('compress/src/odl_compress_common.c', 'compress/src/odl_compress_stub.c')],
   notes='Interface tag: none. odl_nvcomp.cpp and the bench need CUDA and nvCOMP and are not compiled. ' + NOT_RUN),
 F('runtime.odinlink5.driver_core', ['driver/odl_tb5_core.h', 'driver/odl_tb5_service.c', 'driver/odl_tb5_proto.c', 'driver/odl_tb5_xd_proto.h', 'driver/odl_tb5_chardev.c', 'driver/odl_tb5_params.c', 'driver/odl_tb5_debugfs.c', 'driver/odl_tb5_loopback.c', 'driver/odl_tb5_transport.h', 'driver/uapi/', 'driver/Kbuild', 'driver/Makefile', 'driver/71-odl-tb5.rules'],
   'the odl_tb5 kernel driver core: module lifecycle (advertises the OdinLink protocol in XDomain property directories, probe on a matching peer, remove on unplug), the XDomain login / logout handshake that agrees DMA HopIDs, enables paths and ping-pongs the link, the /dev/odl_tb5_N character device (stream ioctls, legacy double-buffer ioctls, mmap of DMA buffers, poll, a context per open file), module parameters, debugfs, a loopback mode, the transport-ops interface and the uapi header',
   'this is what lets a stock kernel carry RDMA-like traffic between two hosts over one cable: a tb_protocol_handler on the unmodified thunderbolt core, no patched core and no blacklist; the uapi is the contract every layer above it uses',
   provides=[P('transport.odl-driver/-/-', '/dev/odl_tb5_N ioctls (driver/uapi/odl_tb5_uapi.h); module params ring_size, loopback, protocol, e2e', 'bytes', 'Linux, thunderbolt / USB4 core, one peer a link')],
   tag=None, spdx='MIT AND GPL-2.0-only',
   notes='Interface tag: none. Not compiled: needs the target kernel\'s headers (/lib/modules/<release>/build). SPDX: MIT for most files, GPL-2.0 for odl_tb5_debugfs.c and odl_tb5_loopback.c; README says the driver as a whole is GPL v2. ' + NOT_RUN),
 F('runtime.odinlink5.driver_nhi', ['driver/odl_tb5_transport_nhi.c', 'driver/odl_tb5_ring_dma.c'],
   'the NHI data path of the driver: the transport backend that wraps the kernel\'s tb_ring API and XDomain path management, and the DMA engine (TX / RX rings of 4-KiB frames, a reusable frame pool, a latency mode that pushes one frame at a time inside the ioctl and a throughput mode of 256-KiB batches, RX reassembly of fragmented messages)',
   'the latency mode is why a 48 KiB decode all-reduce is 13 frames from one syscall with no wait for TX completion; the RX reassembly is where odinlink-local.patch had to preallocate buffers on UMA hosts (catalog/vllm_strix runtime.vllm_strix.odinlink_local_patch)',
   provides=[P('transport.nhi-dma-rings/-/-', 'odl_tb5_transport_ops (driver/odl_tb5_transport.h) for Intel / AMD USB4 NHI', 'bytes; 4-KiB frames with framing / CRC reserve', 'one ring pair a peer')],
   tag=None, requires=['module:runtime.odinlink5.driver_core'],
   notes='Interface tag: none. Not compiled (kernel code). ' + NOT_RUN),
 F('runtime.odinlink5.driver_apple', ['driver/odl_tb5_transport_apple.c', 'driver/odl_tb5_service_apple.c', 'driver/odl_tb5_xd_proto_apple.h', 'driver/apple_tb5_nhi_regs.h'],
   'the Apple-Silicon transport backend: Apple NHI DMA on the ACIO fabric from analysis of macOS\'s AppleThunderboltNHI kext (register layout from firmware tunable device-tree blobs, per-HopID MSI-X routing, a shared TX descriptor buffer, completion callback chain, the stopDMA sequence, XDomain login with Apple\'s protocol UUID 0xFA57), probing on apple,thunderbolt-nhi',
   'it implements the same transport-ops interface as the NHI backend, which shows the seam: a second link type slots in under an unchanged core; relevant to strixflerp only as the pattern for adding a transport backend',
   provides=[P('transport.apple-acio-dma/-/-', 'odl_tb5_transport_ops for apple,thunderbolt-nhi', 'bytes', 'Linux on Apple Silicon')],
   tag=None, requires=['module:runtime.odinlink5.driver_core'], spdx='MIT AND GPL-2.0-only',
   notes='Interface tag: none. Not compiled (kernel code). ' + NOT_RUN),
 F('runtime.odinlink5.mac_rdma', ['mac/'],
   'Linux to Mac memory over Thunderbolt 5: a macOS kext (OdinLinkNHI, OdinLinkRDMA) that posts 4-KiB RX descriptors into a DART-mapped buffer the app mmaps, the shared wire protocol header, a Mac client and odl_tensor_send on the Linux side; two-sided send / recv, Linux posts TX frames',
   'Apple\'s own RDMA is Mac-to-Mac and a stub on current macOS; owning both ends is how the project parks tensors in a Mac\'s unified memory. Outside the pod\'s hardware, kept for completeness of the origin',
   provides=[P('transport.mac-dart-recv/-/-', 'OdinLinkRDMA.kext user client; odl_tensor_send (mac/linux_test)', 'bytes', 'Linux PC (Intel NHI) to a Mac')],
   tag=None, notes='Interface tag: none. Not compiled (macOS kext, Apple SDK). ' + NOT_RUN),
 F('tool.odinlink5.cli', ['cli/'],
   'odl_tb5_cli, the link test tool: server mode answers, client mode runs bandwidth, latency, latency-under-load, jitter, MIMO (multi-stream) and diagnostics tests over the library with a small wire protocol and statistics',
   'link qualification before any collective runs: bandwidth and tail latency by message size are the two numbers that decide which traffic a USB4 pair should carry',
   provides=[P('tool.link-bench/-/-.odl-cli', 'odl_tb5_cli server | client <test> [options]', 'timing', 'one link')],
   tag=None, requires=['module:runtime.odinlink5.lib'],
   cgate=[('cli/src/' + s, LIBINC + ['-I', 'cli/src']) for s in ('odl_tb5_cli_main.c', 'odl_tb5_cli_server.c', 'odl_tb5_cli_client.c', 'odl_tb5_cli_bandwidth.c', 'odl_tb5_cli_latency.c', 'odl_tb5_cli_latency_load.c', 'odl_tb5_cli_mimo.c', 'odl_tb5_cli_jitter.c', 'odl_tb5_cli_proto.c', 'odl_tb5_cli_stats.c', 'odl_tb5_cli_diag.c')],
   notes='Interface tag: none. ' + NOT_RUN),
 F('tool.odinlink5.tests', ['tests/', 'odl_stress.c'], 'smoke and stress tests: device ioctl, library API and plugin suites in one plain-C runner, an RCCL DMA-BUF test, lifecycle and RDMA bulk-transfer stress (mirrors ggml-rpc\'s 256-KiB chunks with 24 pre-posted receives and verifies every byte), and odl_stress, which reproduces the RCCL plugin\'s framing over N concurrent streams sharing one handle with sequence numbers to catch lost, reordered or short messages',
   'odl_stress and odl_rdma_stress are how silent message loss in the stream layer was isolated without RCCL or vLLM in the way; they are the conformance tests any USB4 transport on the pod would have to pass',
   provides=[P('tool.transport-conformance/-/-', 'odl_tb5_test [suite], odl_stress server|client <N> <bytes> <iters>, odl_rdma_stress', 'pass / fail, byte-exact', 'one link')],
   tag=None, requires=['module:runtime.odinlink5.lib', 'module:runtime.odinlink5.rccl_net_plugin'], exclude=['tests/test_odinlink_remote.py'],
   cgate=[(s, LIBINC + ['-I', 'driver/uapi', '-I', 'third_party/rccl']) for s in ('tests/odl_tb5_test_main.c', 'tests/odl_tb5_test_device.c', 'tests/odl_tb5_test_lib_api.c', 'tests/odl_tb5_test_plugin.c', 'tests/odl_tb5_test_rccl_dmabuf.c', 'tests/odl_lifecycle_test.c', 'tests/odl_rdma_stress.c', 'odl_stress.c')],
   notes='Interface tag: none. odl_rdma_stress.c needs libibverbs headers. tests/test_odinlink_remote.py belongs to tool.odinlink5.remote_tensors. ' + NOT_RUN),
 F('tool.odinlink5.daemon', ['daemon/'],
   'odl-tb5-daemon, a GLib background service: D-Bus API for peer status (com.odinlink.Tb5Daemon), cable plug / unplug monitoring, a file-sync engine with its own protocol and a catalogue, export of RCCL shared-memory stats, a FUSE mount of the peer filesystem and a link self-test, with its systemd and D-Bus unit files',
   'operational plumbing around a link (who is connected, how RCCL traffic is doing) that a pod pair would want surfaced; strixflerp would take the stats export, not the sync engine',
   provides=[P('tool.link-daemon/-/-', 'D-Bus com.odinlink.Tb5Daemon (daemon/dbus/com.odinlink.Tb5Daemon.xml)', 'n/a', 'desktop / server Linux')],
   tag=None, requires=['module:runtime.odinlink5.lib'],
   cgate=None, notes='Interface tag: none. Not compiled by the gate: needs GLib / gio, FUSE 3 and gdbus-codegen output that CMake generates. ' + NOT_RUN),
 F('tool.odinlink5.tray', ['tray/'],
   'the GTK / Ayatana AppIndicator tray app: peers window, link tests with charts and result parsing, RCCL stats view, sync status and file actions, talking to the daemon over D-Bus',
   'a desktop front end for the daemon; not for the pod, kept so the origin is complete and the D-Bus contract has its client',
   provides=[P('tool.link-tray/-/-', 'odl-tb5-tray (GTK 3)', 'n/a', 'desktop Linux')],
   tag=None, notes='Interface tag: none. Not compiled: needs GTK 3 and ayatana-appindicator. ' + NOT_RUN),
 F('tool.odinlink5.remote_tensors', ['bridge/', 'odinlink/', 'examples/', 'tests/test_odinlink_remote.py'],
   'Python tensor offload over Thunderbolt IP: the TB-Bridge client / server (pure Python, 25-40 Gb/s over TB5 IP, 50-200 us RTT, MLX helpers on the Mac side), RemoteStore (put / get / prefetch_window of PyTorch tensors into a Mac\'s unified memory), an offload example and its test',
   'the interim, copy-based tier for spilled tensors while the DMA path matures; for strixflerp it is the reference for a host-memory spill tier across a cable',
   provides=[P('transport.tensor-offload-ip/-/-', 'odinlink.RemoteStore; bridge/tb_bridge_client.py, tb_bridge_server.py', 'exact copies', 'IP over Thunderbolt')],
   tag=None, lang='python', notes='Interface tag: none. ' + NOT_RUN),
 F('tool.odinlink5.packaging', ['packaging/', 'CMakeLists.txt', 'scripts/'],
   'build and packaging: the top-level CMake (driver via Kbuild, library, plugins, verbs, daemon and tray with dependency probes), DKMS config and install hooks, meta-package script, and the hardware check, smoke-test and Mac RDMA capture / analysis scripts',
   'the top-level CMake is the authoritative build graph the C gate follows (include paths and targets); DKMS is how the driver tracks kernel updates on a host',
   provides=[P('tool.transport-install/-/-.odl-dkms', 'cmake, dkms.conf, shell', 'n/a', 'Debian / Fedora hosts')],
   tag=None, lang='shell', notes='Interface tag: none. Read, not run. ' + NOT_RUN),
]

LEFT_OUT = {
    'docs/, README.md, AGENTS.md, COMPAT.md, mac/README.md': 'documentation, read for the module records (mac/README.md rides in runtime.odinlink5.mac_rdma)',
    'assets/': 'screenshots',
    '.github/, .grok/, .gitignore, .dockerignore, Dockerfile.test, docker-compose.yml': 'CI, agent workflow and container test plumbing',
}

# Licence texts the tree declares but does not ship: {catalog path: repo-relative canonical copy}.
CANONICAL = {'LICENSES/GPL-2.0': 'docs/harvest/data/licences/linux-LICENSES-preferred-GPL-2.0',
             'LICENSES/MIT': 'docs/harvest/data/licences/linux-LICENSES-preferred-MIT',
             'LICENSES/Apache-2.0.txt': 'catalog/strixfold/LICENSES/Apache-2.0.txt'}

ORIGIN_TOML = '''
# OdinLink-Five: a Thunderbolt 5 / USB4 host-to-host transport for Linux on the stock thunderbolt core: a kernel driver
# with a stream API on /dev/odl_tb5_N, its userspace library, RCCL and NCCL net plugins, a libibverbs provider and
# tools. The fabric under vllm-strix-halo (catalog/vllm_strix; card ORIGIN-VLLM-STRIX d2a7b91dec).
name      = "odinlink5"
repo      = "https://github.com/Geramy/OdinLink-Five"
commit    = "03152258f55507aa3f62aeb6a42fd09084dd8f23"
branch    = "main"
version   = "main HEAD 2026-11-04 (Merge pull request #37 from davidcanar/multi-link-source-xdomain); CMake project version 0.1.0. vllm-strix-halo pins 4534f58 plus odinlink/odinlink-local.patch (five fixes); 4534f58 is not in the shallow ark clone"
reference_checkout = "control-node:/data/ark-inbox/src/github.com__Geramy__OdinLink-Five (an unpinned ark row: shallow clone, .git/packed-refs names 03152258f555); laptop: ~/Code/strixflerp-upstream-supplement/odinlink5 (the same 200 files, by tarball through control-node get); docs/harvest/harvest_odinlink5.py pins the tree digest"

[licence]
spdx      = "MIT AND GPL-2.0-only AND Apache-2.0"
files     = []                                           # the tree has no licence file
# The licences the tree declares (README.md and per-file SPDX lines), as canonical texts copied verbatim
# (docs/harvest/harvest_odinlink5.py CANONICAL; sources in docs/harvest/data/licences/SOURCES.tsv)
canonical_files = ["LICENSES/GPL-2.0", "LICENSES/MIT", "LICENSES/Apache-2.0.txt"]
canonical_source = "LICENSES/GPL-2.0 and LICENSES/MIT: the Linux kernel's LICENSES/preferred texts (control-node ark copies, docs/harvest/data/licences/SOURCES.tsv); LICENSES/Apache-2.0.txt: catalog/strixfold/LICENSES/Apache-2.0.txt (StrixFold fork f64b5ff, verbatim)"
canonical_note = "Upstream ships no licence file. It declares MIT per file (SPDX lines) and in README.md and names no copyright holder; LICENSES/MIT is the canonical text with its template line '<year> <copyright holders>' unchanged, no holder supplied here. The GPL-2.0 driver files and NVIDIA's Apache-2.0 plugin headers keep their own notices."
third_party = "No LICENSE file in the tree. README.md, section License: kernel driver (odl_tb5.ko) GPL v2, all userspace MIT. SPDX lines: MIT in most driver, library, compress and daemon files; GPL-2.0 in driver/odl_tb5_debugfs.c, odl_tb5_loopback.c, odl_tb5_service_apple.c; rccl/official/nccl/*.h are NVIDIA's plugin headers under Apache-2.0 (SPDX in each). Files without an SPDX line follow the README. Module spdx fields follow their files."

[toolchain]
host      = "C (CMake, project LANGUAGES C), gcc matching the kernel for the driver (gcc-14 / gcc-15), pkg-config; optional GLib 2.56, GTK 3, ayatana-appindicator, FUSE 3, libcrypto, libibverbs, CUDA + nvCOMP"
device    = "none: no GPU code (the NCCL plugin dlopens libcuda for DMA-BUF export; nvCOMP is optional)"
build     = "cmake -B build && cmake --build build (driver through Kbuild against /lib/modules/$(uname -r)/build; BUILD_KERNEL_MODULE, BUILD_DAEMON, BUILD_TRAY, ODL_ENABLE_NVCOMP); packaging/dkms.conf for the driver"

[gate]
c         = "2026-11-25, gcc 13.3.0: 7 of 7 modules, 38 of 38 units compile (lib 5, rccl_net_plugin 2, nccl_net_plugin 1, verbs_provider 9, compress 2, cli 11, tests 8; log /data/pod/plans/d2a7b91dec/c-gate.log). python3 docs/harvest/harvest_odinlink5.py --c-gate on control-node CPU: every userspace unit of the lib, rccl_net_plugin, nccl_net_plugin, verbs_provider, compress, cli and tests modules compiled with gcc -std=gnu11 -O2 -c and the include paths of the origin's CMake, from the catalogue copy (gate/results/compile/<module>.json). Not compiled: the driver (kernel headers), daemon and tray (GLib / GTK / FUSE), the macOS kext, nvCOMP"
'''


def sh(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-2000:]


def c_gate():
    """Assemble every module's src/ into one tree (origin-relative paths) and compile the userspace C units."""
    root = os.path.join(harvestlib.REPO, 'catalog', O.name)
    cc = os.environ.get('CC', 'gcc')
    rc, ver = sh([cc, '--version'])
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    tmp = tempfile.mkdtemp(prefix='odl5-c-')
    summary = []
    try:
        tree = os.path.join(tmp, 'src')
        for f in FAMILIES:
            kind, _, name = f['id'].split('.', 2)
            shutil.copytree(os.path.join(root, kind, name, 'src'), tree, dirs_exist_ok=True)
        outdir = os.path.join(harvestlib.REPO, 'gate', 'results', 'compile')
        os.makedirs(outdir, exist_ok=True)
        for f in FAMILIES:
            if not f['cgate']:
                continue
            units = []
            for src, flags in f['cgate']:
                obj = os.path.join(tmp, 'obj', src.replace('/', '_') + '.o')
                os.makedirs(os.path.dirname(obj), exist_ok=True)
                cmd = [cc, '-std=gnu11', '-O2', '-D_GNU_SOURCE', '-c'] + flags + [src, '-o', obj]
                t0 = datetime.datetime.now()
                r, out = sh(cmd, cwd=tree)
                ok = r == 0 and os.path.isfile(obj) and os.path.getsize(obj) > 0
                units.append(dict(unit=src, cmd=' '.join(cmd).replace(tmp + '/', ''), rc=r if ok or r else -1,
                                  seconds=round((datetime.datetime.now() - t0).total_seconds(), 2),
                                  object_sha256=harvestlib.sha256(obj) if ok else None, tail='' if ok else out[-800:]))
            npass = sum(1 for u in units if u['object_sha256'])
            res = dict(module=f['id'], gate='compile-c', arch='x86_64 (host code)', host=os.uname().nodename, date=now,
                       toolchain=ver.splitlines()[0] if ver else cc,
                       tree='catalog/odinlink5 modules assembled into one tree at origin-relative paths',
                       units=units, passed=npass, status='pass' if npass == len(units) else 'fail')
            if res['status'] != 'pass':
                bad = next(u for u in units if not u['object_sha256'])
                res['reason'] = f"{len(units) - npass} of {len(units)} units fail; first: {bad['unit']}: {bad['tail'][-240:]}"
            p = os.path.join(outdir, f['id'] + '.json')
            json.dump(res, open(p, 'w'), indent=1)
            open(p, 'a').write('\n')
            summary.append(f"{f['id']} {npass}/{len(units)}")
        print('c gate: ' + '; '.join(summary) + f'; {ver.splitlines()[0] if ver else cc}')
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


GATE_C = 'c: compiled by the C gate, not gate/compile/run.py (python3 docs/harvest/harvest_odinlink5.py --c-gate on control-node CPU, gcc with the include paths of the origin\'s CMake)'


def mods():
    out = []
    for f in FAMILIES:
        if f['cgate']:
            reason = GATE_C
        elif f['lang'] == 'c':
            reason = 'c: not compiled by the gate (' + f['notes'].split('Not compiled')[1].split('.')[0].lstrip(': ').strip() + ')' if 'Not compiled' in f['notes'] else 'c: not compiled'
        else:
            reason = f"{f['lang']}: not compiled"
        out.append(M(f['id'], f['summary'], f['paths'], provides=f['provides'], requires=f['requires'], lang=f['lang'],
                     sources=(), includes=f['includes'], toolchain='', authors=None, spdx=f['spdx'], notes=f['notes'], why=f['why'],
                     compile='n/a', reason=reason, exclude=f['exclude']))
    return out


def modules_md():
    bt = '\x60'
    o = ['# OdinLink-Five modules by strixflerp interface', '',
         'One row for each module of OdinLink-Five at 0315225. Card ORIGIN-VLLM-STRIX d2a7b91dec. Written by',
         bt + 'docs/harvest/harvest_odinlink5.py' + bt + ' from its ' + bt + 'FAMILIES' + bt + ' table.', '',
         '- **tag**: the strixflerp interface class the module matches, or none. Only the verbs provider matches one',
         '  (' + bt + 'transport.verbs' + bt + '); the catalogue has no USB4 or Thunderbolt transport besides catalog/ciru_link.', '',
         '| module | lang | files | lines | provides | tag |', '|---|---|---:|---:|---|---|']
    tf = tl = 0
    for f in FAMILIES:
        files = O.expand(f['paths'], f['exclude'])
        lines = sum(len(open(O.find(p), errors='replace').read().splitlines()) for p in files)
        tf += len(files)
        tl += lines
        o.append('| %s%s%s | %s | %d | %d | %s | %s |' % (bt, f['id'], bt, f['lang'], len(files), lines,
                 ', '.join(bt + p['interface'] + bt for p in f['provides']), (bt + f['tag'] + bt) if f['tag'] else 'none'))
    o += ['', f'{len(FAMILIES)} modules, {tf} files, {tl} lines.', '']
    open(os.path.join(harvestlib.REPO, 'catalog', O.name, 'MODULES.md'), 'w').write('\n'.join(o) + '\n')


if __name__ == '__main__':
    if '--c-gate' in sys.argv:
        sys.exit(c_gate())
    run(O, mods(), ORIGIN_TOML, LEFT_OUT, [], CANONICAL)
    modules_md()
