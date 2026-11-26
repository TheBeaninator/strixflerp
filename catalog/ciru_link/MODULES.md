# CiruStrixLink modules by strixflerp interface

One row for each module of CiruStrixLink at 1c7f1ae (v0.3.3 line). Card ORIGIN-CIRU-LINK 682cd99ba2. Written by
`docs/harvest/harvest_ciru_link.py` from its `FAMILIES` table.

- **tag**: the strixflerp interface class the module matches, or none. Every module here is none: the catalogue has no
  USB4 transport, and its collectives (`collective.*`) and transports (`transport.*`) run over RoCE verbs or TCP.
  The module's `[[provides]]` names what it would be called.
- The GPU side of the direct path (the HIP bridge that packs, validates and sums the BF16 partials) is not in this
  tree; ORIGIN.toml `[bridge]` records its design and numbers.

| module | lang | files | lines | provides | tag |
|---|---|---:|---:|---|---|
| `runtime.ciru_link.tbstream_zerocopy` | patch | 12 | 2042 | `transport.usb4-stream-zerocopy/-/-` | none |
| `runtime.ciru_link.tbstream_hopid` | patch | 2 | 401 | `transport.usb4-hopid-ownership/-/-` | none |
| `runtime.ciru_link.tbstream_dmabuf` | patch | 2 | 1368 | `transport.usb4-dmabuf-import/-/-` | none |
| `runtime.ciru_link.tbstream_ring_fixes` | patch | 4 | 327 | `transport.usb4-ring-arm/-/-` | none |
| `tool.ciru_link.tbstream_diag` | patch | 1 | 1332 | `transport.usb4-stream-progress/-/-` | none |
| `runtime.ciru_link.nhi_endpoint` | go | 7 | 1365 | `transport.usb4-nhi-endpoint/-/-`, `transport.collective-env-usb4/-/-` | none |
| `runtime.ciru_link.usb4net` | go | 4 | 474 | `transport.usb4net-setup/-/-` | none |
| `tool.ciru_link.linktest` | go | 4 | 701 | `verify.link-test-usb4/-/-`, `verify.link-quality-policy/-/-` | none |
| `tool.ciru_link.prereq` | go | 4 | 720 | `verify.host-prereq-usb4/-/-`, `tool.install-plan-usb4/-/-` | none |
| `serve.ciru_link.console` | go | 12 | 5977 | `serve.operator-console-usb4/-/-`, `serve.peer-agent-usb4/-/-`, `serve.tp2-pair-control/-/-` | none |
| `tool.ciru_link.cli` | go | 3 | 937 | `tool.cli-ciru-strixlink/-/-` | none |

11 modules, 55 files, 15644 lines.

## Nearest kin in the catalogue (same family, different link)

- `collective.*`: `kernel.gufo.qfn_kernels`, `kernel.hipfire.tp`, `kernel.lucebox.collective`, `kernel.radiance.halo_allreduce`, `kernel.strixfold.coll_proxy`, `kernel.tensorfold.cuda_roce_gather`, `runtime.gufo.tp_verbs`, `runtime.hipfire.runtime`, `runtime.strixfold.coll`, `runtime.strixfold.coll_nccl`, `runtime.strixfold.coll_rdma`, `runtime.strixfold.coll_tcp`, `runtime.strixfold_tp4.coll`, `runtime.strixfold_tp4.coll_nccl`, `runtime.strixfold_tp4.coll_rdma`, `runtime.strixfold_tp4.coll_tcp`, `runtime.tensorfold.hip_comm`
- `transport.*`: `runtime.flerp.stage_relay`, `runtime.strixfold.fabric`, `runtime.tensorfold.fabric`

## Where the shapes would go

- The DMA-BUF import (`runtime.ciru_link.tbstream_dmabuf`) with the [bridge] message layout is the shape of a
  two-host `collective.allreduce-p2p` provider over USB4: GPU-owned pools, a footer with a release-stored commit word, exact BF16 sums.
- The pair verdict of `runtime.ciru_link.nhi_endpoint` (inspect both, reconcile, prepare both, inspect again) fits any
  two-node bring-up; the paired load / unload with rollback in `serve.ciru_link.console` fits a TP2 keeper.

## Not done here

Nothing was run: no USB4 link joins pod nodes. The Go modules compile on control-node (`--go-gate`); the patches
are checked against SHA256SUMS and parsed (`--patch-gate`) but not applied: no Linux 7.2.2 tree is on the island.
