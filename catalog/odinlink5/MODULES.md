# OdinLink-Five modules by strixflerp interface

One row for each module of OdinLink-Five at 0315225. Card ORIGIN-VLLM-STRIX d2a7b91dec. Written by
`docs/harvest/harvest_odinlink5.py` from its `FAMILIES` table.

- **tag**: the strixflerp interface class the module matches, or none. Only the verbs provider matches one
  (`transport.verbs`); the catalogue has no USB4 or Thunderbolt transport besides catalog/ciru_link.

| module | lang | files | lines | provides | tag |
|---|---|---:|---:|---|---|
| `runtime.odinlink5.lib` | c | 12 | 987 | `transport.odl-stream/-/-` | none |
| `runtime.odinlink5.rccl_net_plugin` | c | 21 | 2578 | `collective.net-plugin-rccl-odl/-/-` | none |
| `runtime.odinlink5.nccl_net_plugin` | c | 3 | 1084 | `collective.net-plugin-nccl-odl-dmabuf/-/-` | none |
| `runtime.odinlink5.verbs_provider` | c | 18 | 4644 | `transport.verbs-odl/-/-` | `transport.verbs` |
| `runtime.odinlink5.compress` | c | 6 | 682 | `transport.payload-compress/-/-` | none |
| `runtime.odinlink5.driver_core` | c | 13 | 3724 | `transport.odl-driver/-/-` | none |
| `runtime.odinlink5.driver_nhi` | c | 2 | 2506 | `transport.nhi-dma-rings/-/-` | none |
| `runtime.odinlink5.driver_apple` | c | 4 | 2034 | `transport.apple-acio-dma/-/-` | none |
| `runtime.odinlink5.mac_rdma` | c | 11 | 1959 | `transport.mac-dart-recv/-/-` | none |
| `tool.odinlink5.cli` | c | 13 | 2900 | `tool.link-bench-odl-cli/-/-` | none |
| `tool.odinlink5.tests` | c | 9 | 1930 | `tool.transport-conformance/-/-` | none |
| `tool.odinlink5.daemon` | c | 26 | 8148 | `tool.link-daemon/-/-` | none |
| `tool.odinlink5.tray` | c | 21 | 4996 | `tool.link-tray/-/-` | none |
| `tool.odinlink5.remote_tensors` | python | 9 | 1366 | `transport.tensor-offload-ip/-/-` | none |
| `tool.odinlink5.packaging` | shell | 11 | 1566 | `tool.transport-install-odl-dkms/-/-` | none |

15 modules, 179 files, 41104 lines.

