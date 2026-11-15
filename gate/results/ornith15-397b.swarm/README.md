# Ornith-1.5 397B staged for loom-swarm

What is in place on control-node (2026-11-12 UTC) so the four-node window can pull each rank's bytes over the rail.

## Where the manifests live

The origin (`loom-swarm-origin.service`) runs `find /data/models -name "*.swarm.json"` once at start and serves each
manifest's file from `<manifest dir>/<file_name>` (`resolve_artifact` in `crates/loom-swarm/src/net.rs`; there is no
manifest-directory option and `find` does not follow directory symlinks). nv1 is mounted read-only, and no writable USB
NVMe has room for a 227.5 GiB copy (free: nv2 35G, nv3 83G, nv4 56G ro, nv5 119G ro, nv6 179G). So:

    /data/models/ornith-1.5-397b/                          a real directory on /data (scanned by the origin)
      Ornith-1.5-397B-Q4_K_M.gguf -> /mnt/nv1/models/ornith-1.5-397b/Ornith-1.5-397B-Q4_K_M.gguf
      Ornith-1.5-397B-Q4_K_M.gguf.swarm.json               full manifest
      Ornith-1.5-397B-Q4_K_M.gguf.r{0,1,2,3}of4.swarm.json  one layout artifact per world-4 rank

Nothing is written on nv1. The data symlink only has to resolve on control-node; workers read the manifests from the
`/data/models` export. Risk: the origin bails at start if any manifest's file is missing, so if nv1 is not mounted when
the origin (re)starts, the origin serves nothing at all. Remove this directory when the model is no longer staged.

## Rank layouts

`gate/placed_layout.py FILE --world 4 --out DIR` writes each rank's read log (`<file>.r<r>of4.ranges`, "offset length"
lines): the header `[0, 11013920)` plus every tensor of the gguf_placed stage plan for that rank. Plan sha256
`cff5a987...4327d`, the same as `census.json`'s world-4 plan. Tested on a synthetic GGUF (`tests/test_placed_layout.py`,
including a real `loom-swarm manifest --layout` run on control-node).

| rank | blocks | census tensor bytes | layout bytes (+ 11013920 header) |
|---|---|---|---|
| 0 | 0-15 (+ token_embd) | 61235546112 | 61246560032 |
| 1 | 15-30 | 58372590592 | 58383604512 |
| 2 | 30-45 | 58385026048 | 58396039968 |
| 3 | 45-60 (+ head) | 62033054720 | 62044068640 |

The `file_size` of each layout manifest equals its layout bytes (checked, see `check.txt`).

## Measured

- nv1 sequential read, one O_DIRECT reader (`dd bs=16M iflag=direct`, ionice best-effort 4, 20 s), 1-minute load
  7.95 at start: **126 MB/s** (2.52 GB in 20.0 s). A `grep -rl /data/ark-inbox` from another session was reading the
  same disk at the time, and the device is ~110 MB/s in total with it running. nv1 shares the usb8 hub with nv6. For
  comparison, 8 s O_DIRECT reads gave nv5 764, nv2 534, nv3 346, nv6 192 (while nv1 was hashing), nv4 186 MB/s.
- Full manifest (`ionice -c3 nice -n 19`, 16 threads, load 8-29 during the run): 14562 pieces, root `bfce0f976fc77e9c`,
  **3572 s** (59.5 min, 68 MB/s).
- Layout manifests (each one reads its rank's bytes again): r0 1169 s, r1 2130 s, r2 1523 s, r3 639 s (r3 partly from
  page cache). Roots r0 `5d183bdf3884bb98`, r1 `6b388a9c9ea052d6`, r2 `fb16020841143ef9`, r3 `258f6061c1af302e`.
  Each one coalesces to 2 ranges: the header (plus `output.weight` for rank 3, which sits right after it), then the
  rank's contiguous blocks.
- Origin: restarted once with `swarm.sh origin up`. It logs all five Ornith artifacts and "serving 150 artifact(s)".
  From control-node over the rail address 10.20.0.31:7150: a `pull --match` of 3 pieces of the full manifest (0.05 GiB),
  and 8 s of the r3 layout (1.81 GB, 15 extents, each matching the source file), into /dev/shm. Deleted afterwards.
  No worker pulled.
- Expected window pull: every tensor byte (233 GB for the four ranks) comes off nv1 once. At 70-126 MB/s that is
  **31-55 min** for all four ranks together, however they are scheduled. The rail and peers do not change this, and a
  rank's pull from a node's /cache copy is local in later windows.

## Window commands

See docs/PHASE5-FOURNODE-DESIGN.md, "Staging Ornith with loom-swarm".
