# DeepSeek-V4.1-Flash at TP4: recipe `dsv41-flash.strixfold-tp4`

Pod cards bb68604fa9 (FLERP-DSV41-TP4, the recipe at 003f679c28b1) and fe4d694019 (FLERP-TP4-SERVED, retargeted to
the build the pod serves, 72c0f083fca1). The recipe builds the pod's StrixFold TP4 serving engine, the build that
serves the 4-bit model on F D C A, from catalogue modules, and `tools/flerp-stage.py` turns that build into the
deploy directory the ranks start from. The engine is the served one, vision included (rank 0 encodes pictures with
vision.hip); the gate's eight prompts are text.

## What it reproduces

| | |
|---|---|
| serving build | `serve-next-20261123c @ 72c0f083fca1` (TP4-SERVE1 003f679c28b1 + Pillow and body limit 9757a35411, Edge-REAP 023c5d2ad9, world-4 chunk-parity guard c7b0f22dde, loop policy f64f606edd, route records 28153e18f3, per-request acts and the Drafter.at fix 4f9ba05838, logprobs 73bab11681, TP4-LINGER 7aa8487768, expert mask 5bb471d3b9), fork `/data/pod/plans/strixfold/fork` |
| built where, how | `/data/pod/plans/serve-next-20261123c/wt`, `run.sh`: `zig_build.sh native-hip --summary all -Dhip-triton-kit=/data/pod/plans/4d4c5173d2/kit/gfx1151-v6f2c38-tp4 -Dhipcc=<therock-10.2>/bin/hipcc -Drocm=<therock-10.2>` |
| deployed | `~/tfp/deploy/72c0f083fca1` on F D C A from stage `/data/pod/plans/strixfold/deploy-stage/72c0f083fca1` (654 SHA256SUMS lines); conf `tp4-serve1-image-v7.conf` |
| kit | 220 objects (16 HIP sources, 204 Triton), 402 kernels, manifest sha256 `446c43ebb75f0386b6c5ba9ef8de14540c0fbb7f8f814c1faa6ac3888908a0a8` |

003f679c28b1 (what bb68604fa9 matched, kit `4d7b62c5...a4d6`) is an ancestor of 72c0f083fca1: 42 commits, 75 files,
+10,843 lines. `docs/harvest/harvest_strixfold_tp4.py` names the commit once (`HEAD`); moving to the next served build
is: export it, refresh the data files its docstring lists, re-run it, re-read the pins and the two path pins.

## How it is assembled: a sub-origin, not a re-cut

`catalog/strixfold` is cut at f64b5ff, which is the TP2 pair's deployment. Four recipes are pinned to it: tp2, tp2-abi,
tp2-zip2 and flerp-serve-tp2. The TP2 vision work in progress is pinned to it too. The TP4 line is **not** a descendant
of f64b5ff: both branch from upstream 88ea779, and the TP4 side has its own pod commits. Re-cutting `catalog/strixfold`
would move every TP2 recipe off its reference and would stale the zip2 overlay, which holds f64b5ff copies of model.zig
and lanes.zig. So the TP4 tree is a sub-origin (SPEC.md section 3, "an origin has one commit"):

- `catalog/strixfold_tp4` (parent `strixfold`, commit 72c0f083, made by `docs/harvest/harvest_strixfold_tp4.py`) holds
  only what 72c0f083 changes or adds against the catalogue, 28 modules and 400 files:
  - 22 overlay modules `<kind>.strixfold_tp4.<name>`. Each holds the TP4 bytes of the files of
    `<kind>.strixfold.<name>` that changed (92 files in all), verbatim at their origin paths. Each module's
    `overlay.json` gives, per file, the parent's sha256, the TP4 sha256 and the commits that touched it.
  - 66 added files: the import closure of the assembled tree, the build graph's `b.path()` roots and the kit
    registry's new HIP sources. Each is assigned to the overlay module of the file that imports it.
  - 5 kernel modules for the HIP sources the TP4 kit adds: `kernel.strixfold_tp4.aux_markov`, `.exl3_mlp`, `.vision`,
    `.reap` (Edge-REAP statistics, off unless REAP_STATS=1) and `.route_rec` (routing records, off unless ROUTE_RECORDS=1).
  - `tool.strixfold_tp4.dsv41_hip`: the other 242 files of `tools/dsv41-hip` (launcher, serve.conf, checks, kit
    preparation), verbatim. The build never reads them; the deploy directory carries the whole directory, so the
    stage needs them byte for byte.
  - `kernel.strixfold_tp4.triton_kit` with the TP4 Triton kit as its `[[prebuilt]]` record (204 objects, aot.json).
  - `runtime.strixfold_tp4.kit` with the served kit manifest as its record.
- The recipe binds every interface exactly as the TP2 recipe does. The TP4 code enters through
  `[build.overlay] "<id>" = "."` (`flerplib/assemble_zig.py`). With root `"."`, a sub-origin module's files sit at
  their own origin paths. A file that a required module also holds replaces that module's copy, and any other file is
  an addition. An overlay module reaches what the modules it replaces files of reach. A replaced file still counts as
  its old module's when another file imports it.
- One parent file is gone at 72c0f083: `zig/src/families/dsv41/state_file.zig`. It stays in the tree with nothing
  importing it. Apart from that file, the assembled tree (622 files) is byte-identical to `git archive 72c0f083`, path
  for path, and holds every file of its `tools/dsv41-hip`. This was checked after `flerp assemble`.
- Kit pins: `[build.hipcc_output_pins]` holds the 16 cache directory names the reference build compiled each HIP object
  into, read from `/data/pod/plans/serve-next-20261123c/wt/.zig-cache`. Each object's sha256 there equals the served
  manifest's.
- Path pins (new, `[build.path_pins]`): Zig writes the build root (2 strings), a source directory (1) and its global
  cache path (2) into each binary, so a tree built under `build/<recipe>/tree` differs from the reference in those
  bytes. `build.sh` runs Zig under bwrap in a private mount namespace: `/` is bound whole, the nearest existing
  parent of each pinned path gets the namespace's own tmpfs holding binds of its real entries, and tree/ and
  .zig-global are bound at `/data/pod/plans/serve-next-20261123c/wt` and `.../.zig-global-wt`
  (`assemble_zig.py path-view`). The host directory is not created, written or hidden: checked with a listing before
  and after, and a file touched inside the view was absent on the host.
- Build: `build/dsv41-flash.strixfold-tp4/build.sh` on control-node, as a systemd user unit named for the card under
  the same caps as the origin's `zig_build.sh` (MemoryMax 16G, 800% CPU, nice 19; the second build also under
  `taskset -c 8-15,24-31`, zig_build.sh's CPU set).

## The stage, file by file

`tools/flerp-stage.py dsv41-flash.strixfold-tp4 /data/pod/plans/strixfold/deploy-stage/flerp-72c0f083fca1` lays the
deploy directory out as `restore-tfp.sh stage` does, from the recipe's `[deploy.stage]`: `native-hip/` = the build's
zig-out, `tools/dsv41-hip/` = the assembled tree's (catalogue) copy, `pylib/` = Pillow 12.3.0 from
`/data/pod/plans/9757a35411/pylib` checked against `docs/harvest/data/strixfold_tp4/pylib-SHA256SUMS` (a third-party
wheel, recorded by hash, not catalogue source). SHA256SUMS is written the reference's way (`find -type f`, files
named SHA256SUMS or DEPLOY.json left out, sorted under en_US.UTF-8), then DEPLOY.json, then read-only. It then compares
every file with the served stage (`/data/pod/plans/fe4d694019/stage-compare.txt`):

- SHA256SUMS: 654 names on both sides, the same names in the same order (the card's VERIFY `diff` returns 0).
- Byte-equal: 650 of the 654 listed files (220 kit objects and the kit manifest, 263 `tools/dsv41-hip` files,
  166 `pylib` files), and the two unlisted `tools/dsv41-hip/{tri,vision-reference}/SHA256SUMS`.
- Differing: 4 files, every one for the same cause:

| file | flerp sha256 | served sha256 | cause |
|---|---|---|---|
| native-hip/bin/tensorfold-native | 4d11f395f5e9 | dffaa0c9e567 | 11 `__anon_<n>` symbol names in .strtab |
| native-hip/bin/tf-dsv41-lanes | cf41de6aeabf | 6660492721e8 | 10 `__anon_<n>` symbol names in .strtab |
| native-hip/bin/tf-dsv41-layer | 54a66ca4f77d | af85bdf96e7d | 7 `__anon_<n>` symbol names in .strtab |
| native-hip/bin/tf-hip-kit | 4f245df2ef60 | fccfe37d36a7 | 4 `__anon_<n>` symbol names in .strtab |

The cause: Zig 0.17 names an anonymous declaration after its InternPool index, and the pool puts the id of the
compiler thread that interned the value in the high bits of that index; which worker thread does it varies from run
to run. Evidence: each pair has the same size; every differing byte lies inside `.strtab`; `objcopy --strip-all`
of both gives the same sha256 (tensorfold-native a9746708b947..., lanes 0d91038c1b12..., layer 19482d60f3af...,
tf-hip-kit af84c80127ac...); the anonymous numbers are equal in their low bits (the served build has thread id bits
0x14, two flerp builds of the same tree 0x30 and 0x0C); and those two flerp builds also differ from each other in the
same symbols. So the code, data and every loaded section are the served bytes; a rebuild of the fork's own tree would
not reproduce the served binaries either. `flerp-stage.py` says this per file (`symbol_names_only`) and ends
`STAGE-EQUAL-BUT-SYMBOL-NAMES` (exit 3), never `STAGE-EQUAL`.

Not tried: building with one compiler thread on both sides (which should give thread id 0 everywhere and make the
binaries byte-equal); it needs the served stage rebuilt the same way. A one-CPU build here was not started: the
transient unit for it was refused (systemd-run "Access denied").

The serving conf is `configs/dsv41-flash.tp4-serve.conf`: `tp4-serve1-image-v9.conf` with only DEPLOY_DIR changed, to
`$HOME/tfp/deploy/flerp-72c0f083fca1` (line 99, the only line `diff` reports). v9 is v7 with only L2_PREFETCH=0 -> 1
(TP4-SWITCHES 8854252e67: A B B A +3.93 % drafts off, +2.23 % drafts on; the 8 coherence replies byte-equal to the v7
reference `tp4-serve1-ref/tp4-serve1-image-v7.conf.72c0f083fca1.json` in both L2_PREFETCH launches). That evidence ran on
the fork stage 72c0f083fca1; L2_PREFETCH is a launch switch of the same launcher (`serve_lib.sh`, byte-equal in both
stages) driving the same code (the flerp binaries differ only in symbol names, above), and FLERP-TP4-GATE 7f6825bee0 ran
the flerp stage with v7 (8/8 byte-equal), so the flerp stage was not booted with L2_PREFETCH=1 itself (FLERP-TP4-REGISTER
5cf9661e12 moved this conf from v7 to v9).

## The pod's serving entry and its rollback (FLERP-TP4-REGISTER 5cf9661e12)

Installed on control-node as `/data/pod/plans/strixfold/tp4-serve-flerp.conf`, byte-equal to this repo's
`configs/dsv41-flash.tp4-serve.conf` (`cmp` exits 0).

Which file reads it:

- The TP4 keeper `strixfold-pair@tp4` (pod-infra `tools/strixfold-pair.sh`, TP4-CUTOVER cd2c0705b3) reads `SP_CONF` from
  `/data/pod/tools/strixfold-pair-tp4.env`. pod-infra's copy of that file (`tools/strixfold-pair-tp4.env`) names
  `/data/pod/plans/strixfold/tp4-serve-flerp.conf`. It is installed at cutover step T3 and not before, and its timer stays off
  until the cutover, which is owner's call. Before each start the keeper copies the conf to `~/tfp/run/conf/tp4-serving.conf`
  on the four nodes, and `first_light.sh` reads that copy.
- By hand (how the instance is served today, under a hold over F D C A): `tp4-serve1.sh` reads the file named by `CONF=`.

Rollback to the fork stage (`tp4-serve1-image-v9.conf`, `deploy-stage/72c0f083fca1`, the same flags), one command, under the
hold the window holds (`tp4-serve1.sh` refuses without one, and while any strixfold-pair timer is active):

```
S=/data/pod/plans/strixfold; CONF=$S/tp4-serve-flerp.conf $S/tp4-serve1.sh down && CONF=$S/tp4-serve1-image-v9.conf $S/tp4-serve1.sh up
```

Under the keeper (after the cutover), the rollback is one line of the env file the keeper reads:
`sed -i 's|^SP_CONF=.*|SP_CONF=/data/pod/plans/strixfold/tp4-serve1-image-v9.conf|' /data/pod/tools/strixfold-pair-tp4.env`
(make the same change in pod-infra, so the installed file does not drift). A serving keeper does not restart for a conf
change. A pod-quad hold on F D C A makes it take its run down, and after `pod-quad release-owner` it starts on v9.

Not ready under the keeper yet: the keeper's start runs `restore-tfp.sh` with the conf, and `restore-tfp.sh` refuses any
DEPLOY_DIR that does not end in a commit12 (`flerp-72c0f083fca1` does not). Its defaults are also pair 1's (native sha,
links). Also, `mk-tp4-cutover.py` (cutover T1 and T2) refuses any conf whose NODES is not 'F D B A', and every TP4
conf, v7 and v9 included, is 'F D C A'. The GO-FF-NODE.md amendment of 5cf9661e12 lists both gaps.

## Serving fixes: carried and not carried

Source list: `/data/pod/plans/bb68604fa9/LESSONS.md`. Each item has the list's tag (V = VERIFIED, R = REPORTED). The
table was made for 003f679c28b1 (bb68604fa9); 003f679c is an ancestor of 72c0f083, so every fix carried there is in
72c0f083 too, and the two rows that were not carried because 9757a35411 was not deployed are carried now (marked
72c0f083). An item the recipe depends on was re-checked against 003f679c itself: commit ancestry (`git merge-base --is-ancestor`) and
the code at that commit. A sha the list cites may not be the one in 003f679c; where the fix reached 003f679c through a
different commit, both are named. The four ways an item can be carried:

- **tree**: the fix is code in the assembled tree. The module that carries it is named.
- **kit**: the fix lives in the prebuilt record or in the pins.
- **gate**: the fix is in the launcher (`tools/dsv41-hip/serve_lib.sh`, `first_light.sh`) or in the serving conf. Since
  fe4d694019 the launcher is catalogue content (`tool.strixfold_tp4.dsv41_hip`, byte-equal to the served stage's) and
  the conf is `configs/dsv41-flash.tp4-serve.conf`, so these fixes reach the flerp stage from strixflerp itself.
- **not carried**: the reason is given.

### Top 10

| # | lesson (V/R) | carried | where / why |
|---|---|---|---|
| 1 | Collective deadline: 2000 ms killed long prompts; COLL_PROGRESS_MS=15000 (V) | gate (conf, enforced) | Engine default is still 2000 (`tcp_coll.zig:386`), so the value comes from the conf. The TP4 confs ran 2000 until 2026-11-21. The TP4 session (agent) then raised all 14 to 15000: `tp4-serve1*`, `6270c982cc/kit/tp4w-*`, and the generator source `716f35d40a/kit/tp4-fl.conf`, with backups `*.bak-collprog-20261121`. `gate/dsv41_tp4.py` builds both arms from `tp4-serve1.conf`, and `conf_for` refuses a conf below `COLL_PROGRESS_FLOOR = 15000`. The instance running on 2026-11-21 keeps 2000 until its next launch |
| 2 | COLL_SLOT_KIB=4352 made GRAPHS=1 unusable; 8704 everywhere (V) | tree + gate | `rdma_proxy.zig:485 default_slot_kib_n = 8704` (runtime.strixfold_tp4.coll_rdma); serve_lib refuses GRAPHS=1 below the need (`serve_lib.sh:113-118`); conf `COLL_SLOT_KIB=8704` |
| 3 | World-4 transport hardcodes: comm.zig tcp at world 4 (14202a0e), one RDMA device per rank (585e3e71), first_light per-rank devices (f30049ee), device names differ, empty TF_DS_AUX (d2c69b40) (V) | tree + gate | 14202a0e is in 003f679c (`comm.zig`, runtime.strixfold_tp4.coll_nccl). 585e3e71, f30049ee and d2c69b40 are launcher commits, all ancestors of 003f679c (`serve_lib.sh:282-284`, `first_light.sh:143`); they apply through the gate's deployed tools/ |
| 4 | TP4 nodes on the wrong 2D expert prefill path at >= 64 rows; 2531e8fd moved them to w-pf (V) | tree | 2531e8fd is an ancestor of 003f679c; its engine files are in executor.strixfold_tp4.dsv41 |
| 5 | Build with the TP4 Triton kit gfx1151-v6f2c38-tp4, not af9276f296 (V) | kit | `[build.prebuilt] kernel.strixfold_tp4.triton_kit = /data/pod/plans/4d4c5173d2/kit/gfx1151-v6f2c38-tp4`; build.sh checks all 204 objects and aot.json against the record before it builds |
| 6 | hipcc cuid hashes the source path; another worktree gives another kit sha (V) | kit | `[build.hipcc_output_pins]`, 14 pins from the reference build's Zig cache |
| 7 | Rank 0 core-dumped on stop; leave() = fflush then _exit; raw_exit on the HIP engine (V) | tree (partly) | The rank 0 fix reached 003f679c as **6c9bc923** ("server: on HIP the server leaves through _exit ..."), not as the cited b695e264/cc89979e, which are not ancestors: `zig/src/server/leave.zig` (serve.strixfold_tp4.http) and the engine host. Rank 1's `_exit` is 74b715d8 (`lanes_main.zig:44`, executor.strixfold_tp4.dsv41). Not carried: the same exit-139 in strixflerp's own adapter host (serve.flerp.strixfold_host) is a different binary, and this recipe builds the fork's server |
| 8 | A killed rank leaves its flock holder; next up waits LOCK_WAIT_S (V) | gate | 93d28458 (first_light.sh) is an ancestor of 003f679c; the gate downs through first_light.sh and checks for zero rank processes |
| 9 | Rank units need LimitMEMLOCK=infinity; serve_rccl_check refuses without (V) | gate (check) / not carried (drop-in) | The memlock check is in serve_lib (`serve_lib.sh:227-236, 257`). The systemd drop-in is part of the worker image, not of any build |
| 10 | A pod-thermal push killed rank 0 ~1 s later (observation) | gate (etiquette) | Root cause unproven; the gate plan forbids thermal pushes while ranks run |

### (a) Build / kit

| lesson (V/R) | carried | where / why |
|---|---|---|
| RCCL first link ~2 h; ninja install deletes output; patchelf; p2p.h guard (V) | not carried | RCCL is a runtime library (`RCCL_LIB` in the conf), not built by this recipe. The serving conf runs TRANSPORT=rdma |
| Native gfx1151 RCCL replaces torch's; no second HIP runtime; NATIVE_RCCL=1 (V) | not carried | Runtime environment of the RCCL arm (launcher and worker), outside the build |
| Host-suite flake: rdma_proxy4 fault matrix under load (V) | not carried | The recipe runs the `native-hip` step only, not the host tests. The test sources (`rdma_proxy4_test.zig`) are in the tree because Zig loads imported test files |
| Deploy dir left read-only by br-run broke `up` (V) | gate | tp4-serve1.sh's deploy_stage runs `chmod -R u+w` before it replaces a deploy dir |
| Workers lack Pillow; TF_PIL_PATH (9757a35411 @ 03d67fc1, READY, not deployed) (V) | tree + stage (72c0f083) | 03d67fc1 is an ancestor of 72c0f083 (serve.strixfold_tp4.http); the stage's `pylib/` is the [deploy.stage.inputs] Pillow, byte-equal to the served one. Was not carried at 003f679c |
| Bumping frame_protocol breaks the model.zig handshake test (V) | not carried | No protocol change here. The recipe changes no origin file |

### (b) Launch / launcher / conf

| lesson (V/R) | carried | where / why |
|---|---|---|
| Rank order F D C A; engram half follows the rank; NODES, NODE_IPS, NODE_RAILS, RDMA_DEV0-3 (V) | gate | The gate confs are tp4-serve1.conf's bytes (`NODES='F D C A'`) with only DEPLOY_DIR and LOGITS_SHA changed |
| World-4 generalized in serve_lib, first_light, rankf, tfp-node0, serve-rank1.sh; four GPU locks all-or-nothing (V) | gate | Launcher (7f2a4e75 is an ancestor); the deployment's tools/ |
| Scripts piped to ssh swallow stdin; use files with </dev/null (R) | gate | tp4-serve1.sh passes commands with `ssh -n` |
| systemd user units, not setsid; SPAWN_UNIT (V) | gate + build | tp4-serve1.sh starts first_light.sh in a transient user unit. The recipe's own build also runs in a named user unit |
| Worker image: rankf ssh gate, memlock drop-in need an image rebuild (V) | not carried | Worker image, not a build artefact |
| serve_lib forced PROMPT_DISK_GIB=0 at world 4; c96f0318 enables sf-disk at WORLD 4 (V) | tree + gate | e33be4f4 (disk tier at world 4: executor.strixfold_tp4.dsv41, runtime.strixfold_tp4.core) and c96f0318 (serve_lib) are both ancestors; conf `PROMPT_DISK_GIB=2.5` |
| Unsuffixed TP4 serve conf names its rollback confs; per-build stored reference (V) | gate | The gate stores no reference of its own. It compares arm F with arm R in the same window |
| Reference first start fails ~half the time (pair era) (R) | gate | Budget one retry of arm R's up. Recorded, not explained |

### (c) Collectives / rail / RDMA

| lesson (V/R) | carried | where / why |
|---|---|---|
| RCCL 11 ms/round vs RDMA 5.6 ms; TF_COLL_LANES 4 of 8 slots, cold prefill 971 -> 1204 tok/s (V) | tree | 96d9d9ed (`rdma_proxy.zig:486`, runtime.strixfold_tp4.coll_rdma); c6fa0b6f catRank pair exchange (executor.strixfold_tp4.dsv41); conf TRANSPORT=rdma |
| Every RDMA rank must report TF_COLL_KIND=rdma; a wrong grep gave false faults (V) | not carried | A runner's environment check. This gate judges outputs, not environment greps |
| NFS v3 on the rail IP (V) | not carried | Mount policy of the workers; the gate stages nothing |
| Never change NIC pause/PFC/MTU under live RDMA (V) | gate (etiquette) | In the plan's prerequisites |
| SN2700 lossless RoCE: PFC one priority, DSCP 26, ECN (V) | not carried | Switch configuration |
| COLL_SWAP12 opt-in (V) | tree | 4467d9c4 (runtime.strixfold_tp4.coll_rdma, executor.strixfold_tp4.dsv41); off unless the conf sets it |

### (d) State / cache / prompt

| lesson (V/R) | carried | where / why |
|---|---|---|
| Kept states per rank (`<key>.r<rank>`), state bytes identical across ranks (untagged in the list) | tree | Kept-state code of the 003f679c engine (lanes/disk). The identity holds only if the code is byte-equal, and it is: the tree equals the export |
| Disk states at world 4: reconcile ANDs three followers; disk_drop frames; cache import ships rank 0 bytes (V) | tree | e33be4f4 (executor.strixfold_tp4.dsv41, runtime.strixfold_tp4.core; `prompt_disk*.zig` added) |
| /cache routes 401 without CACHE_TRANSFER_KEY bearer; FreeFitty sends none (V) | tree + gate | `--cache-transfer-key` (`server/cli.zig:239`, `serve.zig:105`, `cache_key.zig`, serve.strixfold_tp4.http); conf `CACHE_TRANSFER_KEY`. Not carried: the FreeFitty client side |
| Engram local on each rank's /cache; over rail NFS it stalled prefill (V) | gate | The gate confs keep the served engram paths; nothing is moved |
| Engram placements protected only through their window; renew via PLACE (V) | not carried | PLACE policy; the gate reads placements and changes none |
| Memory: min MemAvailable 16-20 GiB on F at 16 streams, POOL=22000000; queue cap PRs #58-60 (V) | tree + gate | Queue cap reached 003f679c as **37b2db2a** (`server/queue_cap.zig`, serve.strixfold_tp4.http), not as the cited 31415770 (not an ancestor); conf STREAMS=16, POOL=22000000; tp4-serve1.sh smoke checks MemAvailable >= 8 GiB |
| VGATHER=1 byte-equal; TP4_FILL=1 not byte-equal to exact; gate each build against its own reference (V) | tree + gate | 2513d98c/5cb6ff9e (VGATHER) and cd557327 (TP4_FILL) are ancestors. The gate runs the serving conf (TP4_FILL=1, VGATHER=1) on both arms, so equality means the same build, not exactness |
| PREFILL_SPLIT=256 and PREFILL_ZIP2=1 in the serving confs (V) | tree + gate | `zip2d.zig` added (executor.strixfold_tp4.dsv41); conf lines 176, 180 |
| 4-bit decode numbers (G3 29.3 RCCL, RDMA 84.5/48.6 drafts on) (V) | not carried | Measurements, not fixes. The gate records speed and never judges it |

### (e) Vision at TP4

| lesson (V/R) | carried | where / why |
|---|---|---|
| World-agnostic vision: tower on rank 0, rows hidden-wide on every rank, bias_vl whole on every rank (V) | tree (dormant) | 3d24dbf0 (executor.strixfold_tp4.dsv41), vit_hip.zig + vision.hip (kernel.strixfold_tp4.vision): compiled in because the deployment has them. The text-only gate never sends a picture. The TP4 vision recipe is a later card |
| tp4-int predates SF-IMAGES; merge sf-image-cache (IMAGE_TOKENS, ea255f2c) (V) | tree (dormant) | 0fd4db5b (merge) and da090917/ea255f2c (encoder picture cache) are ancestors |
| gate_bias_vl: 40 tensors at 4.0 vs 43 at 2.9; make_bias_vl.py (V) | not carried | Model inputs of the vision recipe (ds-kit-512k-vis4), not of a text-only build |
| Vision files on /cache/dsv41-vision-4p0 via PLACE (V) | not carried | Staging, vision only |
| Body limit 32 -> 256 MiB (9757a35411, not deployed); picture cache 1024 MiB (V) | tree (72c0f083) | 8c30702f is an ancestor of 72c0f083: `http_body.zig` 256 MiB (serve.strixfold_tp4.http); ENCODE_CACHE_MIB=1024 is in the conf. Was not carried at 003f679c |
| Image prompts skip RAM prompt cache, sf-disk and reuse hook (V) | tree (dormant) | Behaviour of the 003f679c code; untested by a text-only gate |

### (f) Process / window etiquette

| lesson (V/R) | carried | where / why |
|---|---|---|
| Extend holds with `OWNER=ff-<card> pod-quad renew`; never pod-window open; leases > 6 h need owner's label (V) | gate (etiquette) | Plan prerequisites: hold owner `ff-bb68604fa9`, renew only |
| pod-quad sweep reaps a hold whose card is in review (V) | gate (etiquette) | Plan: keep the gate's card in an active column during the window |
| Take boxes under boxes.lock; down at first fault; coordinate in window card comments (V) | gate (etiquette) | Plan prerequisites |
| Pod not production during TP4; worker /cache displaceable via PLACE; never touch Engram dirs (V) | gate (etiquette) | Nothing is staged on /cache |
| Stage only via PLACE; B's NVMe collapses after ~29 GB; no second 80 GB rank set fits (V) | gate | Arm F reuses the served rank dirs and bundle. Only code (~0.5 GB: binaries and kit) is deployed, as tp4-serve1.sh deploys a stage |
| PLACE freed-space race; pod-place.timer takes a window's requests (V) | not carried | The gate makes no PLACE request |
| Workers hang on systemctl reboot: sysrq b, dnsmasq, outlet (V) | not carried | Recovery procedure, not part of a build or gate |
| Gate written before the run; PASS = byte-equal logits, never loosened; DRY=1 every kit; `script - DRY=1` passes args (V) | gate | gate/dsv41_tp4.py is written before any run and fixes its criterion in its docstring. `--plan` is its DRY form |

### (g) Shutdown / exit / cleanup

| lesson (V/R) | carried | where / why |
|---|---|---|
| Down order: first_light.sh down, stop leftover tfp-lock* units, zero rank processes on all four (V) | gate | tp4-serve1.sh down after each arm |
| Rank 0 fix branch sf-rank0-exit b695e264 (upstream #33) (V) | tree | Present as 6c9bc923 (see Top 10 #7); b695e264 itself is not an ancestor |
| Upstream runtime-fix PRs #53 timer, #54 keepalive, #55 server _exit, #56 empty AUX, #58-60 queue cap (V) | tree + gate | #53 = 33999a4f (runtime.strixfold_tp4.coll, executor overlay); #54 = **42ae9458** (`link.zig`, runtime.strixfold_tp4.coll; the cited 16a9f775 is not an ancestor); #55 = 6c9bc923; #56 = d2c69b40 (launcher, gate); #58-60 = 37b2db2a. Issue #57 is a proposal, nothing to carry |
| FreeFitty-side leftovers on F A after a crash: d74915b5d6 (R) | not carried | FreeFitty, not the engine |
| Hand back: keep placements or place evict; ask before leaving a hold; leave boxes down (V) | gate (etiquette) | Plan "after" step |

**Count: 58 items. Carried: 43 (in the tree, the kit, the stage, or the catalogue's launcher and conf). Not carried: 15
(each row above gives the reason).** Two of the carried rows also have a part that is not carried (#7: the adapter host;
#9: the image drop-in). One carried row (#1) is carried as the deployment has it, which differs from the lesson.

## Gate

`gate/dsv41_tp4.py`, written before any run, retargeted by fe4d694019. PASS requires the recipe's kit manifest to equal
the served one (446c43eb...), then all 8 frozen prompts must give arm F (the flerp stage `flerp-72c0f083fca1` with
`configs/dsv41-flash.tp4-serve.conf`'s bytes) the same rank-0 prompt-logits sha256 and row count, the same generated
token ids and the same reply text as arm R (served 72c0f083fca1, tp4-serve1-image-v7.conf). `--plan` is the DRY form;
it reads files on control-node and contacts no worker. The hardware run is FLERP-TP4-GATE 7f6825bee0, which can judge
against the stored reply set `tp4-serve1-ref/tp4-serve1-image-v7.conf.72c0f083fca1.json` instead of booting R.

## Not verified

- No GPU run of the recipe's build or of the flerp stage (7f6825bee0).
- The four binaries are not byte-identical to the served ones (the stage section above gives the cause and the proof
  that only symbol names differ).
- `restore-tfp.sh` takes a DEPLOY_DIR that ends in a 12-hex commit; `flerp-72c0f083fca1` does not, so placing the
  stage on the workers needs restore-tfp.sh to accept the name (or a copy step of its own) in 7f6825bee0.
- `kernel.strixfold_tp4.*` objects are not in `gate/compile` (the standalone compile gate). They are compiled only
  inside this recipe, and each one's bytes are proven through the kit manifest.
- `catalog/strixfold_reap` (card 89442dbceb) is a sub-origin cut over strixfold_tp4 at 003f679c (023c5d2ad9 @
  feb5614b). 72c0f083 already holds Edge-REAP (merge 43515ccc) and later edits to some of the same files, so laying
  that overlay over this strixfold_tp4 is not checked here; it is left as is (`flerp check` reports no error) and
  should be re-harvested over 72c0f083 before a recipe combines the two.
