# COLL-MODEL: exhaustive model check of the proxy slot protocol

Written 2026-11-22 (UTC). Code modelled: the StrixFold
serving tree at fork commit 003f679c28b1 (the same tree as deploy-stage/003f679c28b1):
`zig/src/families/dsv41/{tcp_proxy,rdma_proxy,rdma_core,rdma_nic,tcp_coll,tcp_wire}.zig`, `zig/kernels/hip/dsv41/coll.hip`,
COLLECTIVES-DESIGN 2.5, 3.2, 3.6. The model: `tools/coll-model/coll_model.py` (pure Python, no TLC or z3), tables from
`tools/coll-model/report_tables.py`, raw results and counterexample traces in `docs/experiments/coll-model-traces/`.
Runs: control-node CPU, nice 19, no hold, no weights, nothing touched on any worker.

## Result

- **No violation of S1, S2, S3, D1 or T1 in any configuration the code accepts**, at every size the search exhausted
  (every D1 row at world 2 and 4 and every T1 row at world 2; world-4 T1 is bounded, see its table). Every world-2 row (tcp and rdma, proxy threads 1 and 2, wire threads 1 and 2, NSLOT 2..8, D1 and T1) is
  exhausted and clean.
- **The deployed TP4 transport holds.** tp4-serve1*.conf: TRANSPORT=rdma, WORLD=4, TF_COLL_SLOTS and TF_COLL_LANES unset
  (8 slots, 4 lanes: 2 slots a lane), COLL_PROXY_STRIPE_KIB=0, COLL_WIRE_THREADS unset (1). Decode shape (5 calls of gathers, swap12 gathers and both exchange pairings, lane 0): 4,780 states; prompt shape (5 calls of 5-chunk gathers and 2-chunk exchanges, all four lanes in flight, both slots of every lane reused): 12,847 states; one 5-chunk gather: 522 states. All exhausted, no violation of S1, S2, S3 or D1. T1 at world 4 is bounded (its table).
  No live bug, so no message was sent.
- **Hypothesis (lean-spike 2.2): first half confirmed, second half refuted.** One proxy thread: no violation at any swept
  size, world 2 and 4. Two proxy threads: no S1 counterexample either. The only two-thread design in the code,
  tcp_proxy.Second, is handed each gather and the first thread waits for its half before it publishes done
  (tcp_proxy.zig:134-151), so there is still one gather sequence; the same design over a second QP (what rdma would get if
  rdma_proxy.zig:211-215 stopped refusing it) holds S1, S2, D1 and T1 at NSLOT 2..8. The correspondence table puts the
  difference on the review's side: the 2026-11-06 argument assumed a second thread that breaks the single sequence, a
  design the code does not have. The refusal is conservative, not load-bearing, as long as a future rdma stripe keeps the
  hand-and-join; removing the join gives an S2 counterexample in 12 steps (control tcp-stripe-nowait / rdma-stripe-nowait).

## What the model is

Explicit-state, breadth-first, every interleaving of the atomic steps below up to the partial-order reduction described
further down (whose soundness is cross-checked against weaker levels); a run that reaches its state budget is
reported as not exhausted. Faults and deadlines are nondeterministic transitions.

- **Ranks** 2 (tcp_proxy, rdma_proxy RdmaProxy) or 4 (RdmaProxyN over CoreN). Each rank has a GPU stream, one or two
  proxy threads, a NIC (RC queue pair per peer; per direction a FIFO of posted writes and of unacknowledged ones, one send
  and one receive completion queue) or TCP wires (FIFO per direction), and a host thread.
- **GPU** runs the ops of each call in stream order exactly as the launches queue them: `coll.stage` (reads seq, waits
  consumed, copies, publishes bytes/site/seq/ready), `coll.collect` (reads seq once, waits done, copies, publishes
  consumed), at world 4 per lane with Layout.depth's order (stages of chunks 0..lanes-1, then collect c, stage c+lanes)
  and one collect per rank for chunked gathers. Gathers larger than a slot at world 2 are tcp_host calls (two links,
  one or two wire threads).
- **Proxy** steps are those of tcp_proxy Core.run / Second.run, rdma_core Core.run / completions (one completion per
  poll) and CoreN.run / start / check / poison, including the early-trailer branches and the Mismatch exits.
- **Memory**: every TX slot, RX slot (world 2: two halves; world 4: four quarters of the data area and one trailer
  place per rank) carries a tag (generation, writer rank, call) and a *pending* mark from the write until this rank's
  collect (or, for a trailer, the proxy's check) has read it, plus a count of NIC reads in flight when it is a write's
  source.
- **Script**: the calls are chosen nondeterministically and lazily, the same for every rank (the coll.zig contract):
  world 2 from {S slot gather, P slot gather at least the stripe size, H gather larger than a slot}; world 4 from
  {a gather k=1, s gather k=1 swap12, x exchange (0,1)(2,3), z exchange (0,3)(1,2) with the lower rank sending 0 bytes}
  (decode shape, lane 0 only) and {c gather of lanes+1 chunks, y exchange of 2 chunks (0,2)(1,3)} (prompt shape, every
  lane). Script length per row is in the tables.
- **Properties.** S1: a write onto a pending region of a rank that has not failed, and S1-TX: a source region rewritten
  while a NIC read of it is in flight, are violations at the step that does it. S2: every collect checks the tag of every
  region it copies (generation q, the expected writer, this call). S3: a static check of every remote write's byte range
  (`coll_model.py s3`). D1 (no faults, no deadline fires): every terminal state has every rank finished and healthy, so a
  deadlock and a spurious failure (a Mismatch, a skipped collect) are both D1 violations. T1 (one rank dies, wedges or has
  a deadline fire, at any point; afterwards every deadline the code has may fire): every terminal state has every live
  rank finished; a live rank whose calls did not all complete reports PeerLost or Timeout, never OK and never another code.

## Verdicts per size

### World 2 (tcp_proxy; rdma_proxy RdmaProxy)

S1 and S2 are checked on every step of both runs; S3 is the static check (19530 write ranges, no problem); states: D1 run /
T1 run (3 calls) / T1 with NSLOT + 2 calls (wire threads 2 only; + = budget reached, not exhausted).

| transport | proxy threads | wire threads | NSLOT | S1 | S2 | S3 | D1 | T1 | states D1 / T1 / T1 deep |
|---|---|---|---|---|---|---|---|---|---|
| tcp | 1 | 1 | 2 | holds | holds | n/a | holds | holds | 1739 / 89567 / - |
| tcp | 1 | 1 | 3 | holds | holds | n/a | holds | holds | 3124 / 89567 / - |
| tcp | 1 | 1 | 4 | holds | holds | n/a | holds | holds | 4913 / 89567 / - |
| tcp | 1 | 1 | 5 | holds | holds | n/a | holds | holds | 7106 / 89567 / - |
| tcp | 1 | 1 | 6 | holds | holds | n/a | holds | holds | 9703 / 89567 / - |
| tcp | 1 | 1 | 7 | holds | holds | n/a | holds | holds | 12704 / 89567 / - |
| tcp | 1 | 1 | 8 | holds | holds | n/a | holds | holds | 16109 / 89567 / - |
| tcp | 1 | 2 | 2 | holds | holds | n/a | holds | holds | 1881 / 90309 / 245478 |
| tcp | 1 | 2 | 3 | holds | holds | n/a | holds | holds | 3385 / 90309 / 581805 |
| tcp | 1 | 2 | 4 | holds | holds | n/a | holds | holds | 5329 / 90309 / 1281674 |
| tcp | 1 | 2 | 5 | holds | holds | n/a | holds | holds | 7713 / 90309 / 2709853 |
| tcp | 1 | 2 | 6 | holds | holds | n/a | holds | holds | 10537 / 90309 / 5000001+ |
| tcp | 1 | 2 | 7 | holds | holds | n/a | holds | holds | 13801 / 90309 / 5000003+ |
| tcp | 1 | 2 | 8 | holds | holds | n/a | holds | holds | 17505 / 90309 / 5000001+ |
| tcp | 2 | 1 | 2 | holds | holds | n/a | holds | holds | 3504 / 402423 / - |
| tcp | 2 | 1 | 3 | holds | holds | n/a | holds | holds | 6348 / 402423 / - |
| tcp | 2 | 1 | 4 | holds | holds | n/a | holds | holds | 10032 / 402423 / - |
| tcp | 2 | 1 | 5 | holds | holds | n/a | holds | holds | 14556 / 402423 / - |
| tcp | 2 | 1 | 6 | holds | holds | n/a | holds | holds | 19920 / 402423 / - |
| tcp | 2 | 1 | 7 | holds | holds | n/a | holds | holds | 26124 / 402423 / - |
| tcp | 2 | 1 | 8 | holds | holds | n/a | holds | holds | 33168 / 402423 / - |
| tcp | 2 | 2 | 2 | holds | holds | n/a | holds | holds | 3661 / 403959 / 1387250 |
| tcp | 2 | 2 | 3 | holds | holds | n/a | holds | holds | 6637 / 403959 / 4332949 |
| tcp | 2 | 2 | 4 | holds | holds | n/a | holds | holds | 10493 / 403959 / 5000001+ |
| tcp | 2 | 2 | 5 | holds | holds | n/a | holds | holds | 15229 / 403959 / 5000003+ |
| tcp | 2 | 2 | 6 | holds | holds | n/a | holds | holds | 20845 / 403959 / 5000001+ |
| tcp | 2 | 2 | 7 | holds | holds | n/a | holds | holds | 27341 / 403959 / 5000003+ |
| tcp | 2 | 2 | 8 | holds | holds | n/a | holds | holds | 34717 / 403959 / 5000002+ |
| rdma | 1 | 1 | 2 | holds | holds | holds | holds | holds | 2198 / 115283 / - |
| rdma | 1 | 1 | 3 | holds | holds | holds | holds | holds | 3996 / 115283 / - |
| rdma | 1 | 1 | 4 | holds | holds | holds | holds | holds | 6330 / 115283 / - |
| rdma | 1 | 1 | 5 | holds | holds | holds | holds | holds | 9200 / 115283 / - |
| rdma | 1 | 1 | 6 | holds | holds | holds | holds | holds | 12606 / 115283 / - |
| rdma | 1 | 1 | 7 | holds | holds | holds | holds | holds | 16548 / 115283 / - |
| rdma | 1 | 1 | 8 | holds | holds | holds | holds | holds | 21026 / 115283 / - |
| rdma | 1 | 2 | 2 | holds | holds | holds | holds | holds | 2350 / 116013 / 315998 |
| rdma | 1 | 2 | 3 | holds | holds | holds | holds | holds | 4278 / 116013 / 748774 |
| rdma | 1 | 2 | 4 | holds | holds | holds | holds | holds | 6782 / 116013 / 1648053 |
| rdma | 1 | 2 | 5 | holds | holds | holds | holds | holds | 9862 / 116013 / 3481259 |
| rdma | 1 | 2 | 6 | holds | holds | holds | holds | holds | 13518 / 116013 / 5000001+ |
| rdma | 1 | 2 | 7 | holds | holds | holds | holds | holds | 17750 / 116013 / - |
| rdma | 1 | 2 | 8 | holds | holds | holds | holds | holds | 22558 / 116013 / - |
| rdma | 2 (hypothetical) | 1 | 2 | holds | holds | holds | holds | holds | 5220 / 700501 / - |
| rdma | 2 (hypothetical) | 1 | 3 | holds | holds | holds | holds | holds | 9533 / 700501 / - |
| rdma | 2 (hypothetical) | 1 | 4 | holds | holds | holds | holds | holds | 15138 / 700501 / - |
| rdma | 2 (hypothetical) | 1 | 5 | holds | holds | holds | holds | holds | 22035 / 700501 / - |
| rdma | 2 (hypothetical) | 1 | 6 | holds | holds | holds | holds | holds | 30224 / 700501 / - |
| rdma | 2 (hypothetical) | 1 | 7 | holds | holds | holds | holds | holds | 39705 / 700501 / - |
| rdma | 2 (hypothetical) | 1 | 8 | holds | holds | holds | holds | holds | 50478 / 700501 / - |
| rdma | 2 (hypothetical) | 2 | 2 | holds | holds | holds | holds | holds | 5402 / 701655 / 2416837 |
| rdma | 2 (hypothetical) | 2 | 3 | holds | holds | holds | holds | holds | 9871 / 701655 / 3000001+ |
| rdma | 2 (hypothetical) | 2 | 4 | holds | holds | holds | holds | holds | 15680 / 701655 / 3000001+ |
| rdma | 2 (hypothetical) | 2 | 5 | holds | holds | holds | holds | holds | 22829 / 701655 / 3000001+ |
| rdma | 2 (hypothetical) | 2 | 6 | holds | holds | holds | holds | holds | 31318 / 701655 / 3000002+ |
| rdma | 2 (hypothetical) | 2 | 7 | holds | holds | holds | holds | holds | 41147 / 701655 / 3000001+ |
| rdma | 2 (hypothetical) | 2 | 8 | holds | holds | holds | holds | holds | 52316 / 701655 / 3000001+ |

### World 4 (rdma_proxy RdmaProxyN over CoreN: the deployed TP4 transport)

Proxy threads 2: refused (rdma_proxy.zig:509-512). TCP at world 4: refused (comm.zig:24). Wire threads: RdmaProxyN never
opens tcp_host (openN, rdma_proxy.zig:491-579), so TF_COLL_WIRE_THREADS has no code path and every row holds for 1 and 2.
Shapes: decode = 2m+1 calls of {a, s, x, z} (k = 1, lane 0); prompt = 2m+1 calls of {c, y} (every lane, every lane's
slots reused); prompt1 = one gather of lanes + 1 chunks. m = slots a lane. States: decode / prompt / prompt1.

| NSLOT | lanes (m) | S1 | S2 | S3 | D1 decode | D1 prompt | D1 prompt1 | states |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 (2) | holds | holds | holds | holds | holds | holds | 4780 / 2390 / 210 |
| 3 | 1 (3) | holds | holds | holds | holds | holds | holds | 7018 / 3408 / 210 |
| 4 | 1 (4) | holds | holds | holds | holds | holds | holds | 9256 / 4426 / 210 |
| 4 | 2 (2) | holds | holds | holds | holds | holds | holds | 4780 / 8895 / 314 |
| 5 | 1 (5) | holds | holds | holds | holds | holds | holds | 11494 / 5444 / 210 |
| 6 | 1 (6) | holds | holds | holds | holds | holds | holds | 13732 / 6462 / 210 |
| 6 | 2 (3) | holds | holds | holds | holds | holds | holds | 7018 / 17553 / 314 |
| 6 | 3 (2) | holds | holds | holds | holds | holds | holds | 4780 / 10871 / 418 |
| 7 | 1 (7) | holds | holds | holds | holds | holds | holds | 15970 / 7480 / 210 |
| 8 | 1 (8) | holds | holds | holds | holds | holds | holds | 18208 / 8498 / 210 |
| 8 | 2 (4) | holds | holds | holds | holds | holds | holds | 9256 / 29171 / 314 |
| 8 (deployed) | 4 (2) | holds | holds | holds | holds | holds | holds | 4780 / 12847 / 522 |

### T1 at world 4

One slot gather (a) or one pair exchange (x) at 8 slots / 4 lanes; with one call the model is the same for every NSLOT and
lanes setting with 2+ slots a lane. Faults in every visited state; before the fault the reduced steps of the D1 search.

| call | T1 | S1 / S2 under faults | states | seconds |
|---|---|---|---|---|
| x | not exhausted (4000002 states) | none seen | 4000002+ | 762.8 |
| a | not exhausted (4000003 states) | none seen | 4000003+ | 794.2 |

### The reduction against weaker ones

| configuration | full | basic | none |
|---|---|---|---|
| w2 tcp stripe=1 NSLOT=2 D1 6 calls | 3661, clean | 4208, clean | 42848, clean |
| w2 rdma stripe=0 NSLOT=3 D1 8 calls | 4278, clean | 4363, clean | 33958, clean |
| w2 rdma stripe=1 NSLOT=2 D1 6 calls | 5402, clean | 5708, clean | 505232, clean |
| w4 NSLOT=2 lanes=1 D1 5 calls of axzs | 4780, clean | 64260, clean | - |
| w4 NSLOT=3 lanes=1 D1 7 calls of axzs | 7018, clean | 96216, clean | - |
| w4 NSLOT=2 lanes=1 D1 2 calls of cy | 863, clean | 58998, clean | - |
| w4 NSLOT=4 lanes=1 D1 2 calls of cy | 863, clean | 58998, clean | - |
| w4 NSLOT=4 lanes=2 D1 2 calls of cy | 1458, clean | - | - |
| w4 NSLOT=8 lanes=4 D1 1 calls of c | 522, clean | 3000001+, clean | - |
| w4 NSLOT=4 lanes=1 D1 9 calls of axzs | 9256, clean | 128172, clean | - |
| w4 NSLOT=2 lanes=1 D1 1 calls of a | 82, clean | - | - |
| w4 NSLOT=2 lanes=1 D1 2 calls of a | 163, clean | - | - |
| w4 NSLOT=8 lanes=4 D1 2 calls of cy | 2104, clean | 3000001+, clean | - |
| w2 tcp stripe=0 NSLOT=2 T1 3 calls | 90309, clean | 90309, clean | 146431, clean |
| w2 rdma stripe=0 NSLOT=2 T1 3 calls | 116013, clean | 116013, clean | 259966, clean |

### Controls (mutations and refused settings: the checker must find these)

| control | what it changes | expected | found | states |
|---|---|---|---|---|
| rdma-nslot1 | TF_COLL_SLOTS=1, refused at rdma_proxy.zig:204 | S1 | S1: rank 1 RX[0] half 0 still holds gather 1 (call 0) for its collect when gather 2 (call 1) from rank 0 lands | 28 |
| tcp-nslot1 | TF_COLL_SLOTS=1 with tcp (refused at tcp_proxy.zig:486); RX is written by the rank's own recvmsg | none | none (exhausted) | 92 |
| rdma-nogate | coll.stage without its consumed wait (coll.hip:59-64): stream order alone already orders the reuse | none | none (exhausted) | 115 |
| rdma-stream | the next stage runs before this collect (gatherEnd on another stream, refused at rdma_proxy.zig:156) | S2 | S1: rank 1 RX[1] half 0 still holds gather 1 (call 0) for its collect when gather 3 (call 2) from rank 0 lands | 178 |
| rdma-nogate-stream | both: no consumed wait and the next stage ahead of the collect | S1 | S1: rank 1 RX[1] half 0 still holds gather 1 (call 0) for its collect when gather 3 (call 2) from rank 0 lands | 178 |
| tcp-stripe-nowait | the first proxy thread publishes done without waiting for the second cable's half (tcp_proxy.zig:142-151) | S2 | S2: rank 0 collect of call 0 (gather 1) copies RX[1] half 1 holding gather 0 of call -1 written by rank -1 | 12 |
| rdma-stripe-nowait | hypothetical striped rdma whose first thread does not wait for QP 2's half | S2 | S2: rank 0 collect of call 0 (gather 1) copies RX[1] half 1 holding gather 0 of call -1 written by rank -1 | 15 |
| w4-depth | one stage more ahead than Layout.depth (two stages of one lane before its collect) | S2 | S2: rank 0 collect of call 0 chunk 0 (lane 0 gather 2) reads slot 0 quarter 0 holding gather 2 of call (0, 2) from rank 0 (expected rank 0) | 204 |
| w4-lane1slot | one slot a lane (Layout.lanesFit refuses it, rdma_proxy.zig:502) | S1 | S1: rank 0's write of gather 2 lands in rank 1's slot 0 quarter 0, which still holds gather 1 for its collect | 98 |
| rdma-stripe-handoff-T1 | hypothetical striped rdma (tcp_proxy's Second design over a second QP), dead/wedged peer | none | none (exhausted) | 249438 |

Reading the tables: "holds" = the search exhausted the configuration's state space with no violation. The state counts
are after the reduction described below. T1 rows use a shorter script than D1 rows (3 calls at world 2, 1 at world 4):
the deadline paths do not depend on slot reuse (coll.stage's consumed wait never blocks under stream order, finding 1);
the "T1 deep" state count repeats T1 with every slot reused (NSLOT + 2 calls) where the budget allowed (+ = budget
reached, not exhausted, no violation in the states seen).

## Findings

1. **What carries RX-slot reuse is stream order plus two slots, not the consumed gate.** With coll.stage's consumed wait
   deleted (control rdma-nogate) the search still exhausts with no violation: on one stream, stage(q + 1) cannot start
   before collect(q) finished, so at most one slot gather is in flight per rank (per lane at world 4) and the gate never
   blocks. Breaking stream order (control rdma-stream: the next stage runs before this collect, which gatherEnd's
   same-stream check at rdma_proxy.zig:156 / :435 refuses) or using one slot (rdma-nslot1, refused at rdma_proxy.zig:204)
   gives an S1 counterexample. COLLECTIVES-DESIGN 3.6 says this; the model now checks it. The gate stays useful as
   defence in depth (a second stream, a future graph split), but the argument should name stream order first.
2. **At world 4 the rule is per lane, and the deployed setting sits exactly on its edge.** 8 slots in 4 lanes is 2 slots a
   lane, the minimum the transitive argument needs; Layout.lanesFit (rdma_proxy.zig:502) is what keeps a lane at 2 or
   more: one slot a lane is an S1 counterexample (control w4-lane1slot). Layout.depth (one stage ahead a lane,
   rdma_core.zig:330) is load-bearing too: one more stage ahead makes a collect read the next chunk's sequence (control
   w4-depth, S2). Raising TF_COLL_LANES or lowering TF_COLL_SLOTS past 2 a lane is refused at open, so no setting can
   reach the bad case.
3. **TCP needs no second slot for RX.** tcp_proxy's RX is written only by the rank's own recvmsg, gated by its own ready, so
   NSLOT = 1 is clean for tcp (control tcp-nslot1) and not for rdma; tcp_proxy.zig:486 refuses NSLOT < 2 anyway (TX reuse
   would then rest on stream order alone, which finding 1 shows is enough).
4. **T1 holds wherever exhausted, and the slow path is bounded by the deadlines.** Every wait in the modelled code has a
   bound (device 10 s, proxy progress TF_COLL_PROGRESS_MS = 15 s in the TP4 confs, RC retries about 0.5 s), and a
   failure reaches the peers as PeerLost (poison write, RETRY_EXC, closed socket) or Timeout. At world 4 a follower watches
   only rank 0's control socket (rdma_core.zig:559-564), so a follower waiting on another follower's trailer when that
   follower dies leaves through rank 0's poison or through its own deadline (up to the 10 s GPU wait), not within
   milliseconds; correct, just slow.
5. **A contract the model assumes, checked by reading: both pairs of a world-4 exchange take the same number of slot
   chunks.** CoreN expects a trailer from every rank at every q. prompt2d's exchanges are catRank (g ^ 2, each pair moves
   {w0, w1}: symmetric), engramSum's bf16 step (g ^ 2: symmetric) and engramSum's fp32 step (g ^ 1: pair p moves
   n x ew[p] x 4 bytes). ew comes from plan.zig:78-80, cut of (hc + 1) x hidden = 25,600 columns = 200 blocks of 128 into
   12,800 + 12,800, so the deployed model is symmetric. A split with an odd block count would give the two pairs
   different chunk counts for some prompt lengths and CoreN would stop with Mismatch or Timeout (fail-stop, not silent
   corruption). Not live; worth an assert at Two.init if another model is ever split this way.

## Counterexamples

None in any configuration the code accepts. The controls' traces, each step mapped to the code it models, are in
`docs/experiments/coll-model-traces/` (rdma-nslot1, rdma-stream, rdma-nogate-stream, tcp-stripe-nowait,
rdma-stripe-nowait, w4-depth, w4-lane1slot). Two of them, short:

**rdma, TF_COLL_SLOTS=1 (refused at rdma_proxy.zig:204): S1 in 23 steps.** Rank 0 and rank 1 finish gather 1 at their
proxies (rdma_core.zig:74-85); rank 0's collect copies RX[0] (coll.hip:108-114), rank 0 stages gather 2 into the same slot
(coll.hip:59-81: q = 2 > nslot, consumed = 1 passes the gate), its proxy posts the write (rdma_core.zig:68) and rank 0's
NIC places gather 2 in rank 1's RX[0] while rank 1's collect of gather 1 has not run. With two slots the peer's write
of q + 2 needs this rank's trailer of q + 1, posted after this rank's stage(q + 1), which stream order puts after its
collect(q): the review's transitive argument, now checked rather than argued.

**tcp, two proxy threads, first thread not waiting for the second cable (tcp_proxy.zig:142-151 removed): S2 in 12 steps.**

    4. rank 0 proxy sees ready[1] = 1 (call 0), hands half 1 to the second thread   tcp_proxy.zig:134-136, :237-242
    5. rank 0 proxy sends gather 1 (call 0) on cable 1                               tcp_proxy.zig:138
   10. rank 0 proxy receives the peer's gather into RX[1]                            tcp_proxy.zig:138 (half 0 only)
   11. rank 0 proxy: done[1] = 1                                                     tcp_proxy.zig:153 (without :142-151)
   12. rank 0 collect call 0: done[1] = 1, copies RX[1]                              coll.hip:97-114
       -> RX[1] half 1 was never received: the collect copies stale bytes

## Correspondence table

Model state (coll_model.py) to code:

| model | code |
|---|---|
| seq, ready[s], done[s], cons[s], sinfo[s] (bytes/site), abort | tf_ds_coll_shared / tf_ds_coll_slot (abi.h:332-358, tcp_proxy.zig:33-43); world 4: one header a lane (rdma_proxy.zig:295-297, :358-363) |
| failed | TcpProxy/RdmaProxy/RdmaProxyN.failed, status() (tcp_proxy.zig:333-344, rdma_proxy.zig:68-79, :337-347) |
| px (q, phase, flags, split) | tcp_proxy.zig Core.run locals (q :109, second/first :134-135); rdma_core.zig Core.run q :36, completions have_send/have_recv :92-93 |
| sx (todo, last, done, ...) | tcp_proxy.zig Second (:219-235) |
| turns[l] (active, q, have_send, have_recv, early) | rdma_core.zig CoreN Turn (:438-451) |
| tx[s], rx[s] tags, trailer tags | the slab's TX/RX slots (tcp_proxy.zig:326-331, rdma_proxy.zig:233-235, :530-531), the 64-byte trailer (rdma_core.zig:63-65, :612-614) |
| out / unack queues, scq / rcq | the RC QPs' send queues and the send/receive CQs (rdma_nic.zig:135-158, :356-384) |
| wire FIFOs, wire sequence numbers | tcp_wire.zig Wire (header seq/site/bytes/part, check :328) |
| st (L/D/W), faults | process death (QPs destroyed, sockets closed) / wedge (process stopped, NIC up) |

Model steps to code (the full list, used to annotate every trace line, is STEP_CODE in coll_model.py):

| model step | code |
|---|---|
| stage (world 2 / world 4) | coll.hip:47-84; launched by tcp_proxy.zig:426, rdma_proxy.zig:144 / :393-400 (own = send at world 4) |
| collect reads seq; collect | coll.hip:93; coll.hip:86-117; launched by tcp_proxy.zig:442, rdma_proxy.zig:162 / :402-410 |
| launch order at world 4 | RdmaProxyN.gatherBegin/gatherEnd/exchange rdma_proxy.zig:420-458, Layout.depth/gatherStep/exchangeStep/slotIndex rdma_core.zig:287-379 |
| device deadline, abort word | coll.hip:20-29, :62, :100 |
| GPU skips everything after an abort | coll.hip:59, :97 (tf_ds_coll_aborted) + status() at every begin/end/check |
| tcp_host gather (wire threads) | tcp_coll.zig:282-330; tcp_wire.zig:203-224 exchange, :230-265 runBoth + Helper :336-380, :266 run, :328 header check |
| tcp proxy start / send / receive / done | tcp_proxy.zig:111-136, :138, :153 |
| tcp second thread, first thread's wait | tcp_proxy.zig:244-268, :142-151 |
| rdma start (trailer, post) | rdma_core.zig:40-71, rdma_nic.zig:206-211 |
| rdma completions, trailer check, done | rdma_core.zig:91-154, :74-85 |
| rdma deadline / peer gone / abort while waiting | rdma_core.zig:139-145, :146-149 + :169-178, :135-138 |
| CoreN start / poll send / poll receive / finish | rdma_core.zig:465-472 + :592-643, :490-501, :503-531 + check :646-663, :533-543 |
| CoreN abort / deadline / control socket | rdma_core.zig:480-484 + :546-549, :551-558, :559-564 |
| fail, poison, QPs to error | tcp_proxy.zig:160-169 + :346-352; rdma_proxy.zig:81-86; rdma_core.zig:156-159, :665-694; rdma_proxy.zig:380-383 |
| NIC placement, ack, retries exhausted | RC semantics; rdma_nic.zig:443-457 (data unsignaled, then signaled trailer), :460-466 poison, :184/:416 retry_cnt 7, :96-98 statusCode |
| refused settings | rdma_proxy.zig:204, :211-215, :494-512; tcp_proxy.zig:486; comm.zig:24 |

## Reduction, and why the verdicts are still exhaustive

1. **Partial-order reduction** (persistent sets of size one): when a step that commutes with every step that can precede
   it is enabled, only that step is expanded. Three levels, all in the code (`"level"` in a configuration):
   - *basic*: acks and "collect reads seq" always; in D1 also CQ polls, publishing done, a lane's finish, sends, the
     proxy's start and coll.stage when no write touching that slot is queued, and placements or receives onto a region
     that holds nothing pending. A placement onto pending data (the S1 question itself) is never reduced.
   - *full* (the tables): adds three structural rules, each argued from the model's own program order and the trailer
     handshake, not from the properties: a collect at the head of the stream when seq < q + m - 1 (every write of gather
     q is placed before done, and a peer's write of q + m into this slot needs this rank's trailer of q + m - 1, i.e. a
     stage later in this stream); coll.stage with writes of peers in flight toward its slot (they land in their own
     blocks); the world-4 start with early trailers already in. With one slot a lane, or under the stream control, the
     first rule does not apply and the collect is expanded, which is why those controls are still found.
   - *none*: every interleaving.
   T1 before the fault: only steps whose effect is erased by the canonical form once their own rank stops, and that
   commute with a fault of another rank. **Evidence**: the "reduction against weaker ones" table: on every configuration
   where a weaker level finishes it gives the same verdict (clean) as full, with 1.0x to 94x the states; at 8 slots / 4
   lanes the basic level did not finish within 3M states (clean so far), and the world-4 runs at level none did not finish
   in the time this card had (not listed). Every control is found at full (rdma-stream shows S1, the overwrite, one step
   before the S2 its label expected).
2. **Canonical form**: what no later step can read is erased (the call carried by a region nobody will read again, the
   site word of a slot the proxy finished, an idle second thread's last gather, a stopped rank's completion queues and
   loop state), and the wire sequence numbers, which are only compared for equality between the ranks, are rebased. This
   merges states that differ only in history; it does not remove behaviour.
3. **Script length** bounds the run (tables). Every D1 row reuses every slot (world 2: 2 NSLOT + 2 calls; world 4 decode
   shape: 2 m + 1 calls on lane 0, m = NSLOT / lanes; prompt shape: 2 calls of lanes + 1 or 2 chunks, so every lane runs
   and lane 0 wraps).

## Fidelity caveats

- **Memory model.** Flag stores/loads are atomic and ordered (release/acquire, system scope, as coll.hip and the gpuvis
  Phase-0 test establish); a kernel's copy is one step at its completion point, and the pending marks make any write that
  would overlap a copy a violation. Weaker orderings (a missing fence) are outside the model.
- **NIC.** RC in-order placement; a write with immediate's receive completion implies its data and the data write posted
  before it on that QP are placed; receive queues never run dry (256 posted, at most one turn ahead per lane) and CQ
  overflow (cq_depth 512) is not modelled; writes are whole (no torn placement). A death drops every write not yet placed.
- **Contract assumptions.** All ranks issue the same calls with the same sizes (a divergent script is not explored), and
  at world 4 every pair of an exchange call has the same chunk count (CoreN expects a trailer from every rank at every q).
- **T1 fault timing.** One fault a run (a death, a wedge, or a deadline firing early on one rank). At world 2 faults are
  injected in every visited state and only steps argued to commute with a fault are reduced before it. At world 4 the
  T1 runs (tpor) keep the D1 reduction before the fault and add every fault in every visited state, so a death between
  two reduced steps (for example between a write's posting and its placement) is covered only up to that commutation;
  after the fault nothing is reduced. They are one-call runs and reached their state budget (table), so world-4 T1 is
  bounded evidence, not a proof. By reading, every wait in RdmaProxyN/CoreN has a bound (coll.hip device deadline,
  CoreN progress deadline :551-558, RC retries, the control-socket check, poison writes), the same structure the
  exhausted world-2 T1 rows check.
- **Wire threads** (TF_COLL_WIRE_THREADS) change which thread moves tcp_host's second link, not the set of message
  interleavings; the model has the helper's job hand-off but the two settings explore the same orders. At world 4 the
  setting has no code path.
- **Not modelled.** Graph capture and replay (the kernels read seq from the slab, so a replay is the same steps),
  tcp_host's chunked pipelining (only its two-link striping), close() teardown, gatherBegin/End misuse paths other than the
  stream check, idle sleeps and CPU pinning (timing only), the abort word's 2/3 logging values, the 2-rank RdmaProxy's
  fallback for gathers larger than a slot when a graph is capturing (refused, rdma_proxy.zig:130-135).
- **Stripe with rdma** does not exist in the code; the "proxy threads 2" rdma rows model what removing the refusal with
  tcp_proxy's Second design over a second QP would give, so they say what a future change may rely on, not what runs.

## Reproduce

On control-node (or any box with Python 3.12), from the card branch:

    nice -n 19 python3 tools/coll-model/coll_model.py s3 > s3.json
    nice -n 19 python3 tools/coll-model/coll_model.py sweep mut --jobs 2 --out mut.jsonl
    nice -n 19 python3 tools/coll-model/coll_model.py sweep w2 --jobs 6 --out w2.jsonl --max-states 6000000
    nice -n 19 python3 tools/coll-model/coll_model.py sweep w2deep --jobs 2 --out w2deep.jsonl --max-states 5000000
    nice -n 19 python3 tools/coll-model/coll_model.py sweep w4 --jobs 4 --out w4.jsonl --max-states 6000000
    nice -n 19 python3 tools/coll-model/coll_model.py sweep w4t1 --jobs 2 --out w4t1.jsonl --max-states 4000000
    nice -n 19 python3 tools/coll-model/coll_model.py sweep porval --jobs 3 --out porval.jsonl --max-states 8000000
    python3 tools/coll-model/report_tables.py .        # the tables above
    python3 tools/coll-model/coll_model.py traces docs/experiments/coll-model-traces
    python3 tools/coll-model/coll_model.py run '{"world":4,"nslot":8,"lanes":4,"mode":"D1","calls":5,"kinds":"axzs"}'

Run times (control-node, nice 19, CPU seconds summed over the runs): world-2 sweep 657 s for 112 configurations
(largest 701,655 states); world-4 D1 sweep 13 s for 36 configurations (largest 29,171 states). The raw JSON lines of every
run are in docs/experiments/coll-model-traces/*.jsonl. The "T1 deep" rows for tcp and for rdma with one proxy thread come
from a run of an earlier revision of coll_model.py whose world-2 T1 code is the branch's (the later edits touched world 4,
the D1-only reduction rules and the sweep lists); rdma NSLOT 7 and 8 with one thread were not run deep. World-4 T1:
2 x 4M states, 13 min each.
