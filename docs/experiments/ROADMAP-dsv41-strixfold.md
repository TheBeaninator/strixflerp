# DeepSeek-V4.1-Flash 4-bit on StrixFold TP4: experiment roadmap and service plan

agent, 2026-11-24 late PT (card SF-TP4-ROADMAP d2f2a9e2c4). Owner: "get back to grinding improvements out of
deepseek 4-bit on strixflerp, and get strixflerp ready for service". This file replaces the 2026-11-17 roadmap (S0-S10,
written for the 2.9 bpw TP2 pair at f64b5ff); section 2 says what became of each S item. `ROADMAP.md` beside this file
is the Flash-Next roadmap. No hardware was used for this file; every number is copied from the report or card named
beside it.

## 1. Where it stands

The pod's DeepSeek is one 4.0 bpw instance, tensor parallel over four workers F D C A (rank order, 2D split of
prompt2d.zig), RDMA collectives (rdma_proxy), drafts on (MTP), 16 streams, a 22,000,000-position pool.

- Served build: fork `/data/pod/plans/strixfold/fork` branch serve-next-20261123c @ 72c0f083fca1, stage
  `/data/pod/plans/strixfold/deploy-stage/72c0f083fca1` (654 files in SHA256SUMS), conf
  `/data/pod/plans/strixfold/tp4-serve1-image-v7.conf`, stored reference `tp4-serve1-ref/tp4-serve1-image-v7.conf.72c0f083fca1.json`.
  The next launch takes `tp4-serve1-image-v9.conf`: v7 with only L2_PREFETCH=1 (TP4-SWITCHES 8854252e67, item 2), same
  build; v7 stays the rollback. v9 has no stored reference of its own yet: its replies matched v7's 8 of 8 in the window.
  It is TP4-SERVE1's build (003f679c28b1: graphs, swap12, prefill overlap, fast attention TP4_FILL=1, VGATHER) plus
  logprobs, route records, per-request acts, the linger guard, the drafter overflow fix and expert masking (all off or
  exact in this conf).
- It runs in windows. engines.conf has no TP4 entry and FreeFitty is not cut over (TP4-CUTOVER cd2c0705b3 in review,
  keeper timers off under owner's stop). The last windows were KCUT-FIELD round 2 (released 2026-11-25 03:15Z) and
  LIVE-KTRACE (started 04:55Z).

| what | number | source |
|---|---|---|
| one chat, drafts off | 36.94 code / 37.04 prose tok/s (27.07 ms a token); 36.4-37.4 at 8K-128K context | tp4-quickwins.md (arm gr), tp4-serve1-bench.md section 2 |
| one chat, drafts on | 85.51 code / 49.25 prose; VGATHER adds +1.8 % code (smoke 86.7) | tp4-quickwins.md, tp4-serve1.md addendum 11-21 |
| upstream bench, total tok/s at 1..8 chats | 63.58, 88.75, 102.18, 109.46, 113.88, 117.29, 117.31, 122.63 (2.9 pair: 48.55 ... 91.80) | tp4-serve1-bench.md section 1 (build 5ecaf207) |
| greedy concurrency, total | chat 174.7 / code 189.1 at 8 streams, 183.9 / 189.5 at 16; sampled T=1 about 120-126 | tp4-quickwins.md arm b16 |
| cold prompt | 1,333-1,487 tok/s from 8K to 128K (TTFT 93 s at 130K), about 370 tok/s a node | tp4-serve1-bench.md section 2 |
| round cost by rows | 1 row 27.09 ms, 2: 32.84, 3: 37.49, 4: 42.05, 5: 46.95, 6: 49.68; fit 28.99 + 4.31 r for 7-16 | tp4-quickwins.md (rounds_probe) |
| drafted round, phases (eager, 5.49 rows) | MoE 45 % (gate-up m_gu alone 26 %), attention 31 %, gathers 17 % (9.52 ms), Engram/mix/head 7 %; drafter pass 4.4-4.8 ms, about 10 % of a round | tp4-serve1-bench.md section 3, tp4-quickwins.md |
| collectives, 1-row round | 163 a round, 5.64 ms (sync timer, upper bound), about 20 % of the token | tp4-kernel-review.md section 1, KCUT-FIELD task d |
| floor | 2.76 GB a node a token: 12.0 ms at 230 GB/s, about 14.5 ms at the kernels' measured rates, against 27.07 ms | tp4-kernel-review.md section 1 |
| drafts | 78.8 % of drafted tokens accepted; code 4.55 tokens a round, prose 1.94 | tp4-kernel-review.md section 1 |
| kept state | tp4-cc 9 of 9 PASS on every serve1 build; factory slice C7 cached share 0.499 (pair sf-disk 0.056) | tp4-serve1.md |
| start-up, memory | health in 41-65 s; lowest MemAvailable 15,933 MiB (F) | tp4-serve1.md |

The decode token sits at about twice its byte floor: 5.6 ms of collectives, 2.7-5.4 ms of launch boundaries
(~1,500-1,800 launches a round) and kernels below their read rates. In multi-stream rounds the experts dominate: each
added row brings up to 6 new experts, and the experts already run at the read ceiling (WRENCH-M8 e9d0b002da: 194-204
GB/s at M = 1-16 against 199), so the levers there are fewer distinct experts or fewer bytes, not faster expert kernels.
Prompt reading is the weak side: agent turns read about 13 tokens for every token they write.

## 2. The 2026-11-17 items

| # | item (11-17) | status 2026-11-24 | cards |
|---|---|---|---|
| S0 | rebase on upstream main | done; the TP4 work branched from it | SF-REBASE 5fe6c77903 (done) |
| S1 | pair boot and yardstick | superseded by TP4: logits SHA equal to TP2 on 8 of 8 prompts, phase profile, upstream bench, boots in 41-65 s | TP4-FIRST-LIGHT 716f35d40a, TP4-QUICKWINS 034bf2c6cd, TP4-SERVE1 db9e4a4b4d (review). SF-BOOT 5d2db3fdd4 stays open as the owner card of the TP4 window hold (ff-5d2db3fdd4) |
| S2 | agent correctness (librarian loop) | answered: the loop reproduces cold on a fresh instance, a greedy fixed point at temperature 0 with thinking off, not a cache fault. Served conf has LOOP_POLICY=stop | LIB-LOOP 8ed514af56 (todo, comment #62058), TP4-LOOP-POLICY f64f606edd (done), FF-REPEAT-GUARD be05a599f9 (review); SF-LIBREPLAY 50fdeae15b (todo, its hypothesis refuted) |
| S3 | factory replay | done on the pair, refuted: 0.21 of Flash-Next a node, because replayed threads lost their kept states. Not rerun on TP4 | SF-FACTORY-REPLAY 9ebe1d7c24 (review); TP4 rerun = item 4 below |
| S4 | 8 streams and the shared pool | superseded: TP4 serves 16 streams on a 22M pool, 174.7-189.5 tok/s greedy at 8-16 (11-17 target was 100 at 8) | TP4-QUICKWINS 034bf2c6cd arm b16; SF-STREAMS8 76b42a4b8c (pair, todo) can close |
| S5 | prefill split, zip2, bigger units | done at TP4: PREFILL_SPLIT 256 + ZIP2 +55.6 % / +51.1 % prompt tok/s, pipelined prompt gathers, COLL_SLOT_KIB 8704 | TP4-PREFILL 19121aad1c, TP4-RDMA-PIPE 96431609cf (review); SF-SPLIT-AB d158833841 (pair, todo) superseded; next = item 3 |
| S6 | kept state (sf-disk) | done at world 4: a 15,466-token prompt cached 14,336 after a restart. Open: hit rate under live interleaving | TP4-DISK-STATES 602dfeb13b, SF-DISK-SAVE 848ca944f7, SF-DISK-EVICT 49bdf842f2 (review); SF-STATE d9ef7a305e (pair, todo); next = item 4 |
| S7 | same-bits decode kernels | partly done. Pair: SPEED-FUSE-1 -0.91 ms and FUSE-2 -0.32 ms at 1 row, no served gain. TP4: swap12 +2.6 %, VGATHER +1.8 % code drafts on, graphs no gain | SPEED-FUSE-1 b7269694f8 (review), SPEED-FUSE-2 31e3ab5061 (done), SHARED-FUSE 7f9542ef0d (todo, pair-era), TP4-VGATHER e96f0f5fa0; next = items 2, 5, 6 |
| S8 | fewer bytes (CODEBOOK-K1) | parked: about parity at 1 row, about 1.2x in drafted windows, nothing wired; the served pack is EXL3 4.0 | CODEBOOK-K1 96c4a607d2 (review, verdict PARK); item 9 |
| S9 | serving through strixflerp | changed shape: strixflerp rebuilds the served fork tree byte for byte instead of serving through its own front end on the pair. Recipe at 003f679c28b1 exists, kit manifest equal | FLERP-DSV41-TP4 bb68604fa9 (review); FLERP-SF-PAIRPASS b390d523b9, FLERP-SF-STREAMS 2f0b2dfc19, FLERP-SF-REPLY d1443f40e3 superseded (backlog); section 4 |
| S10 | 24 h soak | moved to TP4 on the strixflerp-built stage | SF-SOAK f1c521d530 superseded (backlog) by FLERP-TP4-SOAK 5aa2c82f1e |

## 3. Remaining improvement experiments, in order

Fidelity floor: reply bytes equal to the served build (tp4-serve1.sh gate against the stored reference), except where a
row says the bits change; those need owner's sign-off and a quality gate (FQ-1, coder-eval). Every hardware row takes F D
C A through one pod-quad hold: all four workers leave FreeFitty's strixite slots while it runs. Sizes are against the
served 27.07 ms drafts-off token unless a row says otherwise.

| # | experiment | bits | predicted | window cost | card |
|---|---|---|---|---|---|
| 1 | LIVE-KTRACE: rocprofv3 trace of the served engine under 8 real research streams; top 10 shapes against their microbench rate | none changed | ranks where real multi-stream time goes; reorders items 5-8 | ~70 min hold, running since 2026-11-25 04:55Z | 2007c996cf (doing) |
| 2 | TP4-SWITCHES: L2_PREFETCH 0/1 (KDEC section 4's MALL prefetch, wired on the TP4 round, off in the served conf), plus KCUT-FIELD round 2's DRAFT_PAIR (drafter pairGather as one row-pair exchange) and XINPLACE (one-row cat_rank straight into place) | exact | L2_PREFETCH +2 to +3.5 % drafts off (0.6-1.0 ms), +1 to +2 % drafts on (review-v2-e); DRAFT_PAIR 0.06-0.6 % (review-v2-f); XINPLACE 0.27-0.44 % (task d, not reviewed). DONE 2026-11-25 (window 05:35-05:58Z, base launches 38.55 and 38.56 tok/s drafts off): L2_PREFETCH=1 ADOPT, A B B A +3.93 % drafts off, +2.23 % drafts on, replies 8/8 byte-equal (hypothesis confirmed); DRAFT_PAIR REJECT (first A/B -0.05 % drafts off); XINPLACE REJECT (first A/B +0.66 %, under the +1 % gate). Next launch conf: tp4-serve1-image-v9.conf = v7 + L2_PREFETCH=1 (v7 is the rollback) | done: ~23 min hold, 7 launches | 8854252e67 (review); /data/pod/reports/2026-11/tp4-switches.md |
| 3 | TP4-PF-PROFILE: phase profile of a TP4 prompt chunk at 8K and 64K, and MGU=ra (each expert's gate-up weights read once for all its rows) stacked on the served prefill | exact (MGU replies equal 8/8 on c96f0318) | MGU alone gave +16.1 / +20.5 % prompt tok/s on c96f0318; predicted +8 to +15 % on top of PREFILL_SPLIT/ZIP2; refuted below +5 % | ~60 min hold, 3 launches | a4575d667d |
| 4 | Kept states at TP4: the factory replay at 2, 4 and 8 in flight with sf-disk, on the served conf | exact | prefilled tokens a request fall toward new tokens plus at most one 2,048-row chunk; cached share above C7's 0.499 | ~2 h hold | SF-KEEP-PAIR 3ddb72f4b1 (written for the pair; retarget comment posted), cause work in SF-KEEP-TRACE ef450588a1 |
| 5 | TP4-BALANCE: gate/up split 640/512 to 576/576 columns a rank pair (the round waits for pair 0, F and D) | exact (SHA gated) | 0.35-0.45 ms, about +1.3-1.6 % drafts off; also evens memory (F has the least) | regrouped rank files first (tf-dsv41-rank-regroup, staged through PLACE / pod-weights), then ~60 min hold | d1ab3c837f (review, ready for hardware) |
| 6 | Head per-quarter pick: gather each rank's (max, index) per row instead of fp32 vocabulary quarters | exact greedy; sampled draws change | ~0.1 ms at 1 row, ~0.5 ms a 6-row round, ~1.4 ms a 16-row round (tp4-kernel-review.md item 8) | 2-3 days code, then a window | not carded: card it if LIVE-KTRACE puts the head gather in its top 10 |
| 7 | 16-row drafting rounds: the fixed-order WMMA family for dense glinear and the head (KDEC section 9). WRENCH-M8 measured the 9-16-row two-part path at 0.51-0.80 of the 1-row rate | changes bits at every width | -2 to -4 ms of a ~98 ms 16-row round, 2-4 % at C4-C8 (TP2 estimate -4 ms, a TP4 node holds about half the dense bytes); 1 row +0 to +0.3 ms | owner's decision first; then kernel work, new reference SHAs, quality rerun, A/B at C1-C8 | WMMA16-DECIDE 6792f74358 (decide, in review for owner) |
| 8 | TP4-SPLIT2: K-split inside each box pair, 163 to 83 collectives a round | changes bits (a node-boundary split differs from today's sum in 75.5 % of cases, KCUT-FIELD task d) | 2.5-3.3 ms, about 40-41 tok/s drafts off (tp4-kernel-review.md item 5); off arm measured -0.2 % | built; needs owner's word on giving up TP2 bit parity, then KL + coder-eval | 19d9e36ea4 (review) |
| 9 | CODEBOOK-K1: ROCmFP4 4.25 bpw dense read path | new pack, new bits | about parity at 1 row, about 1.2x in drafted windows on its own bench; a TP4 pack needs four new rank files and a quality gate | parked | 96c4a607d2 (review, PARK) |

Order rule. Item 1 is running and decides how much of 5-8 is worth it at real multi-stream load. Item 2 is done: the
prefetch is adopted (+3.93 % drafts off), the two exchange switches rejected. Items 3 and 4 come before more decode work because
agent turns are prompt-heavy. Items 7 and 8 change bits and wait for owner. Stacked, the exact items 2 and 5 predict about
1.0-1.6 ms off the 27.07 ms token (38-39 tok/s drafts off); item 8 would add 2.5-3.3 ms.

Closed at TP4, do not re-run: HIP graphs (+0.04 tok/s, the round is GPU-bound), a TP4 ROUND_MS table (prose -3.6 %,
chat@4 -7.5 %), MGU for decode (-0.8 % drafts on), TP4-MTILE expert tilings (below the four-box resolution, 30f18bc279),
expert tile order (KDEC section 5, KCUT-FIELD task a), exact row charge in takeTurns (KCUT-FIELD task b, invalid), the
PF-13 panel as a decode lever (task c, invalid), task e's TP4 prefetch budgets (inert: Engine.world is 2 on every TP4
node), multi-row expert kernels at M = 8 (WRENCH-M8: at the read ceiling), RCCL against RDMA (5.64 against 10.92 ms a
round), TF_HC_FUSE on multi-row rounds (slower, KDEC section 9), more than 16 streams for throughput (flat 175-190 tok/s
from 8 to 16), DRAFT_PAIR and XINPLACE (TP4-SWITCHES 8854252e67: -0.05 % and +0.66 % drafts off).

## 4. Ready for service: strixflerp as the serving path for DSV4.1 TP4

What it means: the binary and kit the pod launches for DeepSeek come out of a strixflerp recipe build, byte for byte
the fork's served stage (as `recipes/dsv41-flash.strixfold-tp2.toml` does for the pair's f64b5ff kit); the serving conf
lives in strixflerp; the engine registry deploys that stage, with the fork stage one command back as rollback; and it
has served a day of real threads unattended.

What does not need building: the served tree already carries StrixFold's own HTTP server, so 16 streams with /health
keys, reply split and tool parsing, the prompt cache, shared_messages, sf-disk and the /cache routes all come with the
byte-for-byte rebuild. That is why the pair-era front-end cards (strixflerp's own front end over the TP2 ABI adapter)
are superseded for service: FLERP-SF-PAIRPASS b390d523b9, FLERP-SF-STREAMS 2f0b2dfc19 and FLERP-SF-REPLY d1443f40e3
(backlog, with a comment). FLERP-SF-STATE 152c723a5e and FLERP-SF-CACHE e11fabbf9e fall under the same reasoning and were
left untouched. If the ABI front end returns as research, it first needs a world-4 adapter.

| step | what | card | depends on |
|---|---|---|---|
| 1 | Recipe for the TP4 tree, kit manifest equal to the deployed one (4d7b62c5..., at 003f679c28b1) | FLERP-DSV41-TP4 bb68604fa9 (review: needs a reviewer and merge) | none |
| 2 | Recipe retargeted to the served 72c0f083fca1; a strixflerp tool writes deploy-stage/flerp-72c0f083fca1 over the same 654 file names; every differing file listed with its cause (target 0); configs/dsv41-flash.tp4-serve.conf = tp4-serve1-image-v7.conf with DEPLOY_DIR on the flerp stage | FLERP-TP4-SERVED fe4d694019 | bb68604fa9 |
| 3 | One boot of the flerp stage: gate 8 of 8 byte-equal to the stored 72c0f083fca1 reference, tp4-cc 9 checks | FLERP-TP4-GATE 7f6825bee0 (experiment, ~45 min hold) | fe4d694019 |
| 4 | FreeFitty sends CACHE_TRANSFER_KEY's bearer on /cache routes (moves, parking, probe affinity fail without it) | FF-SF-BEARER c328002352 | none; FreeFitty restart is owner's call |
| 5 | FreeFitty and the registry move to TP4 (its DONE still names F D A B; the nodes are F D C A) | TP4-CUTOVER cd2c0705b3 (review) | none |
| 6 | The TP4 keeper env and the dsv41 engines.conf entry deploy the flerp stage; one-command rollback to the fork stage | FLERP-TP4-REGISTER 5cf9661e12 | 7f6825bee0, cd2c0705b3 |
| 7 | 24 h of FreeFitty threads on the flerp stage: 0 rank deaths, every exit with its cause | FLERP-TP4-SOAK 5aa2c82f1e (experiment) | 5cf9661e12, c328002352 |

Rule once step 6 lands: a new fork serving build reaches the workers only through a recipe bump in strixflerp and the
step 3 gate, and an experiment that adopts a switch changes `configs/dsv41-flash.tp4-serve.conf`, not a copy under
/data/pod/plans.

Owner's calls on this path: going live with the cutover while the keeper timers are off under his stop, the FreeFitty
restart that ships the bearer, and WMMA16-DECIDE / TP4-SPLIT2 for the bit-changing levers.
