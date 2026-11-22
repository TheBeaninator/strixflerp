//! The exact 2D-TP4 split on the prompt path (DESIGN-zig-tp4.md sections 3-4; model.py Comm2D, wo_2d, moe_2d and
//! exl3/experts2d.py on four nodes): node g = 2 p + r is TP2 rank r of pair p. A node computes every output column it
//! holds whole, over its TP2 slice's K in TP2's order, and every exchange only moves bytes, so after an exchange a
//! node holds TP2 rank r's bytes: `quarters` assembles TP2's [2, n, N] gather of the two ranks' partials (which the
//! mixes then add in rank order, as on TP2), `catRank` the rank's whole wo_a output or expert intermediate from the
//! two pairs' parts. Prompt chunks exchange over NCCL across the four (the RDMA rings drop prompt-size bursts on the
//! switch: slots P6, P7b), each as one all-gather of every node's part then copies into place: the bytes Comm2D's
//! NCCL paths (gather4, _p2p) put there. With RDMA rings (Two.rings, ring2d.zig) the exchanges the lane sends over
//! them take them, written straight into place: quarters of fewer than 64 rows (decode windows; TF_DS_2D_PROMPT_NCCL
//! keeps prompt chunks on NCCL) and every cat_rank its column pair's ring takes; the same bytes either way.
//!
//! prompt.zig takes it through thin hooks behind Engine.two; a TP2 engine has none and runs as before. The decode
//! rounds' part is round2d.zig's, on this node state.
const std = @import("std");
const gpu = @import("gpu");
const Config = @import("config.zig").Config;
const plan = @import("plan.zig");
const prompt = @import("prompt.zig");
const weights = @import("weights.zig");
const exl3_experts = @import("exl3_experts.zig");
const exl3_experts2d = @import("exl3_experts2d.zig");
const ring2d = @import("ring2d.zig");
const coll = @import("coll.zig");
const zip2d = @import("zip2d.zig");
/// The RDMA rings' type: ring2d.zig needs verbs RDMA, which a HIP build has none of (it compiles out there, and every
/// exchange goes over the Coll).
pub const Rings = if (gpu.features.verbs_rdma) ring2d.Rings else opaque {};
const round_rows = @import("round.zig").max_rows;
const Link = @import("link.zig").Link;

const nccl = gpu.nccl;

/// The split and the links moved to fan.zig (a TP2 build without this file needs them).
pub const split = @import("fan.zig").split;
pub const Fan = @import("fan.zig").Fan;

fn len(r: ?[2]usize) usize {
    const x = r orelse return 0;
    return x[1] - x[0];
}

/// A 2D node's part of the prompt path and the decode rounds: the pairs' widths of each split output and the exchange
/// buffers.
pub const Two = struct {
    r: usize,
    p: usize,
    heads: usize, // the heads here (its pair's half of the rank's)
    uw: [2]usize, // wo_a's output a pair (its groups x o_lora): pair 0's, pair 1's
    ow: [2]usize, // wo_b's output columns a pair
    gu: [2]usize, // gate / up's columns a pair
    dw: [2]usize, // down's output columns a pair
    ew: [2]usize, // Engram wkv's output columns a pair (0 without an Engram layer)
    hw: [2]usize, // the head's vocabulary columns a pair (253 / 252 blocks of a TP2 half)
    parts: u64, // [4, rows, widest] where the all-gathers land
    pad: u64, // this node's part at the wider pair's width (when the pairs' widths differ)
    part_bytes: usize, // the largest part a node sends
    u_full: u64, // bf16 [cap, uw0 + uw1]: the rank's wo_a output, pair 0's groups then pair 1's
    xd_full: u64, // fp16 [cap * slots, gu0 + gu1]: the rank's expert intermediate, pair 0's blocks then pair 1's
    own: u64, // fp16 [round rows, uw[p]]: a round's half of wo_b's rotated input rows (wo_a's epilogue, round2d.woRot)
    rings: ?*const Rings, // the RDMA rings (TF_DS_2D_INTO, TF_DS_2D_GROUPS), when the caller opened them

    /// The widths from the split (both pairs'), and the buffers for chunks of up to `cap` rows (and rounds) from `a`.
    pub fn init(c: *const Config, w: *const weights.Weights, a: *prompt.Arena, s: plan.Split, cap: usize) !Two {
        const pp = s.pair orelse return error.NotA2DSplit;
        const s0 = s.atPair(0);
        const s1 = s.atPair(1);
        const x0 = s0.expertParts(c.*);
        const x1 = s1.expertParts(c.*);
        var t: Two = undefined;
        t.r = s.rank;
        t.p = pp;
        t.heads = s.headSpan(c.*)[1];
        t.uw = .{ s0.woAGroups(c.*)[1] * c.o_lora, s1.woAGroups(c.*)[1] * c.o_lora };
        t.ow = .{ len(s0.woBCols(c.*)), len(s1.woBCols(c.*)) };
        t.gu = .{ len(x0.gu), len(x1.gu) };
        t.dw = .{ len(x0.dcols), len(x1.dcols) };
        var engram = false;
        for (w.layers) |lay| engram = engram or lay.engram_wkv != null;
        t.ew = if (engram) .{ len(s0.engramCols(c.*)), len(s1.engramCols(c.*)) } else .{ 0, 0 };
        t.hw = .{ len(s0.headCols(c.*)), len(s1.headCols(c.*)) };
        // the weights loaded here are this split's
        const l0 = w.layers[0];
        if (l0.wq_b.n != t.heads * c.head_dim or l0.groups * c.o_lora != t.uw[pp] or l0.wo_b.n != t.ow[pp] or
            l0.experts.width != t.gu[pp] or l0.experts.down_n != t.dw[pp]) return error.WeightsNotThisSplit;
        // prompt.gather assembles either sublayer's partials: both splits must give the same widths
        if (t.ow[0] != t.dw[0] or t.ow[1] != t.dw[1]) return error.UnequalPartialSplits;
        const slots = c.top_k + 1;
        const biggest = @max(@max(@max(cap * @max(t.ow[0], t.ow[1]) * 4, cap * @max(t.ew[0], t.ew[1]) * 4), @max(cap * @max(t.uw[0], t.uw[1]) * 2, cap * slots * @max(t.gu[0], t.gu[1]) * 2)), @max(t.hw[0], t.hw[1]) * 4);
        t.part_bytes = biggest;
        t.parts = try a.take(4 * biggest);
        t.pad = try a.take(biggest);
        t.u_full = try a.take(cap * (t.uw[0] + t.uw[1]) * 2);
        t.xd_full = try a.take(@max(cap, round_rows) * slots * (t.gu[0] + t.gu[1]) * 2);
        t.own = try a.take(round_rows * t.uw[pp] * 2);
        t.rings = null;
        return t;
    }

    /// Every node's part over NCCL into `parts` [4, n, wmax] (bytes), this node's `src` [n, w[p]] padded to wmax.
    fn allParts(t: *const Two, e: *const prompt.Engine, src: u64, n: usize, w: [2]usize, esize: usize) !usize {
        const wmax = @max(w[0], w[1]);
        if (n * wmax * esize > t.part_bytes) return error.PartTooBig;
        var send = src;
        if (w[t.p] != wmax) {
            try e.ops.copyRows(e.s, src, w[t.p] * esize, t.pad, wmax * esize, w[t.p] * esize, n);
            send = t.pad;
        }
        try e.comm.allGather(.at(.two_parts, 0), send, t.parts, n * wmax * esize, .u8, e.s);
        return wmax;
    }

    /// Whether a ring took the exchange: false when there is none or the part does not fit it (nothing sent then).
    fn ringed(r: anyerror!void) !bool {
        r catch |err| switch (err) {
            error.TooLarge, error.NotFloat4 => return false,
            else => return err,
        };
        return true;
    }

    /// quarters (Comm2D.quarters): `src` [n, w[p]] (this node's columns of its TP2 rank's partial) -> `dst`
    /// [2, n, w0 + w1]: rank r's partial, pair 0's columns then pair 1's, for every r.
    pub fn quarters(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, w: [2]usize, esize: usize) !void {
        if (gpu.features.verbs_rdma) if (t.rings) |rs| if (n < 64 and try ringed(rs.quarters(e.s, src, dst, n, w, esize))) return;
        // one row of equal widths: dst [2 r, 2 p, w] is the parts [4 g, w] with ranks 1 and 2 traded (g = 2 p + r), so a
        // swap12 gather writes every part straight into its place and the four copies below are not needed
        if (n == 1 and w[0] == w[1] and e.comm.swap12 and w[0] * esize <= t.part_bytes)
            return e.comm.allGather(.{ .phase = .two_parts, .layer = 0, .swap12 = true }, src, dst, w[0] * esize, .u8, e.s);
        _ = try t.allParts(e, src, n, w, esize);
        return t.assembleQuarters(e, dst, n, w, esize);
    }

    /// Parts are in. TF_DS_VGATHER (default off), equal widths, more than one row: two copy_rows16, one per pair.
    /// Ranks g = 2p and g = 2p + 1 are adjacent in the rank-order parts (this gather is not swap12, so Word.place
    /// is the rank; swap12's permutation is the one-row gather above and never reaches here). Source row
    /// r·n + row is rank g = 2p + r, and it lands on destination row r·n + row at column block p, which is
    /// quarterAt. copyRows uses the byte strided kernel unless every address is a multiple of 16 and rows <= 65535;
    /// this launch's rows are 2·n, so a wider gather or a width that is not a multiple of 16 bytes stays on the
    /// four copies. No other cap: copy_rows16 is one 16-byte word a thread, with no per-thread loop.
    fn assembleQuarters(t: *const Two, e: *const prompt.Engine, dst: u64, n: usize, w: [2]usize, esize: usize) !void {
        const rb = w[0] * esize;
        if (e.comm.vgather and n > 1 and w[0] == w[1] and t.copy16Quarters(dst, n, rb)) return t.placeQuarters(e, dst, n, rb);
        try t.scatterQuarters(e, dst, n, w, esize);
    }

    /// True when both placeQuarters launches take copy_rows16. rows = 2·n must fit grid.y (copyRows' 65535).
    fn copy16Quarters(t: *const Two, dst: u64, n: usize, rb: usize) bool {
        if (rb == 0 or rb % 16 != 0 or n > 65535 / 2) return false;
        return (t.parts | dst) % 16 == 0;
    }

    /// allParts' rank-order parts [4, n, W] into dst [2, n, 2W], two copy_rows16. Pair p: src at 2p·n·rb with
    /// stride rb, dst at column block p with stride 2·rb, 2·n rows of rb bytes.
    fn placeQuarters(t: *const Two, e: *const prompt.Engine, dst: u64, n: usize, rb: usize) !void {
        for (0..2) |p| {
            try e.ops.copyRows(e.s, t.parts + 2 * p * n * rb, rb, dst + p * rb, 2 * rb, rb, 2 * n);
        }
    }

    /// quarters' copies, after the parts have landed. The byte offsets are zip2d.quarterAt (the host test's grid).
    fn scatterQuarters(t: *const Two, e: *const prompt.Engine, dst: u64, n: usize, w: [2]usize, esize: usize) !void {
        for (0..4) |g| {
            const at = zip2d.quarterAt(g, n, w, esize);
            try e.ops.copyRows(e.s, t.parts + at.src, at.src_stride, dst + at.dst, at.dst_stride, at.nbytes, n);
        }
    }

    /// A quarters all-gather begun so other work can run before it ends (fillSplit2d). The same send, the same
    /// parts buffer and the same site as allParts. n < 64 leaves the gather to quarters() at quartersEnd (the CUDA
    /// ring, or allGather): a prompt half is larger than that. One begun gather at a time; the caller ends it before
    /// any other Coll call, and does not write `src` or touch `parts` in between.
    pub const Begun = struct {
        pending: ?coll.Coll.Pending = null,
        src: u64,
        dst: u64,
        n: usize,
        w: [2]usize,
        esize: usize,
    };

    pub fn quartersBegin(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, w: [2]usize, esize: usize) !Begun {
        var b: Begun = .{ .src = src, .dst = dst, .n = n, .w = w, .esize = esize };
        if (n < 64) return b;
        const wmax = @max(w[0], w[1]);
        if (n * wmax * esize > t.part_bytes) return error.PartTooBig;
        var send = src;
        if (w[t.p] != wmax) {
            try e.ops.copyRows(e.s, src, w[t.p] * esize, t.pad, wmax * esize, w[t.p] * esize, n);
            send = t.pad;
        }
        b.pending = try e.comm.gatherBegin(.at(.two_parts, 0), send, t.parts, n * wmax * esize, .u8, e.s);
        return b;
    }

    pub fn quartersEnd(t: *const Two, e: *const prompt.Engine, b: Begun) !void {
        if (b.pending) |p| {
            try e.comm.gatherEnd(p, e.s);
            return t.assembleQuarters(e, b.dst, b.n, b.w, b.esize);
        }
        return t.quarters(e, b.src, b.dst, b.n, b.w, b.esize);
    }

    /// cat_rank (Comm2D.cat_rank): `src` [n, w[p]] -> `dst` [n, w0 + w1]: this node's TP2 rank's whole width, pair 0's
    /// part then pair 1's (from its column partner, the same rank of the other pair). Only the partner's part is used,
    /// so it is one pair exchange with g ^ 2 (uneven widths as they are), not a four-rank all-gather: a third of the
    /// egress and no all-rank barrier (TP4-RDMA-PIPE, kernel review item 3); the same bytes land in dst.
    pub fn catRank(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, w: [2]usize, esize: usize) !void {
        if (gpu.features.verbs_rdma) if (t.rings) |rs| if (try ringed(rs.catRank(e.s, src, dst, n, w, esize))) return;
        const q = 1 - t.p;
        if (n * w[q] * esize > 4 * t.part_bytes) return error.PartTooBig;
        const g: u32 = @intCast(t.r + 2 * t.p);
        try e.comm.exchange(.at(.two_pair, 2), src, n * w[t.p] * esize, t.parts, n * w[q] * esize, .u8, g ^ 2, e.s);
        const row = (w[0] + w[1]) * esize;
        try e.ops.copyRows(e.s, src, w[t.p] * esize, dst + t.p * w[0] * esize, row, w[t.p] * esize, n);
        try e.ops.copyRows(e.s, t.parts, w[q] * esize, dst + q * w[0] * esize, row, w[q] * esize, n);
    }

    /// Comm2D.gather (TP2 inside a pair: the DSpark drafter's sublayer partials): `src` [n, width] -> `dst` [2, n, width],
    /// this pair's two ranks in TP2 rank order: over the row pair's ring, else the four-rank all-gather's pair half
    /// (Comm2D.gather4(x)[2p:2p + 2]).
    pub fn pairGather(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, width: usize, esize: usize) !void {
        if (gpu.features.verbs_rdma) if (t.rings) |rs| if (try ringed(rs.pairGather(e.s, src, dst, n, width, esize))) return;
        const wmax = try t.allParts(e, src, n, .{ width, width }, esize);
        try e.ops.copyRows(e.s, t.parts + 2 * t.p * n * wmax * esize, wmax * esize, dst, width * esize, width * esize, 2 * n);
    }

    /// wo_2d after wo_a: the rank's four groups (this node's two in ch.u [n, uw[p]], the partner's two), then wo_b
    /// for this pair's output columns: ch.pa [n, ow[p]] fp32, the matching columns of TP2's partial.
    pub fn woB(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer) !void {
        try t.catRank(e, ch.u, t.u_full, ch.n, t.uw, 2);
        try prompt.mm(e, ch, lay.wo_b, t.u_full, t.uw[0] + t.uw[1], ch.pa, .fp32, lay.wo_b.n);
    }

    /// moe_2d's experts after the routing: this node's gate / up blocks, the rank's whole intermediate assembled, down
    /// for this node's output columns: ch.pm [n, dw[p]] fp32. A chunk of fewer than EXACT_ROWS rows takes the decode
    /// window's fused path and scratch (experts2d.fused_ok), as the rounds do.
    pub fn experts(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer) !void {
        try t.expertsUp(e, ch, lay);
        try t.expertsDown(e, ch, lay);
    }

    /// Gate / up only (no collective). experts() is this then expertsDown, the same launches either way.
    pub fn expertsUp(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer) !void {
        const c = e.c;
        const n = ch.n;
        if (n < exl3_experts.exact_rows) return exl3_experts2d.decodeGateUp(e.ex, e.s, lay.experts, ch.xsd, ch.x, c.hidden, ch.pick, ch.wts, ch.pm, n, c.swiglu_limit);
        // the AMD kit's prompt experts where TP2's rank takes them (exl3_experts.prompt): the same bits per column
        if (e.ex.pf) |*pf| if (exl3_experts2d.pfPlans(lay.experts, ch.xs, ch.x, c.hidden, ch.pick, ch.wts, t.xd_full, ch.pm, n, c.swiglu_limit)) |pl|
            return exl3_experts.pfGateUp(pf, e.s, pl.gu);
        try exl3_experts2d.gateUp(e.ex, e.s, lay.experts, ch.xs, ch.x, c.hidden, ch.pick, n, c.swiglu_limit);
    }

    /// catRank of the intermediate, then down. The same branch expertsUp took (pfPlans reads the weights, not the device).
    pub fn expertsDown(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer) !void {
        const c = e.c;
        const n = ch.n;
        if (n < exl3_experts.exact_rows) {
            try t.catRank(e, ch.xsd.xd, t.xd_full, n * ch.xsd.slots, t.gu, 2);
            var sc = ch.xsd;
            sc.reap_y = 0;
            try exl3_experts2d.decodeDown(e.ex, e.s, lay.experts, sc, t.xd_full, ch.pick, ch.wts, ch.pm, n);
            if (e.reap_note != null) try prompt.reapNoteCapture(e, lay.idx, n, sc.slots, sc.y, ch.reap_bank, 2);
            return;
        }
        if (e.ex.pf) |*pf| if (exl3_experts2d.pfPlans(lay.experts, ch.xs, ch.x, c.hidden, ch.pick, ch.wts, t.xd_full, ch.pm, n, c.swiglu_limit)) |pl| {
            try t.catRank(e, ch.xs.xd, t.xd_full, n * ch.xs.slots, t.gu, 2);
            try exl3_experts.pfDown(pf, e.s, pl.down);
            return prompt.reapNoteCapture(e, lay.idx, n, ch.xs.slots, ch.xs.z, ch.reap_bank, 2);
        };
        try t.catRank(e, ch.xs.xd, t.xd_full, n * ch.xs.slots, t.gu, 2);
        try exl3_experts2d.down(e.ex, e.s, lay.experts, ch.xs, t.xd_full, ch.pick, ch.wts, ch.pm, n);
        try prompt.reapNoteCapture(e, lay.idx, n, ch.xs.slots, 0, ch.reap_bank, 0);
    }

    /// Engram's projection summed over the TP2 ranks and cast, kv [n, ew0 + ew1] bf16 = bf16(g0 + g1) (model.py esum
    /// then .to(bf16)), from this node's fp32 columns `src` [n, ew[p]] with less traffic than the quarters: the row
    /// partner's columns over NCCL send / recv, one fp32 add (two operands: the same bits in either order) cast to
    /// bf16, then the column partner's bf16 sums the same way. `tmp` holds n * (ew[p] * 6 + ew[1 - p] * 2) bytes.
    pub fn engramSum(t: *const Two, e: *const prompt.Engine, src: u64, tmp: u64, kv: u64, n: usize) !void {
        const w = t.ew;
        const mine = n * w[t.p];
        const other = tmp;
        const half = tmp + mine * 4;
        const their = half + mine * 2;
        const g: u32 = @intCast(t.r + 2 * t.p);
        try e.comm.exchange(.at(.two_pair, 0), src, mine, other, mine, .f32, g ^ 1, e.s);
        try e.ops.add2Bf16(e.s, src, other, half, mine);
        try e.comm.exchange(.at(.two_pair, 1), half, mine, their, n * w[1 - t.p], .bf16, g ^ 2, e.s);
        const row = (w[0] + w[1]) * 2;
        for (0..2) |pg| {
            const from = if (pg == t.p) half else their;
            try e.ops.copyRows(e.s, from, w[pg] * 2, kv + pg * w[0] * 2, row, w[pg] * 2, n);
        }
    }

    /// Engram's projection summed over the ranks from its column quarters (model.py esum): the gathered width.
    pub fn engramWidth(t: *const Two) usize {
        return t.ew[0] + t.ew[1];
    }
};

test "a node's split: TP over two, the 2D split over four" {
    try std.testing.expectEqual(plan.Split{ .rank = 1, .world = 2 }, try split(1, 2));
    try std.testing.expectEqual(plan.Split{ .rank = 0, .world = 2, .pair = 1 }, try split(2, 4));
    try std.testing.expectEqual(plan.Split{ .rank = 1, .world = 2, .pair = 1 }, try split(3, 4));
    try std.testing.expectError(error.UnsupportedWorld, split(0, 3));
}
