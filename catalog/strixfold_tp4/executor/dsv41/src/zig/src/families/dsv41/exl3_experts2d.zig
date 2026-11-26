//! A 2D node's routed experts (exl3/experts2d.py): this node's gate / up blocks of every picked slot with their
//! epilogue, then, once the caller's column exchange has assembled the TP2 rank's whole intermediate, down over it for
//! this node's output columns and the combine. On a prompt chunk (gateup / down) the same eight launches as
//! exl3_experts.prompt, in two calls; on a decode window (gateup_fused / down_fused) TP2's three fused launches
//! (exl3_experts.decodeLaunches), decode_prep and gate / up before the exchange and down after it. Either way the
//! kernel settings are pinned from the TP2 half's shapes (Plan.of: the half's I, not this node's narrower width) and
//! only the launched widths narrowed, so every output column has the bits of the TP2 rank's partial.
const std = @import("std");
const gpu = @import("gpu");
const weights = @import("weights.zig");
const X = @import("exl3_experts.zig");
const reap = @import("reap.zig");

fn launch(f: gpu.Function, s: gpu.Stream, grid: [3]usize, block: usize, args: *gpu.Args) !void {
    try gpu.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block) } }, s, args);
}

fn i32a(a: *gpu.Args, v: usize) void {
    a.add(@as(c_int, @intCast(v)));
}

/// The settings of a 2D node's prompt launches: TP2's for the half (D, I), checked as exl3_experts.prompt checks
/// them, and the widths launched here.
pub const Plan = struct {
    d: usize, // the model width (gate / up K)
    i: usize, // the TP2 half's intermediate (down K)
    ig: usize, // gate / up's output columns here
    dn: usize, // down's output columns here
    cgu: [4]usize,
    cd: [4]usize,

    pub fn of(ex: weights.Experts) !Plan {
        const d: usize = ex.dims;
        const i: usize = ex.down_k;
        const ig: usize = ex.width;
        const dn: usize = ex.down_n;
        if (i % 128 != 0 or d % 128 != 0 or ig % 128 != 0 or dn % 128 != 0 or ig > i or dn > d) return error.UnsupportedShape;
        const cgu = try X.config(d, i, true);
        const cd = try X.config(i, d, false);
        // the served fast path as on TP2: mma for gate / up (K chains a multiple of MMA_KB), grouped_rows for down
        const chain_gu = (d / 16) / (cgu[1] * cgu[2]);
        if ((d / 16) % (cgu[1] * cgu[2]) != 0 or chain_gu % 4 != 0) return error.UnsupportedShape;
        const chain_d = (i / 16) / (cd[1] * cd[2]);
        if ((i / 16) % (cd[1] * cd[2]) == 0 and chain_d % 4 == 0) return error.UnsupportedShape; // down would take mma
        if (cd[2] != 1) return error.UnsupportedShape;
        if (ig % X.mma_cols != 0) return error.UnsupportedShape;
        return .{ .d = d, .i = i, .ig = ig, .dn = dn, .cgu = cgu, .cd = cd };
    }
};

/// A 2D node's prompt experts on the AMD kit's w-pf kernels (exl3_experts.pfPlan): TP2's rank runs those for every
/// prompt chunk its table takes them for, and the mma / rows launches below sum K in another order, so a node takes
/// them too or no column has the rank's bits (TP4-FIRST-LIGHT: prompts of 64 rows and more differed).
/// gu: count, place, rot and gate / up for this node's ig columns (K D) into sc.xd [R * slots, ig], sorted order; then
/// the caller's cat_rank of xd into xd_full; down: down over xd_full [R * slots, I] for this node's dn columns and the
/// combine into out [R, dn]. Every gu / down program owns whole 128-column strips (128-aligned here) and reads only
/// their trellis tiles and scale rows (stacked at this node's widths, weights.zig), over all of K in the rank's order,
/// and the grouping (count, place) is a function of pick alone: each element has the rank's bits. Null when the
/// rank's own plan would be (TP2 then takes the mma launches, and so does the node).
pub const PfPlans = struct { gu: X.PfPlan, down: X.PfPlan };

pub fn pfPlans(ex: weights.Experts, sc: X.Scratch, x: u64, x_stride: usize, pick: u64, wts: u64, xd_full: u64, out: u64, r: usize, limit: f32) ?PfPlans {
    if (r < X.exact_rows or r > sc.rows) return null;
    _ = X.pfPlanOf(ex, sc, x, x_stride, pick, wts, out, r, limit, ex.dims, ex.down_k) orelse return null; // the rank's
    const gu = X.pfPlanOf(ex, sc, x, x_stride, pick, wts, out, r, limit, ex.dims, ex.width) orelse return null;
    var dn = X.pfPlanOf(ex, sc, x, x_stride, pick, wts, out, r, limit, ex.down_n, ex.down_k) orelse return null;
    dn.args.xd = xd_full;
    return .{ .gu = gu, .down = dn };
}

test "a 2D node's prompt experts on the AMD kit: the rank's w-pf launches at this node's widths" {
    // node 3 (pair 1, TP2 rank 1): gate / up columns 1792-2304 of the half (512), down columns 2560-5120
    var ex: weights.Experts = .{ .trellis = 0, .gate_ptr = 1, .up_ptr = 2, .down_ptr = 3, .gate_k2 = 11, .up_k2 = 12, .down_k2 = 13, .suh_g = 4, .suh_u = 5, .svh_g = 6, .svh_u = 7, .suh_d = 8, .svh_d = 9, .count = 385, .dims = 5120, .width = 512, .down_k = 1152, .down_n = 2560, .k2_gu = .{ 6, 8 }, .k2_d = .{ 6, 10 } };
    var sc: X.Scratch = undefined;
    sc.slots = 7;
    sc.rows = 4096;
    sc.xd = 0x500;
    try std.testing.expect(pfPlans(ex, sc, 0x100, 5120, 0x200, 0x300, 0x600, 0x400, 2048, 10) == null); // no masks: mma, as on TP2
    ex.k2_masks = .{ 1 << 6 | 1 << 8, 1 << 6 | 1 << 10 };
    try std.testing.expect(pfPlans(ex, sc, 0x100, 5120, 0x200, 0x300, 0x600, 0x400, 63, 10) == null); // a decode window
    const pl = pfPlans(ex, sc, 0x100, 5120, 0x200, 0x300, 0x600, 0x400, 2048, 10).?;
    // the grouping is the rank's: the same jobs as its own plan
    var rank = ex;
    rank.width = 1152;
    rank.down_n = 5120;
    const rp = X.pfPlan(rank, sc, 0x100, 5120, 0x200, 0x300, 0x400, 2048, 10).?;
    for ([_]X.PfPlan{ pl.gu, pl.down }) |p| {
        try std.testing.expectEqual(rp.nm, p.nm);
        try std.testing.expectEqual(rp.args.maxjobs, p.args.maxjobs);
        try std.testing.expectEqualSlices(u32, rp.k2_gu[0..rp.n_gu], p.k2_gu[0..p.n_gu]);
        try std.testing.expectEqualSlices(u32, rp.k2_d[0..rp.n_d], p.k2_d[0..p.n_d]);
    }
    try std.testing.expectEqual(@as(u32, 5120), pl.gu.args.d);
    try std.testing.expectEqual(@as(u32, 512), pl.gu.args.i);
    try std.testing.expectEqual(@as(u64, 0x500), pl.gu.args.xd);
    try std.testing.expectEqual(@as(u32, 2560), pl.down.args.d);
    try std.testing.expectEqual(@as(u32, 1152), pl.down.args.i);
    try std.testing.expectEqual(@as(u64, 0x600), pl.down.args.xd);
    try std.testing.expectEqual(@as(u32, 2560), pl.down.args.ldo);
    try std.testing.expectEqual(@as(usize, 2048 * 2560 / 4 / 256), pl.down.grids[3]);
    // a half the kernels cannot take (down's columns not whole 256-column pairs): null, the mma launches
    ex.down_n = 2432;
    try std.testing.expect(pfPlans(ex, sc, 0x100, 5120, 0x200, 0x300, 0x600, 0x400, 2048, 10) == null);
}

/// experts2d.gateup on a prompt chunk (R >= 64): the grouping (TP2's, by pick only), the rotated rows, gate and up for
/// this node's columns, the epilogue (TP2's values on them) into sc.xd [R * slots, ig] fp16. x bf16 [R, D] (row stride
/// x_stride elements), pick int32 [R, slots]; sc sized for this node's width.
pub fn gateUp(k: *const X.Kernels, s: gpu.Stream, ex: weights.Experts, sc: X.Scratch, x: u64, x_stride: usize, pick: u64, r: usize, limit: f32) !void {
    const pl = try Plan.of(ex);
    const e: usize = ex.count;
    const slots = sc.slots;
    if (r < X.exact_rows or r > sc.rows) return error.NotAPromptChunk;
    if (slots > 32) return error.UnsupportedShape;
    const p = r * slots;
    const maxu = @min(p, e);
    const maxm = X.memberStride(r);
    const n_gu = X.listLen(p, maxu, X.mma_rows_gu);
    const n_d = X.listLen(p, maxu, X.rows_rows_d);
    const mma = k.mma[X.range(ex.k2_gu[0], ex.k2_gu[1])] orelse return error.MissingKernel;

    var a: gpu.Args = .{};
    a.add(pick);
    a.add(sc.counts);
    i32a(&a, p);
    try launch(k.group_count, s, .{ e, 1, 1 }, X.place_threads, &a);

    a = .{};
    for ([_]u64{ pick, sc.counts, sc.ids, sc.count, sc.members }) |v| a.add(v);
    for ([_]usize{ p, slots, e, maxm }) |v| i32a(&a, v);
    try launch(k.group_place, s, .{ e, 1, 1 }, X.place_threads, &a);

    // both work lists here: down (after the exchange) uses the same grouping
    a = .{};
    for ([_]u64{ sc.counts, sc.ids, sc.count }) |v| a.add(v);
    i32a(&a, maxu);
    a.add(sc.work_gu);
    i32a(&a, X.mma_rows_gu);
    i32a(&a, n_gu);
    a.add(sc.work_d);
    i32a(&a, X.rows_rows_d);
    i32a(&a, n_d);
    a.add(@as(c_int, -1)); // skip_e: no shared-dense GEMM (none on 2D either)
    try launch(k.work_list, s, .{ 1, 1, 1 }, X.list_threads, &a);

    a = .{};
    a.add(x);
    i32a(&a, x_stride);
    for ([_]u64{ pick, ex.suh_g, ex.suh_u, sc.xg, sc.xu }) |v| a.add(v);
    for ([_]usize{ pl.d, slots, e }) |v| i32a(&a, v);
    try launch(k.rot_in, s, .{ p, pl.d / 128, 2 }, 32, &a);

    // gate and up for this node's columns: grid (1, ig / 64, pairs x 2 matrices), TP2's K splits and warps
    a = .{};
    for ([_]u64{ sc.xg, sc.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, sc.ids, sc.count, sc.members, sc.z }) |v| a.add(v);
    for ([_]usize{ pl.d, pl.ig, p, pl.cgu[2], pl.cgu[1], maxm, slots, n_gu }) |v| i32a(&a, v);
    a.add(sc.work_gu);
    if (n_gu > 0) try launch(mma, s, .{ 1, pl.ig / X.mma_cols, n_gu * 2 }, 256, &a);

    a = .{};
    for ([_]u64{ sc.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, sc.xd }) |v| a.add(v);
    for ([_]usize{ p, pl.ig, 1, e }) |v| i32a(&a, v);
    a.add(limit);
    a.add(@as(c_int, X.act_f32));
    try launch(k.gateup_epilogue, s, .{ p, pl.ig / 128, 1 }, 32, &a);
}

/// experts2d.down on a prompt chunk: down over the TP2 rank's whole intermediate xd_full [R * slots, I] fp16 for this
/// node's output columns, then the served combine (slots in order) into out fp32 [R, dn]. Uses the grouping and work
/// lists gateUp made in sc.
pub fn down(k: *const X.Kernels, s: gpu.Stream, ex: weights.Experts, sc: X.Scratch, xd_full: u64, pick: u64, wts: u64, out: u64, r: usize) !void {
    const pl = try Plan.of(ex);
    const e: usize = ex.count;
    const slots = sc.slots;
    if (r < X.exact_rows or r > sc.rows) return error.NotAPromptChunk;
    const p = r * slots;
    const maxu = @min(p, e);
    const maxm = X.memberStride(r);
    const n_d = X.listLen(p, maxu, X.rows_rows_d);
    const dfn = k.rows[X.range(ex.k2_d[0], ex.k2_d[1])] orelse return error.MissingKernel;

    // down: grid (1, dn / 128, pairs), 4 warps, two 16-row member tiles a program, one split (TP2's settings)
    var a: gpu.Args = .{};
    for ([_]u64{ xd_full, xd_full, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, sc.ids, sc.count, sc.members, sc.z }) |v| a.add(v);
    for ([_]usize{ pl.i, pl.dn, p, 1, maxm, slots, n_d }) |v| i32a(&a, v);
    a.add(sc.work_d);
    if (n_d > 0) try launch(dfn, s, .{ 1, pl.dn / 128, n_d }, 4 * 32, &a);

    a = .{};
    for ([_]u64{ sc.z, pick, ex.svh_d, sc.no_y, wts, 0, out }) |v| a.add(v);
    for ([_]usize{ p, pl.dn, 1, e, slots, 0 }) |v| i32a(&a, v);
    try launch(k.down_combine, s, .{ r, pl.dn / 128, 1 }, 32 * slots, &a);
}

/// A decode window's shape on a 2D node, checked as experts2d.fused_ok checks it: TP2's fused settings for the half
/// (GLM's tiles for gate / up, one split for down), whole 128-column blocks here, at most 32 slots.
const Window = struct {
    d: usize, // gate / up's K
    i: usize, // the TP2 half's intermediate (down's K)
    ig: usize, // gate / up's output columns here
    dn: usize, // down's output columns here
    sk: usize, // gate / up's K splits (the half's)
    e: usize,
    slots: usize,
    p: usize,
    maxu: usize, // the window's expert places
    mt: usize, // a place's 16-row member tiles

    fn of(ex: weights.Experts, sc: X.DecodeScratch, r: usize) !Window {
        const d: usize = ex.dims;
        const i: usize = ex.down_k;
        const ig: usize = ex.width;
        const dn: usize = ex.down_n;
        if (r == 0 or r >= X.exact_rows or r > sc.rows) return error.NotADecodeWindow;
        const cgu = try X.config(d, i, true);
        const cd = try X.config(i, d, false);
        if (cgu[0] != 8 or cgu[1] != 4 or cd[0] != 8 or cd[1] != 4 or cd[2] != 1) return error.UnsupportedShape;
        if (i % 128 != 0 or d % 128 != 0 or ig % 128 != 0 or dn % 128 != 0 or ig > i or dn > d or sc.slots > 32) return error.UnsupportedShape;
        const p = r * sc.slots;
        return .{ .d = d, .i = i, .ig = ig, .dn = dn, .sk = cgu[2], .e = ex.count, .slots = sc.slots, .p = p, .maxu = @min(p, ex.count), .mt = (r + 15) / 16 };
    }
};

/// Bytes of a 2D node's decode scratch, in DecodeScratch's field order (xg .. members): TP2's buffers at this node's
/// widths (experts2d.Scratch2D's for the fused launches): xd [P, ig] fp16, z fp32 [max(2 * splits * ig, dn) * P],
/// y [P, dn], the counters a (place, 128 columns of ig) and a (row, 128 columns of dn); epoch and the ready flags stay
/// unused (no readiness across the exchange) but are sized as TP2's.
pub fn decodeSizes(rows: usize, slots: usize, ex: weights.Experts) ![13]usize {
    const d: usize = ex.dims;
    const i: usize = ex.down_k;
    const ig: usize = ex.width;
    const dn: usize = ex.down_n;
    const p = rows * slots;
    const maxu = @min(p, @as(usize, ex.count));
    const one: usize = 1;
    const zrow = @max(2 * (try X.config(d, i, true))[2] * ig, (try X.config(i, d, false))[2] * dn);
    return .{ 2 * p * d, 2 * p * d, 2 * p * ig, 4 * zrow * p, 4 * p * dn, 4 * maxu * @max(one, ig / 128), 4 * rows * @max(one, dn / 128), 4, 4 * @max(one, maxu), 4 * @max(one, maxu), 4 * maxu, 4, 4 * maxu * ((rows + 127) / 128 * 128) };
}

/// gateup_fused's two launches on a decode window (R < 64): decode_prep (the grouping; gate / up's rotated input rows,
/// all D columns; `out`'s rows that route no slot, this node's dn columns) and gate / up with its epilogue for this
/// node's ig columns into sc.xd [R * slots, ig] fp16. TP2's launches (exl3_experts.decodeLaunches) with the widths
/// narrowed and no readiness: prep keeps no epoch and the epilogue publishes no ready flag, since the column exchange
/// stands between gate / up and down. x bf16 [R, D] (row stride x_stride elements), pick int32 and wts fp32
/// [R, slots], out fp32 [R, dn].
pub fn gateUpLaunches(ex: weights.Experts, sc: X.DecodeScratch, x: u64, x_stride: usize, pick: u64, wts: u64, out: u64, r: usize, limit: f32) ![2]X.DecodeLaunch {
    const w = try Window.of(ex, sc, r);
    var prep: gpu.Args = .{};
    prep.add(x);
    i32a(&prep, x_stride);
    for ([_]u64{ pick, ex.suh_g, ex.suh_u, sc.xg, sc.xu }) |v| prep.add(v);
    for ([_]usize{ w.d, w.slots, w.e, r }) |v| i32a(&prep, v);
    for ([_]u64{ sc.ids, sc.count, sc.members }) |v| prep.add(v);
    i32a(&prep, r); // maxm: the member lists' stride
    for ([_]u64{ wts, sc.y, 0, out }) |v| prep.add(v); // no add
    for ([_]usize{ w.dn, 1 }) |v| i32a(&prep, v); // out's columns here; store_y
    prep.add(@as(u64, 0)); // no epoch
    const prep_warps = X.prep_threads / 32;
    const blocks = 1 + (w.p * (w.d / 128) * 2 + prep_warps - 1) / prep_warps;

    var gu: gpu.Args = .{};
    for ([_]u64{ sc.xg, sc.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, sc.ids, sc.count, sc.members, sc.z }) |v| gu.add(v);
    for ([_]usize{ w.d, w.ig, w.p, w.sk, r, w.slots }) |v| i32a(&gu, v);
    gu.add(X.DecodeEpi{ .pick = pick, .E = @intCast(w.e), .svh_g = ex.svh_g, .svh_u = ex.svh_u, .suh_d = ex.suh_d, .xd = sc.xd, .limit = limit, .act_mode = X.act_f32, .cnt = sc.cnt_gu, .discard = @intFromBool(8 * w.sk <= 32) });
    return .{
        .{ .kernel = .prep, .cfg = .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = X.prep_threads }, .shared = @intCast(2 * w.p * 4) }, .args = prep },
        .{ .kernel = .gate_up, .range = X.range(ex.k2_gu[0], ex.k2_gu[1]), .cfg = .{ .grid = .{ .x = @intCast(w.maxu), .y = @intCast(w.ig / 128), .z = @intCast(2 * w.sk * w.mt) }, .block = .{ .x = 4 * 32 }, .pdl = gpu.features.pdl }, .args = gu },
    };
}

/// down_fused's launch: down over the TP2 rank's whole intermediate xd_full [R * slots, I] fp16 (the column exchange's
/// assembly, pair 0's blocks first) for this node's dn columns, the combine (slots in order, by wts) in its epilogue
/// into out fp32 [R, dn]. A programmatic dependent (TF_DS_2D_DOWN_PDL=1, served): with no ready flag to wait for it
/// issues its first trellis words, then waits for the kernel before it (the exchange) before it reads a row. Uses
/// gateUpLaunches' grouping in sc.
pub fn downLaunch(ex: weights.Experts, sc: X.DecodeScratch, xd_full: u64, pick: u64, wts: u64, out: u64, r: usize) !X.DecodeLaunch {
    const w = try Window.of(ex, sc, r);
    var dn: gpu.Args = .{};
    for ([_]u64{ xd_full, xd_full, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, sc.ids, sc.count, sc.members, sc.z }) |v| dn.add(v);
    for ([_]usize{ w.i, w.dn, w.p, 1, r, w.slots }) |v| i32a(&dn, v);
    const dy = reap.downY(sc.y, sc.reap_y, w.slots);
    dn.add(X.DecodeEpi{ .pick = pick, .E = @intCast(w.e), .svh_d = ex.svh_d, .y = dy.y, .wts = wts, .out = out, .cnt = sc.cnt_d, .discard = dy.discard });
    return .{ .kernel = .down, .range = X.range(ex.k2_d[0], ex.k2_d[1]), .cfg = .{ .grid = .{ .x = @intCast(w.maxu), .y = @intCast(w.dn / 128), .z = @intCast(w.mt) }, .block = .{ .x = 4 * 32 }, .pdl = gpu.features.pdl }, .args = dn };
}

fn function(k: *const X.Kernels, l: *const X.DecodeLaunch) !gpu.Function {
    return switch (l.kernel) {
        .prep => k.decode_prep,
        .gate_up => k.cp[l.range][0] orelse error.MissingKernel,
        .down => k.cp[l.range][1] orelse error.MissingKernel,
    };
}

/// gateup_fused on a decode window: decode_prep, then gate / up (this node's columns of the intermediate in sc.xd).
pub fn decodeGateUp(k: *const X.Kernels, s: gpu.Stream, ex: weights.Experts, sc: X.DecodeScratch, x: u64, x_stride: usize, pick: u64, wts: u64, out: u64, r: usize, limit: f32) !void {
    var ls = try gateUpLaunches(ex, sc, x, x_stride, pick, wts, out, r, limit);
    var fs: [2]gpu.Function = undefined;
    for (&ls, &fs) |*l, *f| f.* = try function(k, l);
    for (&ls, fs) |*l, f| try gpu.launch.launch(f, l.cfg, s, &l.args);
}

/// down_fused on a decode window, after the exchange: out fp32 [R, dn].
pub fn decodeDown(k: *const X.Kernels, s: gpu.Stream, ex: weights.Experts, sc: X.DecodeScratch, xd_full: u64, pick: u64, wts: u64, out: u64, r: usize) !void {
    var l = try downLaunch(ex, sc, xd_full, pick, wts, out, r);
    try gpu.launch.launch(try function(k, &l), l.cfg, s, &l.args);
}

test "a 2D node's prompt experts keep the TP2 half's settings" {
    var ex: weights.Experts = undefined;
    ex.dims = 5120;
    ex.down_k = 1152;
    ex.count = 385;
    // node of pair 0: blocks 0-4 of the half, down columns 0-2559
    ex.width = 640;
    ex.down_n = 2560;
    const p0 = try Plan.of(ex);
    try std.testing.expectEqual([4]usize{ 8, 4, 4, 1 }, p0.cgu);
    try std.testing.expectEqual([4]usize{ 8, 4, 1, 1 }, p0.cd);
    try std.testing.expectEqual(@as(usize, 1152), p0.i);
    // pair 1: blocks 5-8
    ex.width = 512;
    const p1 = try Plan.of(ex);
    try std.testing.expectEqual(p0.cgu, p1.cgu);
    try std.testing.expectEqual(@as(usize, 512), p1.ig);
    // TP2 itself: the whole half and every column
    ex.width = 1152;
    ex.down_n = 5120;
    const t = try Plan.of(ex);
    try std.testing.expectEqual(try X.config(5120, 1152, true), t.cgu);
    // a part that is not whole 128-blocks is refused
    ex.width = 576;
    try std.testing.expectError(error.UnsupportedShape, Plan.of(ex));
}

test "a 2D node's decode scratch and windows" {
    // a decoder layer's TP2 half (D 5120, I 1152, 385 experts, 7 slots) split for pair 0's node (gate / up blocks 0-4,
    // down columns 0-2559) and pair 1's (blocks 5-8), the served 64-row scratch
    var ex: weights.Experts = .{ .trellis = 0, .gate_ptr = 0, .up_ptr = 0, .down_ptr = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .count = 385, .dims = 5120, .width = 640, .down_k = 1152, .down_n = 2560, .k2_gu = .{ 6, 10 }, .k2_d = .{ 6, 10 } };
    try std.testing.expectEqual([13]usize{ 4587520, 4587520, 573440, 9175040, 4587520, 7700, 5120, 4, 1540, 1540, 1540, 4, 197120 }, try decodeSizes(64, 7, ex));
    const sc: X.DecodeScratch = .{ .rows = 64, .slots = 7, .xg = 0, .xu = 0, .xd = 0, .z = 0, .y = 0, .cnt_gu = 0, .cnt_d = 0, .epoch = 0, .ready = 0, .ready_cnt = 0, .ids = 0, .count = 0, .members = 0 };
    for ([_]usize{ 0, 64, 65 }) |r| try std.testing.expectError(error.NotADecodeWindow, gateUpLaunches(ex, sc, 0, 5120, 0, 0, 0, r, 10));
    _ = try gateUpLaunches(ex, sc, 0, 5120, 0, 0, 0, 63, 10);
    ex.width = 512;
    try std.testing.expectEqual([13]usize{ 4587520, 4587520, 458752, 7340032, 4587520, 6160, 5120, 4, 1540, 1540, 1540, 4, 197120 }, try decodeSizes(64, 7, ex));
    // a part of whole 128-column blocks only, within the half
    ex.width = 576;
    try std.testing.expectError(error.UnsupportedShape, downLaunch(ex, sc, 0, 0, 0, 0, 1));
    ex.width = 1280;
    try std.testing.expectError(error.UnsupportedShape, downLaunch(ex, sc, 0, 0, 0, 0, 1));
}

test "reap down y redirect on a 2D node changes only the down epilogue" {
    const ex: weights.Experts = .{ .trellis = 0, .gate_ptr = 0, .up_ptr = 0, .down_ptr = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .count = 385, .dims = 5120, .width = 640, .down_k = 1152, .down_n = 2560, .k2_gu = .{ 6, 10 }, .k2_d = .{ 6, 10 } };
    const sc: X.DecodeScratch = .{ .rows = 64, .slots = 7, .xg = 1, .xu = 2, .xd = 3, .z = 4, .y = 0x1000, .cnt_gu = 5, .cnt_d = 6, .epoch = 7, .ready = 8, .ready_cnt = 9, .ids = 10, .count = 11, .members = 12 };
    var r: usize = 1;
    while (r <= 8) : (r += 1) {
        const gu0 = try gateUpLaunches(ex, sc, 0x20, 5120, 0x30, 0x40, 0x50, r, 10);
        const dn0 = try downLaunch(ex, sc, 0x60, 0x30, 0x40, 0x50, r);
        var on_sc = sc;
        on_sc.reap_y = 0xABC0;
        const gu1 = try gateUpLaunches(ex, on_sc, 0x20, 5120, 0x30, 0x40, 0x50, r, 10);
        const dn1 = try downLaunch(ex, on_sc, 0x60, 0x30, 0x40, 0x50, r);
        try std.testing.expectEqualSlices(u8, gu0[0].args.storage[0..gu0[0].args.used], gu1[0].args.storage[0..gu1[0].args.used]);
        try std.testing.expectEqualSlices(u8, gu0[1].args.storage[0..gu0[1].args.used], gu1[1].args.storage[0..gu1[1].args.used]);
        const epi = dn1.args.offsets[dn1.args.count - 1];
        var a = dn0.args;
        var b = dn1.args;
        @memset(a.storage[epi + @offsetOf(X.DecodeEpi, "y") ..][0..8], 0);
        @memset(b.storage[epi + @offsetOf(X.DecodeEpi, "y") ..][0..8], 0);
        @memset(a.storage[epi + @offsetOf(X.DecodeEpi, "discard") ..][0..4], 0);
        @memset(b.storage[epi + @offsetOf(X.DecodeEpi, "discard") ..][0..4], 0);
        try std.testing.expectEqualSlices(u8, a.storage[0..a.used], b.storage[0..b.used]);
        const y = std.mem.readInt(u64, dn1.args.storage[epi + @offsetOf(X.DecodeEpi, "y") ..][0..8], .little);
        const disc = std.mem.readInt(u32, dn1.args.storage[epi + @offsetOf(X.DecodeEpi, "discard") ..][0..4], .little);
        try std.testing.expectEqual(@as(u64, 0xABC0), y);
        try std.testing.expectEqual(@as(u32, 0), disc);
    }
}
