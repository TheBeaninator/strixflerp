//! EXL3 linears of 1 to 128 rows as the served build runs them (exl3/linear.py's Exl3Group, linear_grouped.cu): one
//! rot_many launch rotating every layer's input rows, one glinear launch for each (bits, codebook, warps) set, from
//! the served build's own cubin. Each output has its layer's own plan, K ranges and reduction order.
const std = @import("std");
const gpu = @import("gpu");
const weights = @import("weights.zig");
const elf = @import("elf.zig");

/// The layers one launch takes (linear_grouped.cu GMAX).
pub const gmax = 8;
/// mul1, the codebook of every tensor of our checkpoint (CODEBOOK_IDS).
pub const codebook_mul1 = 2;

pub const DType = enum(c_int) { f16 = 0, bf16 = 1, f32 = 2 };

/// (K splits, warps a program) for a K x N layer (linear.py plan): the shape's alone, so a layer keeps one reduction
/// for every row count.
/// The N a layer's plan is made for: its own, or the TP2 slice's a 2D column part keeps (weights.Linear.plan_n).
fn planN(l: weights.Linear) usize {
    return if (l.plan_n != 0) l.plan_n else l.n;
}

pub fn plan(k: usize, n: usize) struct { sk: u32, wk: u32 } {
    const blocks = 192;
    const min_tiles = 8;
    const kt = k / 16;
    const nb = n / 128;
    var sk: usize = 1;
    var wk: usize = if (nb >= 64) 8 else 4;
    if (nb < 64) {
        while (nb * sk < blocks and sk < 64) sk *= 2;
    }
    while ((wk > 4 or sk > 1) and (kt % (sk * wk) != 0 or kt / (sk * wk) < min_tiles)) {
        if (wk > 4) {
            wk = 4;
        } else if (sk > 1) {
            sk /= 2;
        } else break;
    }
    return .{ .sk = @intCast(sk), .wk = @intCast(wk) };
}

const Plan = @TypeOf(plan(0, 0));

/// Decode windows TF_DS_TP4_FILL may retile. glinear's row passes are 16; a wider call keeps plan().
pub const fill_rows = 16;

/// TF_DS_TP4_FILL=1. More K splits on a decode window (foldTiles moves).
pub const sk_splits: u8 = 1;
/// TF_DS_TP4_FILL=exact. plan()'s sk and wk, launched as exl3.glinear_mlp (the fp32 order stays).
pub const sk_exact: u8 = 2;
/// TF_DS_TP4_FILL=exact_lds. The same plan and the same fp32 order, launched as exl3.glinear_mlp_lds.
pub const sk_exact_lds: u8 = 3;

/// plan(), unless sk_fill is sk_splits and the window is a decode window: then more K splits on the columns
/// actually launched (the N of the tensor, not plan_n). sk_exact and sk_exact_lds keep this plan. The warps stay
/// plan()'s, so the kernel specialisation does not change. The extra splits change the fp32 partial-sum order
/// (foldTiles); sk_fill 0, sk_exact and sk_exact_lds do not.
pub fn planFor(l: weights.Linear, m: usize) Plan {
    const base = plan(l.k, planN(l));
    if (l.sk_fill != sk_splits or m == 0 or m > fill_rows) return base;
    return fillSk(l.k, l.n, base);
}

/// 0: exl3.glinear. 1: exl3.glinear_mlp. 2: exl3.glinear_mlp_lds.
/// m > 16 stays on exl3.glinear, wide path included. exact and exact_lds do not share a launch.
pub fn mlpKind(l: weights.Linear, m: usize) u8 {
    if (m == 0 or m > fill_rows) return 0;
    if (l.sk_fill == sk_exact) return 1;
    if (l.sk_fill == sk_exact_lds) return 2;
    return 0;
}

/// exl3.glinear_mlp or exl3.glinear_mlp_lds for this linear: exact or exact_lds, and a decode window.
pub fn mlpLaunch(l: weights.Linear, m: usize) bool {
    return mlpKind(l, m) != 0;
}

/// Raise K splits until the column blocks reach the 192-block target, or the next split would not divide K or would
/// hold fewer than 4 k tiles a wave. A remainder would be dropped by glinear's integer division (per_warp = kt/SK/WK).
fn fillSk(k: usize, n: usize, base: Plan) Plan {
    const blocks = 192;
    const min_tiles = 4;
    const kt = k / 16;
    const nb = n / 128;
    var sk: usize = base.sk;
    const wk: usize = base.wk;
    while (sk < 64 and nb * sk < blocks) {
        const nsk = sk * 2;
        if (kt % (nsk * wk) != 0 or kt / (nsk * wk) < min_tiles) break;
        sk = nsk;
    }
    return .{ .sk = @intCast(sk), .wk = base.wk };
}

/// glinear's fp32 association of one output (exl3.hip glinear_body). `tiles` is one partial a (split, wave, k tile),
/// already past the four dot2s and the half-wave add. Each wave sums its tiles left to right; waves are added in
/// wave order (wave 0 first); splits are added in split order. exl3_mlp.hip keeps this order (foldQuads).
pub fn foldTiles(tiles: []const f32, sk: usize, wk: usize) f32 {
    const per = tiles.len / sk / wk;
    var acc: f32 = 0;
    for (0..sk) |s| {
        var wave: f32 = 0;
        for (0..wk) |w| {
            var part: f32 = 0;
            const at = (s * wk + w) * per;
            for (tiles[at..][0..per]) |t| part += t;
            wave = if (w == 0) part else wave + part;
        }
        acc = if (s == 0) wave else acc + wave;
    }
    return acc;
}

/// The same association one step finer: each k tile is four left-associated dot2 adds onto the wave's running sum
/// (exl3.hip strip_tiles_1row 310-319 and strip_rows_part 371-387, kept verbatim in exl3_mlp.hip), then waves and
/// splits as foldTiles. The half-wave add is after that chain and does not move with sk. A quad's other three
/// components are the tile's later dot2s; zeros leave the tile equal to its first component.
pub fn foldQuads(quads: []const [4]f32, sk: usize, wk: usize) f32 {
    const per = quads.len / sk / wk;
    var acc: f32 = 0;
    for (0..sk) |s| {
        var wave: f32 = 0;
        for (0..wk) |w| {
            var part: f32 = 0;
            const at = (s * wk + w) * per;
            for (quads[at..][0..per]) |q| {
                part += q[0];
                part += q[1];
                part += q[2];
                part += q[3];
            }
            wave = if (w == 0) part else wave + part;
        }
        acc = if (s == 0) wave else acc + wave;
    }
    return acc;
}

/// (words between k tiles, words between 128-column blocks) of a layer's words in the strips layout.
pub fn strides(l: weights.Linear) [2]i64 {
    const tw: i64 = 4 * @as(i64, l.k2);
    return .{ 8 * tw, @divExact(@as(i64, l.k), 16) * 8 * tw };
}

/// linear_grouped.cu's GLayer, field for field (C layout).
pub const GLayer = extern struct {
    xh: u64 = 0,
    T: u64 = 0,
    svh: u64 = 0,
    bias: u64 = 0,
    y: u64 = 0,
    Z: u64 = 0,
    counters: u64 = 0,
    ldx: i64 = 0,
    ldy: i64 = 0,
    stride_k: i64 = 0,
    stride_nb: i64 = 0,
    y_dtype: c_int = 0,
    K: c_int = 0,
    N: c_int = 0,
    SK: c_int = 0,
    first: c_int = 0,
    rsuh: u64 = 0,
    rxh: u64 = 0,
    ldr: i64 = 0,
    roff: c_int = 0,
    rcos: u64 = 0,
    rsin: u64 = 0,
    rpos: u64 = 0,
    rhd: c_int = 0,
    rrd: c_int = 0,
};

pub const GArgs = extern struct { l: [gmax]GLayer = @splat(.{}), n: c_int = 0, M: c_int = 0, discard: c_int = 0 };

pub const RLayer = extern struct { x: u64 = 0, suh: u64 = 0, xh: u64 = 0, ldx: i64 = 0, x_dtype: c_int = 0, K: c_int = 0, first: c_int = 0 };

pub const RArgs = extern struct { l: [gmax]RLayer = @splat(.{}), n: c_int = 0, M: c_int = 0 };

comptime {
    // the sizes the kernels were compiled with (pointers and long longs 8 bytes, ints 4, C padding)
    std.debug.assert(@sizeOf(GLayer) == 176 and @sizeOf(GArgs) == 1424);
    std.debug.assert(@sizeOf(RLayer) == 48 and @sizeOf(RArgs) == 392);
}

/// The cubin's kernels: glinear_kernel<K2, codebook, warps> by name, and rot_many_kernel.
pub const Kernels = struct {
    module: gpu.Module,
    rot_many: gpu.Function,
    glinear: [17][3][2]?gpu.Function = @splat(@splat(@splat(null))),
    /// exl3.glinear_mlp (exl3_mlp.hip): same ABI, k2 by warps 4 | 8, codebook mul1. Null until the kit has it.
    glinear_mlp: [17][2]?gpu.Function = @splat(@splat(null)),
    /// exl3.glinear_mlp_lds: the LDS-staged family. A null slot falls back to glinear_mlp (same plan, same sum order).
    glinear_mlp_lds: [17][2]?gpu.Function = @splat(@splat(null)),
    /// The functions are the AMD kit's (hip_load.zig): unload leaves its modules alone.
    kit_owned: bool = false,

    /// The served build's linear_grouped cubin (cuobjdump of tensorfold_exl3_linear_v7's .so).
    pub fn load(d: *const gpu.Driver, cubin: []const u8) !Kernels {
        var k: Kernels = .{ .module = try gpu.Module.load(d, cubin), .rot_many = undefined };
        errdefer k.module.unload();
        var names = elf.Symbols.init(cubin) orelse return error.BadCubin;
        var found_rot = false;
        while (names.next()) |name| {
            var buf: [256]u8 = undefined;
            if (std.mem.indexOf(u8, name, "rot_many_kernel") != null) {
                k.rot_many = try k.module.function(try std.mem.printSentinel(&buf, "{s}", .{name}, 0));
                found_rot = true;
                continue;
            }
            const at = std.mem.indexOf(u8, name, "glinear_kernelILi") orelse continue;
            const t = template3(name[at + "glinear_kernel".len ..]) orelse continue;
            if (t[0] > 16 or t[1] > 2 or (t[2] != 4 and t[2] != 8)) continue;
            k.glinear[t[0]][t[1]][if (t[2] == 8) 1 else 0] = try k.module.function(try std.mem.printSentinel(&buf, "{s}", .{name}, 0));
        }
        if (!found_rot) return error.MissingKernel;
        return k;
    }

    pub fn unload(k: *Kernels) void {
        if (!k.kit_owned) k.module.unload();
        k.* = undefined;
    }
};

/// "ILi7ELi2ELi8EE..." -> {7, 2, 8}
fn template3(s: []const u8) ?[3]u32 {
    var out: [3]u32 = undefined;
    var rest = s;
    for (&out) |*v| {
        const i = std.mem.indexOf(u8, rest, "Li") orelse return null;
        rest = rest[i + 2 ..];
        const e = std.mem.indexOfScalar(u8, rest, 'E') orelse return null;
        v.* = std.fmt.parseInt(u32, rest[0..e], 10) catch return null;
        rest = rest[e + 1 ..];
    }
    return out;
}

/// Exl3Group.rotated's `rot` for a layer: its outputs also rotated into the next layer's input rows (rot_many's bits
/// of y): that layer's suh, its rotated rows (fp16, row stride ldr) and the output's column offset there.
pub const RotOut = struct { suh: u64, xh: u64, ldr: i64, off: u32 };

/// Exl3Group.rotated's `rope` for a layer: its bf16 outputs leave with rope_heads applied (cos / sin fp32 [*, rd / 2],
/// positions int64 [M], heads of hd columns, RoPE on their last rd).
pub const RopeOut = struct { cos: u64, sin: u64, pos: u64, hd: u32, rd: u32 };

/// One layer's call: its rotated input rows, its output rows and dtype, its split-K counters (int32 [8 N / 128], left
/// zero by the kernel) and, for SK > 1, its partials' place in the launch's Z; the folds its epilogue makes.
pub const Call = struct {
    layer: weights.Linear,
    x: u64,
    ldx: i64,
    x_dtype: DType,
    xh: u64,
    y: u64,
    ldy: i64,
    y_dtype: DType,
    counters: u64,
    rot: ?RotOut = null,
    rope: ?RopeOut = null,
};

/// rot_many: every layer's input rows rotated into its xh rows [M, K] fp16, one launch.
pub fn rotMany(k: *const Kernels, stream: gpu.Stream, calls: []const Call, m: usize, pdl: bool) !void {
    var a: RArgs = .{ .n = @intCast(calls.len), .M = @intCast(m) };
    var warps: usize = 0;
    for (calls, 0..) |c, i| {
        a.l[i] = .{ .x = c.x, .suh = c.layer.suh, .xh = c.xh, .ldx = c.ldx, .x_dtype = @intFromEnum(c.x_dtype), .K = @intCast(c.layer.k), .first = @intCast(warps) };
        warps += m * (c.layer.k / 128);
    }
    var args: gpu.Args = .{};
    args.add(a);
    try gpu.launch.launch(k.rot_many, .{ .grid = .{ .x = @intCast((warps + 3) / 4) }, .block = .{ .x = 128 }, .pdl = pdl and gpu.features.pdl }, stream, &args);
}

/// The glinear launches of a group (Exl3Group.rotated, with each call's folds): each (bits, codebook, warps) set in
/// first-seen order, its split layers' partials in z (fp32, SK * M * N floats each, in order). An exact linear
/// takes exl3.glinear_mlp, an exact_lds linear takes exl3.glinear_mlp_lds, and neither shares a launch with a
/// normal one or with the other family. With sk_fill 0 every call is kind 0, and the groups are today's.
pub fn glinear(k: *const Kernels, stream: gpu.Stream, calls: []const Call, m: usize, z: u64, pdl: bool, discard: bool) !void {
    var done: [gmax]bool = @splat(false);
    for (calls, 0..) |first, i| {
        if (done[i]) continue;
        const pf = planFor(first.layer, m);
        const want = mlpKind(first.layer, m);
        var a: GArgs = .{ .M = @intCast(m) };
        var blocks: usize = 0;
        var zoff: u64 = 0;
        var any_z = false;
        for (calls[i..], i..) |c, j| {
            const pj = planFor(c.layer, m);
            if (done[j] or c.layer.k2 != first.layer.k2 or pj.wk != pf.wk or mlpKind(c.layer, m) != want) continue;
            done[j] = true;
            const s = strides(c.layer);
            const n: usize = @intCast(a.n);
            a.l[n] = .{ .xh = c.xh, .ldx = @intCast(c.layer.k), .T = c.layer.words, .stride_k = s[0], .stride_nb = s[1], .svh = c.layer.svh, .y = c.y, .y_dtype = @intFromEnum(c.y_dtype), .ldy = c.ldy, .K = @intCast(c.layer.k), .N = @intCast(c.layer.n), .SK = @intCast(pj.sk), .counters = c.counters, .first = @intCast(blocks) };
            if (c.rot) |r| {
                a.l[n].rsuh = r.suh;
                a.l[n].rxh = r.xh;
                a.l[n].ldr = r.ldr;
                a.l[n].roff = @intCast(r.off);
            }
            if (c.rope) |r| {
                a.l[n].rcos = r.cos;
                a.l[n].rsin = r.sin;
                a.l[n].rpos = r.pos;
                a.l[n].rhd = @intCast(r.hd);
                a.l[n].rrd = @intCast(r.rd);
            }
            if (pj.sk > 1) {
                a.l[n].Z = z + zoff * 4;
                zoff += @as(u64, pj.sk) * m * c.layer.n;
                any_z = true;
            }
            blocks += (c.layer.n / 128) * pj.sk;
            a.n += 1;
        }
        a.discard = @intFromBool(discard and any_z);
        const slot: usize = if (pf.wk == 8) 1 else 0;
        const f = switch (want) {
            1 => k.glinear_mlp[first.layer.k2][slot] orelse return error.MissingKernel,
            // a (k2, wk) with no LDS symbol keeps the register kernel: same plan, same sum order
            2 => k.glinear_mlp_lds[first.layer.k2][slot] orelse k.glinear_mlp[first.layer.k2][slot] orelse return error.MissingKernel,
            else => k.glinear[first.layer.k2][codebook_mul1][slot] orelse return error.MissingKernel,
        };
        var args: gpu.Args = .{};
        args.add(a);
        const smem: u32 = @intCast(pf.wk * @min(m, 8) * 128 * 4);
        try gpu.launch.launch(f, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = pf.wk * 32 }, .shared = smem, .pdl = pdl and gpu.features.pdl }, stream, &args);
    }
}

/// Floats of Z a group needs at m rows (every split layer's SK * M * N). A split-filled linear (sk_fill ==
/// sk_splits) whose base plan has no partials still reserves the decode window's, so a lone wq_b at the chunk's
/// 128-row size is honest. sk_fill 0, sk_exact and sk_exact_lds use the base plan's count alone.
pub fn zFloats(layers: []const weights.Linear, m: usize) usize {
    var n: usize = 0;
    for (layers) |l| {
        const p = planFor(l, m);
        var need: usize = if (p.sk > 1) p.sk * m * l.n else 0;
        if (l.sk_fill == sk_splits and m > fill_rows) {
            const pf = planFor(l, fill_rows);
            if (pf.sk > 1) need = @max(need, pf.sk * fill_rows * l.n);
        }
        n += need;
    }
    return n;
}

test "plans as linear.py makes them" {
    // narrow layers take K splits to fill the SMs; wide ones 8 warps; the K tile count bounds both
    try std.testing.expectEqual(@as(u32, 8), plan(1280, 32768).wk);
    try std.testing.expectEqual(@as(u32, 1), plan(1280, 32768).sk);
    const q = plan(5120, 1280);
    try std.testing.expectEqual(@as(u32, 4), q.wk);
    try std.testing.expect(q.sk > 1 and (5120 / 16) % (q.sk * q.wk) == 0);
    try std.testing.expectEqualDeep([3]u32{ 7, 2, 8 }, template3("ILi7ELi2ELi8EEEvNS_5GArgsE").?);
}

test "plans and strides equal every recorded glinear layer of the served build" {
    // (K, N, SK, WK, K2, stride_k, stride_nb) of each distinct layer the recording's glinear calls passed
    const seen = [_][7]u32{
        .{ 512, 128, 1, 4, 16, 512, 16384 },       .{ 1280, 4096, 2, 4, 10, 320, 25600 },     .{ 1280, 16384, 1, 8, 8, 256, 20480 },
        .{ 1280, 16384, 1, 8, 10, 320, 25600 },    .{ 3072, 25600, 1, 8, 8, 256, 49152 },     .{ 3072, 25600, 1, 8, 10, 320, 61440 },
        .{ 4096, 1024, 8, 4, 8, 256, 65536 },      .{ 4096, 1024, 8, 4, 10, 320, 81920 },     .{ 4096, 5120, 8, 4, 8, 256, 65536 },
        .{ 4096, 5120, 8, 4, 10, 320, 81920 },     .{ 5120, 512, 8, 4, 8, 256, 81920 },       .{ 5120, 512, 8, 4, 10, 320, 102400 },
        .{ 5120, 512, 8, 4, 12, 384, 122880 },     .{ 5120, 1280, 8, 4, 8, 256, 81920 },      .{ 5120, 1280, 8, 4, 10, 320, 102400 },
        .{ 5120, 1280, 8, 4, 12, 384, 122880 },    .{ 5120, 64640, 1, 8, 12, 384, 122880 },   .{ 15360, 5120, 8, 4, 8, 256, 245760 },
    };
    for (seen) |r| {
        const p = plan(r[0], r[1]);
        try std.testing.expectEqual(r[2], p.sk);
        try std.testing.expectEqual(r[3], p.wk);
        const s = strides(.{ .words = 0, .suh = 0, .svh = 0, .k = r[0], .n = r[1], .k2 = r[4] });
        try std.testing.expectEqual(@as(i64, r[5]), s[0]);
        try std.testing.expectEqual(@as(i64, r[6]), s[1]);
    }
}

test "a 2D node's column part keeps its TP2 slice's plan" {
    // wq_b of a pair: 8,192 of the rank's 16,384 columns, planned as the 16,384 the TP2 rank launches
    const part: weights.Linear = .{ .words = 0, .suh = 0, .svh = 0, .k = 1024, .n = 8192, .k2 = 2, .plan_n = 16384 };
    try std.testing.expectEqual(plan(1024, 16384), plan(part.k, planN(part)));
    const whole: weights.Linear = .{ .words = 0, .suh = 0, .svh = 0, .k = 1024, .n = 16384, .k2 = 2 };
    try std.testing.expectEqual(plan(1024, 16384), plan(whole.k, planN(whole)));
    // sk_fill defaults to 0: the call sites' plan is plan(), at a decode window too
    try std.testing.expectEqual(plan(1024, 16384), planFor(part, 4));
}

fn lin(k: u32, n: u32, plan_n: u32, fill: u8) weights.Linear {
    return .{ .words = 0, .suh = 0, .svh = 0, .k = k, .n = n, .k2 = 8, .plan_n = plan_n, .sk_fill = fill };
}

test "sk_fill off is plan(), and a window past 16 rows stays plan()" {
    const shapes = [_]weights.Linear{
        lin(4096, 1024, 0, 0), // wo_a, one group
        lin(4096, 2560, 5120, 0), // wo_b, a pair's columns
        lin(1280, 8192, 16384, 0), // wq_b, a pair's heads
    };
    for (shapes) |l| {
        try std.testing.expectEqual(plan(l.k, planN(l)), planFor(l, 4));
        var on = l;
        on.sk_fill = 1;
        try std.testing.expectEqual(plan(l.k, planN(l)), planFor(on, 17));
        try std.testing.expectEqual(plan(l.k, planN(l)), planFor(on, 128));
        try std.testing.expectEqual(plan(l.k, planN(l)), planFor(on, 0));
    }
}

test "TF_DS_TP4_FILL raises K splits on a 2D node's halved attention linears" {
    // wo_a, one group. base sk 8 wk 4 (64 blocks); fill stops at 16 (128). 32 would leave 2 tiles a wave.
    const woa = lin(4096, 1024, 0, 1);
    const pa = planFor(woa, 4);
    try std.testing.expectEqual(@as(u32, 16), pa.sk);
    try std.testing.expectEqual(@as(u32, 4), pa.wk);
    try std.testing.expectEqual(@as(usize, 128), (woa.n / 128) * pa.sk);
    // two groups share one launch when k2 and wk match: 128 blocks today, 256 filled
    try std.testing.expectEqual(@as(usize, 128), 2 * (woa.n / 128) * plan(woa.k, woa.n).sk);
    try std.testing.expectEqual(@as(usize, 256), 2 * (woa.n / 128) * pa.sk);
    // wo_b N 2560 (plan_n 5120 is the TP2 plan, not the grid). 160 blocks, filled 320.
    const wob = lin(4096, 2560, 5120, 1);
    const pb = planFor(wob, 1);
    try std.testing.expectEqual(@as(u32, 16), pb.sk);
    try std.testing.expectEqual(plan(wob.k, planN(wob)).wk, pb.wk);
    try std.testing.expectEqual(@as(usize, 160), (wob.n / 128) * plan(wob.k, planN(wob)).sk);
    try std.testing.expectEqual(@as(usize, 320), (wob.n / 128) * pb.sk);
    // wq_b N 8192 planned as 16384 (sk 1 wk 8). 80 k tiles divide 2*8 and not 4*8. 64 blocks, filled 128.
    const wqb = lin(1280, 8192, 16384, 1);
    const pq = planFor(wqb, 6);
    try std.testing.expectEqual(@as(u32, 2), pq.sk);
    try std.testing.expectEqual(@as(u32, 8), pq.wk);
    try std.testing.expectEqual(@as(usize, 64), (wqb.n / 128) * plan(wqb.k, planN(wqb)).sk);
    try std.testing.expectEqual(@as(usize, 128), (wqb.n / 128) * pq.sk);
    // a full TP2 wo_b (N 5120, 320 blocks already) does not grow even if asked
    const full = lin(4096, 5120, 0, 1);
    try std.testing.expectEqual(plan(full.k, full.n), planFor(full, 4));
    for ([_]weights.Linear{ woa, wob, wqb }) |l| {
        const p = planFor(l, 4);
        const kt = l.k / 16;
        try std.testing.expect(kt % (p.sk * p.wk) == 0);
        try std.testing.expect(kt / (p.sk * p.wk) >= 4);
        try std.testing.expectEqual(plan(l.k, planN(l)).wk, p.wk);
        // rows 1..16 share one plan; 2 and 6 are the profile's neighbours of 3.71
        try std.testing.expectEqual(p, planFor(l, 2));
        try std.testing.expectEqual(p, planFor(l, 16));
    }
}

test "foldTiles is glinear's fp32 association: a K split moves the partial boundaries" {
    // 1e8 + 1 rounds to 1e8. One chain ends at 1; two split chains, each summed before the add, end at 0.
    const tiles = [_]f32{ 1e8, 1, -1e8, 1 };
    try std.testing.expectEqual(@as(f32, 1), foldTiles(&tiles, 1, 1));
    try std.testing.expectEqual(@as(f32, 0), foldTiles(&tiles, 2, 1));
    // waves add the same way after each wave has summed its own tiles (wave 0 is 1e8+1, wave 1 is -1e8+1)
    try std.testing.expectEqual(@as(f32, 0), foldTiles(&tiles, 1, 2));
}

test "zFloats with sk_fill off matches the base plan, and a filled wq_b sizes its decode partials" {
    const woa = lin(4096, 1024, 0, 0);
    try std.testing.expectEqual(8 * 128 * 1024, zFloats(&[_]weights.Linear{woa}, 128));
    const wqb = lin(1280, 8192, 16384, 0);
    try std.testing.expectEqual(@as(usize, 0), zFloats(&[_]weights.Linear{wqb}, 4));
    try std.testing.expectEqual(@as(usize, 0), zFloats(&[_]weights.Linear{wqb}, 128));
    var on = wqb;
    on.sk_fill = 1;
    try std.testing.expectEqual(2 * 4 * 8192, zFloats(&[_]weights.Linear{on}, 4));
    // the 128-row allocation still covers the decode window (the base sk is 1, so the base term is 0)
    try std.testing.expectEqual(2 * fill_rows * 8192, zFloats(&[_]weights.Linear{on}, 128));
    // wo_b's base partials at m 128 already cover the filled window, so the chunk's max does not grow
    const wob = lin(4096, 2560, 5120, 1);
    try std.testing.expectEqual(8 * 128 * 2560, zFloats(&[_]weights.Linear{wob}, 128));
    try std.testing.expectEqual(16 * 4 * 2560, zFloats(&[_]weights.Linear{wob}, 4));
    const base_woa = 8 * 128 * 1024;
    var filled_woa = woa;
    filled_woa.sk_fill = 1;
    try std.testing.expectEqual(base_woa, zFloats(&[_]weights.Linear{filled_woa}, 128));
}

test "TF_DS_TP4_FILL=exact keeps TP2's sk, wk and per-column association" {
    // wo_a one group is the TP2 group's N. wo_b and wq_b are the pair's columns, planned as the TP2 N.
    const shapes = [_]struct { k: u32, n: u32, plan_n: u32, tp2_n: u32 }{
        .{ .k = 4096, .n = 1024, .plan_n = 0, .tp2_n = 1024 },
        .{ .k = 4096, .n = 2560, .plan_n = 5120, .tp2_n = 5120 },
        .{ .k = 1280, .n = 8192, .plan_n = 16384, .tp2_n = 16384 },
    };
    for (shapes) |sh| {
        const exact = lin(sh.k, sh.n, sh.plan_n, sk_exact);
        const filled = lin(sh.k, sh.n, sh.plan_n, sk_splits);
        const off = lin(sh.k, sh.n, sh.plan_n, 0);
        const pe = planFor(exact, 4);
        const tp2 = plan(sh.k, sh.tp2_n);
        try std.testing.expectEqual(plan(sh.k, planN(exact)), pe);
        try std.testing.expectEqual(tp2, pe);
        const kt = sh.k / 16;
        try std.testing.expect(kt % (pe.sk * pe.wk) == 0);
        try std.testing.expectEqual((sh.n / 128) * tp2.sk, (sh.n / 128) * pe.sk);
        const pf = planFor(filled, 4);
        const per_f = kt / pf.sk / pf.wk;
        // {1e8, 1, -1e8, 1} starts at the filled plan's per-2, so the four tiles sit in exact's first wave
        // (result 1) and straddle a wave of the raised plan (result 0). A fold that ignores sk cannot pass both.
        var q: [256][4]f32 = @splat(.{ 0, 0, 0, 0 });
        const vals = [_]f32{ 1e8, 1, -1e8, 1 };
        for (vals, 0..) |v, i| q[per_f - 2 + i] = .{ v, 0, 0, 0 };
        const got_e = foldQuads(q[0..kt], pe.sk, pe.wk);
        try std.testing.expectEqual(foldQuads(q[0..kt], tp2.sk, tp2.wk), got_e);
        try std.testing.expectEqual(@as(f32, 1), got_e);
        try std.testing.expectEqual(@as(f32, 0), foldQuads(q[0..kt], pf.sk, pf.wk));
        const lds = lin(sh.k, sh.n, sh.plan_n, sk_exact_lds);
        const pl = planFor(lds, 4);
        try std.testing.expectEqual(pe, pl);
        // the LDS kernel runs this same plan through the same dot2 chain, so the host association is got_e
        try std.testing.expectEqual(got_e, foldQuads(q[0..kt], pl.sk, pl.wk));
        try std.testing.expectEqual(@as(u8, 1), mlpKind(exact, 4));
        try std.testing.expectEqual(@as(u8, 2), mlpKind(lds, 4));
        try std.testing.expectEqual(@as(u8, 0), mlpKind(lds, 17));
        try std.testing.expectEqual(zFloats(&[_]weights.Linear{exact}, 4), zFloats(&[_]weights.Linear{lds}, 4));
        try std.testing.expectEqual(zFloats(&[_]weights.Linear{exact}, 128), zFloats(&[_]weights.Linear{lds}, 128));
        for ([_]usize{ 1, 2, 4, 6, 16 }) |m| {
            try std.testing.expect(mlpLaunch(exact, m));
            try std.testing.expect(mlpLaunch(lds, m));
        }
        try std.testing.expect(!mlpLaunch(exact, 0));
        try std.testing.expect(!mlpLaunch(exact, 17));
        try std.testing.expect(!mlpLaunch(lds, 0));
        try std.testing.expect(!mlpLaunch(lds, 17));
        try std.testing.expect(!mlpLaunch(off, 4));
        try std.testing.expect(!mlpLaunch(filled, 4));
        try std.testing.expectEqual(plan(sh.k, planN(off)), planFor(off, 4));
        try std.testing.expectEqual(planFor(off, 4), planFor(exact, 17));
        try std.testing.expectEqual(zFloats(&[_]weights.Linear{off}, 4), zFloats(&[_]weights.Linear{exact}, 4));
        try std.testing.expectEqual(zFloats(&[_]weights.Linear{off}, 128), zFloats(&[_]weights.Linear{exact}, 128));
    }
    // four dot2s of one tile, left-associated: ((1e8 + 1) - 1e8) + 1 rounds to 1
    const one = [_][4]f32{.{ 1e8, 1, -1e8, 1 }};
    try std.testing.expectEqual(@as(f32, 1), foldQuads(&one, 1, 1));
    const wob = lin(4096, 2560, 5120, sk_exact);
    try std.testing.expectEqual(@as(usize, 160), (wob.n / 128) * planFor(wob, 4).sk);
    const wob1 = lin(4096, 2560, 5120, sk_splits);
    try std.testing.expectEqual(@as(usize, 320), (wob1.n / 128) * planFor(wob1, 4).sk);
    const woa = lin(4096, 1024, 0, sk_exact);
    try std.testing.expectEqual(@as(usize, 128), 2 * (woa.n / 128) * planFor(woa, 4).sk);
}
