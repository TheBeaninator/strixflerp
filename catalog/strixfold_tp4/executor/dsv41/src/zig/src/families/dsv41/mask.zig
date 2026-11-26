//! MASK_EXPERTS (serve.conf; TF_DS_MASK_EXPERTS on every rank): routed experts of the target
//! layers taken out of the router at load, so a launch routes as a checkpoint with those experts deleted would, without
//! building one. The file names (layer, expert) pairs, one layer a line: 'layer:expert' or 'layer:e1,e2,...'; '#' starts
//! a comment (route_join.py --mask-out writes a '#' header, then one 'layer:expert' a line); blank lines are skipped.
//!
//! How: each masked expert's f32 in its layer's router bias (gate.bias, and gate.bias_vl when the image bias is loaded)
//! becomes -inf before the first launch. The router (tri_kernels _route_row / _route_parts / _rowmm_route) picks top-k on
//! sqrt(softplus(logit)) + bias, so a masked expert's selection score is -inf and it is never picked while at least top_k
//! experts of the layer stay unmasked (refused otherwise); the picked weights are the unbiased scores normalised over the
//! picked (unmasked) set, times routed_scaling_factor: a pruned checkpoint's routing, bit for bit (the test below runs
//! both on the host reference). Only the target layers (0 .. num_hidden_layers-1): the DSpark/MTP blocks are never
//! masked, and the shared expert is not a routed expert (its id, n_routed_experts, is refused as out of range).
//!
//! An empty file (or one with only comments) masks nothing: no bias is touched and the handshake's plan text is the stock
//! one, so the launch is the base build's byte for byte. A non-empty mask adds " mask=<16 hex>" to the plan text (the
//! SHA-256 of the canonical list below, so comments and line order do not matter), and peers that differ are refused.
const std = @import("std");

pub const ParseError = error{ BadMaskLine, MaskLayerOutOfRange, MaskExpertOutOfRange, MaskLeavesTooFew };

pub const Mask = struct {
    layers: usize,
    experts: usize,
    off: []bool, // [layers * experts]: true where the expert is masked
    count: usize, // distinct (layer, expert) pairs masked
    sha256: [64]u8, // of the canonical text (canonicalHash): id() is its first 16 characters

    pub fn deinit(m: *Mask, gpa: std.mem.Allocator) void {
        gpa.free(m.off);
        m.off = &.{};
    }

    pub fn masked(m: *const Mask, layer: usize, expert: usize) bool {
        return m.off[layer * m.experts + expert];
    }

    /// The masked experts of the layer.
    pub fn inLayer(m: *const Mask, layer: usize) usize {
        var n: usize = 0;
        for (m.off[layer * m.experts ..][0..m.experts]) |b| n += @intFromBool(b);
        return n;
    }

    /// The mask's name in the handshake's plan text (16 hex characters).
    pub fn id(m: *const Mask) []const u8 {
        return m.sha256[0..16];
    }

    /// Writes -inf over each masked expert's f32 in bias, the layer's router bias [experts] as little-endian bytes
    /// (gate.bias or gate.bias_vl); every other byte stays as it is (an unmasked layer: all of them).
    pub fn applyBias(m: *const Mask, layer: usize, bias: []u8) error{BadBiasLength}!void {
        if (bias.len != m.experts * 4 or layer >= m.layers) return error.BadBiasLength;
        const neg: u32 = @bitCast(-std.math.inf(f32));
        for (0..m.experts) |e| if (m.masked(layer, e)) std.mem.writeInt(u32, bias[4 * e ..][0..4], neg, .little);
    }
};

fn digits(s: []const u8) ?usize {
    if (s.len == 0 or s.len > 9) return null;
    for (s) |ch| if (ch < '0' or ch > '9') return null;
    return std.fmt.parseInt(usize, s, 10) catch null;
}

/// The mask in text for a model of 'layers' target layers of 'experts' routed experts, top_k picked: a line that is not
/// 'layer:e[,e...]' (decimal, spaces around the numbers allowed), a layer past the target layers (an MTP block's index),
/// an expert past the routed ones (the shared expert's id) or a layer left with fewer than top_k experts is refused, its
/// 1-based line number in bad_line (0 for a layer left too few). Repeats are one entry.
pub fn parse(gpa: std.mem.Allocator, text: []const u8, layers: usize, experts: usize, top_k: usize, bad_line: *usize) (ParseError || std.mem.Allocator.Error)!Mask {
    const off = try gpa.alloc(bool, layers * experts);
    errdefer gpa.free(off);
    @memset(off, false);
    var count: usize = 0;
    var it = std.mem.splitScalar(u8, text, '\n');
    var line_no: usize = 0;
    bad_line.* = 0;
    while (it.next()) |raw| {
        line_no += 1;
        const hash = std.mem.indexOfScalar(u8, raw, '#') orelse raw.len;
        const line = std.mem.trim(u8, raw[0..hash], " \t\r");
        if (line.len == 0) continue;
        errdefer bad_line.* = line_no;
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return error.BadMaskLine;
        const layer = digits(std.mem.trim(u8, line[0..colon], " \t")) orelse return error.BadMaskLine;
        if (layer >= layers) return error.MaskLayerOutOfRange;
        var es = std.mem.splitScalar(u8, line[colon + 1 ..], ',');
        while (es.next()) |part| {
            const e = digits(std.mem.trim(u8, part, " \t")) orelse return error.BadMaskLine;
            if (e >= experts) return error.MaskExpertOutOfRange;
            if (!off[layer * experts + e]) count += 1;
            off[layer * experts + e] = true;
        }
    }
    var m: Mask = .{ .layers = layers, .experts = experts, .off = off, .count = count, .sha256 = undefined };
    for (0..layers) |l| if (experts - m.inLayer(l) < top_k) return error.MaskLeavesTooFew;
    m.sha256 = canonicalHash(&m);
    return m;
}

/// SHA-256 (hex) of the canonical list: for each layer with a masked expert, ascending, "layer:e1,e2,...\n" with the
/// experts ascending. Two files that mask the same pairs have the same hash whatever their comments or order.
pub fn canonicalHash(m: *const Mask) [64]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var nb: [24]u8 = undefined;
    for (0..m.layers) |l| {
        if (m.inLayer(l) == 0) continue;
        h.update(std.fmt.bufPrint(&nb, "{d}:", .{l}) catch unreachable);
        var first = true;
        for (0..m.experts) |e| if (m.masked(l, e)) {
            h.update(std.fmt.bufPrint(&nb, "{s}{d}", .{ if (first) "" else ",", e }) catch unreachable);
            first = false;
        };
        h.update("\n");
    }
    var out: [32]u8 = undefined;
    h.final(&out);
    return std.fmt.bytesToHex(out, .lower);
}

/// Reads and parses the file at path (refusals logged with the path and line).
pub fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8, layers: usize, experts: usize, top_k: usize) !Mask {
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24)) catch |e| {
        std.log.err("MASK_EXPERTS {s}: {s}", .{ path, @errorName(e) });
        return e;
    };
    defer gpa.free(text);
    var bad: usize = 0;
    return parse(gpa, text, layers, experts, top_k, &bad) catch |e| {
        if (bad > 0)
            std.log.err("MASK_EXPERTS {s} line {d}: {s} (layers 0..{d}, experts 0..{d})", .{ path, bad, @errorName(e), layers - 1, experts - 1 })
        else
            std.log.err("MASK_EXPERTS {s}: {s} (a layer keeps fewer than top_k = {d} experts)", .{ path, @errorName(e), top_k });
        return e;
    };
}

/// " mask=<16 hex>" for a mask that masks something, else "" (the stock plan text). Leading space included.
pub fn planSuffix(buf: []u8, m: ?*const Mask) []const u8 {
    const mk = m orelse return "";
    if (mk.count == 0) return "";
    return std.fmt.bufPrint(buf, " mask={s}", .{mk.id()}) catch "";
}

// -- the host reference of the router --------------------------------------------------------------------------

/// One row of tri_kernels' _route_row on the host in f32: score sc = sqrt(softplus(logit)) (softplus = the logit past
/// 20), selection sc + bias, top_k picks by the highest selection (the lowest index among ties), each pick's weight its
/// unbiased score over the picks' sum (+1e-20, summed in pick order) times scale; slots past top_k hold shared_id with
/// weight 1. picks and wts are [slots].
pub fn routeRow(logits: []const f32, bias: []const f32, top_k: usize, scale: f32, shared_id: i32, picks: []i32, wts: []f32) void {
    const ne = logits.len;
    var sc: [1024]f32 = undefined;
    var sel: [1024]f32 = undefined;
    std.debug.assert(ne <= sc.len and bias.len == ne and picks.len == wts.len and picks.len >= top_k);
    for (0..ne) |e| {
        const lg = logits[e];
        const sp = if (lg > 20.0) lg else @log(1.0 + @exp(lg));
        sc[e] = @sqrt(sp);
        sel[e] = sc[e] + bias[e];
    }
    @memset(picks, shared_id);
    @memset(wts, 1.0);
    var tot: f32 = 0.0;
    for (0..top_k) |k| {
        var best: f32 = -std.math.inf(f32);
        for (sel[0..ne]) |v| best = @max(best, v);
        var idx: usize = ne;
        for (0..ne) |e| if (sel[e] == best) {
            idx = e;
            break;
        };
        picks[k] = @intCast(idx);
        wts[k] = sc[idx];
        tot += sc[idx];
        sel[idx] = -std.math.inf(f32);
    }
    for (0..top_k) |k| wts[k] = wts[k] / (tot + 1e-20) * scale;
}

fn biasBytes(bias: []const f32, out: []u8) void {
    for (bias, 0..) |v, e| std.mem.writeInt(u32, out[4 * e ..][0..4], @bitCast(v), .little);
}

fn biasFloats(bytes: []const u8, out: []f32) void {
    for (out, 0..) |*v, e| v.* = @bitCast(std.mem.readInt(u32, bytes[4 * e ..][0..4], .little));
}

/// The reference that deletes the layer's masked experts: the router over the kept experts only (their logits and
/// biases, in index order), its picks mapped back to the full index.
fn routePruned(m: *const Mask, layer: usize, logits: []const f32, bias: []const f32, top_k: usize, scale: f32, shared_id: i32, picks: []i32, wts: []f32) void {
    var kl: [1024]f32 = undefined;
    var kb: [1024]f32 = undefined;
    var keep: [1024]usize = undefined;
    var n: usize = 0;
    for (0..m.experts) |e| if (!m.masked(layer, e)) {
        kl[n] = logits[e];
        kb[n] = bias[e];
        keep[n] = e;
        n += 1;
    };
    routeRow(kl[0..n], kb[0..n], top_k, scale, shared_id, picks, wts);
    for (0..top_k) |k| picks[k] = @intCast(keep[@intCast(picks[k])]);
}

const t_layers = 40;
const t_experts = 384;
const t_topk = 6;
const t_slots = 7; // top_k + the shared expert
const t_scale: f32 = 1.5;
const t_shared: i32 = 384;

fn expectSameRoute(a: []const i32, aw: []const f32, b: []const i32, bw: []const f32) !void {
    try std.testing.expectEqualSlices(i32, a, b);
    for (aw, bw) |x, y| try std.testing.expectEqual(@as(u32, @bitCast(x)), @as(u32, @bitCast(y)));
}

test "a mask file: route_join's header, 'layer:expert' and 'layer:e1,e2' lines, comments, blanks and repeats" {
    const gpa = std.testing.allocator;
    const text = "# route_join mask: suspects with q < 0.05, method=perm-fdr, permutations=1000, seed=7, min_attempts=20, strata=domain+task\n" ++
        "3:17\n\n  5:1, 2 ,9  # three of layer 5\r\n39:383\n3:17\n# the end\n";
    var bad: usize = 0;
    var m = try parse(gpa, text, t_layers, t_experts, t_topk, &bad);
    defer m.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 5), m.count);
    try std.testing.expect(m.masked(3, 17) and m.masked(5, 1) and m.masked(5, 2) and m.masked(5, 9) and m.masked(39, 383));
    try std.testing.expect(!m.masked(3, 16) and !m.masked(0, 17));
    try std.testing.expectEqual(@as(usize, 3), m.inLayer(5));
    // the same pairs in another order, other comments, one line: the same id
    var m2 = try parse(gpa, "39:383\n5:9,2,1,1 # x\n3:17", t_layers, t_experts, t_topk, &bad);
    defer m2.deinit(gpa);
    try std.testing.expectEqualStrings(m.id(), m2.id());
    // sha256("3:17\n5:1,2,9\n39:383\n"): strixflerp tools/ablate.py's test asserts the same id for the same pairs
    try std.testing.expectEqualStrings("7672079e14c6b86a", m.id());
    var m3 = try parse(gpa, "39:383\n5:9,2\n3:17", t_layers, t_experts, t_topk, &bad);
    defer m3.deinit(gpa);
    try std.testing.expect(!std.mem.eql(u8, m.id(), m3.id()));
    var buf: [32]u8 = undefined;
    const suf = planSuffix(&buf, &m);
    try std.testing.expect(std.mem.startsWith(u8, suf, " mask=") and suf.len == 22);
    try std.testing.expectEqualStrings(m.id(), suf[6..]);
}

test "a mask file is refused for a bad line, an MTP layer, the shared expert's id or a layer left below top_k" {
    const gpa = std.testing.allocator;
    var bad: usize = 0;
    const Case = struct { text: []const u8, err: anyerror, line: usize };
    const cases = [_]Case{
        .{ .text = "1:2\n40:1\n", .err = error.MaskLayerOutOfRange, .line = 2 }, // 40: the first DSpark/MTP block
        .{ .text = "1:384", .err = error.MaskExpertOutOfRange, .line = 1 }, // 384: the shared expert's slot
        .{ .text = "# h\n1:", .err = error.BadMaskLine, .line = 2 },
        .{ .text = "x:1", .err = error.BadMaskLine, .line = 1 },
        .{ .text = "1:2:3", .err = error.BadMaskLine, .line = 1 },
        .{ .text = "1:-1", .err = error.BadMaskLine, .line = 1 },
        .{ .text = "1:+2", .err = error.BadMaskLine, .line = 1 },
        .{ .text = "1:1_0", .err = error.BadMaskLine, .line = 1 },
        .{ .text = "17", .err = error.BadMaskLine, .line = 1 },
        .{ .text = "1:2,,3", .err = error.BadMaskLine, .line = 1 },
    };
    for (cases) |cs| {
        try std.testing.expectError(cs.err, parse(gpa, cs.text, t_layers, t_experts, t_topk, &bad));
        try std.testing.expectEqual(cs.line, bad);
    }
    // a layer keeping exactly top_k experts is taken; one fewer is refused
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(gpa);
    try list.appendSlice(gpa, "7:0");
    for (1..t_experts - t_topk) |e| try list.print(gpa, ",{d}", .{e});
    var m = try parse(gpa, list.items, t_layers, t_experts, t_topk, &bad);
    defer m.deinit(gpa);
    try std.testing.expectEqual(@as(usize, t_experts - t_topk), m.inLayer(7));
    try list.print(gpa, ",{d}", .{t_experts - t_topk});
    try std.testing.expectError(error.MaskLeavesTooFew, parse(gpa, list.items, t_layers, t_experts, t_topk, &bad));
    try std.testing.expectEqual(@as(usize, 0), bad);
}

test "an empty mask (no lines, or comments only) touches no bias byte and leaves the plan text stock" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5bb471d3b9);
    const r = prng.random();
    var bias: [t_experts]f32 = undefined;
    for (&bias) |*b| b.* = r.float(f32) * 2.0 - 1.0;
    var before: [t_experts * 4]u8 = undefined;
    biasBytes(&bias, &before);
    var bad: usize = 0;
    for ([_][]const u8{ "", "\n\n", "# route_join mask: suspects with q < 0.05, method=perm-fdr\n", "  # only a comment\r\n" }) |text| {
        var m = try parse(gpa, text, t_layers, t_experts, t_topk, &bad);
        defer m.deinit(gpa);
        try std.testing.expectEqual(@as(usize, 0), m.count);
        var buf: [32]u8 = undefined;
        try std.testing.expectEqualStrings("", planSuffix(&buf, &m));
        for (0..t_layers) |l| {
            var after = before;
            try m.applyBias(l, &after);
            try std.testing.expectEqualSlices(u8, &before, &after);
        }
    }
    var buf: [32]u8 = undefined;
    try std.testing.expectEqualStrings("", planSuffix(&buf, null));
}

test "masking expert set S routes exactly as the reference with S deleted: picks, renormalised weights, shared slot" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x28153e18f3);
    const r = prng.random();
    var bad: usize = 0;
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    var rows_checked: usize = 0;
    var hit_rows: usize = 0;
    for (0..24) |trial| {
        const layer = r.uintLessThan(usize, t_layers);
        var bias: [t_experts]f32 = undefined;
        var bias_vl: [t_experts]f32 = undefined;
        for (&bias, &bias_vl) |*b, *bv| {
            b.* = (r.float(f32) * 2.0 - 1.0) * 0.5;
            bv.* = (r.float(f32) * 2.0 - 1.0) * 0.5;
        }
        var rows: [8][t_experts]f32 = undefined;
        for (&rows) |*row| for (row) |*v| {
            v.* = (r.float(f32) * 2.0 - 1.0) * 6.0;
        };
        if (trial % 4 == 0) rows[1] = rows[0]; // a repeated row
        if (trial % 3 == 0) for (0..12) |e| {
            rows[2][e * 7] = 1.25; // ties in the selection score (equal logits and equal biases)
            bias[e * 7] = 0.0;
            bias_vl[e * 7] = 0.0;
        };
        // S: two of row 0's top picks (the case that changes routing) plus random experts; trial 0 masks all but a few
        var full_p: [t_slots]i32 = undefined;
        var full_w: [t_slots]f32 = undefined;
        routeRow(&rows[0], &bias, t_topk, t_scale, t_shared, &full_p, &full_w);
        text.clearRetainingCapacity();
        try text.print(gpa, "# trial {d}\n{d}:{d},{d}", .{ trial, layer, full_p[0], full_p[2] });
        const extra = if (trial == 0) t_experts else r.uintLessThan(usize, 120);
        var listed: usize = 2;
        for (0..extra) |i| {
            if (listed + 2 + t_topk >= t_experts) break;
            const e = if (trial == 0) i else r.uintLessThan(usize, t_experts);
            try text.print(gpa, ",{d}", .{e});
            listed += 1;
        }
        try text.append(gpa, '\n');
        var m = try parse(gpa, text.items, t_layers, t_experts, t_topk, &bad);
        defer m.deinit(gpa);
        try std.testing.expect(m.masked(layer, @intCast(full_p[0])));
        // the masked biases as the engine makes them: the f32 bytes with -inf written over S
        var bb: [t_experts * 4]u8 = undefined;
        var masked_b: [t_experts]f32 = undefined;
        var masked_vl: [t_experts]f32 = undefined;
        biasBytes(&bias, &bb);
        try m.applyBias(layer, &bb);
        biasFloats(&bb, &masked_b);
        biasBytes(&bias_vl, &bb);
        try m.applyBias(layer, &bb);
        biasFloats(&bb, &masked_vl);
        for (0..t_experts) |e| if (!m.masked(layer, e)) {
            try std.testing.expectEqual(@as(u32, @bitCast(bias[e])), @as(u32, @bitCast(masked_b[e])));
        } else try std.testing.expect(masked_b[e] == -std.math.inf(f32) and masked_vl[e] == -std.math.inf(f32));
        const Pair = struct { m: []const f32, o: []const f32 };
        for (&rows) |*row| for ([_]Pair{ .{ .m = &masked_b, .o = &bias }, .{ .m = &masked_vl, .o = &bias_vl } }) |pair| {
            var mp: [t_slots]i32 = undefined;
            var mw: [t_slots]f32 = undefined;
            var pp: [t_slots]i32 = undefined;
            var pw: [t_slots]f32 = undefined;
            routeRow(row, pair.m, t_topk, t_scale, t_shared, &mp, &mw);
            routePruned(&m, layer, row, pair.o, t_topk, t_scale, t_shared, &pp, &pw);
            try expectSameRoute(&mp, &mw, &pp, &pw);
            var sum: f32 = 0;
            for (0..t_topk) |k| {
                try std.testing.expect(!m.masked(layer, @intCast(mp[k])));
                sum += mw[k];
            }
            try std.testing.expectApproxEqAbs(t_scale, sum, 1e-5);
            try std.testing.expectEqual(t_shared, mp[t_topk]);
            try std.testing.expectEqual(@as(f32, 1.0), mw[t_topk]);
            var up: [t_slots]i32 = undefined;
            var uw: [t_slots]f32 = undefined;
            routeRow(row, pair.o, t_topk, t_scale, t_shared, &up, &uw);
            if (!std.mem.eql(i32, &up, &mp)) hit_rows += 1;
            rows_checked += 1;
        };
        // a layer the mask leaves alone keeps its bias bytes (routes as the base)
        const other = (layer + 1) % t_layers;
        biasBytes(&bias, &bb);
        var copy = bb;
        try m.applyBias(other, &copy);
        try std.testing.expectEqualSlices(u8, &bb, &copy);
    }
    try std.testing.expectEqual(@as(usize, 24 * 8 * 2), rows_checked);
    try std.testing.expect(hit_rows >= 24); // every trial's row 0 lost two of its picks under gate.bias
}
