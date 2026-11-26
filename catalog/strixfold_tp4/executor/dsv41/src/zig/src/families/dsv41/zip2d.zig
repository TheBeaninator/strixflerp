//! The world-4 prompt zip (model.fillSplit2d), as a list of steps a host test can walk.
//!
//! fillWhole's order, per half: Engram (when the layer has one), attention mixes, attention through wo_a, wo_b
//! (catRank), the attention quarters, FFN mixes and expert gate/up, expert down (catRank), the MoE quarters.
//! The zip keeps that order on each half. Across halves it lets one half's quarters move while the other half
//! does work that does not call Coll and does not touch the in-flight send or the parts buffer: the other half's
//! attention prefix (mixes, wo_a, the ring and the kv source; no collective), or its FFN mixes and gate/up.
//! A begun gather is ended before any catRank, Engram exchange, or other Coll call. TP2's fillSplit is a different
//! list (its attention and MoE have no collective) and is not described here.
const std = @import("std");

pub const Step = enum {
    end_b_moe, // end half B's MoE quarters from the previous layer (no-op when none is open)
    engram_a,
    engram_b,
    attn_mix_a,
    attn_prefix_a,
    attn_suffix_a, // catRank + wo_b
    mid_save_a, // zip2: the first chunk's layer state, after its attention has written it
    begin_attn_a,
    attn_mix_b,
    attn_prefix_b,
    end_attn_a,
    attn_suffix_b,
    begin_attn_b,
    ffn_up_a, // FFN mixes + routing + gate/up
    end_attn_b,
    moe_down_a,
    begin_moe_a,
    ffn_up_b,
    end_moe_a,
    end_layer_a,
    moe_down_b,
    begin_moe_b,
    end_layer_b,
    replay_a, // replayCut of the first half, while B's last MoE quarters may still be open
    mid_tail, // zip2: decoder-layer state before the second half's cut writes it
    replay_b,
    // the second half alone past the cut, the same calls as fillWhole (a gather opens and closes in one step)
    dec_mix,
    dec_prefix,
    dec_suffix,
    dec_attn_gather,
    dec_ffn_up,
    dec_down,
    dec_moe_gather,
    dec_end,
    head,
};

/// Where node g's part lands in quarters' destination [2, n, w0 + w1]: rank row g % 2, columns at pair g / 2.
pub const QuarterAt = struct {
    src: usize, // byte offset in parts [4, n, wmax]
    dst: usize, // byte offset in the destination
    src_stride: usize,
    dst_stride: usize,
    nbytes: usize, // the pair's own width (the gather row is padded to wmax)
};

pub fn quarterAt(g: usize, n: usize, w: [2]usize, esize: usize) QuarterAt {
    const wmax = @max(w[0], w[1]);
    const rr = g % 2;
    const pg = g / 2;
    const row = (w[0] + w[1]) * esize;
    return .{
        .src = g * n * wmax * esize,
        .dst = rr * n * row + pg * w[0] * esize,
        .src_stride = wmax * esize,
        .dst_stride = row,
        .nbytes = w[pg] * esize,
    };
}

/// Bytes one node sends for a quarters or catRank all-gather: n rows by the wider pair, esize bytes an element.
pub fn partBytes(n: usize, w: [2]usize, esize: usize) usize {
    return n * @max(w[0], w[1]) * esize;
}

/// One pre-cut layer. `engram`: the layer runs Engram before the mixes. `unit`: zip2 saves the first half's state.
pub fn layerActs(buf: []Step, engram: bool, unit: bool) []const Step {
    var n: usize = 0;
    if (engram) {
        buf[n] = .end_b_moe;
        n += 1;
        buf[n] = .engram_a;
        n += 1;
        buf[n] = .engram_b;
        n += 1;
    }
    const body = [_]Step{
        .attn_mix_a,   .attn_prefix_a, .end_b_moe,     .attn_suffix_a, .mid_save_a, .begin_attn_a,
        .attn_mix_b,   .attn_prefix_b, .end_attn_a,    .attn_suffix_b, .begin_attn_b, .ffn_up_a,
        .end_attn_b,   .moe_down_a,    .begin_moe_a,   .ffn_up_b,      .end_moe_a, .end_layer_a,
        .moe_down_b,   .begin_moe_b,   .end_layer_b,
    };
    for (body) |s| {
        if (s == .mid_save_a and !unit) continue;
        buf[n] = s;
        n += 1;
    }
    return buf[0..n];
}

fn isColl(s: Step) bool {
    return switch (s) {
        .attn_suffix_a, .attn_suffix_b, .moe_down_a, .moe_down_b, .engram_a, .engram_b, .dec_suffix, .dec_down, .dec_attn_gather, .dec_moe_gather, .head => true,
        else => false,
    };
}

const Open = enum { attn_a, attn_b, moe_a, moe_b };

/// Walk a from-start prompt. `decoder`: the second half holds rows at or past the replay cut, so it runs the
/// decoder layers (fillWhole's tail). False: both halves are encoder-only and the walk stops at the cut.
pub fn walk(layers: usize, cut: usize, engram_at: []const usize, unit: bool, decoder: bool) !void {
    var open: ?Open = null;
    var a_gm = true; // no MoE gather is pending at the first layer
    var b_gm = true;
    var a_ga = false;
    var b_ga = false;
    var a_prefixed = false;
    var buf: [40]Step = undefined;
    for (0..cut) |li| {
        var engram = false;
        for (engram_at) |e| engram = engram or e == li;
        a_prefixed = false;
        a_ga = false;
        b_ga = false;
        for (layerActs(&buf, engram, unit)) |s| {
            try step(&open, &a_gm, &b_gm, &a_ga, &b_ga, &a_prefixed, s);
        }
    }
    // the cut: A's replayCut may overlap B's last MoE quarters; B's waits until they have landed
    try step(&open, &a_gm, &b_gm, &a_ga, &b_ga, &a_prefixed, .replay_a);
    try step(&open, &a_gm, &b_gm, &a_ga, &b_ga, &a_prefixed, .end_b_moe);
    if (unit) try step(&open, &a_gm, &b_gm, &a_ga, &b_ga, &a_prefixed, .mid_tail);
    try step(&open, &a_gm, &b_gm, &a_ga, &b_ga, &a_prefixed, .replay_b);
    if (!decoder) {
        if (open != null) return error.GatherLeftOpen;
        return;
    }
    for (cut..layers) |_| {
        a_ga = false;
        b_ga = false;
        for ([_]Step{ .dec_mix, .dec_prefix, .dec_suffix, .dec_attn_gather, .dec_ffn_up, .dec_down, .dec_moe_gather, .dec_end }) |s| {
            try step(&open, &a_gm, &b_gm, &a_ga, &b_ga, &a_prefixed, s);
        }
    }
    try step(&open, &a_gm, &b_gm, &a_ga, &b_ga, &a_prefixed, .head);
    if (open != null) return error.GatherLeftOpen;
}

fn step(open: *?Open, a_gm: *bool, b_gm: *bool, a_ga: *bool, b_ga: *bool, a_prefixed: *bool, s: Step) !void {
    if (open.* != null and isColl(s)) return error.CollDuringGather;
    switch (s) {
        .end_b_moe => {
            if (open.*) |o| {
                if (o != .moe_b) return error.EndedOther;
                open.* = null;
            }
            b_gm.* = true;
        },
        .end_attn_a => {
            if (open.*) |o| if (o != .attn_a) return error.EndedOther;
            if (open.* == null) return error.EndedNothing;
            open.* = null;
            a_ga.* = true;
        },
        .end_attn_b => {
            if (open.*) |o| if (o != .attn_b) return error.EndedOther;
            if (open.* == null) return error.EndedNothing;
            open.* = null;
            b_ga.* = true;
        },
        .end_moe_a => {
            if (open.*) |o| if (o != .moe_a) return error.EndedOther;
            if (open.* == null) return error.EndedNothing;
            open.* = null;
            a_gm.* = true;
        },
        .begin_attn_a => try begin(open, .attn_a),
        .begin_attn_b => try begin(open, .attn_b),
        .begin_moe_a => {
            try begin(open, .moe_a);
            a_gm.* = false;
        },
        .begin_moe_b => {
            try begin(open, .moe_b);
            b_gm.* = false;
        },
        .engram_a, .attn_mix_a, .replay_a => if (!a_gm.*) return error.GmNotReady,
        .engram_b, .attn_mix_b, .replay_b, .dec_mix, .head => if (!b_gm.*) return error.GmNotReady,
        .ffn_up_a => if (!a_ga.*) return error.GaNotReady,
        .ffn_up_b => if (!b_ga.*) return error.GaNotReady,
        .dec_ffn_up => if (!b_ga.*) return error.GaNotReady,
        .attn_prefix_a => a_prefixed.* = true,
        .attn_prefix_b, .dec_prefix => if (!a_prefixed.* and s == .attn_prefix_b) return error.RingOrder,
        .dec_attn_gather => b_ga.* = true,
        .dec_moe_gather => b_gm.* = true,
        .mid_save_a => if (!a_prefixed.*) return error.SaveBeforeAttention,
        else => {},
    }
}

fn begin(open: *?Open, o: Open) !void {
    if (open.* != null) return error.TwoGathers;
    open.* = o;
}

/// The math steps of one half, gathers dropped: fillWhole's order.
fn mathIs(acts: []const Step, half_a: bool, engram: bool) bool {
    const a = [_]Step{ .engram_a, .attn_mix_a, .attn_prefix_a, .attn_suffix_a, .ffn_up_a, .moe_down_a, .end_layer_a };
    const b = [_]Step{ .engram_b, .attn_mix_b, .attn_prefix_b, .attn_suffix_b, .ffn_up_b, .moe_down_b, .end_layer_b };
    const want = if (half_a) a[if (engram) 0 else 1 ..] else b[if (engram) 0 else 1 ..];
    var i: usize = 0;
    for (acts) |s| {
        const mine = switch (s) {
            .engram_a, .attn_mix_a, .attn_prefix_a, .attn_suffix_a, .ffn_up_a, .moe_down_a, .end_layer_a => half_a,
            .engram_b, .attn_mix_b, .attn_prefix_b, .attn_suffix_b, .ffn_up_b, .moe_down_b, .end_layer_b => !half_a,
            else => false,
        };
        if (!mine) continue;
        if (i >= want.len or want[i] != s) return false;
        i += 1;
    }
    return i == want.len;
}

test "quarters: rank row g % 2, columns at pair g / 2, inside [2, n, w0 + w1]" {
    const n: usize = 4;
    const es: usize = 4;
    const row = (2560 + 2560) * es;
    const dst = [_]usize{ 0, n * row, 2560 * es, n * row + 2560 * es };
    const src = [_]usize{ 0, n * 2560 * es, 2 * n * 2560 * es, 3 * n * 2560 * es };
    for (0..4) |g| {
        const at = quarterAt(g, n, .{ 2560, 2560 }, es);
        try std.testing.expectEqual(dst[g], at.dst);
        try std.testing.expectEqual(src[g], at.src);
        try std.testing.expectEqual(2560 * es, at.nbytes);
        try std.testing.expectEqual(row, at.dst_stride);
    }
    // the head: pair 0 holds 32,384 columns, pair 1 holds 32,256, one row
    const hw = [2]usize{ 32384, 32256 };
    const wide = 32384 * 4;
    const head_row = (32384 + 32256) * 4;
    for (0..4) |g| {
        const at = quarterAt(g, 1, hw, 4);
        const pg = g / 2;
        const rr = g % 2;
        try std.testing.expectEqual(rr * head_row + pg * wide, at.dst);
        try std.testing.expectEqual(hw[pg] * 4, at.nbytes);
        try std.testing.expectEqual(g * wide, at.src);
    }
    try std.testing.expectEqual(@as(usize, 2048 * 2560 * 4), partBytes(2048, .{ 2560, 2560 }, 4));
    try std.testing.expectEqual(@as(usize, 2048 * 7 * 640 * 2), partBytes(2048 * 7, .{ 640, 512 }, 2));
}

test "each half's math steps are fillWhole's, and no collective runs during a begun gather" {
    var buf: [40]Step = undefined;
    for ([_]bool{ false, true }) |engram| {
        for ([_]bool{ false, true }) |unit| {
            const acts = layerActs(&buf, engram, unit);
            try std.testing.expect(mathIs(acts, true, engram));
            try std.testing.expect(mathIs(acts, false, engram));
            if (unit) {
                var saw_suffix = false;
                var saw_save = false;
                var saw_b = false;
                for (acts) |s| {
                    if (s == .attn_suffix_a) saw_suffix = true;
                    if (s == .mid_save_a) {
                        try std.testing.expect(saw_suffix);
                        saw_save = true;
                    }
                    if (s == .attn_prefix_b) {
                        try std.testing.expect(!unit or saw_save);
                        saw_b = true;
                    }
                }
                try std.testing.expect(saw_b);
            }
        }
    }
    const engram_at = [_]usize{ 1, 14 };
    try walk(40, 20, &engram_at, false, false);
    try walk(40, 20, &engram_at, true, false);
    try walk(40, 20, &engram_at, false, true);
    try walk(40, 20, &engram_at, true, true);
    // a carried MoE gather is closed before the next layer's Engram exchange
    try walk(4, 4, &[_]usize{0}, false, false);
}
