//! Per-token logprobs: what a request asking OpenAI `logprobs` gets for each reply token. A
//! backend that draws on the host puts each verify (or first-token) row in the stream's Ring, keyed by the absolute
//! position the row draws at, from the same logits row its sampler reads; the lane host reads the ring for the tokens
//! it delivers. The distribution is the model's own: log_softmax of the raw logits (temperature 1, before temperature,
//! top_k, top_p, min_p or a grammar shape the draw), as OpenAI reports it, greedy replies included.
const std = @import("std");

/// OpenAI's limit on top_logprobs.
pub const max_top = 20;

/// A token's logprob and its row's best alternatives, best first (of equal logits the lower id first).
pub const TokenLogprob = struct {
    token: u32,
    /// NaN: no row of the stream drew at the token's position, or the token is a forced one its row does not list.
    logprob: f32,
    n: u8 = 0,
    ids: [max_top]u32 = undefined,
    logprobs: [max_top]f32 = undefined,
};

/// One row's record: the drawn token, the token the window fed after it (a draft or a forced token; `none`: the
/// window's last row), each one's logprob, and the row's top `n`.
pub const Row = struct {
    position: u64 = std.math.maxInt(u64),
    drawn: u32 = 0,
    drawn_lp: f32 = 0,
    next: u32 = none,
    next_lp: f32 = 0,
    n: u8 = 0,
    ids: [max_top]u32 = undefined,
    logprobs: [max_top]f32 = undefined,
};

pub const none = std.math.maxInt(u32);

/// A token's logprob from a row's max and log-sum-exp, as f32 (-inf, a masked logit, as the smallest f32: JSON has
/// no infinity).
fn lpOf(x: f32, lse: f64) f32 {
    const v: f32 = @floatCast(@as(f64, x) - lse);
    return if (std.math.isNegativeInf(v)) -std.math.floatMax(f32) else v;
}

/// a before b: the larger logit, then the lower id (the samplers' order).
fn before(va: f32, ia: u32, vb: f32, ib: u32) bool {
    return va > vb or (va == vb and ia < ib);
}

/// The row's log-sum-exp over its finite logits (f64), its best `k` (a partial insertion sort, k <= max_top).
pub fn record(row: []const f32, k: u8, drawn: u32, next: u32) Row {
    var mx: f32 = -std.math.inf(f32);
    for (row) |v| if (v > mx) {
        mx = v;
    };
    var sum: f64 = 0;
    for (row) |v| sum += @exp(@as(f64, v) - @as(f64, mx));
    const lse = @as(f64, mx) + @log(sum);
    var r: Row = .{ .drawn = drawn, .next = next };
    if (drawn < row.len) r.drawn_lp = lpOf(row[drawn], lse) else r.drawn_lp = std.math.nan(f32);
    if (next != none) r.next_lp = if (next < row.len) lpOf(row[next], lse) else std.math.nan(f32);
    const want: usize = @min(k, max_top, row.len);
    var vals: [max_top]f32 = undefined;
    var n: usize = 0;
    if (want > 0) for (row, 0..) |v, i| {
        const id: u32 = @intCast(i);
        if (n == want and !before(v, id, vals[want - 1], r.ids[want - 1])) continue;
        var j: usize = if (n < want) n else want - 1;
        while (j > 0 and before(v, id, vals[j - 1], r.ids[j - 1])) : (j -= 1) {
            vals[j] = vals[j - 1];
            r.ids[j] = r.ids[j - 1];
        }
        vals[j] = v;
        r.ids[j] = id;
        if (n < want) n += 1;
    };
    r.n = @intCast(n);
    for (0..n) |j| r.logprobs[j] = lpOf(vals[j], lse);
    return r;
}

/// A stream's rows by position. A round writes only positions past the stream's committed tokens, and the lane host
/// reads a round's tokens before the next round runs, so a committed token's slot holds the row that drew it.
pub const Ring = struct {
    top: u8,
    rows: [slots]Row = @splat(.{}),

    pub const slots = 256;

    /// A row drawn at `position` (its logits row, the token drawn and the token fed after it).
    pub fn put(ring: *Ring, position: u64, row: []const f32, drawn: u32, next: u32) void {
        var r = record(row, ring.top, drawn, next);
        r.position = position;
        ring.rows[position % slots] = r;
    }

    /// The committed `token` at `position`: the drawn token's logprob, else the fed one's (an accepted forced token),
    /// else its entry in the row's top; NaN when none of those is it or no row drew there.
    pub fn get(ring: *const Ring, position: u64, token: u32) TokenLogprob {
        const r = &ring.rows[position % slots];
        var out: TokenLogprob = .{ .token = token, .logprob = std.math.nan(f32) };
        if (r.position != position) return out;
        out.n = r.n;
        out.ids = r.ids;
        out.logprobs = r.logprobs;
        if (token == r.drawn) {
            out.logprob = r.drawn_lp;
        } else if (token == r.next) {
            out.logprob = r.next_lp;
        } else for (r.ids[0..r.n], r.logprobs[0..r.n]) |id, lp| if (id == token) {
            out.logprob = lp;
        };
        return out;
    }
};

test "record: log_softmax of the raw row, the best k by logit then id" {
    const row = [_]f32{ 1, 3, 3, -std.math.inf(f32), 0.5 };
    const r = record(&row, 3, 4, 0);
    var sum: f64 = 0;
    for (row) |v| sum += @exp(@as(f64, v) - 3.0);
    const lse = 3.0 + @log(sum);
    try std.testing.expectEqual(@as(u8, 3), r.n);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 0 }, r.ids[0..3]);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(3.0 - lse)), r.logprobs[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(1.0 - lse)), r.logprobs[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(0.5 - lse)), r.drawn_lp, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(1.0 - lse)), r.next_lp, 1e-6);
    // probabilities sum to 1 (the masked logit adds nothing)
    var p: f64 = 0;
    for (row) |v| p += @exp(@as(f64, v) - lse);
    try std.testing.expectApproxEqAbs(@as(f64, 1), p, 1e-9);
    // a masked token: the smallest f32, never -inf
    try std.testing.expectEqual(-std.math.floatMax(f32), record(&row, 0, 3, none).drawn_lp);
    // k = 0: no alternatives; k past the row: the whole row
    try std.testing.expectEqual(@as(u8, 0), record(&row, 0, 1, none).n);
    try std.testing.expectEqual(@as(u8, 5), record(&row, 20, 1, none).n);
}

test "ring: the slot of a committed position holds the row that drew it; a later round overwrites rejected rows" {
    var ring: Ring = .{ .top = 2 };
    const a = [_]f32{ 0, 5, 1 };
    const b = [_]f32{ 4, 0, 1 };
    ring.put(10, &a, 1, 0); // drew 1, the window fed draft 0 after it
    ring.put(11, &a, 2, none); // a rejected row past the draft
    ring.put(11, &b, 0, none); // the next round draws position 11 again
    try std.testing.expectEqual(ring.rows[10].drawn_lp, ring.get(10, 1).logprob);
    try std.testing.expectEqual(ring.rows[10].next_lp, ring.get(10, 0).logprob); // a forced token fed after the row
    try std.testing.expectEqual(ring.rows[10].logprobs[1], ring.get(10, 2).logprob); // in the row's top
    try std.testing.expectEqual(ring.rows[11].drawn_lp, ring.get(11, 0).logprob);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2 }, ring.get(11, 0).ids[0..2]);
    try std.testing.expect(std.math.isNan(ring.get(12, 0).logprob)); // no row drew there
    try std.testing.expect(std.math.isNan(ring.get(10 + Ring.slots, 1).logprob)); // a slot another position holds
}
