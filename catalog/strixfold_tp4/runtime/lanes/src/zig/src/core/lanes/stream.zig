//! One stream (Python LaneStream) and the round loop's state for it (Python's dicts keyed by stream id).
const std = @import("std");
const shape = @import("shape.zig");
const plan_lanes = @import("plan_lanes.zig");
const image_key = @import("image_key.zig");
const acts_mod = @import("acts.zig");
const logprobs_mod = @import("logprobs.zig");
const Allocator = std.mem.Allocator;
const Proposer = @import("proposer.zig").Proposer;
const Sampling = @import("sampling.zig").Sampling;
const State = @import("depth.zig").State;
const Structure = @import("grammar/grammar.zig").Structure;

pub const Reason = enum {
    none,
    stop,
    length,
    @"error",
    cancelled,
    loop, // the loop policy `stop` ended the reply where an exact cycle was found

    pub fn name(r: Reason) []const u8 {
        return if (r == .none) "" else @tagName(r);
    }
};

pub const Mode = enum { pipe, drain, verify, exit };

// the longest exact cycle the loop detector looks for
pub const loop_max_period: usize = 64;
pub const loop_min_run: usize = 256;
pub const loop_warm_in: usize = 64;

/// Return the shortest exact cycle at the end of a combined token prefix.
fn cycleAt(head: []const u32, tail: []const u32, n: usize) ?u32 {
    if (n < loop_min_run + 1) return null;
    for (1..loop_max_period + 1) |period| {
        if (n < loop_min_run + period) break;
        if (n - loop_min_run - period < loop_warm_in) break;
        var matches = true;
        for (1..loop_min_run + 1) |k| {
            const right = n - k;
            const left = right - period;
            const right_token = if (right < head.len) head[right] else tail[right - head.len];
            const left_token = if (left < head.len) head[left] else tail[left - head.len];
            if (right_token != left_token) {
                matches = false;
                break;
            }
        }
        if (matches) return @intCast(period);
    }
    return null;
}

/// What a thinking loop does to the reply. Every policy but `off` detects and reports (Stream.loop, Stream.answer_loop);
/// `close` forces `think_close` so the model answers (the old --loop-guard), `stop` ends the reply at once with
/// Reason.loop, `count` never acts. A loop in the answer (after `think_end`) is counted under every policy, never acted on.
pub const LoopPolicy = enum {
    off,
    close,
    stop,
    count,

    /// Whether the policy changes the reply when a thinking loop is found.
    pub fn acts(p: LoopPolicy) bool {
        return p == .close or p == .stop;
    }

    pub fn parse(text: []const u8) ?LoopPolicy {
        return std.meta.stringToEnum(LoopPolicy, text);
    }
};

/// Where a loop was found: its period and the emitted index (0-based) of the token that completed the cycle's run.
pub const LoopHit = struct { period: u32, at_token: u32 };

/// Return the first token index in a candidate batch that would fire an acting policy (close, stop).
pub fn loopCut(s: *const Stream, tokens: []const u32) ?usize {
    if (!s.loop_policy.acts() or !s.think_open or s.think_end < 0 or s.loop != null) return null;
    for (tokens, 0..) |token, i| {
        if (@as(i64, token) == s.think_end) return null;
        if (cycleAt(s.emitted(), tokens, s.emitted().len + i + 1) != null) return i;
    }
    return null;
}

/// Return the detected period at the end of committed tokens, or null.
pub fn detectLoop(emitted: []const u32) ?u32 {
    return cycleAt(emitted, &.{}, emitted.len);
}

/// A stop-string check over the emitted tokens (the server's StopPolicy decodes their tail).
pub const StopCheck = struct {
    ptr: *anyopaque,
    check: *const fn (ptr: *anyopaque, emitted: []const u32) bool,
};

/// The host's answer to whether the stream's request was cancelled, asked between prompt chunks.
pub const CancelCheck = struct {
    ptr: *anyopaque,
    check: *const fn (ptr: *anyopaque) bool,
};

/// A kept prompt state the backend restores before its prompt pass (core/prompt_cache.zig), and where it keeps new ones.
pub const Reuse = struct {
    saved: ?*anyopaque = null, // the backend's copy of the state after `at` prompt tokens (null: start at 0)
    at: u32 = 0,
    marks: []const u32 = &.{}, // ascending: the pass cuts a chunk at each and calls `hook` there
    hook: ?MarkHook = null,
};

/// Called between prompt chunks, the pass standing at `at` (the GPU done with the chunk that ends there).
pub const MarkHook = struct {
    ptr: *anyopaque,
    at: *const fn (ptr: *anyopaque, s: *Stream, at: u32) void,
};

/// Drafts the backend holds for the stream's next round; a tree keeps its tokens and parents on the host.
pub const Held = struct {
    count: u32,
    tokens: ?[]u32 = null,
    parents: ?[]i32 = null,
};

/// An image in the prompt (a vision family's): the span of `tokens` positions from `at` it fills (each position carries
/// the image token id), and the payload its family's Vision.prepare made, or with `rows` its span's embedding rows
/// ready (a gate's input).
pub const Image = struct {
    at: u32,
    tokens: u32,
    bytes: []const u8,
    rows: bool = false,
    digest: ?image_key.Digest = null, // the picture's identity for the prompt caches (null: not taken yet)
};

/// A vision family's image input as a server renders it (engine_api.Info.vision): each image part becomes
/// `placeholder` (one token: `token`) in the template's text, and `prepare` turns the image's bytes into its span's
/// length and the payload the backend reads (Image.bytes), on the request's thread; on BadImage `problem` says why.
pub const Vision = struct {
    ctx: *anyopaque,
    placeholder: []const u8,
    token: u32,
    prepare: *const fn (ctx: *anyopaque, a: Allocator, bytes: []const u8, problem: *[]const u8) PrepareError!Prepared,

    /// A tower that encodes off the engine thread (HIP's); null: the prompt pass runs it.
    encode: ?Encode = null,
    /// The most image tokens one request may hold (the engine's reserved rows); 0: no limit.
    max_tokens: u32 = 0,
    /// The most decoded pixels one request's images may hold together (Prepared.pixels summed); 0: no limit.
    max_pixels: u64 = 0,

    /// A picture's span length, its payload, and the pixels its decode held (0: not counted).
    pub const Prepared = struct { tokens: u32, payload: []const u8, pixels: u64 = 0 };
    pub const PrepareError = error{ BadImage, OutOfMemory };
};

/// Off-thread encodes: `start` at submit, `ready` before admission (false: it waits), `forget` once the rows are taken.
pub const Encode = struct {
    ctx: *anyopaque,
    // spans that end by `from` (where the prompt cache would resume) are encoded only if the prompt pass asks for them
    start: *const fn (ctx: *anyopaque, images: []const Image, from: u32, wake: Wake) void,
    ready: *const fn (ctx: *anyopaque, images: []const Image) bool,
    forget: *const fn (ctx: *anyopaque, images: []const Image) void,
};

/// Called on the encoder's thread once a request's rows are ready, with none of its own locks held.
pub const Wake = struct { ctx: *anyopaque, call: *const fn (ctx: *anyopaque) void };

pub const Spec = struct {
    id: []const u8,
    prompt: []const u32,
    max_new: u32,
    eos: []const u32 = &.{},
    sampling: ?Sampling = null,
    drafts: bool = true,
    proposer: ?Proposer = null,
    stop_check: ?StopCheck = null,
    cancel_check: ?CancelCheck = null,
    think_budget: u32 = 0,
    think_close: []const u32 = &.{},
    think_end: i64 = -1,
    think_open: ?bool = null, // null: open when a budget is set (the server's rule)
    loop_policy: LoopPolicy = .off,
    chunks: []const u32 = &.{}, // where prefill chunks start after 0 (Python's PrefillPlan); empty: the backend's step
    reuse: Reuse = .{},
    images: []const Image = &.{}, // in prompt order; the caller's until the stream is released
    cache_key: []const u32 = &.{}, // with images: the ids the prompt caches key it by (empty: none kept or resumed)
    cache_spans: []const image_key.KeySpan = &.{}, // with images: each picture's span and whole digest
    structure: ?Structure = null, // the grammar every reply token keeps to (a backend's draws enforce it)
    reap_bucket: u32 = 0, // Edge-REAP domain index (0: "default"). The text bucket byte is index * 2.
    route_record: []const u8 = "", // routing record id (empty: none), the client's tag and the reply id
    route_tag: []const u8 = "",
    route_request: []const u8 = "",
    acts: acts_mod.Spec = .{}, // per-request dumps and row injection; off: none
    logprobs: ?u8 = null, // per-token logprobs with this many alternatives; null: none kept
};

pub const Stream = struct {
    id: []const u8,
    prompt_len: usize,
    max_new: u32,
    eos: []const u32,
    sampling: ?Sampling,
    drafts: bool,
    proposer: ?Proposer,
    stop_check: ?StopCheck,
    cancel_check: ?CancelCheck,
    think_budget: u32,
    think_close: []const u32,
    think_end: i64,
    think_open: bool,
    loop_policy: LoopPolicy,
    loop: ?LoopHit = null, // the thinking loop found (latched once)
    answer_loop: ?LoopHit = null, // a loop in the answer, counted only (latched once)
    answer_from: ?usize = null, // emitted index where the answer starts (after think_end; 0 when thinking never opened)
    chunks: []const u32,
    reuse: Reuse = .{},
    images: []const Image = &.{},
    cache_key: []const u32 = &.{},
    cache_spans: []const image_key.KeySpan = &.{},
    structure: ?Structure = null,
    reap_bucket: u32 = 0,
    route_record: []const u8 = "",
    route_tag: []const u8 = "",
    route_request: []const u8 = "",
    acts: acts_mod.Spec = .{},
    logprobs: ?*logprobs_mod.Ring = null, // the rows a host-drawing backend keeps for the reply's logprobs
    cached: u32 = 0, // prompt tokens the backend restored from `reuse` (its prompt pass started there)
    reuse_failed: bool = false, // the backend's restore of `reuse` failed: it prefilled from 0
    context: std.ArrayList(u32) = .empty,
    pending: ?u32 = null,
    force: std.ArrayList(u32) = .empty,
    cache_len: u64 = 0,
    finished: bool = false,
    reason: Reason = .none,
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    branch_rows: u64 = 0, // suffix-match branch rows verified (lanes/fill.zig)
    // the backend's own counters for this request as one JSON object (written at release; the reply's telemetry)
    telemetry_buf: [3072]u8 = undefined,
    telemetry_len: usize = 0,
    branch_accepted: u64 = 0, // of those, rows on the accepted path
    graft_hit: f64 = 1, // recent share of grafted rows kept; a read of the chain that grafted none counts as a miss
    odds: ?shape.Odds = null, // how often the target took the head's rank-r lane at each depth (head trees)
    lanes: ?shape.Shape = null, // the shape of the held tree drafts
    landed: [shape.max_depth][shape.ranks][2]u32 = @splat(@splat(.{ 0, 0 })), // tree lanes verified and taken
    picks: [plan_lanes.sizes.len + 1]u32 = @splat(0), // the planner's picks: each tree size, then chain + grafts
    graft_room: ?u32 = null, // grafted rows the planner left this round's window (null: no planner)
    held_levels: u32 = 0, // head levels drafted for the held window
    round_over: ?f64 = null, // the stream's round time beyond its window and head levels (ms, recent): chains
    tree_over: ?f64 = null, // and trees
    measured: [plan_lanes.sizes.len + 1]plan_lanes.Measured = @splat(.{}), // each pick's round times past its table price
    priced: ?struct { choice: usize, ms: f64 } = null, // the pick that shaped the held window, at its table price
    min_rows: u32 = 0,
    paused: bool = false,
    depth: ?State = null,
    mode: ?Mode = null,
    next: ?Held = null,
    inflight: ?u64 = null,
    served: ?u64 = null,
    granted: ?u32 = null,
    copy_width: ?u32 = null,

    pub fn init(gpa: Allocator, spec: Spec) !Stream {
        var s: Stream = .{
            .id = spec.id,
            .prompt_len = spec.prompt.len,
            .max_new = spec.max_new,
            .eos = spec.eos,
            .sampling = spec.sampling,
            .drafts = spec.drafts,
            .proposer = if (spec.drafts) spec.proposer else null,
            .stop_check = spec.stop_check,
            .cancel_check = spec.cancel_check,
            .think_budget = spec.think_budget,
            .think_close = spec.think_close,
            .think_end = spec.think_end,
            .think_open = spec.think_open orelse (spec.think_budget > 0 or (spec.loop_policy != .off and spec.think_end >= 0)),
            .loop_policy = spec.loop_policy,
            .chunks = spec.chunks,
            .reuse = spec.reuse,
            .images = spec.images,
            .cache_key = spec.cache_key,
            .cache_spans = spec.cache_spans,
            .structure = spec.structure,
            .reap_bucket = spec.reap_bucket,
            .route_record = spec.route_record,
            .route_tag = spec.route_tag,
            .route_request = spec.route_request,
            .acts = spec.acts,
        };
        if (!s.think_open) s.answer_from = 0;
        if (spec.logprobs) |top| {
            const ring = try gpa.create(logprobs_mod.Ring);
            ring.* = .{ .top = top };
            s.logprobs = ring;
        }
        errdefer if (s.logprobs) |ring| gpa.destroy(ring);
        try s.context.appendSlice(gpa, spec.prompt);
        return s;
    }

    pub fn deinit(s: *Stream, gpa: Allocator) void {
        if (s.logprobs) |ring| gpa.destroy(ring);
        s.logprobs = null;
        s.context.deinit(gpa);
        s.force.deinit(gpa);
        s.dropRoundState(gpa);
    }

    /// Forget the round loop's state (Python `_release_stream_state`).
    pub fn dropRoundState(s: *Stream, gpa: Allocator) void {
        if (s.depth) |*d| d.deinit(gpa);
        s.dropHeld(gpa);
        s.depth = null;
        s.mode = null;
        s.inflight = null;
        s.served = null;
        s.granted = null;
        s.copy_width = null;
    }

    pub fn dropHeld(s: *Stream, gpa: Allocator) void {
        if (s.next) |h| {
            if (h.tokens) |t| gpa.free(t);
            if (h.parents) |p| gpa.free(p);
        }
        s.next = null;
        if (s.lanes) |l| l.deinit(gpa);
        s.lanes = null;
    }

    pub fn prompt(s: *const Stream) []const u32 {
        return s.context.items[0..s.prompt_len];
    }

    pub fn emitted(s: *const Stream) []const u32 {
        return s.context.items[s.prompt_len..];
    }

    pub fn budgetLeft(s: *const Stream) i64 {
        return @as(i64, s.max_new) - @as(i64, @intCast(s.emitted().len));
    }

    /// Whether the request was cancelled (false with no hook); a prompt pass asks before each chunk.
    pub fn isCancelled(s: *const Stream) bool {
        const c = s.cancel_check orelse return false;
        return c.check(c.ptr);
    }

    fn budgetActive(s: *const Stream) bool {
        return s.think_open and s.think_budget > 0 and s.think_close.len > 0;
    }

    /// The index the thinking budget replaces with `think_close[0]`, or null if the block closed first.
    pub fn thinkCut(s: *const Stream, tokens: anytype) ?usize {
        if (!s.budgetActive()) return null;
        const done = s.emitted().len;
        for (tokens, 0..) |t, i| {
            if (done + i + 1 >= s.think_budget) return i;
            if (@as(i64, t) == s.think_end) return null;
        }
        return null;
    }

    /// Python `think_cut([-1]) == 0`: the next position is the budget's cut.
    pub fn cutsNext(s: *const Stream) bool {
        return s.budgetActive() and s.emitted().len + 1 >= s.think_budget;
    }

    /// Begin the thinking budget's close: its first token now, the rest forced after it.
    pub fn startClose(s: *Stream, gpa: Allocator) !u32 {
        s.think_open = false;
        s.force.clearRetainingCapacity();
        try s.force.appendSlice(gpa, s.think_close[1..]);
        return s.think_close[0];
    }

    /// Tokens a round may commit before the length limit or the thinking budget's cut.
    pub fn draftRoom(s: *const Stream) i64 {
        var room = s.budgetLeft();
        if (s.budgetActive()) room = @min(room, @as(i64, s.think_budget) - @as(i64, @intCast(s.emitted().len)));
        return room;
    }

    pub fn isEos(s: *const Stream, token: u32) bool {
        return std.mem.indexOfScalar(u32, s.eos, token) != null;
    }

    /// Append tokens until the stream finishes; how many landed (a prefix of `tokens`).
    pub fn commit(s: *Stream, gpa: Allocator, tokens: []const u32) !usize {
        var landed: usize = 0;
        for (tokens) |t| {
            if (s.finished) break;
            try s.context.append(gpa, t);
            landed += 1;
            if (@as(i64, t) == s.think_end) {
                s.think_open = false;
                if (s.answer_from == null) s.answer_from = s.emitted().len;
            }
            const fire = if (s.loop_policy != .off and s.think_open and s.think_end >= 0 and s.loop == null) detectLoop(s.emitted()) else null;
            s.noteAnswerLoop();
            if (s.isEos(t) or s.stopped()) {
                s.finished = true;
                s.reason = .stop;
            } else if (fire) |period| {
                s.loop = .{ .period = period, .at_token = @intCast(s.emitted().len - 1) };
                if (s.emitted().len >= s.max_new) {
                    s.finished = true;
                    s.reason = .length;
                } else if (s.loop_policy == .count) {
                    continue; // recorded, never acted on: the stream goes on as with the policy off
                } else if (s.loop_policy == .stop) {
                    s.finished = true;
                    s.reason = .loop;
                } else if (s.think_close.len == 0) {
                    s.finished = true;
                    s.reason = .stop;
                } else { // close
                    s.think_open = false;
                    s.force.clearRetainingCapacity();
                    try s.force.appendSlice(gpa, s.think_close);
                }
                break;
            } else if (s.emitted().len >= s.max_new) {
                s.finished = true;
                s.reason = .length;
            }
        }
        return landed;
    }

    /// Count (never act on) an exact cycle in the answer: the same rule as the thinking loop, over the answer's tokens.
    fn noteAnswerLoop(s: *Stream) void {
        if (s.loop_policy == .off or s.answer_loop != null or s.think_open) return;
        const from = s.answer_from orelse return;
        const all = s.emitted();
        if (detectLoop(all[from..])) |period| s.answer_loop = .{ .period = period, .at_token = @intCast(all.len - 1) };
    }

    fn stopped(s: *const Stream) bool {
        const c = s.stop_check orelse return false;
        return c.check(c.ptr, s.emitted());
    }

    pub fn popForce(s: *Stream) ?u32 {
        if (s.force.items.len == 0) return null;
        return s.force.orderedRemove(0);
    }
};

test "commit stops at eos and length" {
    const gpa = std.testing.allocator;
    var s = try Stream.init(gpa, .{ .id = "a", .prompt = &.{ 1, 2 }, .max_new = 3, .eos = &.{9} });
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), try s.commit(gpa, &.{ 5, 9, 7 }));
    try std.testing.expectEqualSlices(u32, &.{ 5, 9 }, s.emitted());
    try std.testing.expect(s.finished and s.reason == .stop);
}

test "loop policy close stops at the cap even when no cycle fires" {
    const gpa = std.testing.allocator;
    var tokens: [20]u32 = undefined;
    for (&tokens, 0..) |*token, i| token.* = @intCast(100 + i);
    var guarded = try Stream.init(gpa, .{ .id = "guarded", .prompt = &.{1}, .max_new = 10, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_policy = .close });
    defer guarded.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 10), try guarded.commit(gpa, &tokens));
    try std.testing.expect(guarded.finished and guarded.reason == .length);
}

test "loop policy close sees a long sentence cycle (29 tokens) and not one past the longest period" {
    const gpa = std.testing.allocator;
    const Case = struct { period: usize, fires: bool };
    for ([_]Case{ .{ .period = 29, .fires = true }, .{ .period = loop_max_period, .fires = true }, .{ .period = loop_max_period + 1, .fires = false } }) |c| {
        var tokens: std.ArrayList(u32) = .empty;
        defer tokens.deinit(gpa);
        for (0..loop_warm_in) |i| try tokens.append(gpa, @intCast(1000 + i)); // the reasoning before the loop
        const n = loop_warm_in + loop_min_run + c.period + 3 * c.period; // the cycle a few times past the guard's run
        while (tokens.items.len < n) try tokens.append(gpa, @intCast(2000 + (tokens.items.len - loop_warm_in) % c.period));
        var s = try Stream.init(gpa, .{ .id = "loop", .prompt = &.{1}, .max_new = 4096, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_policy = .close });
        defer s.deinit(gpa);
        const landed = try s.commit(gpa, tokens.items);
        if (c.fires) {
            // it fires at the first token that completes loop_min_run tokens of the cycle after its first round
            try std.testing.expectEqual(loop_warm_in + c.period + loop_min_run, landed);
            try std.testing.expectEqual(@as(u32, @intCast(c.period)), s.loop.?.period);
            try std.testing.expectEqual(@as(u32, @intCast(landed - 1)), s.loop.?.at_token);
            try std.testing.expect(!s.finished and !s.think_open and s.force.items.len == 3);
        } else {
            try std.testing.expectEqual(n, landed);
            try std.testing.expect(s.loop == null and s.think_open);
        }
    }
}

test "loop policy close fires, closes thinking, and latches once" {
    const gpa = std.testing.allocator;
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    for (0..loop_warm_in) |i| try tokens.append(gpa, @intCast(1000 + i));
    for (0..257) |_| try tokens.append(gpa, 7);
    var s = try Stream.init(gpa, .{ .id = "loop", .prompt = &.{1}, .max_new = 400, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_policy = .close });
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 321), try s.commit(gpa, tokens.items));
    try std.testing.expectEqual(@as(u32, 1), s.loop.?.period);
    try std.testing.expect(!s.finished and !s.think_open and s.force.items.len == 3);
    while (s.popForce()) |token| _ = try s.commit(gpa, &.{token});
    try std.testing.expect(!s.finished and s.force.items.len == 0);
    _ = try s.commit(gpa, &.{8});
    try std.testing.expect(!s.finished and s.loop.?.period == 1);
}

/// 64 distinct reasoning tokens, then one token repeated: the period-1 cycle fires at emitted index 64 + 256 = 320.
fn loopTokens(gpa: Allocator, extra: usize) !std.ArrayList(u32) {
    var tokens: std.ArrayList(u32) = .empty;
    for (0..loop_warm_in) |i| try tokens.append(gpa, @intCast(1000 + i));
    for (0..loop_min_run + 1 + extra) |_| try tokens.append(gpa, 7);
    return tokens;
}

test "loop policy stop ends the reply at detection with reason loop" {
    const gpa = std.testing.allocator;
    var tokens = try loopTokens(gpa, 40);
    defer tokens.deinit(gpa);
    var s = try Stream.init(gpa, .{ .id = "loop", .prompt = &.{1}, .max_new = 4096, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_policy = .stop });
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 321), try s.commit(gpa, tokens.items));
    try std.testing.expect(s.finished and s.reason == .loop);
    try std.testing.expectEqualStrings("loop", s.reason.name());
    try std.testing.expectEqual(LoopHit{ .period = 1, .at_token = 320 }, s.loop.?);
    try std.testing.expect(s.force.items.len == 0 and s.think_open); // no forced close, no answer
    try std.testing.expectEqual(@as(?usize, null), loopCut(&s, &.{7})); // latched: nothing more is cut
}

test "loop policy stop: the round loop cuts a drafted batch at the firing token" {
    const gpa = std.testing.allocator;
    var tokens = try loopTokens(gpa, 0);
    defer tokens.deinit(gpa);
    var s = try Stream.init(gpa, .{ .id = "loop", .prompt = &.{1}, .max_new = 4096, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_policy = .stop });
    defer s.deinit(gpa);
    _ = try s.commit(gpa, tokens.items[0..300]);
    const sevens: [30]u32 = @splat(7);
    try std.testing.expectEqual(@as(?usize, 20), loopCut(&s, &sevens));
    var counted = try Stream.init(gpa, .{ .id = "count", .prompt = &.{1}, .max_new = 4096, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_policy = .count });
    defer counted.deinit(gpa);
    _ = try counted.commit(gpa, tokens.items[0..300]);
    try std.testing.expectEqual(@as(?usize, null), loopCut(&counted, &sevens)); // count never cuts
}

test "loop policy count records the loop and leaves the stream as policy off" {
    const gpa = std.testing.allocator;
    var tokens = try loopTokens(gpa, 100);
    defer tokens.deinit(gpa);
    try tokens.appendSlice(gpa, &.{ 90, 91, 92, 40, 41 });
    var off = try Stream.init(gpa, .{ .id = "off", .prompt = &.{1}, .max_new = 4096, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .think_open = true });
    defer off.deinit(gpa);
    var counted = try Stream.init(gpa, .{ .id = "count", .prompt = &.{1}, .max_new = 4096, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_policy = .count });
    defer counted.deinit(gpa);
    try std.testing.expectEqual(try off.commit(gpa, tokens.items), try counted.commit(gpa, tokens.items));
    try std.testing.expectEqualSlices(u32, off.emitted(), counted.emitted());
    try std.testing.expectEqual(off.finished, counted.finished);
    try std.testing.expectEqual(off.think_open, counted.think_open);
    try std.testing.expect(off.loop == null and off.answer_loop == null);
    try std.testing.expectEqual(LoopHit{ .period = 1, .at_token = 320 }, counted.loop.?);
    try std.testing.expect(counted.force.items.len == 0 and !counted.finished);
}

test "a loop in the answer is counted under every policy and never acted on" {
    const gpa = std.testing.allocator;
    for ([_]LoopPolicy{ .close, .stop, .count }) |policy| {
        var tokens: std.ArrayList(u32) = .empty;
        defer tokens.deinit(gpa);
        for (0..10) |i| try tokens.append(gpa, @intCast(1000 + i)); // short reasoning
        try tokens.appendSlice(gpa, &.{ 90, 91, 92 }); // closed by the model: the answer starts at emitted index 12 (the 92)
        for (0..loop_warm_in) |i| try tokens.append(gpa, @intCast(3000 + i));
        for (0..loop_min_run + 2 + 50) |i| try tokens.append(gpa, @intCast(40 + i % 2)); // a period-2 answer cycle
        var s = try Stream.init(gpa, .{ .id = "answer", .prompt = &.{1}, .max_new = 4096, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_policy = policy });
        defer s.deinit(gpa);
        try std.testing.expectEqual(tokens.items.len, try s.commit(gpa, tokens.items));
        try std.testing.expect(!s.finished and s.loop == null and s.force.items.len == 0);
        // the cycle starts at answer index 65; its run (256 tokens matching 2 back) completes at answer index 322: emitted 12 + 322
        try std.testing.expectEqual(LoopHit{ .period = 2, .at_token = 334 }, s.answer_loop.?);
    }
    // with no think block at all (thinking off), the whole reply is the answer
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    for (0..loop_warm_in) |i| try tokens.append(gpa, @intCast(3000 + i));
    for (0..loop_min_run + 1) |_| try tokens.append(gpa, 5);
    var plain = try Stream.init(gpa, .{ .id = "plain", .prompt = &.{1}, .max_new = 4096, .loop_policy = .stop });
    defer plain.deinit(gpa);
    _ = try plain.commit(gpa, tokens.items);
    try std.testing.expect(!plain.finished and plain.loop == null);
    try std.testing.expectEqual(LoopHit{ .period = 1, .at_token = 320 }, plain.answer_loop.?);
    var off = try Stream.init(gpa, .{ .id = "off", .prompt = &.{1}, .max_new = 4096 });
    defer off.deinit(gpa);
    _ = try off.commit(gpa, tokens.items);
    try std.testing.expect(off.answer_loop == null); // policy off: nothing is detected
}

test "loop policy names parse; only close and stop act" {
    try std.testing.expectEqual(LoopPolicy.stop, LoopPolicy.parse("stop").?);
    try std.testing.expectEqual(@as(?LoopPolicy, null), LoopPolicy.parse("guard"));
    try std.testing.expect(LoopPolicy.close.acts() and LoopPolicy.stop.acts() and !LoopPolicy.count.acts() and !LoopPolicy.off.acts());
}

test "loop policy close with no think_close ends the reply with reason stop, as the old guard did" {
    const gpa = std.testing.allocator;
    var tokens = try loopTokens(gpa, 10);
    defer tokens.deinit(gpa);
    var s = try Stream.init(gpa, .{ .id = "loop", .prompt = &.{1}, .max_new = 4096, .think_end = 91, .loop_policy = .close });
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 321), try s.commit(gpa, tokens.items));
    try std.testing.expect(s.finished and s.reason == .stop and s.force.items.len == 0);
    try std.testing.expectEqual(LoopHit{ .period = 1, .at_token = 320 }, s.loop.?);
}
