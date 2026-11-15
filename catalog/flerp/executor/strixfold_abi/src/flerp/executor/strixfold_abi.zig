//! StrixFold's dsv41 executor behind include/flerp/executor.h: the C ABI exported over the lane backend vtable
//! (zig/src/core/lanes/backend.zig) as the family's native.open hands it out, with the kept-state functions
//! (native.Snaps). docs/EXECUTOR-ABI.md has the call-by-call mapping and what does not map.
//!
//! The process is rank 0 of the tensor-parallel pair: flerp_open loads this rank and links rank 1 (tf-dsv41-lanes on
//! the other node), exactly as the server's engine host does. Everything native.zig reads from the environment
//! (TF_TP_MASTER, TF_TP_PORT, TF_DS_RANK_CACHE, TF_DS_KIT, TF_DS_ENGRAM, TF_DS_TOKEN_MAP, TF_COLL_*, ...) is read
//! there, unchanged. flerp_open keys: model_dir, kit (the AMD kernel kit: manifest.json and objects/), context
//! (the pool's positions), drafts (on|off).
//!
//! The backend draws tokens on the device and keeps its caches by position, so:
//!   - only FLERP_OUT_TOKENS is offered; a verify's rows come back as the tokens drawn at each row;
//!   - a prompt is one prompt pass: forwards before the first asked result are gathered and run when it is asked;
//!   - keep, keep_prefix and drop move the seq's cache length (rows past it are written over by the next window);
//!   - a state is kept only while the prompt pass stands at a chunk end, and restored only into a prompt pass.
//! Since fork commits d21ae2e and f64b5ff the family can also write a kept state to each rank's disk and hand it out
//! as bytes (native.Snaps write / read / forget, export_state / import_state; the server's /cache/export and
//! /cache/import). This adapter does not map them yet: a state that comes back needs the token ids of its prefix
//! for the prompt pass that resumes from it, and flerp_state_import / flerp_snapshot_read carry none (an ABI v1 gap).
const std = @import("std");
const gpu = @import("gpu");
const lanes = @import("lanes");
const dsv41 = @import("dsv41");
const c = @import("flerp_abi");

const be = lanes.backend;
const max_rows = 64;
const max_snaps = 4;

const OK: c_int = 0;
const E_ARGUMENT: c_int = -1;
const E_STATE: c_int = -2;
const E_UNSUPPORTED: c_int = -3;
const E_FULL: c_int = -4;
const E_FAILED: c_int = -5;
const E_CONFIG: c_int = -6;

const Snap = struct {
    used: bool = false,
    at: u32 = 0,
    saved: ?*anyopaque = null,
    prefix: []u32 = &.{},
};

const Seq = struct {
    open: bool = false,
    live: bool = false, // the prompt pass ran: `stream` is the backend's
    prompt: std.ArrayList(u32) = .empty,
    stream: lanes.Stream = undefined,
    sampling: ?lanes.Sampling = null,
    drafts: bool = true,
    max_new: u32 = 1024,
    // the pending verify
    pending: bool = false,
    v_rows: usize = 0,
    v_start: u64 = 0,
    v_ids: [max_rows]u32 = undefined, // the pending token, then the drafts the window ran (held or the caller's)
    v_sampled: [max_rows]u32 = undefined,
    v_drafts: [max_rows]u32 = undefined,
    v_positions: [max_rows]u64 = undefined,
    // the last kept window, as the draft head's next request names it
    have_kept: bool = false,
    kept_rows: usize = 0,
    kept_start: u64 = 0,
    follow: [max_rows]u32 = undefined,
    path: [max_rows]u32 = undefined,
    // kept states
    marks: [max_snaps]u32 = undefined,
    n_marks: usize = 0,
    snaps: [max_snaps]Snap = @splat(.{}),
    reuse_saved: ?*anyopaque = null,
    reuse_at: u32 = 0,
};

const Ex = struct {
    driver: gpu.Driver,
    ctx: gpu.Context,
    loaded: dsv41.native.Loaded,
    seqs: []Seq,
    context: usize,
    describe: [:0]u8,
    forwards: i64 = 0,
    forward_tokens: i64 = 0,
    verifies: i64 = 0,
    verify_rows: i64 = 0,
    kept_rows: i64 = 0,
    dropped_rows: i64 = 0,
    draft_tokens: i64 = 0,
    saved: i64 = 0,
    restored: i64 = 0,
    failed: bool = false,
};

var g_gpa: std.mem.Allocator = undefined;
var g_io: std.Io = undefined;
var g_err_buf: [640]u8 = undefined;
var g_err: [*:0]const u8 = "";

/// The C or C++ program this adapter is linked into (its main, compiled with -Dmain=flerp_host_main).
extern fn flerp_host_main(argc: c_int, argv: [*]const [*:0]const u8) c_int;

/// A root file other than this one (flerp_serve_strixfold.zig) hands the process's allocator and io over.
pub fn setProcess(gpa: std.mem.Allocator, io: std.Io) void {
    g_gpa = gpa;
    g_io = io;
}

/// Zig owns the process start (allocator, io); the driver's C main runs as a function and calls back into the ABI.
pub fn main(init: std.process.Init) !u8 {
    g_gpa = init.gpa;
    g_io = init.io;
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    const argv = try a.alloc([*:0]const u8, args.len);
    for (args, argv) |arg, *p| p.* = arg.ptr;
    const rc = flerp_host_main(@intCast(args.len), argv.ptr);
    std.debug.print("host: main returned {d}; the process exits\n", .{rc});
    return @intCast(@as(u32, @bitCast(rc)) & 0xff);
}

fn fail(st: c_int, comptime fmt: []const u8, args: anytype) c_int {
    const msg = std.fmt.bufPrintSentinel(&g_err_buf, fmt, args, 0) catch blk: {
        g_err_buf[g_err_buf.len - 1] = 0;
        break :blk g_err_buf[0 .. g_err_buf.len - 1 :0];
    };
    g_err = msg.ptr;
    return st;
}

/// A backend error: the pair's failure poisons the collective (coll.zig), so the executor is done.
fn broke(ex: *Ex, what: []const u8, err: anyerror) c_int {
    ex.failed = true;
    return fail(E_FAILED, "{s}: {s}{s}{s}", .{ what, @errorName(err), if (dsv41.native.explain(null, err) != null) ": " else "", dsv41.native.explain(null, err) orelse "" });
}

fn of(p: ?*anyopaque) *Ex {
    return @ptrCast(@alignCast(p.?));
}

fn seqOf(ex: *Ex, seq: i32) ?*Seq {
    if (seq < 0 or seq >= ex.seqs.len) return null;
    const q = &ex.seqs[@intCast(seq)];
    return if (q.open) q else null;
}

fn text(p: [*c]const u8) []const u8 {
    return if (p == null) "" else std.mem.span(@as([*:0]const u8, @ptrCast(p)));
}

export fn flerp_open(args: ?*const c.flerp_open_args, out: ?*?*anyopaque) c_int {
    const a = args orelse return fail(E_ARGUMENT, "flerp_open: null arguments", .{});
    if (out == null or a.size < @sizeOf(c.flerp_open_args)) return fail(E_ARGUMENT, "flerp_open: null or short arguments", .{});
    var model_dir: []const u8 = "";
    var kit: []const u8 = "";
    var context: usize = 8192;
    var drafts = true;
    var i: usize = 0;
    while (i < a.n_config) : (i += 1) {
        const key = text(a.config[i].key);
        const value = text(a.config[i].value);
        if (std.mem.eql(u8, key, "model_dir")) {
            model_dir = value;
        } else if (std.mem.eql(u8, key, "kit")) {
            kit = value;
        } else if (std.mem.eql(u8, key, "context")) {
            context = std.fmt.parseInt(usize, value, 10) catch return fail(E_CONFIG, "flerp_open (strixfold): context '{s}' is not a number", .{value});
        } else if (std.mem.eql(u8, key, "drafts")) {
            if (!std.mem.eql(u8, value, "on") and !std.mem.eql(u8, value, "off")) return fail(E_CONFIG, "flerp_open (strixfold): drafts must be on or off", .{});
            drafts = std.mem.eql(u8, value, "on");
        } else return fail(E_CONFIG, "flerp_open (strixfold): unknown key '{s}'", .{key});
    }
    if (model_dir.len == 0 or kit.len == 0) return fail(E_CONFIG, "flerp_open (strixfold): model_dir and kit are required", .{});
    const ex = g_gpa.create(Ex) catch return fail(E_FAILED, "flerp_open: out of memory", .{});
    ex.* = undefined;
    ex.driver = gpu.Driver.open() catch |e| {
        g_gpa.destroy(ex);
        return fail(E_FAILED, "flerp_open (strixfold): the HIP driver did not open: {s}", .{@errorName(e)});
    };
    ex.ctx = gpu.Context.init(&ex.driver, 0) catch |e| {
        ex.driver.close();
        g_gpa.destroy(ex);
        return fail(E_FAILED, "flerp_open (strixfold): no device context: {s}", .{@errorName(e)});
    };
    ex.loaded = dsv41.native.open(g_gpa, g_io, &ex.ctx, model_dir, kit, .{ .context = context, .drafts = drafts }) catch |e| {
        const why = dsv41.native.explain(null, e) orelse "";
        ex.ctx.deinit();
        ex.driver.close();
        g_gpa.destroy(ex);
        return fail(E_FAILED, "flerp_open (strixfold): {s} with kit {s} did not load: {s} {s}", .{ model_dir, kit, @errorName(e), why });
    };
    const n: usize = @max(1, ex.loaded.facts.max_streams);
    ex.seqs = g_gpa.alloc(Seq, n) catch return fail(E_FAILED, "flerp_open: out of memory", .{});
    for (ex.seqs) |*q| q.* = .{};
    ex.context = ex.loaded.context_window orelse context;
    ex.describe = std.fmt.allocPrintSentinel(g_gpa, "strixfold dsv41 (lane backend, rank 0 of the pair): {d} positions, {d} rows a round, drafts {s}", .{ ex.context, ex.loaded.rows, if (ex.loaded.facts.mtp) "on" else "off" }, 0) catch return fail(E_FAILED, "flerp_open: out of memory", .{});
    ex.forwards = 0;
    ex.forward_tokens = 0;
    ex.verifies = 0;
    ex.verify_rows = 0;
    ex.kept_rows = 0;
    ex.dropped_rows = 0;
    ex.draft_tokens = 0;
    ex.saved = 0;
    ex.restored = 0;
    ex.failed = false;
    out.?.* = ex;
    return OK;
}

/// Each step of the close is named on stderr as it is reached: the adapter's first run on the pair (2026-11-11)
/// passed every check of the conformance driver and then died of SIGSEGV somewhere in here, and nothing said where.
fn closing(comptime step: []const u8) void {
    std.debug.print("flerp_close (strixfold): " ++ step ++ "\n", .{});
}

export fn flerp_close(p: ?*anyopaque) void {
    const ex = of(p orelse return);
    closing("freeing the open seqs");
    for (ex.seqs, 0..) |*q, i| if (q.open) {
        _ = flerp_seq_free(ex, @intCast(i));
    };
    // as the origin's engine host closes (zig/src/native/hip.zig, Host.close): the context current on this thread,
    // the family released, then the context and the driver
    closing("making the device context current");
    ex.ctx.makeCurrent() catch {};
    closing("releasing the family (rank 1 hears done)");
    ex.loaded.deinit(ex.loaded.ctx);
    closing("destroying the device context");
    ex.ctx.deinit();
    // Host.close closes the driver last (a dlclose of libamdhip64). A build that left the runtime loaded instead
    // crashed just the same (2026-11-11T02:50Z: every step here done, SIGSEGV after the host's main returned), so the
    // close is the origin's again and the crash is looked for after it.
    closing("closing the HIP driver");
    ex.driver.close();
    g_gpa.free(ex.seqs);
    g_gpa.free(ex.describe);
    g_gpa.destroy(ex);
    closing("done");
}

export fn flerp_get_caps(p: ?*const anyopaque, out: ?*c.flerp_caps) c_int {
    const ex: *const Ex = @ptrCast(@alignCast(p.?));
    const o = out orelse return fail(E_ARGUMENT, "flerp_caps is null", .{});
    // a version 1 caller's struct ends at `flags`: it is filled as far as it goes and never written past
    const v1_size = @offsetOf(c.flerp_caps, "flags") + @sizeOf(u32);
    if (o.size < v1_size) return fail(E_ARGUMENT, "flerp_caps is short", .{});
    var caps = std.mem.zeroes(c.flerp_caps);
    // a version 2 executor (the version 3 calls are FLERP-SF-STATE's): written for at most the version 2 struct
    caps.size = @min(o.size, @offsetOf(c.flerp_caps, "top_k_max") + @sizeOf(i32));
    caps.abi = 2;
    caps.positions = @intCast(ex.context);
    caps.max_forward = @intCast(ex.context);
    caps.max_verify_rows = @intCast(@min(ex.loaded.rows, max_rows));
    caps.logits_row = 0;
    caps.candidates = 0;
    caps.seqs = @intCast(ex.seqs.len);
    caps.snapshots = if (ex.loaded.snaps != null) max_snaps else 0;
    caps.max_draft = if (ex.loaded.facts.mtp) @intCast(@min(ex.loaded.facts.drafts, ex.loaded.rows - 1)) else 0;
    caps.deliveries = c.FLERP_OUT_TOKENS;
    var flags: u32 = c.FLERP_CAP_VERIFY_BATCH | c.FLERP_CAP_KEEP_PREFIX | c.FLERP_CAP_SAMPLED_TOKENS;
    if (ex.loaded.facts.mtp) flags |= c.FLERP_CAP_DRAFT_HELD;
    if (ex.loaded.snaps != null) flags |= c.FLERP_CAP_SNAPSHOT;
    if (ex.loaded.vision != null) flags |= c.FLERP_CAP_VISION;
    caps.flags = flags;
    // version 2: what the keyed sampler takes (lanes.Sampling: seed, temperature, top_k, top_p, min_p) and the
    // family's bound on top_k (native.zig top_k_max, needs_top_k: the server refuses a sampled request without one)
    caps.sampling = c.FLERP_SAMPLE_TOP_K | c.FLERP_SAMPLE_TOP_P | c.FLERP_SAMPLE_MIN_P | c.FLERP_SAMPLE_SEED |
        (if (dsv41.native.needs_top_k) c.FLERP_SAMPLE_NEEDS_TOP_K else 0);
    caps.top_k_max = @intCast(dsv41.native.top_k_max);
    @memcpy(@as([*]u8, @ptrCast(o))[0..caps.size], @as([*]const u8, @ptrCast(&caps))[0..caps.size]);
    return OK;
}

export fn flerp_describe(p: ?*const anyopaque) [*:0]const u8 {
    const ex: *const Ex = @ptrCast(@alignCast(p.?));
    return ex.describe.ptr;
}

export fn flerp_state_fingerprint(p: ?*const anyopaque) [*:0]const u8 {
    _ = p;
    // kept states never leave the process as bytes; what they fit is the launch's identity (TF_DS_DEPLOY, TF_DS_BUNDLE)
    return "strixfold-dsv41: states stay in the executor";
}

export fn flerp_last_error(p: ?*const anyopaque) [*:0]const u8 {
    _ = p;
    return g_err;
}

export fn flerp_failed(p: ?*const anyopaque) c_int {
    const ex: *const Ex = @ptrCast(@alignCast(p.?));
    return @intFromBool(ex.failed);
}

export fn flerp_seq_new(p: ?*anyopaque, args: ?*const c.flerp_seq_args, seq: ?*i32) c_int {
    const ex = of(p);
    const out = seq orelse return fail(E_ARGUMENT, "flerp_seq_new: null seq", .{});
    for (ex.seqs, 0..) |*q, i| {
        if (q.open) continue;
        q.* = .{};
        q.open = true;
        if (args) |a| {
            q.drafts = a.drafts != 0;
            if (a.max_new > 0) q.max_new = @intCast(a.max_new);
            if (a.sampling != null and a.sampling.*.temperature > 0) {
                const s = a.sampling.*;
                q.sampling = .{ .seed = s.seed, .temperature = s.temperature, .top_k = @intCast(@max(0, s.top_k)), .top_p = s.top_p, .min_p = s.min_p };
            }
        }
        out.* = @intCast(i);
        return OK;
    }
    return fail(E_FULL, "every seq is open ({d})", .{ex.seqs.len});
}

fn release(ex: *Ex, q: *Seq) void {
    if (q.live) {
        ex.loaded.backend.release(&q.stream);
        q.stream.deinit(g_gpa);
    }
    q.live = false;
    q.pending = false;
    q.have_kept = false;
    q.n_marks = 0;
    q.reuse_saved = null;
    q.reuse_at = 0;
    q.prompt.clearRetainingCapacity();
}

fn dropSnap(ex: *Ex, s: *Snap) void {
    if (s.saved) |saved| if (ex.loaded.snaps) |sn| sn.drop(sn.ptr, saved);
    if (s.prefix.len > 0) g_gpa.free(s.prefix);
    s.* = .{};
}

export fn flerp_seq_free(p: ?*anyopaque, seq: i32) c_int {
    const ex = of(p);
    const q = seqOf(ex, seq) orelse return fail(E_ARGUMENT, "seq {d} is not open", .{seq});
    release(ex, q);
    for (&q.snaps) |*s| if (s.used) dropSnap(ex, s);
    q.prompt.deinit(g_gpa);
    q.* = .{};
    return OK;
}

export fn flerp_seq_reset(p: ?*anyopaque, seq: i32) c_int {
    const ex = of(p);
    const q = seqOf(ex, seq) orelse return fail(E_ARGUMENT, "seq {d} is not open", .{seq});
    release(ex, q);
    return OK;
}

export fn flerp_seq_pos(p: ?*const anyopaque, seq: i32) i64 {
    const ex: *Ex = @constCast(@ptrCast(@alignCast(p.?)));
    const q = seqOf(ex, seq) orelse return E_ARGUMENT;
    return if (q.live) @intCast(q.stream.cache_len) else @intCast(q.prompt.items.len);
}

/// The prompt pass stands at `at` (a chunk end): keep the state for every snapshot asked there.
fn markAt(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
    const q: *Seq = @ptrCast(@alignCast(ptr));
    const ex = g_marking orelse return;
    const sn = ex.loaded.snaps orelse return;
    for (&q.snaps) |*k| {
        if (!k.used or k.saved != null or k.at != at) continue;
        k.saved = sn.save(sn.ptr, s, at) catch null;
        if (k.saved != null) ex.saved += 1;
    }
}
var g_marking: ?*Ex = null;

/// The gathered prompt as one prompt pass, then its first token drawn at the prompt's length.
fn begin(ex: *Ex, q: *Seq) !u32 {
    q.stream = try lanes.Stream.init(g_gpa, .{
        .id = "flerp",
        .prompt = q.prompt.items,
        .max_new = q.max_new,
        .sampling = q.sampling,
        .drafts = q.drafts,
        // marks cut the prompt pass's chunks: none unless a snapshot was asked, so a plain prompt runs as the server's
        .reuse = .{ .saved = q.reuse_saved, .at = q.reuse_at, .marks = q.marks[0..q.n_marks], .hook = if (q.n_marks > 0) .{ .ptr = q, .at = markAt } else null },
    });
    q.live = true;
    const s = &q.stream;
    const b = ex.loaded.backend;
    g_marking = ex;
    defer g_marking = null;
    try b.prefill(s);
    s.context.shrinkRetainingCapacity(s.prompt_len);
    s.pending = null;
    s.cache_len = s.prompt_len;
    const token = try b.read(try b.first(s, s.prompt_len));
    _ = try s.commit(g_gpa, &.{token});
    s.pending = token;
    if (q.reuse_saved != null and s.cached > 0) ex.restored += 1;
    return token;
}

/// One window of the seq: `ids[0]` the pending token, then `held` drafts the backend holds or the caller's ids.
fn window(q: *Seq, ids: []const u32, held: u32) !be.Window {
    const rows = ids.len + held;
    if (rows > max_rows) return error.WindowTooWide;
    const s = &q.stream;
    for (0..rows) |r| q.v_positions[r] = s.cache_len + 1 + r;
    @memcpy(q.v_ids[0..ids.len], ids);
    q.v_rows = rows;
    q.v_start = s.cache_len;
    return .{ .stream = s, .pending = ids[0], .held = held, .tokens = q.v_ids[1..ids.len], .parents = null, .positions = q.v_positions[0..rows] };
}

/// Keep the first `rows` rows of the seq's window (0: none): the cache length moves, the reply's tokens are committed.
fn resolve(ex: *Ex, q: *Seq, rows: usize, carry: i32) !void {
    const s = &q.stream;
    if (rows > 0) {
        for (0..rows) |r| q.path[r] = @intCast(r);
        const w = be.Window{ .stream = s, .pending = q.v_ids[0], .held = 0, .tokens = q.v_ids[1..q.v_rows], .parents = null, .positions = q.v_positions[0..q.v_rows] };
        try ex.loaded.backend.keep(&.{w}, &.{q.path[0..rows]});
        const next: u32 = if (carry >= 0) @intCast(carry) else q.v_sampled[rows - 1];
        @memcpy(q.follow[0 .. rows - 1], q.v_ids[1..rows]);
        q.follow[rows - 1] = next;
        s.cache_len += rows;
        _ = try s.commit(g_gpa, q.follow[0..rows]);
        s.pending = next;
        q.have_kept = true;
        q.kept_rows = rows;
        q.kept_start = q.v_start;
    }
    ex.kept_rows += @intCast(rows);
    ex.dropped_rows += @intCast(q.v_rows - rows);
    q.pending = false;
}

export fn flerp_forward(p: ?*anyopaque, seq: i32, ids: [*c]const i32, n: usize, want: ?*const c.flerp_want, out: ?*c.flerp_rows, lookahead: [*c]const i32, n_lookahead: usize) c_int {
    _ = .{ lookahead, n_lookahead }; // the prompt pass cuts and looks ahead itself
    const ex = of(p);
    const q = seqOf(ex, seq) orelse return fail(E_ARGUMENT, "seq {d} is not open", .{seq});
    if (q.pending) return fail(E_STATE, "a verify is pending on seq {d}: keep, keep_prefix or drop first", .{seq});
    if (ids == null or n == 0) return fail(E_ARGUMENT, "flerp_forward: no ids", .{});
    const rows: i64 = if (want) |w| w.rows else 0;
    if (rows > 1) return fail(E_UNSUPPORTED, "strixfold returns a forward's last row only (caps: no FLERP_CAP_FORWARD_ROWS)", .{});
    if (rows == 1 and want.?.delivery != c.FLERP_OUT_TOKENS) return fail(E_UNSUPPORTED, "strixfold draws on the device: only FLERP_OUT_TOKENS", .{});
    if (rows == 1 and (out == null or out.?.tokens == null or out.?.tokens_cap < 1)) return fail(E_ARGUMENT, "flerp_rows.tokens is too small", .{});
    const src: []const u32 = @ptrCast(ids[0..n]);
    var token: u32 = 0;
    if (!q.live) {
        q.prompt.appendSlice(g_gpa, src) catch return fail(E_FAILED, "out of memory", .{});
        if (q.prompt.items.len + q.max_new > ex.context) return fail(E_FULL, "the prompt and the seq's max_new do not fit the context window", .{});
        if (rows == 0) return OK; // gathered: the prompt pass runs when a result is asked
        token = begin(ex, q) catch |e| return broke(ex, "prompt pass", e);
        ex.forwards += 1;
        ex.forward_tokens += @intCast(q.prompt.items.len);
    } else {
        // decode: the ids as one window, every row kept
        if (n > ex.loaded.rows) return fail(E_FULL, "more ids than a round's rows", .{});
        const w = window(q, src, 0) catch |e| return broke(ex, "window", e);
        var v = [_]be.Verified{.{ .sampled = q.v_sampled[0..n], .drafts = q.v_drafts[0 .. n - 1] }};
        ex.loaded.backend.verify(&.{w}, &v) catch |e| return broke(ex, "forward", e);
        resolve(ex, q, n, -1) catch |e| return broke(ex, "keep", e);
        token = q.v_sampled[n - 1];
        ex.forwards += 1;
        ex.forward_tokens += @intCast(n);
        ex.kept_rows -= @intCast(n); // a forward's rows are not verify rows
    }
    if (rows == 1) {
        out.?.tokens[0] = @intCast(token);
        out.?.rows = 1;
    }
    return OK;
}

export fn flerp_draft_step(p: ?*anyopaque, seq: i32, token: i32, step: i32, out: ?*c.flerp_draft) c_int {
    _ = .{ p, seq, token, step, out };
    return fail(E_UNSUPPORTED, "strixfold's drafter keeps its drafts on the device: flerp_draft_hold", .{});
}

export fn flerp_draft_hold(p: ?*anyopaque, seq: i32, pending: i32, depth: i32) c_int {
    const ex = of(p);
    const q = seqOf(ex, seq) orelse return fail(E_ARGUMENT, "seq {d} is not open", .{seq});
    if (!q.live) return fail(E_STATE, "seq {d} has no prompt pass yet", .{seq});
    if (q.pending) return fail(E_STATE, "a verify is pending on seq {d}", .{seq});
    if (!ex.loaded.facts.mtp or !q.drafts) return fail(E_UNSUPPORTED, "no draft head for this seq", .{});
    if (depth < 0 or depth > ex.loaded.facts.drafts) return fail(E_ARGUMENT, "depth outside 0..caps.max_draft", .{});
    const s = &q.stream;
    const tok: u32 = @intCast(pending);
    if (q.have_kept) q.follow[q.kept_rows - 1] = tok;
    const req = be.DraftRequest{
        .stream = s,
        .follow = if (q.have_kept) q.follow[0..q.kept_rows] else &.{},
        .first = if (q.have_kept) null else .{ .value = tok },
        .rows = if (q.have_kept) q.path[0..q.kept_rows] else null,
        .start = if (q.have_kept) q.kept_start else s.prompt_len,
        .position = s.cache_len + 1,
        .depth = @intCast(depth),
    };
    ex.loaded.backend.draft(&.{req}) catch |e| return broke(ex, "draft", e);
    ex.draft_tokens += depth;
    return OK;
}

export fn flerp_verify(p: ?*anyopaque, runs: [*c]c.flerp_verify_run, n_runs: usize, want: ?*const c.flerp_want) c_int {
    const ex = of(p);
    if (runs == null or n_runs == 0 or want == null) return fail(E_ARGUMENT, "flerp_verify: no runs or no want", .{});
    if (want.?.delivery != c.FLERP_OUT_TOKENS) return fail(E_UNSUPPORTED, "strixfold draws on the device: only FLERP_OUT_TOKENS", .{});
    if (n_runs > 16) return fail(E_FULL, "more runs than streams", .{});
    var windows: [16]be.Window = undefined;
    var outs: [16]be.Verified = undefined;
    var total: usize = 0;
    for (runs[0..n_runs], 0..) |*r, k| {
        const q = seqOf(ex, r.seq) orelse return fail(E_ARGUMENT, "seq {d} is not open", .{r.seq});
        if (!q.live) return fail(E_STATE, "seq {d} has no prompt pass yet", .{r.seq});
        if (q.pending) return fail(E_STATE, "a verify is pending on seq {d}", .{r.seq});
        for (runs[0..k]) |*e| if (e.seq == r.seq) return fail(E_ARGUMENT, "flerp_verify: two runs of one seq", .{});
        if (r.ids == null or r.n_ids == 0) return fail(E_ARGUMENT, "flerp_verify: a run without ids", .{});
        if (r.held < 0 or (r.held > 0 and r.n_ids != 1)) return fail(E_ARGUMENT, "flerp_verify: a run names held drafts or its own, not both", .{});
        const held: u32 = @intCast(r.held);
        const rows = r.n_ids + held;
        if (r.out == null or r.out.*.tokens == null or r.out.*.tokens_cap < rows) return fail(E_ARGUMENT, "flerp_rows.tokens is too small", .{});
        if (held > 0 and (r.held_out == null or r.held_cap < held)) return fail(E_ARGUMENT, "run.held_out is too small", .{});
        const src: []const u32 = @ptrCast(r.ids[0..r.n_ids]);
        windows[k] = window(q, src, held) catch return fail(E_FULL, "a run wider than {d} rows", .{max_rows});
        outs[k] = .{ .sampled = q.v_sampled[0..rows], .drafts = q.v_drafts[0 .. rows - 1] };
        total += rows;
    }
    if (total > ex.loaded.rows) return fail(E_FULL, "more rows than caps.max_verify_rows", .{});
    ex.loaded.backend.verify(windows[0..n_runs], outs[0..n_runs]) catch |e| return broke(ex, "verify", e);
    ex.verifies += 1;
    ex.verify_rows += @intCast(total);
    for (runs[0..n_runs]) |*r| {
        const q = seqOf(ex, r.seq).?;
        const held: usize = @intCast(r.held);
        q.pending = true;
        // the rows the window ran after the pending token: the held drafts' values, or the caller's echoed back
        @memcpy(q.v_ids[1..q.v_rows], q.v_drafts[0 .. q.v_rows - 1]);
        for (0..q.v_rows) |i| r.out.*.tokens[i] = @intCast(q.v_sampled[i]);
        r.out.*.rows = @intCast(q.v_rows);
        for (0..held) |i| r.held_out[i] = @intCast(q.v_drafts[i]);
    }
    return OK;
}

fn settle(p: ?*anyopaque, seq: i32, rows: i64, carry: i32, all: bool) c_int {
    const ex = of(p);
    const q = seqOf(ex, seq) orelse return fail(E_ARGUMENT, "seq {d} is not open", .{seq});
    if (!q.pending) return fail(E_STATE, "no verify is pending on seq {d}", .{seq});
    const n: usize = if (all) q.v_rows else @intCast(rows);
    resolve(ex, q, n, carry) catch |e| return broke(ex, "keep", e);
    return OK;
}

export fn flerp_keep(p: ?*anyopaque, seq: i32) c_int {
    return settle(p, seq, 0, -1, true);
}

export fn flerp_keep_prefix(p: ?*anyopaque, seq: i32, rows: i64, carry: i32) c_int {
    const ex = of(p);
    const q = seqOf(ex, seq) orelse return fail(E_ARGUMENT, "seq {d} is not open", .{seq});
    if (q.pending and (rows < 1 or rows >= q.v_rows)) return fail(E_ARGUMENT, "flerp_keep_prefix: rows must be 1 .. the run's rows - 1", .{});
    return settle(p, seq, rows, carry, false);
}

export fn flerp_drop(p: ?*anyopaque, seq: i32) c_int {
    return settle(p, seq, 0, -1, false);
}

fn snapOf(q: *Seq, snap: u64) ?*Snap {
    if (snap < 1 or snap > max_snaps) return null;
    const s = &q.snaps[@intCast(snap - 1)];
    return if (s.used) s else null;
}

export fn flerp_snapshot_save(p: ?*anyopaque, seq: i32, out: ?*u64) c_int {
    const ex = of(p);
    const q = seqOf(ex, seq) orelse return fail(E_ARGUMENT, "seq {d} is not open", .{seq});
    if (ex.loaded.snaps == null) return fail(E_UNSUPPORTED, "this executor keeps no prompt states", .{});
    if (q.live) return fail(E_STATE, "strixfold keeps a state only inside the prompt pass: save before the first asked result", .{});
    const at = q.prompt.items.len;
    if (at == 0 or at <= q.reuse_at) return fail(E_STATE, "nothing new to keep at position {d}", .{at});
    for (&q.snaps, 0..) |*s, i| {
        if (s.used) continue;
        if (q.n_marks == max_snaps) break;
        s.* = .{ .used = true, .at = @intCast(at), .prefix = g_gpa.dupe(u32, q.prompt.items) catch return fail(E_FAILED, "out of memory", .{}) };
        q.marks[q.n_marks] = @intCast(at);
        q.n_marks += 1;
        out.?.* = i + 1;
        return OK;
    }
    return fail(E_FULL, "the seq holds its {d} snapshots: drop one", .{max_snaps});
}

export fn flerp_snapshot_restore(p: ?*anyopaque, seq: i32, snap: u64) c_int {
    const ex = of(p);
    const q = seqOf(ex, seq) orelse return fail(E_ARGUMENT, "seq {d} is not open", .{seq});
    const s = snapOf(q, snap) orelse return fail(E_ARGUMENT, "not a snapshot of this seq", .{});
    if (q.live or q.prompt.items.len != 0) return fail(E_STATE, "strixfold restores a state only into a new prompt pass: flerp_seq_reset first", .{});
    if (s.saved == null) return fail(E_STATE, "the prompt pass kept no state at {d}: states are kept at chunk ends only (multiples of {d}, a window before the prompt's end)", .{ s.at, dsv41.model.chunk_rows });
    q.prompt.appendSlice(g_gpa, s.prefix) catch return fail(E_FAILED, "out of memory", .{});
    q.reuse_saved = s.saved;
    q.reuse_at = s.at;
    return OK;
}

export fn flerp_snapshot_drop(p: ?*anyopaque, seq: i32, snap: u64) c_int {
    const ex = of(p);
    const q = seqOf(ex, seq) orelse return fail(E_ARGUMENT, "seq {d} is not open", .{seq});
    const s = snapOf(q, snap) orelse return fail(E_ARGUMENT, "not a snapshot of this seq", .{});
    if (q.reuse_saved != null and q.reuse_saved == s.saved and !q.live) return fail(E_STATE, "the snapshot is the seq's restore point: run or reset the seq first", .{});
    dropSnap(ex, s);
    return OK;
}

export fn flerp_snapshot_pos(p: ?*const anyopaque, seq: i32, snap: u64) i64 {
    const ex: *Ex = @constCast(@ptrCast(@alignCast(p.?)));
    const q = seqOf(ex, seq) orelse return E_ARGUMENT;
    const s = snapOf(q, snap) orelse return E_ARGUMENT;
    if (q.live and s.saved == null) return E_STATE; // the prompt pass ran and kept nothing there
    return s.at;
}

export fn flerp_snapshot_bytes(p: ?*const anyopaque, seq: i32, snap: u64, from: i64) i64 {
    _ = .{ p, seq, snap, from };
    return fail(E_UNSUPPORTED, "not mapped: the family's export_state exists (f64b5ff) but flerp_state_import carries no prefix ids (caps: no FLERP_CAP_SNAPSHOT_BYTES)", .{});
}

export fn flerp_snapshot_export(p: ?*anyopaque, seq: i32, snap: u64, from: i64, bytes: ?*anyopaque, cap: usize, written: ?*usize) c_int {
    _ = .{ p, seq, snap, from, bytes, cap, written };
    return fail(E_UNSUPPORTED, "not mapped: the family's export_state exists (f64b5ff); see the file comment", .{});
}

export fn flerp_state_import(p: ?*anyopaque, seq: i32, bytes: ?*const anyopaque, n_bytes: usize, n: i64, from: i64) c_int {
    _ = .{ p, seq, bytes, n_bytes, n, from };
    return fail(E_UNSUPPORTED, "not mapped: the family's import_state exists (f64b5ff) but needs the prefix's token ids, which this call does not carry", .{});
}

export fn flerp_snapshot_write(p: ?*anyopaque, seq: i32, snap: u64, dir: [*c]const u8, key: u64) c_int {
    _ = .{ p, seq, snap, dir, key };
    return fail(E_UNSUPPORTED, "not mapped: native.Snaps.write exists since d21ae2e (states on each rank's disk); see the file comment", .{});
}

export fn flerp_snapshot_read(p: ?*anyopaque, seq: i32, dir: [*c]const u8, key: u64, at: i64, out: ?*u64) c_int {
    _ = .{ p, seq, dir, key, at, out };
    return fail(E_UNSUPPORTED, "not mapped: native.Snaps.read exists since d21ae2e but the state's prefix ids are not in this call", .{});
}

export fn flerp_get_stats(p: ?*const anyopaque, out: ?*c.flerp_stats) c_int {
    const ex: *const Ex = @ptrCast(@alignCast(p.?));
    const o = out orelse return fail(E_ARGUMENT, "flerp_stats is null", .{});
    if (o.size < @sizeOf(c.flerp_stats)) return fail(E_ARGUMENT, "flerp_stats is short", .{});
    const size = o.size;
    o.* = std.mem.zeroes(c.flerp_stats);
    o.size = size;
    o.forwards = ex.forwards;
    o.forward_tokens = ex.forward_tokens;
    o.verifies = ex.verifies;
    o.verify_rows = ex.verify_rows;
    o.kept_rows = ex.kept_rows;
    o.dropped_rows = ex.dropped_rows;
    o.draft_tokens = ex.draft_tokens;
    o.snapshots_saved = ex.saved;
    o.snapshots_restored = ex.restored;
    o.device_bytes = -1;
    return OK;
}

export fn flerp_stats_json(p: ?*const anyopaque, out: [*c]u8, cap: usize) usize {
    const ex: *Ex = @constCast(@ptrCast(@alignCast(p.?)));
    // the drafter's own counters are written per stream at its release (Stream.telemetry_buf): the last freed seq's
    var best: []const u8 = "{}";
    for (ex.seqs) |*q| if (q.live and q.stream.telemetry_len > 0) {
        best = q.stream.telemetry_buf[0..q.stream.telemetry_len];
    };
    if (out != null and cap > 0) {
        const m = @min(best.len, cap - 1);
        @memcpy(out[0..m], best[0..m]);
        out[m] = 0;
    }
    return best.len;
}
