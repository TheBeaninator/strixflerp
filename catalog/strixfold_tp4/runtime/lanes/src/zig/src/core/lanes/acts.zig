//! Per-request research switches: hidden-state dumps at tap layers, picture-row injection and the
//! first answer token's logits, asked for in a request's `acts` field (server/acts_fields.zig) and honoured only with
//! the admin key. This file holds what the server and the engine share: the request's switches (Spec), what an engine
//! that takes them tells the server (Info), and the Writer that puts dumps on disk off the engine thread.
//!
//! Bounds. The engine thread never waits on the Writer: it asks for room (reserve), copies its rows into memory it
//! then owns, and hands the job over (commit). Room is refused, and counted, when the queue's bytes (ACTS_QUEUE_MB),
//! its job slots, the request's bytes (ACTS_REQUEST_MB) or the dump directory's total (ACTS_TOTAL_MB) would be passed.
//! A refused dump is dropped; the request's reply is unchanged and its done.json says what was dropped and why.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// The most tap layers a request may name (the acts build's TF_DS_HIDDEN_LAYERS cap).
pub const max_taps = 16;
/// The tap after the embedding (the streams as embedded), `embed` in a request.
pub const embed_tap: i32 = -1;

/// A request's switches. Off (every bool false): the request is served exactly as without this file.
pub const Spec = struct {
    /// `acts-<boot seconds, hex>-<sequence>`: the request's subdirectory of the dump directory.
    id: []const u8 = "",
    /// The image-span rows of every chunk at the tap layers; the image prompt runs at replay 0 (every row through
    /// every layer), in chunks the split never cuts.
    dump: bool = false,
    /// A text-only prompt of at most the window's rows: every row at the tap layers (its served pass is replay 0).
    text: bool = false,
    /// Pictures whose digest names a file in the rows directory had their span rows replaced (the server did it).
    rows: bool = false,
    /// The first answer token's logits (f32, the vocabulary) as request.f32 and request.json.
    logits: bool = false,
    taps: [max_taps]i32 = @splat(0),
    n_taps: u8 = 0,
    /// The Writer's record of this request (the server opens it before submitting and closes it after the engine's
    /// end); null: nothing is written.
    req: ?*Req = null,

    /// Any switch on: the request is never kept in or resumed from a prompt cache, kept state or sf-disk.
    pub fn active(s: *const Spec) bool {
        return s.dump or s.text or s.rows or s.logits;
    }

    /// The taps are read (a dump or a text dump).
    pub fn tapsOn(s: *const Spec) bool {
        return s.dump or s.text;
    }

    pub fn tapList(s: *const Spec) []const i32 {
        return s.taps[0..s.n_taps];
    }
};

/// Whether `v` is a tap of a model with `n_layers` decoder layers: -1 (embed) to n_layers - 1 (the head's input).
pub fn tapOk(v: i64, n_layers: usize) bool {
    return v >= embed_tap and v < @as(i64, @intCast(n_layers));
}

/// The taps when a request names none: four evenly spaced layers (i * L / 5 - 1, i = 1..4) and the head's input, as
/// the acts build's TF_DS_HIDDEN_LAYERS unset.
pub fn defaultTaps(n_layers: usize, out: *[max_taps]i32) u8 {
    for (1..5) |i| out[i - 1] = @as(i32, @intCast(i * n_layers / 5)) - 1;
    out[4] = @as(i32, @intCast(n_layers)) - 1;
    return 5;
}

/// The bytes a dump file of `taps` taps over `rows` rows of `row_bytes` bytes takes (SFHD v1: model.zig Acts.encode).
pub fn dumpBytes(taps: usize, rows: usize, row_bytes: usize) u64 {
    return 40 + 4 * @as(u64, taps) + 16 * @as(u64, rows) + @as(u64, taps) * rows * row_bytes;
}

/// What an engine that takes `acts` tells the server (engine_api.Info.acts). Null there: the field is refused.
pub const Info = struct {
    /// The model's decoder layers (taps -1 .. layers - 1) and its window (a text dump's longest prompt).
    layers: u32,
    window: u32,
    /// The bytes of one picture row (BF16 hidden): an injected rows file is exactly tokens x row_bytes.
    row_bytes: u32,
    /// ACTS_ROWS_DIR: where `rows` looks for <digest hex>.bin (empty: `rows` is refused).
    rows_dir: []const u8 = "",
    writer: *Writer,
};

pub const Limits = struct {
    /// Bytes one request may write (ACTS_REQUEST_MB).
    request_bytes: u64 = 1 << 30,
    /// Bytes the dump directory may hold (ACTS_TOTAL_MB): past it new requests are refused and dumps dropped.
    total_bytes: u64 = 16 << 30,
    /// Bytes of copied rows waiting for the Writer (ACTS_QUEUE_MB): the engine thread's only memory for dumps.
    queue_bytes: u64 = 512 << 20,
    /// Jobs waiting at most.
    jobs: u32 = 64,
};

/// ACTS_REQUEST_MB, ACTS_TOTAL_MB, ACTS_QUEUE_MB as the conf gives them (TF_DS_ACTS_*): whole MiB, unset or empty the
/// default (1024, 16384, 512); 0 or anything else BadActsSetting. The queue must hold a request's cap or less.
pub fn limitsFrom(request_mb: ?[]const u8, total_mb: ?[]const u8, queue_mb: ?[]const u8) !Limits {
    const mib = struct {
        fn of(v: ?[]const u8, default: u64) !u64 {
            const t = v orelse return default << 20;
            if (t.len == 0) return default << 20;
            const n = std.fmt.parseInt(u64, t, 10) catch return error.BadActsSetting;
            if (n == 0 or n > 1 << 24) return error.BadActsSetting;
            return n << 20;
        }
    };
    const l: Limits = .{ .request_bytes = try mib.of(request_mb, 1024), .total_bytes = try mib.of(total_mb, 16384), .queue_bytes = try mib.of(queue_mb, 512) };
    if (l.request_bytes > l.total_bytes) return error.BadActsSetting;
    return l;
}

/// Why a dump was not written (done.json "drops").
pub const Drop = enum { queue, request_cap, total_cap, memory, tap };

/// A request's record: opened by the server, written to by the engine thread's jobs, closed by the server once the
/// engine is done with the request, then finished by the Writer (done.json) once its last job is written.
pub const Req = struct {
    id: [64]u8 = undefined,
    id_len: u8 = 0,
    pending: u32 = 0, // jobs reserved or queued, not yet written
    closed: bool = false,
    finish: [24]u8 = undefined,
    finish_len: u8 = 0,
    charged: u64 = 0, // bytes reserved for this request (its cap)
    files: u32 = 0,
    bytes: u64 = 0,
    drops: [5]u32 = .{ 0, 0, 0, 0, 0 },
    errors: u32 = 0,
    first_error: [48]u8 = undefined,
    first_error_len: u8 = 0,
    next: ?*Req = null, // the Writer's list of closed requests waiting for done.json

    pub fn name(r: *const Req) []const u8 {
        return r.id[0..r.id_len];
    }

    fn dropped(r: *const Req) u32 {
        var n: u32 = 0;
        for (r.drops) |d| n += d;
        return n;
    }
};

/// What a job wrote: its files and their bytes.
pub const Wrote = struct { files: u32 = 0, bytes: u64 = 0 };

/// A dump handed to the Writer. `write` runs on the Writer's thread (it encodes and writes into `dir`, the request's
/// directory, which exists); `free` always runs after it (or instead of it when the Writer stops first).
pub const Job = struct {
    ctx: *anyopaque,
    req: *Req,
    held: u64, // queue bytes it holds until written
    est: u64, // bytes it will write (charged to the request and the total at reserve)
    write: *const fn (ctx: *anyopaque, gpa: Allocator, io: std.Io, dir: []const u8) anyerror!Wrote,
    free: *const fn (ctx: *anyopaque, gpa: Allocator) void,
};

/// The bounded queue and its thread. Every field past `limits` is under `mu` (a spin lock held only for bookkeeping,
/// never across I/O), except the atomics.
pub const Writer = struct {
    gpa: Allocator,
    io: std.Io,
    dir: []const u8, // ACTS_DIR (owned)
    limits: Limits,
    mu: std.atomic.Mutex = .unlocked,
    ring: []Job = &.{},
    head: u32 = 0,
    count: u32 = 0,
    reserved: u32 = 0, // slots promised to reserve()s not yet committed
    queued_bytes: u64 = 0,
    disk_bytes: u64 = 0, // the directory's bytes: a scan at open (and rescans), plus what this Writer wrote since
    inflight: u64 = 0, // bytes reserved and not yet written
    finals: ?*Req = null,
    live: u32 = 0, // requests opened and not yet finished
    scanned_ns: i96 = 0,
    drops: [5]std.atomic.Value(u64) = .{ .init(0), .init(0), .init(0), .init(0), .init(0) },
    stop: std.atomic.Value(u32) = .init(0),
    thread: ?std.Thread = null,

    /// The Writer on the heap with its thread running; `dir` is created (one level) and its bytes counted.
    pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, limits: Limits) !*Writer {
        if (dir.len == 0 or dir.len > 400) return error.BadActsDir;
        const w = try gpa.create(Writer);
        errdefer gpa.destroy(w);
        w.* = .{ .gpa = gpa, .io = io, .dir = try gpa.dupe(u8, dir), .limits = limits };
        errdefer gpa.free(w.dir);
        std.Io.Dir.cwd().createDirPath(io, w.dir) catch return error.BadActsDir;
        w.ring = try gpa.alloc(Job, @max(limits.jobs, 1));
        errdefer gpa.free(w.ring);
        w.disk_bytes = scanBytes(io, w.dir);
        w.scanned_ns = std.Io.Clock.awake.now(io).toNanoseconds();
        w.thread = try std.Thread.spawn(.{}, run, .{w});
        return w;
    }

    /// Drains the queue, finishes every closed request, and frees the Writer (after the server and the engine stopped:
    /// the server closes every request it opened once the engine's end has come).
    pub fn close(w: *Writer) void {
        w.stop.store(1, .release);
        if (w.thread) |t| t.join();
        w.thread = null;
        w.gpa.free(w.ring);
        w.gpa.free(w.dir);
        w.gpa.destroy(w);
    }

    fn lock(w: *Writer) void {
        while (!w.mu.tryLock()) std.Thread.yield() catch {};
    }

    /// The server, before submitting a request with `acts` on: its record (null: out of memory).
    pub fn openReq(w: *Writer, id: []const u8) ?*Req {
        if (id.len == 0 or id.len > 64) return null;
        const r = w.gpa.create(Req) catch return null;
        r.* = .{};
        @memcpy(r.id[0..id.len], id);
        r.id_len = @intCast(id.len);
        w.lock();
        defer w.mu.unlock();
        w.live += 1;
        return r;
    }

    /// The server, once the engine's end has come (nothing reserves for `r` after this): done.json follows its last job.
    pub fn closeReq(w: *Writer, r: *Req, finish: []const u8) void {
        w.lock();
        defer w.mu.unlock();
        const n = @min(finish.len, r.finish.len);
        @memcpy(r.finish[0..n], finish[0..n]);
        r.finish_len = @intCast(n);
        r.closed = true;
        r.next = w.finals;
        w.finals = r;
    }

    /// The engine thread: room for a job holding `held` queue bytes that writes `est` bytes. Never waits; false (and
    /// counted on the request and the Writer) when a bound would be passed.
    pub fn reserve(w: *Writer, r: *Req, held: u64, est: u64) bool {
        w.lock();
        defer w.mu.unlock();
        const why: ?Drop = if (r.charged + est > w.limits.request_bytes)
            .request_cap
        else if (w.disk_bytes + w.inflight + est > w.limits.total_bytes)
            .total_cap
        else if (w.queued_bytes + held > w.limits.queue_bytes or w.count + w.reserved >= w.ring.len)
            .queue
        else
            null;
        if (why) |d| {
            w.noteDropLocked(r, d);
            return false;
        }
        w.reserved += 1;
        w.queued_bytes += held;
        w.inflight += est;
        r.charged += est;
        r.pending += 1;
        return true;
    }

    /// The engine thread: a reserve() whose job could not be made (its memory): the room goes back, counted.
    pub fn unreserve(w: *Writer, r: *Req, held: u64, est: u64, why: Drop) void {
        w.lock();
        defer w.mu.unlock();
        w.reserved -= 1;
        w.queued_bytes -= held;
        w.inflight -= est;
        r.charged -= est;
        r.pending -= 1;
        w.noteDropLocked(r, why);
    }

    /// The engine thread: the reserved job, now owning its memory, to the queue (a slot was promised: never fails).
    pub fn commit(w: *Writer, job: Job) void {
        w.lock();
        defer w.mu.unlock();
        w.reserved -= 1;
        w.ring[(w.head + w.count) % w.ring.len] = job;
        w.count += 1;
    }

    /// The engine thread: a dump that was not made (a chunk not tappable whole), counted on the request and the Writer.
    pub fn drop(w: *Writer, r: *Req, d: Drop) void {
        w.lock();
        defer w.mu.unlock();
        w.noteDropLocked(r, d);
    }

    fn noteDropLocked(w: *Writer, r: *Req, d: Drop) void {
        r.drops[@intFromEnum(d)] += 1;
        _ = w.drops[@intFromEnum(d)].fetchAdd(1, .monotonic);
    }

    /// Dumps dropped since the Writer opened, by reason.
    pub fn dropped(w: *Writer, d: Drop) u64 {
        return w.drops[@intFromEnum(d)].load(.monotonic);
    }

    /// The server, before admitting a request: the bytes the directory holds with what is in flight, and whether that
    /// is past the total (ACTS_TOTAL_MB). Past it, the directory is counted again at most every 10 s (files a person
    /// deleted come off).
    pub fn overTotal(w: *Writer, used: *u64) bool {
        w.lock();
        var now_used = w.disk_bytes + w.inflight;
        const stale = std.Io.Clock.awake.now(w.io).toNanoseconds() - w.scanned_ns > 10 * std.time.ns_per_s;
        w.mu.unlock();
        if (now_used >= w.limits.total_bytes and stale) {
            const disk = scanBytes(w.io, w.dir);
            w.lock();
            w.disk_bytes = disk;
            w.scanned_ns = std.Io.Clock.awake.now(w.io).toNanoseconds();
            now_used = w.disk_bytes + w.inflight;
            w.mu.unlock();
        }
        used.* = now_used;
        return now_used >= w.limits.total_bytes;
    }

    fn pop(w: *Writer) ?Job {
        w.lock();
        defer w.mu.unlock();
        if (w.count == 0) return null;
        const j = w.ring[w.head];
        w.head = @intCast((w.head + 1) % w.ring.len);
        w.count -= 1;
        return j;
    }

    /// One closed request whose jobs are all written, taken off the list (null: none).
    fn takeFinal(w: *Writer) ?*Req {
        w.lock();
        defer w.mu.unlock();
        var link: *?*Req = &w.finals;
        while (link.*) |r| {
            if (r.pending == 0) {
                link.* = r.next;
                w.live -= 1;
                return r;
            }
            link = &r.next;
        }
        return null;
    }

    fn reqDir(w: *const Writer, buf: []u8, r: *const Req) ![]const u8 {
        return std.fmt.bufPrint(buf, "{s}/{s}", .{ w.dir, r.name() });
    }

    fn runJob(w: *Writer, job: Job) void {
        defer job.free(job.ctx, w.gpa);
        var db: [512]u8 = undefined;
        var got: Wrote = .{};
        var failed: ?anyerror = null;
        if (w.reqDir(&db, job.req)) |dir| {
            if (std.Io.Dir.cwd().createDirPath(w.io, dir)) |_| {
                got = job.write(job.ctx, w.gpa, w.io, dir) catch |e| blk: {
                    failed = e;
                    break :blk .{};
                };
            } else |e| failed = e;
        } else |e| failed = e;
        w.lock();
        defer w.mu.unlock();
        w.queued_bytes -= job.held;
        w.inflight -= job.est;
        w.disk_bytes += got.bytes;
        job.req.pending -= 1;
        job.req.files += got.files;
        job.req.bytes += got.bytes;
        if (failed) |e| {
            job.req.errors += 1;
            if (job.req.first_error_len == 0) {
                const n = @min(@errorName(e).len, job.req.first_error.len);
                @memcpy(job.req.first_error[0..n], @errorName(e)[0..n]);
                job.req.first_error_len = @intCast(n);
            }
        }
    }

    /// done.json in the request's directory (created when no job made it), then the record freed.
    fn finishReq(w: *Writer, r: *Req) void {
        defer w.gpa.destroy(r);
        var db: [512]u8 = undefined;
        const dir = w.reqDir(&db, r) catch return;
        std.Io.Dir.cwd().createDirPath(w.io, dir) catch return;
        var tb: [768]u8 = undefined;
        const text = doneJson(&tb, r) catch return;
        var pb: [600]u8 = undefined;
        const path = std.fmt.bufPrint(&pb, "{s}/done.json", .{dir}) catch return;
        std.Io.Dir.cwd().writeFile(w.io, .{ .sub_path = path, .data = text }) catch return;
        w.lock();
        w.disk_bytes += text.len;
        w.mu.unlock();
    }

    fn run(w: *Writer) void {
        while (true) {
            if (w.pop()) |job| {
                w.runJob(job);
                continue;
            }
            if (w.takeFinal()) |r| {
                w.finishReq(r);
                continue;
            }
            if (w.stop.load(.acquire) != 0) break;
            std.Io.sleep(w.io, .fromMilliseconds(2), .awake) catch {};
        }
        // stopping: the closed requests left are finished as they stand
        while (true) {
            w.lock();
            const r = w.finals;
            if (r) |x| w.finals = x.next;
            w.mu.unlock();
            const x = r orelse break;
            w.finishReq(x);
        }
    }
};

/// done.json: {"id", "finish", "files", "bytes", "dropped", "drops": {queue, request_cap, total_cap, memory, tap},
/// "errors", "error"}.
pub fn doneJson(buf: []u8, r: *const Req) ![]const u8 {
    return std.fmt.bufPrint(buf, "{{\"id\": \"{s}\", \"finish\": \"{s}\", \"files\": {d}, \"bytes\": {d}, \"dropped\": {d}, " ++
        "\"drops\": {{\"queue\": {d}, \"request_cap\": {d}, \"total_cap\": {d}, \"memory\": {d}, \"tap\": {d}}}, \"errors\": {d}, \"error\": \"{s}\"}}\n", .{
        r.name(),     r.finish[0..r.finish_len], r.files,     r.bytes,     r.dropped(),
        r.drops[0],   r.drops[1],                r.drops[2],  r.drops[3],  r.drops[4], r.errors,
        r.first_error[0..r.first_error_len],
    });
}

/// The bytes of the regular files in `dir` and its subdirectories one level down (each request's directory).
pub fn scanBytes(io: std.Io, dir: []const u8) u64 {
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var total: u64 = 0;
    var it = d.iterate();
    while (it.next(io) catch null) |e| switch (e.kind) {
        .file => total += fileSize(io, d, e.name),
        .directory => {
            var sub = d.openDir(io, e.name, .{ .iterate = true }) catch continue;
            defer sub.close(io);
            var it2 = sub.iterate();
            while (it2.next(io) catch null) |f| if (f.kind == .file) {
                total += fileSize(io, sub, f.name);
            };
        },
        else => {},
    };
    return total;
}

fn fileSize(io: std.Io, dir: std.Io.Dir, name: []const u8) u64 {
    var file = dir.openFile(io, name, .{}) catch return 0;
    defer file.close(io);
    const st = file.stat(io) catch return 0;
    return @intCast(st.size);
}

var boot_s: std.atomic.Value(u64) = .init(0);
var seq: std.atomic.Value(u64) = .init(0);

/// `acts-<boot unix seconds, hex>-<sequence>`, unique across restarts of this process.
pub fn formatId(buf: []u8) []const u8 {
    var cur = boot_s.load(.acquire);
    if (cur == 0) {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.REALTIME, &ts);
        const raw: u64 = if (ts.sec <= 0) 1 else @intCast(ts.sec);
        cur = if (boot_s.cmpxchgStrong(0, raw, .acq_rel, .acquire)) |old| old else raw;
    }
    const n = seq.fetchAdd(1, .monotonic) + 1;
    return std.fmt.bufPrint(buf, "acts-{x}-{d}", .{ cur, n }) catch "acts-0-1";
}

// --- tests ---------------------------------------------------------------------

const testing = std.testing;

fn tmpDirPath(a: Allocator, tmp: *const testing.TmpDir) ![]u8 {
    return std.fmt.allocPrint(a, "{s}/{s}", .{ testing.TmpDir.parent_dir_path, &tmp.sub_path });
}

const TestJob = struct {
    data: []u8,
    name: []const u8,
    fn write(ctx: *anyopaque, _: Allocator, io: std.Io, dir: []const u8) anyerror!Wrote {
        const j: *TestJob = @ptrCast(@alignCast(ctx));
        var pb: [600]u8 = undefined;
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&pb, "{s}/{s}", .{ dir, j.name }), .data = j.data });
        return .{ .files = 1, .bytes = j.data.len };
    }
    fn free(ctx: *anyopaque, gpa: Allocator) void {
        const j: *TestJob = @ptrCast(@alignCast(ctx));
        gpa.free(j.data);
        gpa.destroy(j);
    }
};

fn submitTest(w: *Writer, r: *Req, name: []const u8, n: usize) !bool {
    if (!w.reserve(r, n, n)) return false;
    const j = try w.gpa.create(TestJob);
    j.* = .{ .data = try w.gpa.alloc(u8, n), .name = name };
    @memset(j.data, 7);
    w.commit(.{ .ctx = j, .req = r, .held = n, .est = n, .write = TestJob.write, .free = TestJob.free });
    return true;
}

fn readDone(a: Allocator, io: std.Io, dir: []const u8, id: []const u8) ![]u8 {
    var pb: [600]u8 = undefined;
    const path = try std.fmt.bufPrint(&pb, "{s}/{s}/done.json", .{ dir, id });
    var tries: usize = 0;
    while (tries < 2000) : (tries += 1) {
        if (std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(4096))) |t| return t else |_| {}
        std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
    }
    return error.NoDoneJson;
}

test "acts taps: -1 (embed) to layers - 1 (head); the default is four evenly spaced layers and the head input" {
    try testing.expect(tapOk(-1, 43) and tapOk(0, 43) and tapOk(42, 43));
    try testing.expect(!tapOk(-2, 43) and !tapOk(43, 43) and !tapOk(1 << 40, 43));
    var t: [max_taps]i32 = undefined;
    const n = defaultTaps(43, &t);
    try testing.expectEqualSlices(i32, &.{ 7, 16, 24, 33, 42 }, t[0..n]);
    try testing.expectEqual(@as(u64, 40 + 8 + 16 * 3 + 2 * 3 * 10), dumpBytes(2, 3, 10));
    var s: Spec = .{};
    try testing.expect(!s.active() and !s.tapsOn());
    s.logits = true;
    try testing.expect(s.active() and !s.tapsOn());
    var b: [64]u8 = undefined;
    const id1 = try testing.allocator.dupe(u8, formatId(&b));
    defer testing.allocator.free(id1);
    try testing.expect(std.mem.startsWith(u8, id1, "acts-"));
    try testing.expect(!std.mem.eql(u8, id1, formatId(&b)));
}

test "acts limits: MiB from the conf, the defaults when unset, nonsense refused" {
    const d = try limitsFrom(null, "", null);
    try testing.expectEqual(@as(u64, 1024 << 20), d.request_bytes);
    try testing.expectEqual(@as(u64, 16384 << 20), d.total_bytes);
    try testing.expectEqual(@as(u64, 512 << 20), d.queue_bytes);
    const l = try limitsFrom("64", "2048", "128");
    try testing.expectEqual(@as(u64, 64 << 20), l.request_bytes);
    try testing.expectError(error.BadActsSetting, limitsFrom("0", null, null));
    try testing.expectError(error.BadActsSetting, limitsFrom("1x", null, null));
    try testing.expectError(error.BadActsSetting, limitsFrom("4096", "2048", null));
}

test "acts writer: jobs land in the request's directory, done.json follows the last one, and the bytes are counted" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const dir = try tmpDirPath(a, &tmp);
    defer a.free(dir);
    try tmp.dir.writeFile(io, .{ .sub_path = "older.bin", .data = "0123456789" });
    const w = try Writer.open(a, io, dir, .{ .request_bytes = 1000, .total_bytes = 10_000, .queue_bytes = 1000, .jobs = 4 });
    defer w.close();
    var used: u64 = 0;
    try testing.expect(!w.overTotal(&used));
    try testing.expectEqual(@as(u64, 10), used); // the file already there
    const r = w.openReq("acts-t-1").?;
    try testing.expect(try submitTest(w, r, "001-0-4.bin", 100));
    try testing.expect(try submitTest(w, r, "002-4-4.bin", 200));
    w.closeReq(r, "stop");
    const done = try readDone(a, io, dir, "acts-t-1");
    defer a.free(done);
    try testing.expect(std.mem.indexOf(u8, done, "\"files\": 2, \"bytes\": 300, \"dropped\": 0") != null);
    try testing.expect(std.mem.indexOf(u8, done, "\"finish\": \"stop\"") != null);
    const f = try tmp.dir.readFileAlloc(io, "acts-t-1/002-4-4.bin", a, .limited(1000));
    defer a.free(f);
    try testing.expectEqual(@as(usize, 200), f.len);
}

test "acts writer: the request cap, the total cap and the queue refuse without waiting, each counted" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const dir = try tmpDirPath(a, &tmp);
    defer a.free(dir);
    const w = try Writer.open(a, io, dir, .{ .request_bytes = 250, .total_bytes = 400, .queue_bytes = 150, .jobs = 2 });
    defer w.close();
    const r = w.openReq("acts-t-2").?;
    // the queue: 150 bytes held at most (nothing written yet: reserve directly, so the thread cannot drain it)
    try testing.expect(w.reserve(r, 100, 100));
    try testing.expect(!w.reserve(r, 100, 10)); // 200 queue bytes > 150
    // the request: 100 charged, 160 more passes 250
    try testing.expect(!w.reserve(r, 10, 160));
    w.unreserve(r, 100, 100, .memory); // the job could not be made
    try testing.expectEqual(@as(u32, 0), r.pending);
    // a second request: the directory's total (400): 300 fits, then 150 more does not
    const r2 = w.openReq("acts-t-3").?;
    try testing.expect(w.reserve(r2, 1, 240));
    try testing.expect(!w.reserve(r, 1, 200)); // 240 + 200 > 400
    w.unreserve(r2, 1, 240, .memory);
    try testing.expectEqual(@as(u64, 1), w.dropped(.queue));
    try testing.expectEqual(@as(u64, 1), w.dropped(.request_cap));
    try testing.expectEqual(@as(u64, 1), w.dropped(.total_cap));
    try testing.expectEqual(@as(u64, 2), w.dropped(.memory));
    w.closeReq(r, "length");
    w.closeReq(r2, "stop");
    const done = try readDone(a, io, dir, "acts-t-2");
    defer a.free(done);
    try testing.expect(std.mem.indexOf(u8, done, "\"dropped\": 4, \"drops\": {\"queue\": 1, \"request_cap\": 1, \"total_cap\": 1, \"memory\": 1, \"tap\": 0}") != null);
    const done2 = try readDone(a, io, dir, "acts-t-3");
    defer a.free(done2);
    try testing.expect(std.mem.indexOf(u8, done2, "\"files\": 0") != null);
}

test "acts writer: the job slots are a bound too, and a closed request with nothing written still gets done.json" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const dir = try tmpDirPath(a, &tmp);
    defer a.free(dir);
    const w = try Writer.open(a, io, dir, .{ .jobs = 1 });
    defer w.close();
    const r = w.openReq("acts-t-4").?;
    try testing.expect(w.reserve(r, 1, 1));
    try testing.expect(!w.reserve(r, 1, 1)); // the one slot is promised
    w.unreserve(r, 1, 1, .memory);
    try testing.expect(w.reserve(r, 1, 1));
    w.unreserve(r, 1, 1, .memory);
    w.closeReq(r, "cancelled");
    const done = try readDone(a, io, dir, "acts-t-4");
    defer a.free(done);
    try testing.expect(std.mem.indexOf(u8, done, "\"queue\": 1") != null);
}
