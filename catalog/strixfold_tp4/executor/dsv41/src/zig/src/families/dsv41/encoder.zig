//! Rank 0's picture encodes on a thread of their own (HIP), so the engine thread keeps decoding the other streams.
const std = @import("std");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;
const Image = lanes.stream.Image;
const Wake = lanes.stream.Wake;
const image_key = lanes.image_key;
const Digest = image_key.Digest;

/// The tower the thread drives: `bind` once on that thread, `rows` appends a picture's span rows, returns its tokens.
pub const Tower = struct {
    ctx: *anyopaque,
    bind: *const fn (ctx: *anyopaque) anyerror!void,
    rows: *const fn (ctx: *anyopaque, payload: []const u8, gpa: Allocator, out: *std.ArrayList(u8)) anyerror!usize,
};

const Job = struct {
    key: usize, // the request's images.ptr
    payloads: [][]align(16) u8, // copies of the pictures to encode (empty: rows came ready, skipped, or encoded)
    tokens: []usize, // each picture's span, as its encode returned it
    ranges: [][2]usize, // where each image's rows begin and end in `rows`
    skipped: []bool, // a resume covers its span: encoded only when a prompt pass asks for it (take)
    digests: []Digest, // each picture's identity (Image.digest, else its payload's): the picture cache's key
    rows: std.ArrayList(u8) = .empty,
    took: u32 = 0, // pictures whose rows came from the picture cache (Picture), for the job's log line
    encoded: u32 = 0, // pictures the tower ran for
    t0: i128 = 0, // when the job was queued
    state: enum { queued, running, done } = .queued,
    urgent: bool = false, // a prompt pass waits for it: the thread takes it first
    err: ?anyerror = null,
    gone: bool = false, // forgotten while it ran: the thread frees it
    wake: Wake,
};

/// A picture's rows kept by its digest after an encode (serve.conf ENCODE_CACHE_MIB, TF_DS_ENCODE_CACHE_MIB): a later
/// request with the same picture, at any position (a later turn that resumes inside or before its span, a picture
/// sent again, a regenerated or edited conversation, the same picture twice in one prompt) copies these instead of
/// running the tower (about 3.9 s a picture on the pair). The rows are the tower's output for the prepared payload
/// whose BLAKE3-256 is the key, so equal digests give equal rows.
/// The picture cache's budget when TF_DS_ENCODE_CACHE_MIB is unset: 512 MiB, about 50 pictures of 1,024 tokens (10 KiB of
/// rows a token), more than one request at IMAGE_TOKENS 32768 holds (320 MiB).
pub const default_picture_cache: usize = 512 << 20;

pub const Picture = struct {
    digest: Digest,
    rows: []u8,
    tokens: usize,
    used: u64, // the cache's clock at its last take or keep: the least recent goes first
};

pub const Encoder = struct {
    gpa: Allocator,
    io: std.Io,
    tower: Tower,
    cap: usize, // a request's image rows at most (Model.image_rows, serve.conf IMAGE_TOKENS): more fail at once, nothing encoded
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    done: std.Io.Condition = .init, // a job finished (a prompt pass waits in take)
    jobs: std.ArrayList(*Job) = .empty,
    closing: bool = false,
    thread: ?std.Thread = null,
    /// The picture cache (the lock held): at most `picture_cap` bytes of rows (0: off), least recently used first out.
    pictures: std.ArrayList(Picture) = .empty,
    picture_cap: usize = 0,
    picture_bytes: usize = 0,
    clock: u64 = 0,
    counts: struct { hits: u64 = 0, encoded: u64 = 0, kept: u64 = 0, evicted: u64 = 0 } = .{},

    pub fn init(gpa: Allocator, io: std.Io, tower: Tower, cap: usize) Encoder {
        return .{ .gpa = gpa, .io = io, .tower = tower, .cap = cap };
    }

    /// The picture cache's budget in MiB (TF_DS_ENCODE_CACHE_MIB as text: a whole number from 0 to 65536; 0 off).
    pub fn parseCacheMib(text: []const u8) error{BadEncodeCache}!usize {
        const mib = std.fmt.parseInt(usize, std.mem.trim(u8, text, " "), 10) catch return error.BadEncodeCache;
        if (mib > 65536) return error.BadEncodeCache;
        return mib << 20;
    }

    pub fn start(e: *Encoder) !void {
        e.thread = try std.Thread.spawn(.{}, run, .{e});
    }

    /// Ends the thread once its current picture is done, then frees every job.
    pub fn stop(e: *Encoder) void {
        e.lock();
        e.closing = true;
        e.cond.broadcast(e.io);
        e.done.broadcast(e.io);
        e.unlock();
        if (e.thread) |t| t.join();
        e.thread = null;
        for (e.jobs.items) |j| e.free(j);
        e.jobs.deinit(e.gpa);
        for (e.pictures.items) |p| e.gpa.free(p.rows);
        e.pictures.deinit(e.gpa);
        e.picture_bytes = 0;
    }

    /// The cached rows of picture `d`, marked used (the lock held); null: not kept.
    fn cached(e: *Encoder, d: *const Digest) ?*Picture {
        for (e.pictures.items) |*p| if (std.mem.eql(u8, &p.digest, d)) {
            e.clock += 1;
            p.used = e.clock;
            return p;
        };
        return null;
    }

    /// Keeps a copy of picture `d`'s rows (the lock held), the least recently used going first past the budget; a
    /// picture larger than the whole budget, or a failed copy, is not kept (it is encoded again next time).
    fn keepPicture(e: *Encoder, d: *const Digest, rows: []const u8, tokens: usize) void {
        if (rows.len == 0 or rows.len > e.picture_cap) return;
        if (e.cached(d) != null) return;
        while (e.picture_bytes + rows.len > e.picture_cap) {
            var lru: usize = 0;
            for (e.pictures.items, 0..) |p, i| if (p.used < e.pictures.items[lru].used) {
                lru = i;
            };
            const p = e.pictures.swapRemove(lru);
            e.picture_bytes -= p.rows.len;
            e.gpa.free(p.rows);
            e.counts.evicted += 1;
        }
        e.pictures.ensureUnusedCapacity(e.gpa, 1) catch return;
        const copy = e.gpa.dupe(u8, rows) catch return;
        e.clock += 1;
        e.pictures.appendAssumeCapacity(.{ .digest = d.*, .rows = copy, .tokens = tokens, .used = e.clock });
        e.picture_bytes += copy.len;
        e.counts.kept += 1;
    }

    /// Picture `k` of job `j` from the cache (the lock held): its rows appended, its span set; false when not kept.
    fn takeCached(e: *Encoder, j: *Job, k: usize) !bool {
        if (e.picture_cap == 0) return false;
        const p = e.cached(&j.digests[k]) orelse return false;
        const begin = j.rows.items.len;
        try j.rows.appendSlice(e.gpa, p.rows);
        j.ranges[k] = .{ begin, j.rows.items.len };
        j.tokens[k] = p.tokens;
        j.took += 1;
        e.counts.hits += 1;
        return true;
    }

    /// The lane host's view (lanes.stream.Encode).
    pub fn hook(e: *Encoder) lanes.stream.Encode {
        return .{ .ctx = e, .start = startFn, .ready = readyFn, .forget = forgetFn };
    }

    fn of(ctx: *anyopaque) *Encoder {
        return @ptrCast(@alignCast(ctx));
    }

    fn startFn(ctx: *anyopaque, images: []const Image, from: u32, wake: Wake) void {
        of(ctx).submit(images, from, wake) catch |err| std.log.err("encoder: a request's pictures not queued ({s})", .{@errorName(err)});
    }

    fn readyFn(ctx: *anyopaque, images: []const Image) bool {
        return of(ctx).ready(images);
    }

    fn forgetFn(ctx: *anyopaque, images: []const Image) void {
        of(ctx).forget(images);
    }

    fn lock(e: *Encoder) void {
        e.mutex.lockUncancelable(e.io);
    }

    fn unlock(e: *Encoder) void {
        e.mutex.unlock(e.io);
    }

    /// Queues a request's pictures past `from` (a resume covers the rest), payloads copied (the request may end first).
    pub fn submit(e: *Encoder, images: []const Image, from: u32, wake: Wake) !void {
        const j = try e.gpa.create(Job);
        j.* = .{ .key = @intFromPtr(images.ptr), .payloads = &.{}, .tokens = &.{}, .ranges = &.{}, .skipped = &.{}, .digests = &.{}, .wake = wake, .t0 = std.Io.Clock.awake.now(e.io).toNanoseconds() };
        errdefer e.free(j);
        j.digests = try e.gpa.alloc(Digest, images.len);
        for (images, j.digests) |img, *d| d.* = img.digest orelse if (img.rows) @as(Digest, @splat(0)) else image_key.imageDigest(img.bytes);
        j.payloads = try e.gpa.alloc([]align(16) u8, images.len);
        for (j.payloads) |*p| p.* = &.{};
        j.tokens = try e.gpa.alloc(usize, images.len);
        @memset(j.tokens, 0);
        j.ranges = try e.gpa.alloc([2]usize, images.len);
        @memset(j.ranges, .{ 0, 0 });
        j.skipped = try e.gpa.alloc(bool, images.len);
        @memset(j.skipped, false);
        var rows: usize = 0;
        for (images) |img| rows += img.tokens;
        var any = false;
        e.lock();
        defer e.unlock();
        if (rows > e.cap) {
            j.state = .done;
            j.err = error.TooManyImageRows;
        } else for (images, j.payloads, j.skipped, 0..) |img, *p, *skip, k| {
            skip.* = !img.rows and @as(usize, img.at) + img.tokens <= from;
            if (img.rows or skip.*) continue;
            if (try e.takeCached(j, k)) continue; // encoded before: its rows now, no tower
            p.* = try e.gpa.alignedAlloc(u8, .@"16", img.bytes.len);
            @memcpy(p.*, img.bytes);
            any = true;
        }
        if (!any and j.err == null) {
            j.state = .done; // nothing to encode now
            e.logJob(j);
        }
        try e.jobs.append(e.gpa, j);
        e.cond.signal(e.io);
    }

    /// One line a job when its pictures are ready (the lock held): how many a resume skipped, how many the cache gave,
    /// how many the tower ran for (a skipped picture a pass asks for after all logs the job again).
    fn logJob(e: *Encoder, j: *const Job) void {
        if (j.digests.len == 0 or j.err != null) return;
        var skipped: usize = 0;
        for (j.skipped) |s| skipped += @intFromBool(s);
        const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(e.io).toNanoseconds() - j.t0)) / 1e6;
        std.log.info("encoder: {d} pictures, {d} skipped, {d} from the picture cache, {d} encoded, ready in {d:.1} ms; cache {d} pictures {d} MiB of {d}, hits {d}, evicted {d}", .{ j.digests.len, skipped, j.took, j.encoded, ms, e.pictures.items.len, e.picture_bytes >> 20, e.picture_cap >> 20, e.counts.hits, e.counts.evicted });
    }

    /// Whether a request's rows are ready, or its encode failed, or it never queued (its prompt pass then says why).
    pub fn ready(e: *Encoder, images: []const Image) bool {
        e.lock();
        defer e.unlock();
        const j = e.find(images) orelse return true;
        return j.state == .done;
    }

    /// Appends image `k`'s rows to `out` and returns its span (a skipped one encoded now, the caller waiting).
    pub fn take(e: *Encoder, images: []const Image, k: usize, gpa: Allocator, out: *std.ArrayList(u8)) !usize {
        e.lock();
        defer e.unlock();
        const j = e.find(images) orelse return error.NotEncoded;
        if (j.state != .done) return error.NotEncoded;
        if (j.err) |err| return err;
        if (j.skipped[k]) try e.encodeNow(j, k, images[k].bytes);
        try out.appendSlice(gpa, j.rows.items[j.ranges[k][0]..j.ranges[k][1]]);
        return j.tokens[k];
    }

    /// Picture `k` of a done job queued first and waited for (the lock held).
    fn encodeNow(e: *Encoder, j: *Job, k: usize, bytes: []const u8) !void {
        if (try e.takeCached(j, k)) { // encoded before: no wait
            j.skipped[k] = false;
            return e.logJob(j);
        }
        const p = try e.gpa.alignedAlloc(u8, .@"16", bytes.len);
        @memcpy(p, bytes);
        j.payloads[k] = p;
        j.skipped[k] = false;
        j.state = .queued;
        j.urgent = true;
        e.cond.signal(e.io);
        while (j.state != .done) {
            if (e.closing) return error.Closed;
            e.done.waitUncancelable(e.io, &e.mutex);
        }
        if (j.err) |err| return err;
    }

    /// Drops a request's job; one being encoded is freed by the thread when its picture is done.
    pub fn forget(e: *Encoder, images: []const Image) void {
        e.lock();
        defer e.unlock();
        const j = e.find(images) orelse return;
        if (j.state == .running) {
            j.gone = true;
            return;
        }
        e.remove(j);
        e.free(j);
    }

    fn find(e: *Encoder, images: []const Image) ?*Job {
        if (images.len == 0) return null;
        for (e.jobs.items) |j| if (j.key == @intFromPtr(images.ptr) and !j.gone) return j;
        return null;
    }

    fn remove(e: *Encoder, j: *Job) void {
        const i = std.mem.indexOfScalar(*Job, e.jobs.items, j) orelse return;
        _ = e.jobs.orderedRemove(i);
    }

    fn freePayloads(e: *Encoder, j: *Job) void {
        for (j.payloads) |*p| {
            e.gpa.free(p.*);
            p.* = &.{};
        }
    }

    /// Whether the thread should leave job `j` before its next picture: forgotten, or the encoder stopping.
    fn stopped(e: *Encoder, j: *const Job) bool {
        e.lock();
        defer e.unlock();
        return j.gone or e.closing;
    }

    fn free(e: *Encoder, j: *Job) void {
        e.freePayloads(j);
        e.gpa.free(j.payloads);
        e.gpa.free(j.tokens);
        e.gpa.free(j.ranges);
        e.gpa.free(j.skipped);
        e.gpa.free(j.digests);
        j.rows.deinit(e.gpa);
        e.gpa.destroy(j);
    }

    fn run(e: *Encoder) void {
        const bound = e.tower.bind(e.tower.ctx);
        e.lock();
        defer e.unlock();
        while (!e.closing) {
            const j = e.next() orelse {
                e.cond.waitUncancelable(e.io, &e.mutex);
                continue;
            };
            j.state = .running;
            e.unlock();
            var err: ?anyerror = null;
            if (bound) |_| {
                e.encode(j) catch |x| {
                    err = x;
                };
            } else |x| err = x;
            e.lock();
            j.state = .done;
            j.urgent = false;
            j.err = err;
            e.freePayloads(j);
            if (err == null) e.logJob(j);
            e.done.broadcast(e.io);
            if (j.gone) {
                e.remove(j);
                e.free(j);
                continue;
            }
            const wake = j.wake;
            e.unlock();
            wake.call(wake.ctx);
            e.lock();
        }
    }

    /// The queued job a prompt pass waits for, else the first queued (the lock held).
    fn next(e: *Encoder) ?*Job {
        var first: ?*Job = null;
        for (e.jobs.items) |x| if (x.state == .queued) {
            if (x.urgent) return x;
            if (first == null) first = x;
        };
        return first;
    }

    fn encode(e: *Encoder, j: *Job) !void {
        for (j.payloads, j.tokens, j.ranges, 0..) |p, *t, *r, k| if (p.len > 0) {
            if (e.stopped(j)) return error.EncodeCancelled;
            { // another job (or this one, earlier) encoded the same picture since it queued: its rows, no tower
                e.lock();
                defer e.unlock();
                if (try e.takeCached(j, k)) continue;
            }
            const begin = j.rows.items.len;
            t.* = try e.tower.rows(e.tower.ctx, p, e.gpa, &j.rows);
            r.* = .{ begin, j.rows.items.len };
            j.encoded += 1;
            e.lock();
            defer e.unlock();
            e.counts.encoded += 1;
            if (e.picture_cap > 0) e.keepPicture(&j.digests[k], j.rows.items[begin..], t.*);
        };
    }
};

/// A tower whose pictures are [n, v]: n rows of four bytes v; it holds each encode while `hold` is set.
const Fake = struct {
    hold: std.atomic.Value(bool) = .init(false),
    entered: std.atomic.Value(u32) = .init(0), // pictures whose encode has begun
    encodes: std.atomic.Value(u32) = .init(0),
    fail: bool = false,

    fn tower(f: *Fake) Tower {
        return .{ .ctx = f, .bind = bind, .rows = rows };
    }

    fn bind(_: *anyopaque) anyerror!void {}

    fn rows(ctx: *anyopaque, payload: []const u8, gpa: Allocator, out: *std.ArrayList(u8)) anyerror!usize {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        _ = f.entered.fetchAdd(1, .acq_rel);
        while (f.hold.load(.acquire)) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        _ = f.encodes.fetchAdd(1, .acq_rel);
        if (f.fail) return error.DeviceFault;
        try out.appendNTimes(gpa, payload[1], 4 * @as(usize, payload[0]));
        return payload[0];
    }
};

const Woken = struct {
    n: std.atomic.Value(u32) = .init(0),

    fn call(ctx: *anyopaque) void {
        const w: *Woken = @ptrCast(@alignCast(ctx));
        _ = w.n.fetchAdd(1, .acq_rel);
    }

    fn wake(w: *Woken) Wake {
        return .{ .ctx = w, .call = call };
    }
};

fn waitReady(e: *Encoder, images: []const Image) void {
    while (!e.ready(images)) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
}

test "an encoder hands each request its pictures' rows in order, the rows that came ready skipped" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 8);
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    const a = [_]Image{ .{ .at = 0, .tokens = 2, .bytes = &.{ 2, 7 } }, .{ .at = 3, .tokens = 1, .bytes = "rows", .rows = true }, .{ .at = 5, .tokens = 3, .bytes = &.{ 3, 9 } } };
    try e.submit(&a, 0, woken.wake());
    waitReady(&e, &a);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), try e.take(&a, 2, gpa, &out));
    try std.testing.expectEqual(@as(usize, 2), try e.take(&a, 0, gpa, &out));
    try std.testing.expectEqual(@as(usize, 20), out.items.len);
    try std.testing.expect(std.mem.allEqual(u8, out.items[0..12], 9) and std.mem.allEqual(u8, out.items[12..], 7));
    // the wake comes after ready
    while (woken.n.load(.acquire) == 0) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    try std.testing.expectEqual(@as(u32, 1), woken.n.load(.acquire));
    e.forget(&a);
    try std.testing.expect(e.ready(&a)); // no job: its prompt pass says why
    try std.testing.expectError(error.NotEncoded, e.take(&a, 0, gpa, &out));
}

test "a request forgotten while its picture encodes is freed by the thread, unwoken; a queued one at once" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    fake.hold.store(true, .release);
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 8);
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    const a = [_]Image{.{ .at = 0, .tokens = 1, .bytes = &.{ 1, 1 } }};
    const b = [_]Image{.{ .at = 0, .tokens = 1, .bytes = &.{ 1, 2 } }};
    try e.submit(&a, 0, woken.wake());
    try e.submit(&b, 0, woken.wake());
    while (true) { // the thread has taken a
        e.lock();
        const running = e.jobs.items[0].state == .running;
        e.unlock();
        if (running) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expect(!e.ready(&a) and !e.ready(&b));
    e.forget(&a);
    e.forget(&b);
    fake.hold.store(false, .release);
    while (true) {
        e.lock();
        const left = e.jobs.items.len;
        e.unlock();
        if (left == 0) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
    try std.testing.expectEqual(@as(u32, 0), woken.n.load(.acquire));
}

test "a failed encode is the request's error at its prompt pass" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{ .fail = true };
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 8);
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    const a = [_]Image{.{ .at = 0, .tokens = 1, .bytes = &.{ 1, 1 } }};
    try e.submit(&a, 0, woken.wake());
    waitReady(&e, &a);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try std.testing.expectError(error.DeviceFault, e.take(&a, 0, gpa, &out));
    e.forget(&a);
}

test "a request over the image row cap fails at once, its pictures never encoded" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 4);
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    const a = [_]Image{ .{ .at = 0, .tokens = 3, .bytes = &.{ 3, 1 } }, .{ .at = 4, .tokens = 2, .bytes = &.{ 2, 1 } } };
    try e.submit(&a, 0, woken.wake());
    try std.testing.expect(e.ready(&a));
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try std.testing.expectError(error.TooManyImageRows, e.take(&a, 0, gpa, &out));
    e.forget(&a);
    try std.testing.expectEqual(@as(u32, 0), fake.encodes.load(.acquire));
}

fn waitEntered(f: *Fake) void {
    while (f.entered.load(.acquire) == 0) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
}

const four = [_]Image{ .{ .at = 0, .tokens = 1, .bytes = &.{ 1, 1 } }, .{ .at = 2, .tokens = 1, .bytes = &.{ 1, 2 } }, .{ .at = 4, .tokens = 1, .bytes = &.{ 1, 3 } }, .{ .at = 6, .tokens = 1, .bytes = &.{ 1, 4 } } };

test "a request of four pictures forgotten during its first stops before the second" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    fake.hold.store(true, .release);
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 8);
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    try e.submit(&four, 0, woken.wake());
    waitEntered(&fake);
    e.forget(&four);
    fake.hold.store(false, .release);
    while (true) {
        e.lock();
        const left = e.jobs.items.len;
        e.unlock();
        if (left == 0) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
    try std.testing.expectEqual(@as(u32, 0), woken.n.load(.acquire));
}

test "stopping the encoder during the first of four pictures waits for that picture only" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    fake.hold.store(true, .release);
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 8);
    try e.start();
    var joined = false;
    errdefer if (!joined) {
        fake.hold.store(false, .release);
        e.stop();
    };
    var woken: Woken = .{};
    try e.submit(&four, 0, woken.wake());
    waitEntered(&fake);
    const stopper = try std.Thread.spawn(.{}, Encoder.stop, .{&e});
    while (true) {
        e.lock();
        const closing = e.closing;
        e.unlock();
        if (closing) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    fake.hold.store(false, .release);
    stopper.join();
    joined = true;
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
}

test "pictures a resume covers are not encoded at submit; one its pass still asks for is encoded then" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 16);
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    const a = [_]Image{ .{ .at = 0, .tokens = 2, .bytes = &.{ 2, 7 } }, .{ .at = 3, .tokens = 3, .bytes = &.{ 3, 8 } }, .{ .at = 8, .tokens = 1, .bytes = &.{ 1, 9 } } };
    try e.submit(&a, 6, woken.wake()); // a resume at 6 covers the first two
    waitReady(&e, &a);
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), try e.take(&a, 2, gpa, &out));
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
    try std.testing.expectEqual(@as(usize, 3), try e.take(&a, 1, gpa, &out)); // the resume fell short: encoded now
    try std.testing.expectEqual(@as(u32, 2), fake.encodes.load(.acquire));
    try std.testing.expectEqual(@as(usize, 4 + 12), out.items.len);
    try std.testing.expect(std.mem.allEqual(u8, out.items[0..4], 9) and std.mem.allEqual(u8, out.items[4..], 8));
    out.clearRetainingCapacity();
    try std.testing.expectEqual(@as(usize, 3), try e.take(&a, 1, gpa, &out)); // again: its rows kept, no second encode
    try std.testing.expectEqual(@as(usize, 12), out.items.len);
    try std.testing.expectEqual(@as(u32, 2), fake.encodes.load(.acquire));
    const b = [_]Image{.{ .at = 0, .tokens = 2, .bytes = &.{ 2, 5 } }};
    try e.submit(&b, 2, woken.wake()); // every picture covered: ready at once, nothing encoded
    try std.testing.expect(e.ready(&b));
    try std.testing.expectEqual(@as(u32, 2), fake.encodes.load(.acquire));
    e.forget(&a);
    e.forget(&b);
}

test "a resume inside a picture's span still encodes it at submit; a resume that falls back into a skipped one encodes it then" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 1024);
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    // a picture the resume covers, and one at [4000, 4200) the resume at 4096 lands inside
    const a = [_]Image{ .{ .at = 100, .tokens = 200, .bytes = &.{ 200, 7 } }, .{ .at = 4000, .tokens = 200, .bytes = &.{ 200, 8 } } };
    try e.submit(&a, 4096, woken.wake());
    waitReady(&e, &a);
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire)); // the second only, though it began before 4096
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 200), try e.take(&a, 1, gpa, &out)); // its whole span's rows, already there
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
    try std.testing.expect(out.items.len == 4 * 200 and std.mem.allEqual(u8, out.items, 8));
    e.forget(&a);
    // the hint said 6144 (past the picture) and the pass resumed at 4096 after all: the skipped picture on request
    const b = [_]Image{.{ .at = 4000, .tokens = 200, .bytes = &.{ 200, 9 } }};
    try e.submit(&b, 6144, woken.wake());
    try std.testing.expect(e.ready(&b));
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
    out.clearRetainingCapacity();
    try std.testing.expectEqual(@as(usize, 200), try e.take(&b, 0, gpa, &out));
    try std.testing.expectEqual(@as(u32, 2), fake.encodes.load(.acquire));
    try std.testing.expect(out.items.len == 4 * 200 and std.mem.allEqual(u8, out.items, 9));
    e.forget(&b);
}


/// A picture the Fake tower reads as [n, v]: n rows of four bytes v (its digest is its two bytes').
fn pic(at: u32, comptime n: u8, comptime v: u8) Image {
    return .{ .at = at, .tokens = n, .bytes = &.{ n, v } };
}

fn rowsOf(e: *Encoder, images: []const Image, gpa: Allocator) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    for (0..images.len) |k| if (!images[k].rows) {
        _ = try e.take(images, k, gpa, &out);
    };
    return out.toOwnedSlice(gpa);
}

test "picture cache: a later request with the same pictures at other positions takes their rows, the tower runs for new ones only" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 64);
    e.picture_cap = 1 << 20;
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    const a = [_]Image{ pic(0, 2, 7), pic(5, 3, 9) };
    try e.submit(&a, 0, woken.wake());
    waitReady(&e, &a);
    const ra = try rowsOf(&e, &a, gpa);
    defer gpa.free(ra);
    e.forget(&a);
    try std.testing.expectEqual(@as(u32, 2), fake.encodes.load(.acquire));
    // the next request: both pictures again, moved, and one new between them
    const b = [_]Image{ pic(1, 3, 9), pic(6, 1, 5), pic(9, 2, 7) };
    try e.submit(&b, 0, woken.wake());
    waitReady(&e, &b);
    const rb = try rowsOf(&e, &b, gpa);
    defer gpa.free(rb);
    e.forget(&b);
    try std.testing.expectEqual(@as(u32, 3), fake.encodes.load(.acquire)); // the new picture only
    try std.testing.expectEqual(@as(u64, 2), e.counts.hits);
    // the same rows a cold encoder gives
    var cold_fake: Fake = .{};
    var cold = Encoder.init(gpa, std.testing.io, cold_fake.tower(), 64);
    try cold.start();
    defer cold.stop();
    try cold.submit(&b, 0, woken.wake());
    waitReady(&cold, &b);
    const rc = try rowsOf(&cold, &b, gpa);
    defer gpa.free(rc);
    cold.forget(&b);
    try std.testing.expectEqualSlices(u8, rc, rb);
}

test "picture cache: a turn that resumes inside the pictures takes a skipped one's rows from the cache, no tower and no wait" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 64);
    e.picture_cap = 1 << 20;
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    const t1 = [_]Image{ pic(2, 3, 4), pic(8, 3, 6) };
    try e.submit(&t1, 0, woken.wake());
    waitReady(&e, &t1);
    const r1 = try rowsOf(&e, &t1, gpa);
    defer gpa.free(r1);
    e.forget(&t1);
    // turn 2: the cache guessed a resume at 6 (past the first picture): it is skipped, the second taken from the cache
    const t2 = [_]Image{ pic(2, 3, 4), pic(8, 3, 6) };
    try e.submit(&t2, 6, woken.wake());
    try std.testing.expect(e.ready(&t2)); // nothing left for the tower
    // the pass resumed earlier after all (its kept state went): the skipped picture's rows come from the cache
    const r2 = try rowsOf(&e, &t2, gpa);
    defer gpa.free(r2);
    e.forget(&t2);
    try std.testing.expectEqualSlices(u8, r1, r2);
    try std.testing.expectEqual(@as(u32, 2), fake.encodes.load(.acquire));
    try std.testing.expectEqual(@as(u64, 2), e.counts.hits);
}

test "picture cache: another picture of the same size at the same position is encoded, never the cached rows" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 64);
    e.picture_cap = 1 << 20;
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    const t1 = [_]Image{pic(2, 3, 4)};
    try e.submit(&t1, 0, woken.wake());
    waitReady(&e, &t1);
    e.forget(&t1);
    const t2 = [_]Image{pic(2, 3, 5)};
    try e.submit(&t2, 0, woken.wake());
    waitReady(&e, &t2);
    const r2 = try rowsOf(&e, &t2, gpa);
    defer gpa.free(r2);
    e.forget(&t2);
    try std.testing.expectEqual(@as(u32, 2), fake.encodes.load(.acquire));
    try std.testing.expectEqual(@as(u64, 0), e.counts.hits);
    try std.testing.expect(std.mem.allEqual(u8, r2, 5));
    // the server's digest is the key when it set one: equal bytes under another digest are another picture
    var d = image_key.imageDigest(&.{ 3, 4 });
    d[31] ^= 1;
    const t3 = [_]Image{.{ .at = 2, .tokens = 3, .bytes = &.{ 3, 4 }, .digest = d }};
    try e.submit(&t3, 0, woken.wake());
    waitReady(&e, &t3);
    e.forget(&t3);
    try std.testing.expectEqual(@as(u32, 3), fake.encodes.load(.acquire));
}

test "picture cache: past its budget the least recently used picture goes; one larger than the budget is never kept" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 64);
    e.picture_cap = 24; // three pictures of two rows (8 bytes each)
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    const Run = struct {
        fn one(enc: *Encoder, w: *Woken, img: Image) !void {
            const r = [_]Image{img};
            try enc.submit(&r, 0, w.wake());
            waitReady(enc, &r);
            var out: std.ArrayList(u8) = .empty;
            defer out.deinit(std.testing.allocator);
            _ = try enc.take(&r, 0, std.testing.allocator, &out);
            enc.forget(&r);
        }
    };
    try Run.one(&e, &woken, pic(0, 2, 1)); // A
    try Run.one(&e, &woken, pic(0, 2, 2)); // B
    try Run.one(&e, &woken, pic(0, 2, 3)); // C
    try std.testing.expectEqual(@as(usize, 24), e.picture_bytes);
    try Run.one(&e, &woken, pic(0, 2, 1)); // A again: a hit, now the most recent
    try std.testing.expectEqual(@as(u32, 3), fake.encodes.load(.acquire));
    try Run.one(&e, &woken, pic(0, 2, 4)); // D: B (least recent) goes
    try std.testing.expectEqual(@as(u64, 1), e.counts.evicted);
    try std.testing.expectEqual(@as(usize, 24), e.picture_bytes);
    try Run.one(&e, &woken, pic(0, 2, 1)); // A still kept
    try std.testing.expectEqual(@as(u32, 4), fake.encodes.load(.acquire));
    try Run.one(&e, &woken, pic(0, 2, 2)); // B encoded again
    try std.testing.expectEqual(@as(u32, 5), fake.encodes.load(.acquire));
    try Run.one(&e, &woken, pic(0, 7, 9)); // 28 bytes: over the whole budget, not kept, nothing evicted for it
    try std.testing.expectEqual(@as(usize, 24), e.picture_bytes);
    try Run.one(&e, &woken, pic(0, 7, 9));
    try std.testing.expectEqual(@as(u32, 7), fake.encodes.load(.acquire));
}

test "picture cache: the same picture twice in one request runs the tower once; off (0) it runs for each" {
    const gpa = std.testing.allocator;
    for ([_]usize{ 1 << 20, 0 }) |cap| {
        var fake: Fake = .{};
        var e = Encoder.init(gpa, std.testing.io, fake.tower(), 64);
        e.picture_cap = cap;
        try e.start();
        defer e.stop();
        var woken: Woken = .{};
        const a = [_]Image{ pic(0, 2, 7), pic(4, 2, 7) };
        try e.submit(&a, 0, woken.wake());
        waitReady(&e, &a);
        const r = try rowsOf(&e, &a, gpa);
        defer gpa.free(r);
        e.forget(&a);
        try std.testing.expect(r.len == 16 and std.mem.allEqual(u8, r, 7));
        try std.testing.expectEqual(@as(u32, if (cap > 0) 1 else 2), fake.encodes.load(.acquire));
        try std.testing.expectEqual(@as(usize, if (cap > 0) 1 else 0), e.pictures.items.len);
    }
}

test "picture cache: ENCODE_CACHE_MIB parses 0 to 65536 MiB and refuses the rest" {
    try std.testing.expectEqual(@as(usize, 0), try Encoder.parseCacheMib("0"));
    try std.testing.expectEqual(@as(usize, 512 << 20), try Encoder.parseCacheMib("512"));
    try std.testing.expectEqual(@as(usize, 65536 << 20), try Encoder.parseCacheMib("65536"));
    for ([_][]const u8{ "65537", "-1", "512M", "" }) |bad| try std.testing.expectError(error.BadEncodeCache, Encoder.parseCacheMib(bad));
}
