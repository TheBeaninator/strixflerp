//! The prompt cache over dsv41's pool placement, on the host: a factory replay's prompt passes as rank 0 logged them
//! (dsv41_keep_replay.tsv) run through the real Store (core/prompt_cache.zig) and the real placement (lanes.zig
//! usable, keptOver, placeKept, touch, staleOver). No model and no device: a pass here is the lines of fillStart that
//! seat a stream, and the keeps its chunks make.
const std = @import("std");
const api = @import("engine_api");
const dsv41 = @import("dsv41");
const pc = api.prompt_cache;
const lb = dsv41.lanes;
const Lanes = lb.Lanes;
const Snap = lb.Snap;

const c = 2048; // model.chunk_rows
const window = 128; // the checkpoint's sliding_window
const pool = 524288; // POOL as served
const max_new = 256; // the replay's max_tokens
const round_rows = 16; // ROUND_ROWS as served
/// Tokens the conversations with the shared system block have in common: they resume one state at 4,096 and none
/// past it, so any length from 4,096 to 6,143 replays the log.
const shared_start = 5096;

/// Whether this tree lets a family tell the cache that a kept state no longer restores.
const has_fix = @hasField(pc.Snapshots.VTable, "restores") and @hasDecl(Lanes, "snapRestoresFn");

const Row = struct { phase: []const u8, stream: []const u8, turn: u32, len: u32, live: u32, base: usize, at: u32, shares: bool, marks: [2]u32 = .{ 0, 0 }, n_marks: usize = 0 };
const Seat = struct { base: usize, at: u32 };

fn parse(line: []const u8) !Row {
    var it = std.mem.tokenizeScalar(u8, line, ' ');
    const phase = it.next().?;
    const stream = it.next().?;
    const turn = try std.fmt.parseInt(u32, it.next().?, 10);
    const len = try std.fmt.parseInt(u32, it.next().?, 10);
    const live = try std.fmt.parseInt(u32, it.next().?, 10);
    const base = try std.fmt.parseInt(usize, it.next().?, 10);
    const at = try std.fmt.parseInt(u32, it.next().?, 10);
    const shares = it.next().?[0] == '1';
    var r: Row = .{ .phase = phase, .stream = stream, .turn = turn, .len = len, .live = live, .base = base, .at = at, .shares = shares };
    const marks = it.next().?;
    if (marks[0] != '-') {
        var mi = std.mem.tokenizeScalar(u8, marks, ',');
        while (mi.next()) |x| {
            r.marks[r.n_marks] = try std.fmt.parseInt(u32, x, 10);
            r.n_marks += 1;
        }
    }
    return r;
}

/// A conversation's token at i (each turn's prompt is a prefix of the next turn's, as the log's resumes show).
fn token(key: u32, i: usize) u32 {
    return (@as(u32, @truncate(i)) *% 2654435761 +% key *% 0x9E3779B1) >> 12;
}

fn clashes(used: []const [2]usize, lo: usize, hi: usize) bool {
    for (used) |u| if (lo < u[1] and u[0] < hi) return true;
    return false;
}

fn markEnds(marks: []const u32, at: usize) bool {
    for (marks) |w| if (w / c * c == at) return true;
    return false;
}

const Replay = struct {
    gpa: std.mem.Allocator,
    ln: *Lanes,
    lane: [2]usize = .{ 0, 0 }, // the extent of the pass in progress (a keep's rows are there)
    prompt: []const u32 = &.{},
    live: std.ArrayList(Live) = .empty, // the streams that hold an extent, oldest pass first
    // the place replay (SF-PLACE-VICTIM): the streams that hold an extent at this pass are named
    others: ?[]const u8 = null, // comma separated ("-": none); null: the most recent live ones (sf-keep-trace)
    lanes_of: std.StringHashMapUnmanaged([2]usize) = .empty, // each stream's last extent
    shadow: ?[]u64 = null, // what each pool row holds: the rowKey of the prompt through it (prefill), or a pass's decode
    keys: []const u64 = &.{}, // this pass's rowKey of each token
    n_pass: u64 = 0,
    plant: bool = false,

    const Live = struct { stream: []const u8, lane: [2]usize };

    fn of(ptr: *anyopaque) *Replay {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytes(_: *anyopaque, _: u32) u64 {
        return 6 << 20;
    }
    /// lanes.zig snapSave without the device copy and rank 1's frame.
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const r = of(ptr);
        const sn = try r.gpa.create(Snap);
        errdefer r.gpa.destroy(sn);
        sn.* = .{ .id = r.ln.next_snap, .at = at, .base = r.lane[0], .end = r.lane[1], .path = lb.pathKey(r.prompt[0..at], &.{}), .buf = undefined };
        if (r.ln.place_victim) sn.chain = try lb.pathChain(r.gpa, r.prompt[0..at], &.{});
        errdefer r.gpa.free(sn.chain);
        try r.ln.snaps.append(r.gpa, sn);
        r.ln.next_snap += 1;
        return sn;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.BackendRestores;
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const r = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        if (std.mem.indexOfScalar(*Snap, r.ln.snaps.items, sn)) |i| _ = r.ln.snaps.orderedRemove(i);
        r.gpa.free(sn.chain);
        r.gpa.destroy(sn);
    }
    fn restores(ptr: *anyopaque, saved: pc.Saved, len: u32) bool {
        return Lanes.snapRestoresFn(of(ptr).ln, saved, len);
    }

    /// One request's prompt pass: the host's lookup, fillStart's seat for the stream, the host's report, the keeps.
    fn pass(r: *Replay, a: std.mem.Allocator, store: *pc.Store, row: Row, prompt: []const u32) !Seat {
        // the streams that still hold an extent: as many as the log counted, the most recent other ones
        for (r.live.items, 0..) |l, i| if (std.mem.eql(u8, l.stream, row.stream)) {
            _ = r.live.orderedRemove(i);
            break;
        };
        while (r.live.items.len > row.live) _ = r.live.orderedRemove(0);
        var taken_buf: [8][2]usize = undefined;
        for (r.live.items, 0..) |l, i| taken_buf[i] = l.lane;
        var nt = r.live.items.len;
        if (r.others) |o| { // (the place replay: those the row names, at the extents they last got here)
            nt = 0;
            var it = std.mem.tokenizeScalar(u8, o, ',');
            while (it.next()) |name| if (name[0] != '-') {
                taken_buf[nt] = r.lanes_of.get(name) orelse return error.UnknownOther;
                nt += 1;
            };
        }
        const taken = taken_buf[0..nt];
        const l = try store.lookup(a, prompt, &.{}, 0, row.marks[0..row.n_marks], &.{});
        const len = prompt.len;
        const size = std.mem.alignForward(usize, len + max_new + round_rows + 2, c);
        var kept: ?*Snap = null;
        var moved = false;
        var failed = false;
        if (l.entry) |e| {
            const sn: *Snap = @ptrCast(@alignCast(e.saved));
            if (r.ln.usable(sn, len)) {
                kept = sn;
                moved = sn.base + size > pool or clashes(taken, sn.base, sn.base + size) or r.ln.keptOver(sn.base + sn.at, sn.base + size, sn, prompt, &.{});
            } else failed = true; // "from 0: its kept state is not usable"
        }
        // (a source a live stream holds makes placeKept read the model's clock: the replay has none)
        const held = if (kept) |sn| moved and clashes(taken, sn.base, sn.base + sn.at) else false;
        if (held and r.others == null) return error.HeldSource;
        // (the place replay: a held copy that finds no room waits; the host tries it again until the wait runs out)
        const base = if (kept != null and !moved) kept.?.base else r.ln.placeKept(taken, size, kept) catch |err| switch (err) {
            error.ContextFull => if (held) again: {
                kept.?.wanted = 1;
                break :again try r.ln.placeKept(taken, size, kept);
            } else return err,
            else => return err,
        };
        if (kept) |sn| if (!moved) {
            sn.end = @max(sn.end, base + size);
        };
        if (kept) |sn| if (moved and sn.stale) {
            kept = null;
            failed = true;
        };
        if (kept) |sn| r.ln.touch(sn);
        r.ln.staleOver(if (kept != null and !moved) base + kept.?.at else base, base + size, kept);
        if (r.plant) for (r.ln.snaps.items) |sn| {
            sn.stale = false; // (a planted miss: every state given or written over counted whole again)
        };
        r.lane = .{ base, base + size };
        r.prompt = prompt;
        try r.live.append(r.gpa, .{ .stream = row.stream, .lane = r.lane });
        if (r.others != null) try r.lanes_of.put(r.gpa, row.stream, r.lane);
        const start: u32 = if (kept) |sn| @intCast(sn.at) else 0;
        if (r.shadow) |sh| {
            // the rows the pass reads (the kept state's, in place or copied from its extent to this one) hold this
            // prompt's tokens through each row, as a fresh pass of it writes them; then the pass writes its own
            r.n_pass += 1;
            const src = if (kept) |sn| sn.base else base;
            for (0..start) |j| if (sh[src + j] != r.keys[j]) return error.RowsChanged;
            if (src != base) {
                if (src < base + start and base < src + start) return error.CopyOverlaps;
                @memcpy(sh[base..][0..start], sh[src..][0..start]);
            }
            for (start..len) |j| sh[base + j] = r.keys[j];
            for (len..len + max_new) |j| sh[base + j] = (1 << 63) | r.n_pass; // (a reply's rows: no prompt's)
        }
        // lane_host.zig Job.reported, before the pass's first keep
        if (l.entry) |e| {
            if (failed) store.resumed(e, prompt, false) else if (kept != null) store.resumed(e, prompt, true);
        }
        // fillChunks: a keep at the last chunk end before the replay and at each mark's chunk end
        const replay = len -| window;
        const last_end = @min(replay, len - 1) / c * c;
        var at: usize = (start / c + 1) * c;
        while (at < len and at <= replay) : (at += c) {
            if (at == last_end or markEnds(l.marks, at)) _ = store.keep(prompt, @intCast(at), null, &.{});
        }
        return .{ .base = base, .at = start };
    }
};

fn run(gpa: std.mem.Allocator, rows: []const Row, fixed: bool, out: []Seat) !void {
    const Model = std.meta.Child(@FieldType(Lanes, "m"));
    var m: Model = undefined;
    m.pool_cap = pool;
    m.cfg.window = window;
    var ln: Lanes = .{ .gpa = gpa, .m = &m, .peer = null };
    defer ln.snaps.deinit(gpa);
    var r: Replay = .{ .gpa = gpa, .ln = &ln };
    defer r.live.deinit(gpa);
    var vt: pc.Snapshots.VTable = .{ .bytes = Replay.bytes, .save = Replay.save, .restore = Replay.restore, .drop = Replay.drop };
    if (has_fix) {
        if (fixed) vt.restores = Replay.restores;
    }
    var store = pc.Store.init(gpa, .{ .ptr = &r, .vtable = &vt }, .{}, 4 << 30);
    defer store.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var phase: []const u8 = "";
    for (rows, out) |row, *o| {
        _ = arena.reset(.retain_capacity);
        const a = arena.allocator();
        if (!std.mem.eql(u8, phase, row.phase)) {
            phase = row.phase;
            r.live.clearRetainingCapacity();
        }
        const key: u32 = @as(u32, @truncate(std.hash.Wyhash.hash(0, row.stream))) | 1;
        const prompt = try a.alloc(u32, row.len);
        for (prompt, 0..) |*t, i| t.* = token(if (row.shares and i < shared_start) 0 else key, i);
        o.* = try r.pass(a, &store, row, prompt);
    }
}

test "dsv41 factory replay: a turn whose longest kept state lost its rows resumes from the longest that restores" {
    const gpa = std.testing.allocator;
    var rows: std.ArrayList(Row) = .empty;
    defer rows.deinit(gpa);
    var lines = std.mem.tokenizeScalar(u8, @embedFile("dsv41_keep_replay.tsv"), '\n');
    while (lines.next()) |line| if (line[0] != '#') try rows.append(gpa, try parse(line));
    try std.testing.expectEqual(@as(usize, 3 + 68 + 68), rows.items.len);
    const seats = try gpa.alloc(Seat, rows.items.len);
    defer gpa.free(seats);

    // as served (the cache hands the longest kept state whatever became of its rows): every pass sits and resumes
    // where rank 0 logged it, and 33 of rep-1's 55 warm turns read their prompt from 0
    try run(gpa, rows.items, false, seats);
    var from0: usize = 0;
    for (rows.items, seats) |row, s| {
        try std.testing.expectEqual(row.base, s.base);
        try std.testing.expectEqual(row.at, s.at);
        if (std.mem.eql(u8, row.phase, "rep-1") and row.turn > 0 and s.at == 0) from0 += 1;
    }
    try std.testing.expectEqual(@as(usize, 33), from0);

    // with the family's word on what still restores: the turns one at a time are as logged, rep-1's first miss
    // (s0006 turn 2: solo-1's state at 18,432 lost its rows, rep-1's own at 16,384 has them) resumes at 16,384, and
    // no warm turn of rep-1 reads from 0
    try run(gpa, rows.items, true, seats);
    from0 = 0;
    var first: ?Seat = null;
    for (rows.items, seats) |row, s| {
        if (!std.mem.eql(u8, row.phase, "rep-1")) {
            try std.testing.expectEqual(row.base, s.base);
            try std.testing.expectEqual(row.at, s.at);
            continue;
        }
        if (row.turn > 0 and s.at == 0) from0 += 1;
        if (row.turn == 2 and std.mem.eql(u8, row.stream, "s0006")) first = s;
    }
    try std.testing.expectEqual(@as(u32, 16384), first.?.at);
    try std.testing.expectEqual(@as(usize, 0), from0);
}



// ---- SF-PLACE-VICTIM: placement's victims over the runs with the kept-state fix

const PRow = struct { run: []const u8, row: Row, others: []const u8 };

/// A line of dsv41_place_replay.tsv: run phase stream turn tokens others extent_base resumed_at shares_start marks.
fn parsePlace(line: []const u8) !PRow {
    var it = std.mem.tokenizeScalar(u8, line, ' ');
    const run_name = it.next().?;
    const phase = it.next().?;
    const stream = it.next().?;
    const turn = try std.fmt.parseInt(u32, it.next().?, 10);
    const len = try std.fmt.parseInt(u32, it.next().?, 10);
    const others = it.next().?;
    const base = try std.fmt.parseInt(usize, it.next().?, 10);
    const at = try std.fmt.parseInt(u32, it.next().?, 10);
    const shares = it.next().?[0] == '1';
    var r: Row = .{ .phase = phase, .stream = stream, .turn = turn, .len = len, .live = 0, .base = base, .at = at, .shares = shares };
    const marks = it.next().?;
    if (marks[0] != '-') {
        var mi = std.mem.tokenizeScalar(u8, marks, ',');
        while (mi.next()) |x| {
            r.marks[r.n_marks] = try std.fmt.parseInt(u32, x, 10);
            r.n_marks += 1;
        }
    }
    return .{ .run = run_name, .row = r, .others = others };
}

/// A prompt's row keys: key[i] stands for its tokens [0, i] (what the pool row at i holds after a pass of it).
fn rowKeys(a: std.mem.Allocator, prompt: []const u32) ![]u64 {
    const out = try a.alloc(u64, prompt.len);
    var h: u64 = 0x243F6A8885A308D3;
    for (prompt, out) |t, *o| {
        h = (h +% t +% 1) *% 0x9E3779B97F4A7C15;
        h ^= h >> 29;
        o.* = h & ~(@as(u64, 1) << 63);
    }
    return out;
}

const Tally = struct {
    passes: usize = 0,
    own_lost: usize = 0, // a turn past its stream's first in a phase resumes short of its previous turn's last keep
    own_tokens: usize = 0, // the prompt tokens those turns read again
    own_from0: usize = 0,
    read: usize = 0, // prompt tokens read by every pass
    differ: usize = 0, // passes whose seat (extent base, resume point) is not the row's
};

/// One run of the place replay in a fresh pool and cache; with shadow, every pass's rows are checked (RowsChanged).
fn runPlace(gpa: std.mem.Allocator, rows: []const PRow, place_victim: bool, shadow: bool, plant: bool) !Tally {
    const Model = std.meta.Child(@FieldType(Lanes, "m"));
    var m: Model = undefined;
    m.pool_cap = pool;
    m.cfg.window = window;
    m.io = std.testing.io; // (placeKept reads the clock for a held copy's wait)
    var ln: Lanes = .{ .gpa = gpa, .m = &m, .peer = null };
    ln.place_victim = place_victim;
    defer ln.snaps.deinit(gpa);
    var r: Replay = .{ .gpa = gpa, .ln = &ln, .plant = plant };
    defer r.live.deinit(gpa);
    defer r.lanes_of.deinit(gpa);
    const sh = try gpa.alloc(u64, pool);
    defer gpa.free(sh);
    @memset(sh, 1 << 63);
    if (shadow) r.shadow = sh;
    var vt: pc.Snapshots.VTable = .{ .bytes = Replay.bytes, .save = Replay.save, .restore = Replay.restore, .drop = Replay.drop, .restores = Replay.restores };
    var store = pc.Store.init(gpa, .{ .ptr = &r, .vtable = &vt }, .{}, 4 << 30);
    defer store.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var prev: std.StringHashMapUnmanaged(usize) = .empty; // each stream's last keep in this phase
    defer prev.deinit(gpa);
    var phase: []const u8 = "";
    var t: Tally = .{};
    for (rows) |pr| {
        const row = pr.row;
        _ = arena.reset(.retain_capacity);
        const a = arena.allocator();
        if (!std.mem.eql(u8, phase, row.phase)) {
            phase = row.phase;
            prev.clearRetainingCapacity();
        }
        const key: u32 = @as(u32, @truncate(std.hash.Wyhash.hash(0, row.stream))) | 1;
        const prompt = try a.alloc(u32, row.len);
        for (prompt, 0..) |*tk, i| tk.* = token(if (row.shares and i < shared_start) 0 else key, i);
        r.others = pr.others;
        r.keys = if (shadow) try rowKeys(a, prompt) else &.{};
        const s = try r.pass(a, &store, row, prompt);
        t.passes += 1;
        t.read += row.len - s.at;
        if (s.base != row.base or s.at != row.at) t.differ += 1;
        const last = @min(row.len -| window, row.len - 1) / c * c;
        if (row.turn > 0) if (prev.get(row.stream)) |want| if (want <= row.len -| window and s.at < want) {
            t.own_lost += 1;
            t.own_tokens += want - s.at;
            if (s.at == 0) t.own_from0 += 1;
        };
        try prev.put(gpa, row.stream, last);
    }
    return t;
}

fn tallyRuns(gpa: std.mem.Allocator, rows: []const PRow, synthetic: bool, place_victim: bool, shadow: bool, plant: bool) !Tally {
    var sum: Tally = .{};
    var i: usize = 0;
    while (i < rows.len) {
        var j = i;
        while (j < rows.len and std.mem.eql(u8, rows[j].run, rows[i].run)) j += 1;
        if (std.mem.startsWith(u8, rows[i].run, "syn-") == synthetic) {
            const t = try runPlace(gpa, rows[i..j], place_victim, shadow, plant);
            sum.passes += t.passes;
            sum.own_lost += t.own_lost;
            sum.own_tokens += t.own_tokens;
            sum.own_from0 += t.own_from0;
            sum.read += t.read;
            sum.differ += t.differ;
        }
        i = j;
    }
    return sum;
}

test "dsv41 place replay: superseded states give room first and a state clear of the placed extent keeps its rows" {
    const gpa = std.testing.allocator;
    var rows: std.ArrayList(PRow) = .empty;
    defer rows.deinit(gpa);
    var lines = std.mem.tokenizeScalar(u8, @embedFile("dsv41_place_replay.tsv"), '\n');
    while (lines.next()) |line| if (line[0] != '#') try rows.append(gpa, try parsePlace(line));
    try std.testing.expectEqual(@as(usize, 2 * 205 + 6 * 137 + 2 * 207), rows.items.len);
    // [own-state losses, their tokens, of them from 0, prompt tokens read] off then on: the 8 logged runs, then the
    // synthetic schedule (27 streams of the frozen set, 8 conversations in flight against 4 streams)
    const want = [2][2][4]usize{
        .{ .{ 1, 38912, 1, 11662188 }, .{ 0, 0, 0, 11621228 } },
        .{ .{ 40, 1615872, 5, 4373444 }, .{ 10, 395264, 0, 3169220 } },
    };
    for ([_]bool{ false, true }, want) |syn, w| {
        // as served: every pass of the 8 logged runs sits and resumes where rank 0 logged it (the synthetic one where
        // pv.py placed it); off and on, every row a pass reads holds its own prompt's tokens through it (the shadow
        // pool: else error.RowsChanged), so a pass computes what it computes as served from the same chunk end
        const off = try tallyRuns(gpa, rows.items, syn, false, true, false);
        const on = try tallyRuns(gpa, rows.items, syn, true, true, false);
        std.debug.print("place replay {s}: off passes {d} differ {d} own lost {d} ({d} tokens, {d} from 0) read {d}; on differ {d} own lost {d} ({d} tokens, {d} from 0) read {d}\n", .{ if (syn) "synthetic n27 k8" else "8 logged runs", off.passes, off.differ, off.own_lost, off.own_tokens, off.own_from0, off.read, on.differ, on.own_lost, on.own_tokens, on.own_from0, on.read });
        try std.testing.expectEqual(@as(usize, 0), off.differ);
        try std.testing.expectEqual(w[0], [4]usize{ off.own_lost, off.own_tokens, off.own_from0, off.read });
        try std.testing.expectEqual(w[1], [4]usize{ on.own_lost, on.own_tokens, on.own_from0, on.read });
    }
    // the shadow pool fails a pass that reads rows another stream took: a planted miss (every state counted whole
    // after each placement) resumes from rows that are not its prompt's
    try std.testing.expectError(error.RowsChanged, tallyRuns(gpa, rows.items, false, false, true, true));
}
