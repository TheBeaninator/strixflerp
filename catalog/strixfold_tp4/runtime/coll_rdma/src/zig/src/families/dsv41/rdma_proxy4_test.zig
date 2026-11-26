//! TP4-RDMA-N: rdma_proxy at world 4, GPU-free. The slab protocol of rdma_core.CoreN over a fake fabric (one QP a
//! peer, writes with immediate, completions, a dead or frozen rank), with host threads doing what coll.stage and
//! coll.collect do from the very launch steps RdmaProxyN issues (rdma_core.Layout); the fault matrix of
//! tools/dsv41-hip/coll_fault_matrix.sh at four ranks; the star handshake over loopback links; and, with
//! TF_RDMA_TEST_DEVICE, the same protocol over four ranks' real QPs on one HCA (loopback).
const std = @import("std");
const coll = @import("coll.zig");
const wire = @import("tcp_wire.zig");
const nic = @import("rdma_nic.zig");
const rdma_core = @import("rdma_core.zig");
const tcp_proxy = @import("tcp_proxy.zig");
const tcp_coll = @import("tcp_coll.zig");
const rdma_proxy = @import("rdma_proxy.zig");
const Fan = @import("fan.zig").Fan;
const Shared = tcp_proxy.Shared;
const Layout = rdma_core.Layout;
const Word = rdma_core.Word;
const Cqe = rdma_core.Cqe;
const Seg = rdma_core.Seg;

const c = std.c;
const testing = std.testing;
const W = rdma_core.max_world;

fn sleepNs(ns: u64) void {
    var ts: c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = c.nanosleep(&ts, null);
}

// ---- the fake fabric: what the verbs Nics do, in host memory ----------------------------------------------------------

const Fabric = struct {
    lock: std.atomic.Value(u32) = .init(0),
    world: u32,
    ports: [W]Port = undefined,

    fn acquire(f: *Fabric) void {
        while (f.lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }
    fn release(f: *Fabric) void {
        f.lock.store(0, .release);
    }

    fn kill(f: *Fabric, r: u32) void {
        f.acquire();
        defer f.release();
        f.ports[r].dead = true;
    }

    const Ring = struct {
        buf: [4096]Cqe = undefined,
        head: usize = 0,
        tail: usize = 0,
        fn push(r: *Ring, e: Cqe) void {
            r.buf[r.tail % r.buf.len] = e;
            r.tail += 1;
        }
        fn pop(r: *Ring, out: []Cqe) usize {
            var n: usize = 0;
            while (n < out.len and r.head < r.tail) : (n += 1) {
                out[n] = r.buf[r.head % r.buf.len];
                r.head += 1;
            }
            return n;
        }
    };

    /// One rank's NIC (rdma_core.CoreN's Net).
    const Port = struct {
        fab: *Fabric,
        rank: u32,
        rx: []u8,
        dead: bool = false,
        err: bool = false,
        /// a stopped process (SIGSTOP): its threads make no calls; the peers' writes still land
        frozen: std.atomic.Value(bool) = .init(false),
        sendq: Ring = .{},
        recvq: Ring = .{},

        fn thaw(p: *Port) void {
            while (p.frozen.load(.acquire)) sleepNs(100 * std.time.ns_per_us);
        }

        fn deliver(p: *Port, j: u32, data: ?Seg, trailer: ?Seg, wr_id: u64, imm: u32) void {
            const f = p.fab;
            f.acquire();
            defer f.release();
            if (p.err) return p.sendq.push(.{ .status = 5, .wr_id = wr_id, .peer = j }); // flushed: this QP left RTS
            const t = &f.ports[j];
            if (t.dead or t.err) return p.sendq.push(.{ .status = 12, .wr_id = wr_id, .peer = j }); // no ack: RETRY_EXC
            if (data) |d| @memcpy(t.rx[@intCast(d.remote_off)..][0..d.local.len], d.local);
            var len: u32 = 0;
            if (trailer) |tr| {
                @memcpy(t.rx[@intCast(tr.remote_off)..][0..tr.local.len], tr.local);
                len = @intCast(tr.local.len);
            }
            t.recvq.push(.{ .peer = p.rank, .imm = imm, .byte_len = len, .with_imm = true });
            p.sendq.push(.{ .wr_id = wr_id, .peer = j });
        }

        pub fn post(p: *Port, j: u32, data: ?Seg, trailer: Seg, wr_id: u64, imm: u32) nic.Error!void {
            p.thaw();
            if (p.dead) return error.PostFailed;
            p.deliver(j, data, trailer, wr_id, imm);
        }

        pub fn postPoison(p: *Port, j: u32, code: u32, wr_id: u64) nic.Error!void {
            if (p.dead) return error.PostFailed;
            p.deliver(j, null, null, wr_id, code);
        }

        fn pollQ(p: *Port, q: *Ring, out: []Cqe) isize {
            p.thaw();
            p.fab.acquire();
            defer p.fab.release();
            return @intCast(q.pop(out));
        }
        pub fn pollSend(p: *Port, out: []Cqe) isize {
            return p.pollQ(&p.sendq, out);
        }
        pub fn pollRecv(p: *Port, out: []Cqe) isize {
            return p.pollQ(&p.recvq, out);
        }
        pub fn statusText(_: *const Port, status: c_int) []const u8 {
            return switch (status) {
                5 => "fake: work request flushed",
                12 => "fake: transport retry counter exceeded",
                else => "fake: error",
            };
        }
        /// QPs to error: what is posted flushes; the peers' writes get no acknowledgement.
        pub fn toErr(p: *Port) void {
            p.fab.acquire();
            defer p.fab.release();
            if (!p.err) p.recvq.push(.{ .status = 5, .peer = p.rank });
            p.err = true;
        }
    };
};

// ---- the program four ranks run: the 2D split's shapes, plus chunked ones -----------------------------------------------

const Call = struct {
    exchange: bool = false,
    bytes: u64 = 0,
    peer: u32 = 0,
    rbytes: u64 = 0,
    phase: coll.Phase = .two_parts,
    layer: u16 = 0,
    /// a swap12 gather (Word.gatherSwap12): rank g's block at place 2 (g % 2) + g / 2
    swap: bool = false,
};

/// Call idx of rank g: a quarters gather, the g^1 and g^2 pair exchanges (uneven pair widths), a 16-byte Markov gather,
/// an empty gather, and every fifth round a gather and an exchange larger than a slot.
fn callAt(idx: usize, g: u32, l: Layout) Call {
    const round = idx / 7;
    const big = round % 5 == 0;
    const h = l.exchangeHalf();
    return switch (idx % 7) {
        0 => .{ .bytes = 20480, .phase = .two_parts, .swap = round % 2 == 1 },
        1 => .{ .exchange = true, .bytes = 8192, .peer = g ^ 1, .rbytes = 8192, .phase = .two_pair, .layer = 0 },
        2 => .{ .exchange = true, .bytes = if (g < 2) 5120 else 4096, .peer = g ^ 2, .rbytes = if (g < 2) 4096 else 5120, .phase = .two_pair, .layer = 1 },
        3 => .{ .bytes = 16, .phase = .draft_markov, .layer = @intCast(round % 3) },
        4 => .{ .bytes = 0, .phase = .round_head },
        5 => if (big) .{ .bytes = 3 * l.gatherChunk() + 100, .phase = .prompt_moe, .layer = 7, .swap = round % 10 == 5 } else .{ .bytes = 4096, .phase = .round_moe, .layer = 3 },
        else => if (big)
            .{ .exchange = true, .bytes = if (g % 2 == 0) 3 * h + 7 else h + 1, .peer = g ^ 1, .rbytes = if (g % 2 == 0) h + 1 else 3 * h + 7, .phase = .two_pair, .layer = 2 }
        else
            .{ .exchange = true, .bytes = 64, .peer = g ^ 2, .rbytes = 64, .phase = .two_pair, .layer = 3 },
    };
}

fn pat(idx: usize, g: u32, i: usize) u8 {
    return @truncate((idx *% 131) ^ (@as(usize, g) << 7) ^ i ^ (i >> 8));
}

/// A host thread doing what RdmaProxyN's launches do on the GPU: the same Layout steps in the same stream order
/// (depth stages ahead, then collect and the next stage), each kernel as coll.hip writes it. With ways to misbehave.
const FakeGpu4 = struct {
    /// one header a lane (layout.lanes)
    shared: [*]Shared,
    rx: []u8,
    layout: Layout,
    rank: u32,
    calls: usize,
    send: []u8,
    recv: []u8,
    deadline_ns: u64 = 2 * std.time.ns_per_s,
    frozen: ?*std.atomic.Value(bool) = null,
    // what went wrong
    bad: usize = 0,
    bad_call: ?usize = null,
    bad_block: ?u32 = null,
    timed_out: bool = false,
    stopped_at: ?usize = null,
    finished: usize = 0,
    // faults
    stall_at: ?usize = null,
    stall_ns: u64 = 0,
    kill_at: ?usize = null,
    kill: ?*const fn (ctx: *anyopaque, rank: u32) void = null,
    kill_ctx: *anyopaque = undefined,
    site_at: ?usize = null,
    rbytes_at: ?usize = null,
    peer_at: ?usize = null,
    flip_at: ?usize = null,
    hold_collect_from: usize = std.math.maxInt(usize),
    collect_delay_ns: u64 = 0,
    /// every call this one instead of the program's (the timing test)
    fixed: ?Call = null,
    /// fill send with the pattern and verify recv (off: the timing test's proxy-only figure)
    check: bool = true,
    /// ready -> done per call (ns), for the timing test
    rtt: ?[]u64 = null,
    /// a hash of every finished call's recv bytes (the lanes comparison)
    hashes: ?[]u64 = null,
    t_ready: u64 = 0,
    cur: usize = 0,

    fn thaw(f: *FakeGpu4) void {
        if (f.frozen) |z| while (z.load(.acquire)) sleepNs(100 * std.time.ns_per_us);
    }

    fn giveUp(f: *FakeGpu4, hs: *Shared) void {
        _ = @cmpxchgStrong(u32, &hs.abort, 0, 2, .release, .monotonic);
        f.timed_out = true;
    }

    /// coll.stage on the step's lane (its header, nslot = the lane's slots, tx = the lane's first slot): wait for the
    /// slot (consumed + nslot >= q), the block into its place, publish ready = q.
    fn stage(f: *FakeGpu4, s: Layout.Step) bool {
        f.thaw();
        const l = f.layout;
        const hs = &f.shared[s.lane];
        const m = l.laneSlots();
        const q = @atomicLoad(u64, &hs.seq, .acquire) + 1;
        const ls: usize = @intCast(q % m);
        const slot = l.slotIndex(s.lane, q);
        const t0 = wire.monotonicNs();
        while (true) {
            if (@atomicLoad(u32, &hs.abort, .acquire) != 0) return false;
            const cons = @atomicLoad(u64, &hs.slot[ls].consumed, .acquire);
            if (q <= m or cons + m >= q) break;
            if (wire.monotonicNs() - t0 >= f.deadline_ns) {
                f.giveUp(hs);
                return false;
            }
            std.atomic.spinLoopHint();
        }
        const src: [*]const u8 = @ptrFromInt(s.send);
        @memcpy(f.rx[slot * @as(usize, @intCast(l.slot_bytes)) + @as(usize, @intCast(s.tx_off)) ..][0..@intCast(s.bytes)], src[0..@intCast(s.bytes)]);
        hs.slot[ls].bytes = s.bytes;
        hs.slot[ls].site = s.site;
        @atomicStore(u64, &hs.seq, q, .release);
        @atomicStore(u64, &hs.slot[ls].ready, q, .release);
        f.t_ready = wire.monotonicNs();
        return true;
    }

    /// coll.collect: wait for done == seq, the block(s) to recv, release the slot (consumed = q).
    fn collect(f: *FakeGpu4, s: Layout.Step) bool {
        const l = f.layout;
        const hs = &f.shared[s.lane];
        for (s.collects[0..s.ncollect]) |cl| {
            f.thaw();
            const q = @atomicLoad(u64, &hs.seq, .acquire);
            const ls: usize = @intCast(q % l.laneSlots());
            const slot = l.slotIndex(s.lane, q);
            const t0 = wire.monotonicNs();
            while (true) {
                if (@atomicLoad(u32, &hs.abort, .acquire) != 0) return false;
                const d = @atomicLoad(u64, &hs.slot[ls].done, .acquire);
                if (d == q) break;
                if (d > q or wire.monotonicNs() - t0 >= f.deadline_ns) {
                    f.giveUp(hs);
                    return false;
                }
                std.atomic.spinLoopHint();
            }
            if (f.rtt) |r| r[f.cur] = wire.monotonicNs() - f.t_ready;
            if (f.collect_delay_ns > 0) sleepNs(f.collect_delay_ns);
            const dst: [*]u8 = @ptrFromInt(cl.dst);
            @memcpy(dst[0..@intCast(cl.bytes)], f.rx[slot * @as(usize, @intCast(l.slot_bytes)) + @as(usize, @intCast(cl.rx_off)) ..][0..@intCast(cl.bytes)]);
            @atomicStore(u64, &hs.slot[ls].consumed, q, .release);
        }
        return true;
    }

    fn steps(f: *FakeGpu4, k: u64, comptime stepAt: anytype, args: anytype, hold: bool) bool {
        const w = f.layout.depth(k);
        for (0..w) |ci| if (!f.stage(@call(.auto, stepAt, args ++ .{@as(u64, ci)}))) return false;
        for (0..k) |ci| {
            if (hold) return true;
            if (!f.collect(@call(.auto, stepAt, args ++ .{@as(u64, ci)}))) return false;
            if (ci + w < k) if (!f.stage(@call(.auto, stepAt, args ++ .{@as(u64, ci + w)}))) return false;
        }
        return true;
    }

    fn run(f: *FakeGpu4) void {
        const l = f.layout;
        for (0..f.calls) |idx| {
            if (f.kill_at != null and f.kill_at.? == idx) {
                f.stopped_at = idx;
                if (f.kill) |k| k(f.kill_ctx, f.rank);
                return;
            }
            if (f.stall_at != null and f.stall_at.? == idx) sleepNs(f.stall_ns);
            f.cur = idx;
            var call = f.fixed orelse callAt(idx, f.rank, l);
            if (f.fixed != null and call.exchange) call.peer = f.rank ^ 1;
            if (f.rbytes_at != null and f.rbytes_at.? == idx) call.rbytes += 64;
            if (f.peer_at != null and f.peer_at.? == idx) call.peer = (f.rank + 2) % l.world;
            if (f.check) for (f.send[0..@intCast(call.bytes)], 0..) |*b, i| {
                b.* = pat(idx, f.rank, i);
            };
            if (f.flip_at != null and f.flip_at.? == idx) f.send[@intCast(call.bytes / 2)] ^= 0x10;
            const send = @intFromPtr(f.send.ptr);
            const recv = @intFromPtr(f.recv.ptr);
            const hold = idx >= f.hold_collect_from;
            var ok = true;
            if (call.exchange) {
                const k = l.exchangeChunks(call.bytes, call.rbytes);
                if (k > 0) ok = f.steps(k, Layout.exchangeStep, .{ l, f.rank, call.peer, send, call.bytes, recv, call.rbytes, @as(u16, @intFromEnum(call.phase)), call.layer }, hold);
            } else {
                var word = if (call.swap) Word.gatherSwap12(@intFromEnum(call.phase), call.layer) else Word.gather(@intFromEnum(call.phase), call.layer);
                if (f.site_at != null and f.site_at.? == idx) word +%= 1 << 20; // another layer
                const k = l.gatherChunks(call.bytes);
                if (k > 0) ok = f.steps(k, Layout.gatherStep, .{ l, f.rank, send, recv, call.bytes, word }, hold);
            }
            if (!ok) {
                f.stopped_at = idx;
                return;
            }
            if (hold) continue;
            if (f.check) f.verify(idx, call);
            if (f.hashes) |hh| hh[idx] = std.hash.Wyhash.hash(idx, f.recv[0..@intCast(if (call.exchange) call.rbytes else f.layout.world * call.bytes)]);
            f.finished = idx + 1;
        }
    }

    fn verify(f: *FakeGpu4, idx: usize, call: Call) void {
        const n: usize = @intCast(call.bytes);
        if (call.exchange) {
            for (f.recv[0..@intCast(call.rbytes)], 0..) |b, i| if (b != pat(idx, call.peer, i)) f.noteBad(idx, call.peer);
            return;
        }
        const word = if (call.swap) Word.gatherSwap12(@intFromEnum(call.phase), call.layer) else Word.gather(@intFromEnum(call.phase), call.layer);
        for (0..f.layout.world) |g| {
            const p: usize = Word.place(word, @intCast(g));
            for (f.recv[p * n ..][0..n], 0..) |b, i| if (b != pat(idx, @intCast(g), i)) f.noteBad(idx, @intCast(g));
        }
    }

    fn noteBad(f: *FakeGpu4, idx: usize, g: u32) void {
        f.bad += 1;
        if (f.bad_call == null) {
            f.bad_call = idx;
            f.bad_block = g;
        }
    }
};

// ---- four ranks over the fake fabric --------------------------------------------------------------------------------------

const FakeCore = rdma_core.CoreN(Fabric.Port);

const Rank4 = struct {
    shared: [rdma_core.max_lanes]Shared = std.mem.zeroes([rdma_core.max_lanes]Shared),
    core: FakeCore = undefined,
    stop: std.atomic.Value(u32) = .init(0),
    exchanged: std.atomic.Value(u64) = .init(0),
    /// RdmaProxyN.failed: the first failure (the proxy's code, or 4 for an abort from another thread)
    failed: std.atomic.Value(u32) = .init(0),
    code: u32 = 0,
    gpu: FakeGpu4 = undefined,

    /// RdmaProxyN.proxyLoop: the core's code stays the failure unless one is set, then the QPs go to error.
    fn runCore(r: *Rank4) void {
        r.code = r.core.run();
        if (r.code != 0) {
            _ = r.failed.cmpxchgStrong(0, r.code, .acq_rel, .acquire);
            r.core.net.toErr();
        }
    }

    /// RdmaProxyN.status as the rank's client reads it.
    fn reported(r: *Rank4) []const u8 {
        return switch (r.failed.load(.acquire)) {
            0 => "ok",
            1 => "PeerLost",
            2 => "Timeout",
            3 => "Mismatch",
            4 => "Aborted",
            else => "CollectiveFailed",
        };
    }

    /// RdmaProxyN.abort from another thread.
    fn abort(r: *Rank4) void {
        _ = r.failed.cmpxchgStrong(0, 4, .acq_rel, .acquire);
        for (&r.shared) |*hs| tcp_proxy.raiseAbort(hs);
    }

    /// RdmaProxyN.close: the abort word (the proxy poisons its peers and leaves), then the QPs to error.
    fn closeColl(r: *Rank4) void {
        for (&r.shared) |*hs| @atomicStore(u32, &hs.abort, 1, .release);
    }
};

const World4 = struct {
    fab: Fabric,
    rk: [W]Rank4,
    layout: Layout,
    rx: [W][]u8,
    bufs: [W][2][]u8,

    /// Lanes as many as the slots carry (2 a lane, at most max_lanes): 8 slots run the default 4 lanes.
    fn init(world: u32, nslot: u32, slot_bytes: u64, progress_ns: u64, calls: usize) !*World4 {
        return initLanes(world, nslot, @min(rdma_core.max_lanes, @max(1, nslot / 2)), slot_bytes, progress_ns, calls);
    }

    fn initLanes(world: u32, nslot: u32, lanes: u32, slot_bytes: u64, progress_ns: u64, calls: usize) !*World4 {
        const a = std.heap.page_allocator;
        const w = try a.create(World4);
        w.* = .{ .fab = .{ .world = world }, .rk = undefined, .layout = .{ .world = world, .slot_bytes = slot_bytes, .nslot = nslot, .lanes = lanes }, .rx = undefined, .bufs = undefined };
        const epoch: [16]u8 = @splat(5);
        for (0..world) |g| {
            w.rx[g] = try a.alloc(u8, @intCast(nslot * slot_bytes));
            @memset(w.rx[g], 0);
            w.bufs[g] = .{ try a.alloc(u8, 256 << 10), try a.alloc(u8, 512 << 10) };
            w.fab.ports[g] = .{ .fab = &w.fab, .rank = @intCast(g), .rx = w.rx[g] };
        }
        for (0..world) |g| {
            const r = &w.rk[g];
            r.* = .{};
            r.core = .{ .shared = &r.shared, .rx = w.rx[g], .layout = w.layout, .net = &w.fab.ports[g], .rank = @intCast(g), .epoch = epoch, .stop = &r.stop, .exchanged = &r.exchanged, .progress_ns = progress_ns };
            r.gpu = .{ .shared = &r.shared, .rx = w.rx[g], .layout = w.layout, .rank = @intCast(g), .calls = calls, .send = w.bufs[g][0], .recv = w.bufs[g][1], .frozen = &w.fab.ports[g].frozen };
        }
        return w;
    }

    fn deinit(w: *World4) void {
        const a = std.heap.page_allocator;
        for (0..w.layout.world) |g| {
            a.free(w.rx[g]);
            a.free(w.bufs[g][0]);
            a.free(w.bufs[g][1]);
        }
        a.destroy(w);
    }

    /// The kill hook: the rank's process is gone (its QPs answer nothing, its proxy thread ends).
    fn killRank(ctx: *anyopaque, g: u32) void {
        const w: *World4 = @ptrCast(@alignCast(ctx));
        w.fab.kill(g);
        w.rk[g].stop.store(1, .release);
    }

    fn run(w: *World4, act: ?*const fn (w: *World4) void) !void {
        const n = w.layout.world;
        var cores: [W]std.Thread = undefined;
        var gpus: [W]std.Thread = undefined;
        for (0..n) |g| {
            w.rk[g].gpu.kill_ctx = w;
            w.rk[g].gpu.kill = killRank;
            cores[g] = try std.Thread.spawn(.{}, Rank4.runCore, .{&w.rk[g]});
            gpus[g] = try std.Thread.spawn(.{}, FakeGpu4.run, .{&w.rk[g].gpu});
        }
        if (act) |f| f(w);
        for (gpus[0..n]) |t| t.join();
        for (0..n) |g| w.rk[g].stop.store(1, .release);
        for (cores[0..n]) |t| t.join();
    }

    fn line(w: *World4, name: []const u8, verdict: bool) void {
        std.debug.print("{s} world 4 fault matrix: {s} |", .{ if (verdict) "PASS" else "FAIL", name });
        for (0..w.layout.world) |g| std.debug.print(" rank{d}: {s} (proxy code {d}, {d} gathers, gpu stopped at {?d})", .{ g, w.rk[g].reported(), w.rk[g].code, w.rk[g].exchanged.load(.acquire), w.rk[g].gpu.stopped_at });
        std.debug.print("\n", .{});
    }

    fn is(w: *World4, g: usize, names: []const []const u8) bool {
        for (names) |x| if (std.mem.eql(u8, w.rk[g].reported(), x)) return true;
        return false;
    }
};

const slot_small: u64 = 32768;

test "world 4: Word names the call, the pair and the sizes alike on both ranks of an exchange" {
    const a = Word.exchange(@intFromEnum(coll.Phase.two_pair), 1, 0, 2, 5120, 4096);
    const b = Word.exchange(@intFromEnum(coll.Phase.two_pair), 1, 2, 0, 4096, 5120);
    try testing.expectEqual(a, b);
    try testing.expect(Word.isExchange(a) and !Word.isExchange(Word.gather(13, 1)));
    try testing.expectEqual(@as(?u32, 2), Word.partner(a, 0));
    try testing.expectEqual(@as(?u32, 0), Word.partner(a, 2));
    try testing.expectEqual(@as(?u32, null), Word.partner(a, 1));
    try testing.expect(!Word.names(a, 3));
    // a different size on either side changes the word
    try testing.expect(a != Word.exchange(@intFromEnum(coll.Phase.two_pair), 1, 0, 2, 5120, 4160));
    try testing.expect(a != Word.exchange(@intFromEnum(coll.Phase.two_pair), 1, 0, 2, 5184, 4096));
    // the other pair's word of the same call: the same call bits, not naming rank 0
    const o = Word.exchange(@intFromEnum(coll.Phase.two_pair), 1, 1, 3, 5120, 4096);
    try testing.expectEqual(Word.call(a), Word.call(o));
    try testing.expect(!Word.names(o, 0));
    try testing.expectEqual(@as(u32, 15), Word.phaseOf(Word.gather(0xffff, 0)));
    try testing.expectEqual(@as(u32, 61), Word.layerOf(Word.gather(3, 61)));
}

test "world 4: the slot layout keeps blocks, exchange halves and trailers apart, and chunks cover every byte once" {
    const l: Layout = .{ .world = 4, .slot_bytes = rdma_proxy.default_slot_kib_n << 10, .nslot = 8 };
    try testing.expect(l.gatherChunk() >= 16 * 32384 * 4); // the 16-row head quarter fits one slot gather
    try testing.expectEqual(@as(u64, 1), l.gatherChunks(16 * 32384 * 4));
    try testing.expect(4 * l.gatherChunk() <= l.trailerOff(0));
    try testing.expect(2 * l.exchangeHalf() <= l.trailerOff(0));
    try testing.expectEqual(l.slot_bytes - 64, l.trailerOff(3));
    // one slot: a single collect of the four blocks in rank order, the stage at the rank's own place
    const s = l.gatherStep(2, 0x1000, 0x10_0000, 20480, 7, 0);
    try testing.expectEqual(@as(u64, 2 * 20480), s.tx_off);
    try testing.expectEqual(@as(u32, 1), s.ncollect);
    try testing.expectEqual(Layout.Collect{ .rx_off = 0, .dst = 0x10_0000, .bytes = 4 * 20480 }, s.collects[0]);
    // chunks: every rank's every byte lands once
    const small: Layout = .{ .world = 4, .slot_bytes = slot_small, .nslot = 4 };
    const bytes = 3 * small.gatherChunk() + 100;
    const k = small.gatherChunks(bytes);
    try testing.expectEqual(@as(u64, 4), k);
    var cover: [4]u64 = @splat(0);
    for (0..k) |ci| {
        const st = small.gatherStep(1, 0, 0, bytes, 7, ci);
        try testing.expectEqual(@as(u32, 4), st.ncollect);
        for (st.collects[0..4], 0..) |cl, g| {
            try testing.expectEqual(@as(u64, g) * bytes + ci * small.gatherChunk(), cl.dst);
            cover[g] += cl.bytes;
        }
    }
    for (cover) |n| try testing.expectEqual(bytes, n);
    // exchange: the lower rank at 0, the higher at the half; chunk counts agree from both ends
    const h = small.exchangeHalf();
    try testing.expectEqual(small.exchangeChunks(3 * h + 7, h + 1), small.exchangeChunks(h + 1, 3 * h + 7));
    const e0 = small.exchangeStep(0, 1, 0, 3 * h + 7, 0, h + 1, 14, 2, 1);
    const e1 = small.exchangeStep(1, 0, 0, h + 1, 0, 3 * h + 7, 14, 2, 1);
    try testing.expectEqual(e0.site, e1.site);
    try testing.expectEqual(@as(u64, 0), e0.tx_off);
    try testing.expectEqual(h, e1.tx_off);
    try testing.expectEqual(e1.tx_off, e0.collects[0].rx_off);
    try testing.expectEqual(@as(u64, 1), e1.bytes); // the higher rank's last byte
    try testing.expectEqual(@as(u64, 0), e1.collects[0].rx_off);
    try testing.expectEqual(h, e1.collects[0].bytes);
    try testing.expectEqual(@as(u64, 0), small.depth(0));
    try testing.expectEqual(@as(u64, 1), small.depth(9)); // coll.collect reads the last stage's seq: one in flight
    // lanes (TP4-RDMA-PIPE): one in flight a lane, chunk c on lane c % lanes, lane l owns slots l*m .. l*m+m-1
    const piped: Layout = .{ .world = 4, .slot_bytes = slot_small, .nslot = 8, .lanes = 4 };
    try testing.expectEqual(@as(u32, 2), piped.laneSlots());
    try testing.expectEqual(@as(u64, 4), piped.depth(9));
    try testing.expectEqual(@as(u64, 1), piped.depth(1));
    try testing.expectEqual(@as(u32, 0), piped.gatherStep(1, 0, 0, 20480 / 8, 7, 0).lane); // one slot: lane 0 (decode)
    try testing.expectEqual(@as(u32, 2), piped.gatherStep(1, 0, 0, 8 * piped.gatherChunk(), 7, 6).lane);
    try testing.expectEqual(@as(u32, 3), piped.exchangeStep(0, 1, 0, 3 * h + 7, 0, h + 1, 14, 2, 3).lane);
    try testing.expectEqual(@as(usize, 7), piped.slotIndex(3, 5));
    try testing.expectEqual(@as(usize, 0), piped.slotIndex(0, 4));
    try testing.expect(Layout.lanesFit(8, 4) and Layout.lanesFit(8, 1) and Layout.lanesFit(2, 1));
    try testing.expect(!Layout.lanesFit(8, 3) and !Layout.lanesFit(2, 2) and !Layout.lanesFit(16, 8) and !Layout.lanesFit(8, 0));
    try testing.expectEqual((@as(u32, 3) << 30) | 5, rdma_core.immOf(3, (1 << 30) | 5));
    try testing.expectEqual(@as(u32, 5), rdma_core.immOf(0, 5)); // one lane: the gather's low bits, as before
    try testing.expect(rdma_core.wrOf(9, 1, 2) != rdma_core.wrOf(9, 2, 1) and rdma_core.wrOf(1 << 50, 3, 3) & rdma_core.poison_bit == 0);
}

test "world 4: RdmaProxyN and its open type-check in the host build" {
    std.testing.refAllDecls(rdma_proxy.RdmaProxyN);
    _ = &rdma_proxy.open;
}

test "world 4: a swap12 gather word trades ranks 1 and 2, stays a gather of the same call, and differs from the plain word" {
    const w = Word.gatherSwap12(@intFromEnum(coll.Phase.two_parts), 0);
    const plain = Word.gather(@intFromEnum(coll.Phase.two_parts), 0);
    try testing.expect(w != plain);
    try testing.expect(!Word.isExchange(w));
    try testing.expect(Word.isSwap12(w) and !Word.isSwap12(plain));
    try testing.expectEqual(Word.call(plain), Word.call(w));
    const want = [4]u32{ 0, 2, 1, 3 };
    for (0..4) |g| {
        try testing.expectEqual(want[g], Word.place(w, @intCast(g)));
        try testing.expectEqual(@as(u32, @intCast(g)), Word.place(plain, @intCast(g)));
    }
    // an exchange word never reads as swapped, whatever its pair bits
    try testing.expect(!Word.isSwap12(Word.exchange(1, 2, 1, 3, 64, 64)));
    // one row, equal widths w: prompt2d's dst [2 r, 2 p, w] block of node g = 2 p + r is 2 r + p = place(g)
    for (0..4) |g| try testing.expectEqual(@as(u32, @intCast(2 * (g % 2) + g / 2)), Word.place(w, @intCast(g)));
    // the step: the block at its place, one collect of the whole slot in place order
    const l: Layout = .{ .world = 4, .slot_bytes = 1 << 16, .nslot = 2 };
    const s = l.gatherStep(1, 0x1000, 0x9000, 256, w, 0);
    try testing.expectEqual(@as(u64, 2 * 256), s.tx_off);
    try testing.expectEqual(@as(u32, 1), s.ncollect);
    try testing.expectEqual(@as(u64, 0x9000), s.collects[0].dst);
}

test "world 2 still resolves to the two-rank rdma_proxy; 3 and 4 to the per-peer one" {
    try testing.expectEqual(rdma_proxy.Path.two, rdma_proxy.pathFor(2));
    try testing.expectEqual(rdma_proxy.Path.per_peer, rdma_proxy.pathFor(4));
    try testing.expectEqual(rdma_proxy.Path.per_peer, rdma_proxy.pathFor(3));
    try testing.expectEqual(rdma_proxy.Path.two, rdma_proxy.pathFor(1)); // refused where it always was (tcp_coll.handshake)
}

test "world 4 fake fabric: gathers, uneven pair exchanges and chunked calls move every byte, in reused slots" {
    const calls = 7 * 300;
    var w = try World4.init(4, 8, slot_small, 2 * std.time.ns_per_s, calls);
    defer w.deinit();
    try w.run(null);
    var slots: u64 = 0;
    for (0..4) |g| {
        const r = &w.rk[g];
        try testing.expectEqual(@as(usize, 0), r.gpu.bad);
        try testing.expectEqual(@as(usize, calls), r.gpu.finished);
        try testing.expectEqual(@as(u32, 0), r.code);
        if (g > 0) try testing.expectEqual(w.rk[0].exchanged.load(.acquire), r.exchanged.load(.acquire));
        slots = r.exchanged.load(.acquire);
        try testing.expectEqual(3 * slots, r.core.sends);
        try testing.expectEqual(3 * slots, r.core.recvs);
    }
    std.debug.print("world 4 fake fabric: {d} calls a rank as {d} slot gathers, 0 bad bytes on 4 ranks\n", .{ calls, slots });
    try testing.expect(slots > calls); // the chunked calls took several
}

test "world 4 fake fabric: RX reuse with NSLOT=2 and a slow collector: nothing is overwritten" {
    var w = try World4.init(4, 2, slot_small, 2 * std.time.ns_per_s, 7 * 40);
    defer w.deinit();
    w.rk[2].gpu.collect_delay_ns = 200 * std.time.ns_per_us;
    try w.run(null);
    for (0..4) |g| {
        try testing.expectEqual(@as(usize, 0), w.rk[g].gpu.bad);
        try testing.expectEqual(@as(u32, 0), w.rk[g].code);
    }
}

test "world 4 lanes (TP4-RDMA-PIPE): 1, 2 and 4 lanes give byte-identical results; with 4 the chunks overlap" {
    const calls = 7 * 200;
    const a = std.heap.page_allocator;
    var ref: [W][]u64 = undefined;
    var cur: [W][]u64 = undefined;
    for (0..W) |g| {
        ref[g] = try a.alloc(u64, calls);
        cur[g] = try a.alloc(u64, calls);
    }
    defer for (0..W) |g| {
        a.free(ref[g]);
        a.free(cur[g]);
    };
    for ([_]u32{ 1, 2, 4 }) |lanes| {
        var w = try World4.initLanes(4, 8, lanes, slot_small, 2 * std.time.ns_per_s, calls);
        defer w.deinit();
        for (0..4) |g| w.rk[g].gpu.hashes = if (lanes == 1) ref[g] else cur[g];
        try w.run(null);
        var inflight: u32 = 0;
        for (0..4) |g| {
            const r = &w.rk[g];
            try testing.expectEqual(@as(usize, 0), r.gpu.bad);
            try testing.expectEqual(@as(usize, calls), r.gpu.finished);
            try testing.expectEqual(@as(u32, 0), r.code);
            if (lanes > 1) try testing.expectEqualSlices(u64, ref[g], cur[g]);
            try testing.expectEqual(3 * r.exchanged.load(.acquire), r.core.sends);
            inflight = @max(inflight, r.core.max_inflight);
        }
        std.debug.print("world 4 lanes={d}: {d} calls a rank as {d} slot gathers, 0 bad bytes, recv byte-identical to lanes=1 on 4 ranks, most turns in flight {d}\n", .{ lanes, calls, w.rk[0].exchanged.load(.acquire), inflight });
        if (lanes == 1) try testing.expectEqual(@as(u32, 1), inflight);
        if (lanes == 4) try testing.expect(inflight >= 2);
    }
}

test "world 4 lanes (TP4-RDMA-PIPE): a 12-chunk prompt gather and a 4-chunk exchange every call over 4 lanes, every byte once" {
    for ([_]bool{ false, true }) |xchg| {
        var w = try World4.initLanes(4, 8, 4, slot_small, 2 * std.time.ns_per_s, 150);
        defer w.deinit();
        const l = w.layout;
        const fixed: Call = if (xchg)
            .{ .exchange = true, .bytes = 3 * l.exchangeHalf() + 5, .rbytes = 3 * l.exchangeHalf() + 5, .phase = .two_pair, .layer = 4 }
        else
            .{ .bytes = 12 * l.gatherChunk() - 3, .phase = .prompt_moe, .layer = 9 };
        for (0..4) |g| w.rk[g].gpu.fixed = fixed;
        try w.run(null);
        for (0..4) |g| {
            try testing.expectEqual(@as(usize, 0), w.rk[g].gpu.bad);
            try testing.expectEqual(@as(usize, 150), w.rk[g].gpu.finished);
            try testing.expectEqual(@as(u32, 0), w.rk[g].code);
        }
        std.debug.print("world 4 lanes=4 fixed {s} of {d} B a rank: 150 calls as {d} slot gathers, 0 bad bytes, most turns in flight {d}\n", .{ if (xchg) "exchange" else "gather", fixed.bytes, w.rk[0].exchanged.load(.acquire), w.rk[0].core.max_inflight });
    }
}

test "world 4 fake fabric: a stalled collect stops the stage, not the data" {
    var w = try World4.init(4, 4, slot_small, 4 * std.time.ns_per_s, 60);
    defer w.deinit();
    w.rk[1].gpu.hold_collect_from = 10;
    w.rk[1].gpu.deadline_ns = 500 * std.time.ns_per_ms;
    try w.run(null);
    for (0..4) |g| try testing.expectEqual(@as(usize, 0), w.rk[g].gpu.bad);
    try testing.expect(w.rk[1].gpu.timed_out);
    try testing.expect(w.is(1, &.{"Timeout"}));
    for ([_]usize{ 0, 2, 3 }) |g| try testing.expect(w.is(g, &.{ "PeerLost", "Timeout" }));
}

fn expectMatrix(w: *World4, name: []const u8, ok: bool) !void {
    w.line(name, ok);
    try testing.expect(ok);
}

const P = 500 * std.time.ns_per_ms; // progress deadline in the matrix (TF_COLL_PROGRESS_MS)

test "world 4 fault matrix: kill rank 1 at call 500 -> every other rank PeerLost" {
    var w = try World4.init(4, 8, slot_small, P, 700);
    defer w.deinit();
    w.rk[1].gpu.kill_at = 500;
    try w.run(null);
    try expectMatrix(w, "kill rank 1 at call 500", w.is(0, &.{"PeerLost"}) and w.is(2, &.{"PeerLost"}) and w.is(3, &.{"PeerLost"}));
}

test "world 4 fault matrix: kill rank 0 at call 500 -> every other rank PeerLost" {
    var w = try World4.init(4, 8, slot_small, P, 700);
    defer w.deinit();
    w.rk[0].gpu.kill_at = 500;
    try w.run(null);
    try expectMatrix(w, "kill rank 0 at call 500", w.is(1, &.{"PeerLost"}) and w.is(2, &.{"PeerLost"}) and w.is(3, &.{"PeerLost"}));
}

test "world 4 fault matrix: stall rank 1 (2 s) at call 500 -> the others Timeout (or PeerLost from the first), rank 1 PeerLost" {
    var w = try World4.init(4, 8, slot_small, P, 700);
    defer w.deinit();
    w.rk[1].gpu.stall_at = 500;
    w.rk[1].gpu.stall_ns = 2 * std.time.ns_per_s;
    try w.run(null);
    var timeouts: usize = 0;
    for ([_]usize{ 0, 2, 3 }) |g| timeouts += @intFromBool(w.is(g, &.{"Timeout"}));
    const ok = timeouts >= 1 and w.is(0, &.{ "Timeout", "PeerLost" }) and w.is(2, &.{ "Timeout", "PeerLost" }) and w.is(3, &.{ "Timeout", "PeerLost" }) and w.is(1, &.{ "PeerLost", "Timeout" });
    try expectMatrix(w, "stall rank 1 (2 s) at call 500", ok);
}

test "world 4 fault matrix: a different site from rank 0 -> Mismatch on rank 0 and on the ranks that see it first" {
    var w = try World4.init(4, 8, slot_small, P, 700);
    defer w.deinit();
    w.rk[0].gpu.site_at = 497; // a quarters gather (497 % 7 == 0)
    try w.run(null);
    var mism: usize = 0;
    for (0..4) |g| mism += @intFromBool(w.is(g, &.{"Mismatch"}));
    const ok = w.is(0, &.{"Mismatch"}) and mism >= 2 and w.is(1, &.{ "Mismatch", "PeerLost" }) and w.is(2, &.{ "Mismatch", "PeerLost" }) and w.is(3, &.{ "Mismatch", "PeerLost" });
    try expectMatrix(w, "different site from rank 0", ok);
}

test "world 4 fault matrix: an exchange of a different size on rank 0 -> Mismatch on rank 0 and its partner" {
    var w = try World4.init(4, 8, slot_small, P, 700);
    defer w.deinit();
    w.rk[0].gpu.rbytes_at = 498; // the g^1 exchange (498 % 7 == 1)
    try w.run(null);
    const ok = w.is(0, &.{"Mismatch"}) and w.is(1, &.{"Mismatch"}) and w.is(2, &.{ "PeerLost", "Mismatch" }) and w.is(3, &.{ "PeerLost", "Mismatch" });
    try expectMatrix(w, "exchange size differs on rank 0", ok);
}

test "world 4 fault matrix: rank 0 names the wrong peer -> Mismatch where the pairing breaks" {
    var w = try World4.init(4, 8, slot_small, P, 700);
    defer w.deinit();
    w.rk[0].gpu.peer_at = 498; // rank 0 exchanges with 2 while 1 waits for it and 2 for 3
    try w.run(null);
    // rank 2 sees rank 0's word name it; rank 1 sees it unless rank 2's poison overtakes the trailer (rank 1 can still
    // be finishing the gather before when rank 0's trailer comes in early)
    var mism: usize = 0;
    for (0..4) |g| mism += @intFromBool(w.is(g, &.{"Mismatch"}));
    const ok = mism >= 2 and w.is(2, &.{"Mismatch"}) and w.is(0, &.{ "Mismatch", "PeerLost" }) and w.is(1, &.{ "Mismatch", "PeerLost" }) and w.is(3, &.{ "Mismatch", "PeerLost" });
    try expectMatrix(w, "rank 0 names the wrong peer", ok);
}

fn abortRank0(w: *World4) void {
    while (w.rk[0].gpu.finished < 100) sleepNs(100 * std.time.ns_per_us);
    w.rk[0].abort();
}

test "world 4 fault matrix: abort on rank 0 from a thread -> rank 0 Aborted, the others PeerLost" {
    var w = try World4.init(4, 8, slot_small, P, 7 * 3000);
    defer w.deinit();
    try w.run(abortRank0);
    try expectMatrix(w, "abort on rank 0 from a thread", w.is(0, &.{"Aborted"}) and w.is(1, &.{"PeerLost"}) and w.is(2, &.{"PeerLost"}) and w.is(3, &.{"PeerLost"}));
}

fn closeRank1(w: *World4) void {
    while (w.rk[1].gpu.finished < 500) sleepNs(100 * std.time.ns_per_us);
    w.rk[1].closeColl();
}

test "world 4 fault matrix: rank 1 closes its collective -> the others PeerLost" {
    var w = try World4.init(4, 8, slot_small, P, 7 * 3000);
    defer w.deinit();
    try w.run(closeRank1);
    try expectMatrix(w, "rank 1 closes its collective", w.is(0, &.{"PeerLost"}) and w.is(2, &.{"PeerLost"}) and w.is(3, &.{"PeerLost"}));
}

fn freezeRank1(w: *World4) void {
    while (w.rk[1].gpu.finished < 300) sleepNs(100 * std.time.ns_per_us);
    w.fab.ports[1].frozen.store(true, .release);
    sleepNs(2 * std.time.ns_per_s);
    w.fab.ports[1].frozen.store(false, .release);
}

test "world 4 fault matrix: SIGSTOP rank 1 for 2 s, then SIGCONT -> the others Timeout or PeerLost, rank 1 PeerLost or Timeout" {
    var w = try World4.init(4, 8, slot_small, P, 7 * 3000);
    defer w.deinit();
    try w.run(freezeRank1);
    var timeouts: usize = 0;
    for ([_]usize{ 0, 2, 3 }) |g| timeouts += @intFromBool(w.is(g, &.{"Timeout"}));
    const ok = timeouts >= 1 and w.is(0, &.{ "Timeout", "PeerLost" }) and w.is(2, &.{ "Timeout", "PeerLost" }) and w.is(3, &.{ "Timeout", "PeerLost" }) and w.is(1, &.{ "PeerLost", "Timeout" });
    try expectMatrix(w, "SIGSTOP rank 1 for 2 s, then SIGCONT", ok);
}

test "world 4 fault matrix negative controls: one flipped bit in one rank's send is found exactly once, in that rank's block, by every rank" {
    for ([_][2]usize{ .{ 0, 497 }, .{ 3, 87 } }) |spec| { // a 20 KiB quarters gather; a 16 B Markov gather (87 % 7 == 3)
        var w = try World4.init(4, 8, slot_small, 2 * std.time.ns_per_s, 7 * 80);
        defer w.deinit();
        w.rk[spec[0]].gpu.flip_at = spec[1];
        try w.run(null);
        var ok = true;
        for (0..4) |g| {
            const f = &w.rk[g].gpu;
            ok = ok and f.bad == 1 and f.bad_call.? == spec[1] and f.bad_block.? == spec[0] and w.rk[g].code == 0;
        }
        var name_buf: [96]u8 = undefined;
        try expectMatrix(w, try std.fmt.bufPrint(&name_buf, "negative control: rank {d} flips a bit at call {d}", .{ spec[0], spec[1] }), ok);
    }
}

// ---- the star handshake over loopback links ------------------------------------------------------------------------------

const StarRank = struct {
    rank: u32,
    port: u16,
    plan: []const u8,
    slots: u32 = 4,
    hs: ?anyerror = null,
    info: ?anyerror = null,
    epoch: [16]u8 = @splat(0),
    got: [W]nic.Info = undefined,

    fn run(s: *StarRank) void {
        var fan = Fan.open(testing.io, .{ 127, 0, 0, 1 }, s.port, s.rank, 4) catch |e| {
            s.hs = e;
            return;
        };
        defer fan.close();
        const id: coll.Identity = .{ .rank = s.rank, .world = 4, .kit_abi = 1, .kit_sha256 = @splat('0'), .modalities = 0, .plan = s.plan };
        const se = tcp_coll.handshakeStar(&fan, &id, "kind=rdma") catch |e| {
            s.hs = e;
            return;
        };
        s.epoch = se.epoch;
        var mine: [W]nic.Info = std.mem.zeroes([W]nic.Info);
        for (0..4) |j| if (j != s.rank) {
            mine[j] = .{ .qpn = @intCast(100 * s.rank + j), .psn = 1, .rkey = s.rank, .gid = @splat(0), .active_mtu = 5, .nslot = s.slots, .rx_base = 0x1000 * s.rank, .rx_len = 4 * 4096, .slot_bytes = 4096 };
        };
        s.got = nic.exchangeInfoStar(&fan, s.rank, 4, mine) catch |e| {
            s.info = e;
            return;
        };
        nic.startBarrierStar(&fan, s.rank, 4) catch |e| {
            s.info = e;
        };
    }
};

fn star(port: u16, plans: [4][]const u8, slots: [4]u32) ![4]StarRank {
    var s: [4]StarRank = undefined;
    var th: [4]std.Thread = undefined;
    for (&s, &th, 0..) |*x, *t, g| {
        x.* = .{ .rank = @intCast(g), .port = port, .plan = plans[g], .slots = slots[g] };
        t.* = try std.Thread.spawn(.{}, StarRank.run, .{x});
    }
    for (th) |t| t.join();
    return s;
}

test "world 4 star handshake: one epoch on every rank, every QP's Info reaches its peer, a barrier; a differing plan or slot count is refused on all" {
    // below the ephemeral range (32768-), so no connect's source port takes one of them
    const base: u16 = @intCast(20000 + @mod(c.getpid(), 1000) * 10);
    const good = try star(base, .{ "p=1", "p=1", "p=1", "p=1" }, .{ 4, 4, 4, 4 });
    for (good) |s| if (s.hs != null or s.info != null) std.debug.print("star handshake rank {d}: handshake {?} info {?}\n", .{ s.rank, s.hs, s.info });
    for (good) |s| {
        try testing.expectEqual(@as(?anyerror, null), s.hs);
        try testing.expectEqual(@as(?anyerror, null), s.info);
        try testing.expectEqualSlices(u8, &good[0].epoch, &s.epoch);
        for (0..4) |j| if (j != s.rank) {
            try testing.expectEqual(@as(u32, @intCast(100 * j + s.rank)), s.got[j].qpn); // rank j's QP toward this rank
            try testing.expectEqual(@as(u64, 0x1000 * j), s.got[j].rx_base);
        };
    }
    try testing.expect(!std.mem.allEqual(u8, &good[0].epoch, 0));
    const bad_plan = try star(base + 3, .{ "p=1", "p=1", "p=2", "p=1" }, .{ 4, 4, 4, 4 });
    for (bad_plan) |s| try testing.expectEqual(@as(?anyerror, error.Mismatch), s.hs);
    const bad_slots = try star(base + 6, .{ "p=1", "p=1", "p=1", "p=1" }, .{ 4, 4, 4, 8 });
    for (bad_slots) |s| {
        try testing.expectEqual(@as(?anyerror, null), s.hs);
        try testing.expectEqual(@as(?anyerror, error.Mismatch), s.info);
    }
}

// ---- the same protocol over real QPs: four ranks on one HCA (loopback), TF_RDMA_TEST_DEVICE --------------------------

const RealCore = rdma_core.CoreN(nic.Nics);

const RealRank = struct {
    shared: [rdma_core.max_lanes]Shared = std.mem.zeroes([rdma_core.max_lanes]Shared),
    nics: nic.Nics = undefined,
    core: RealCore = undefined,
    stop: std.atomic.Value(u32) = .init(0),
    exchanged: std.atomic.Value(u64) = .init(0),
    code: u32 = 0,
    gpu: FakeGpu4 = undefined,
    rx: []u8 = &.{},
    bufs: [2][]u8 = undefined,
    /// its ends of the star's control sockets (rank 0: one a follower; a follower: rank 0's), as the fan's links
    fds: [W - 1]c.fd_t = .{ -1, -1, -1 },

    fn runCore(r: *RealRank) void {
        r.code = r.core.run();
        if (r.code != 0) r.nics.toErr();
    }
};

fn realKill(ctx: *anyopaque, g: u32) void {
    const rs: *[W]RealRank = @ptrCast(@alignCast(ctx));
    rs[g].stop.store(1, .release);
    rs[g].nics.toErr(); // its QPs answer nothing any more, as a dead process's
    for (&rs[g].fds) |*fd| if (fd.* >= 0) { // and the kernel closes its sockets
        _ = c.close(fd.*);
        fd.* = -1;
    };
}

test "rdma loopback world 4: four ranks' QPs on one HCA move every gather and exchange; a dead rank is PeerLost to the others" {
    const dev = c.getenv("TF_RDMA_TEST_DEVICE") orelse return error.SkipZigTest;
    const gid_index = try nic.parseGidIndex(if (c.getenv("TF_RDMA_GID_INDEX")) |v| std.mem.span(v) else null);
    const a = std.heap.page_allocator;
    for ([_]?usize{ null, 300 }) |kill_at| {
        var v = try nic.Verbs.open(null);
        defer v.close();
        const layout: Layout = .{ .world = 4, .slot_bytes = slot_small, .nslot = 4, .lanes = 2 };
        const rs = try a.create([W]RealRank);
        defer a.destroy(rs);
        const epoch: [16]u8 = @splat(9);
        for (rs, 0..) |*r, g| {
            r.* = .{};
            r.rx = try a.alloc(u8, @intCast(layout.nslot * layout.slot_bytes));
            @memset(r.rx, 0);
            r.bufs = .{ try a.alloc(u8, 256 << 10), try a.alloc(u8, 512 << 10) };
            r.nics = try nic.Nics.open(&v, std.mem.span(dev), gid_index, @intCast(g), 4, r.rx);
        }
        defer for (rs) |*r| {
            r.nics.close();
            a.free(r.rx);
            a.free(r.bufs[0]);
            a.free(r.bufs[1]);
        };
        for (rs, 0..) |*r, g| for (0..4) |j| if (j != g) try r.nics.connect(@intCast(j), rs[j].nics.info(@intCast(g), layout.nslot, layout.slot_bytes));
        for (1..4) |g| { // the fan's star: rank 0 to each follower
            var sv: [2]c.fd_t = undefined;
            if (c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &sv) != 0) return error.SocketFailed;
            rs[0].fds[g - 1] = sv[0];
            rs[g].fds[0] = sv[1];
        }
        defer for (rs) |*r| for (r.fds) |fd| if (fd >= 0) {
            _ = c.close(fd);
        };
        const calls: usize = if (kill_at == null) 7 * 400 else 7 * 100;
        for (rs, 0..) |*r, g| {
            // past the QP's retry budget (TF_RDMA_QP_TIMEOUT 14: ~3.6 s): a dead rank is found by its closed control
            // socket (rank 0's peek) or by RETRY_EXC, never by the progress deadline
            r.core = .{ .shared = &r.shared, .rx = r.rx, .layout = layout, .net = &r.nics, .rank = @intCast(g), .epoch = epoch, .stop = &r.stop, .exchanged = &r.exchanged, .progress_ns = 10 * std.time.ns_per_s, .ctl_fds = r.fds };
            r.gpu = .{ .shared = &r.shared, .rx = r.rx, .layout = layout, .rank = @intCast(g), .calls = calls, .send = r.bufs[0], .recv = r.bufs[1], .kill = realKill, .kill_ctx = rs };
        }
        if (kill_at) |k| rs[2].gpu.kill_at = k;
        var cores: [W]std.Thread = undefined;
        var gpus: [W]std.Thread = undefined;
        const t0 = wire.monotonicNs();
        for (rs, 0..) |*r, g| {
            cores[g] = try std.Thread.spawn(.{}, RealRank.runCore, .{r});
            gpus[g] = try std.Thread.spawn(.{}, FakeGpu4.run, .{&r.gpu});
        }
        for (gpus) |t| t.join();
        const ms = (wire.monotonicNs() - t0) / std.time.ns_per_ms;
        for (rs) |*r| r.stop.store(1, .release);
        for (cores) |t| t.join();
        if (kill_at == null) {
            for (rs) |*r| {
                try testing.expectEqual(@as(usize, 0), r.gpu.bad);
                try testing.expectEqual(@as(u32, 0), r.code);
                try testing.expectEqual(calls, r.gpu.finished);
            }
            std.debug.print("rdma loopback world 4: {d} calls a rank ({d} slot gathers) in {d} ms, 0 bad bytes\n", .{ calls, rs[0].exchanged.load(.acquire), ms });
        } else {
            for ([_]usize{ 0, 1, 3 }) |g| try testing.expectEqual(@as(u32, 1), rs[g].code);
            std.debug.print("rdma loopback world 4: rank 2 dead at call 300, the others PeerLost after {d} ms\n", .{ms});
        }
    }
}

test "rdma loopback world 4 timing: a 20 KiB quarters gather and an 8 KiB pair exchange, ready -> done on rank 0" {
    const dev = c.getenv("TF_RDMA_TEST_DEVICE") orelse return error.SkipZigTest;
    const gid_index = try nic.parseGidIndex(if (c.getenv("TF_RDMA_GID_INDEX")) |v| std.mem.span(v) else null);
    const a = std.heap.page_allocator;
    const n = 2500;
    const Case = struct { call: Call, check: bool };
    for ([_]Case{
        .{ .call = .{ .bytes = 20480, .phase = .two_parts }, .check = true }, // as the world-2 timing test runs: host fill and verify between calls
        .{ .call = .{ .bytes = 20480, .phase = .two_parts }, .check = false },
        .{ .call = .{ .exchange = true, .bytes = 8192, .rbytes = 8192, .phase = .two_pair }, .check = false },
    }) |case| {
        const fixed = case.call;
        var v = try nic.Verbs.open(null);
        defer v.close();
        const layout: Layout = .{ .world = 4, .slot_bytes = rdma_proxy.default_slot_kib_n << 10, .nslot = 8 };
        const rs = try a.create([W]RealRank);
        defer a.destroy(rs);
        const epoch: [16]u8 = @splat(7);
        for (rs, 0..) |*r, g| {
            r.* = .{};
            r.rx = try a.alloc(u8, @intCast(layout.nslot * layout.slot_bytes));
            r.bufs = .{ try a.alloc(u8, 256 << 10), try a.alloc(u8, 512 << 10) };
            r.nics = try nic.Nics.open(&v, std.mem.span(dev), gid_index, @intCast(g), 4, r.rx);
        }
        defer for (rs) |*r| {
            r.nics.close();
            a.free(r.rx);
            a.free(r.bufs[0]);
            a.free(r.bufs[1]);
        };
        for (rs, 0..) |*r, g| for (0..4) |j| if (j != g) try r.nics.connect(@intCast(j), rs[j].nics.info(@intCast(g), layout.nslot, layout.slot_bytes));
        const rtt = try a.alloc(u64, n);
        defer a.free(rtt);
        for (rs, 0..) |*r, g| {
            r.core = .{ .shared = &r.shared, .rx = r.rx, .layout = layout, .net = &r.nics, .rank = @intCast(g), .epoch = epoch, .stop = &r.stop, .exchanged = &r.exchanged, .progress_ns = 4 * std.time.ns_per_s };
            r.gpu = .{ .shared = &r.shared, .rx = r.rx, .layout = layout, .rank = @intCast(g), .calls = n, .send = r.bufs[0], .recv = r.bufs[1], .fixed = fixed, .check = case.check, .rtt = if (g == 0) rtt else null };
        }
        var cores: [W]std.Thread = undefined;
        var gpus: [W]std.Thread = undefined;
        for (rs, 0..) |*r, g| {
            cores[g] = try std.Thread.spawn(.{}, RealRank.runCore, .{r});
            gpus[g] = try std.Thread.spawn(.{}, FakeGpu4.run, .{&r.gpu});
        }
        for (gpus) |t| t.join();
        for (rs) |*r| r.stop.store(1, .release);
        for (cores) |t| t.join();
        for (rs) |*r| {
            try testing.expectEqual(@as(usize, 0), r.gpu.bad);
            try testing.expectEqual(@as(u32, 0), r.code);
        }
        std.mem.sort(u64, rtt, {}, std.sort.asc(u64));
        var sum: u64 = 0;
        for (rtt) |x| sum += x;
        std.debug.print("rdma loopback world 4 {s} {d} B{s}: n={d} p50={d:.1} p99={d:.1} min={d:.1} max={d:.1} mean={d:.1} us (ready -> done on rank 0)\n", .{
            if (fixed.exchange) "exchange" else "gather", fixed.bytes, if (case.check) " (host fill+verify)" else " (no host fill)", n, us(rtt[n / 2]), us(rtt[n * 99 / 100]), us(rtt[0]), us(rtt[n - 1]), us(sum / n),
        });
    }
}

test "rdma loopback world 4 prompt gather (TP4-RDMA-PIPE): 8 slot chunks a call, 1 lane against 4, wall time a call on rank 0" {
    const dev = c.getenv("TF_RDMA_TEST_DEVICE") orelse return error.SkipZigTest;
    const gid_index = try nic.parseGidIndex(if (c.getenv("TF_RDMA_GID_INDEX")) |v| std.mem.span(v) else null);
    const a = std.heap.page_allocator;
    const n = 40;
    const base: Layout = .{ .world = 4, .slot_bytes = rdma_proxy.default_slot_kib_n << 10, .nslot = 8 };
    const bytes = 8 * base.gatherChunk();
    var ms: [2]f64 = undefined;
    for ([_]u32{ 1, 4 }, 0..) |lanes, li| {
        var v = try nic.Verbs.open(null);
        defer v.close();
        var layout = base;
        layout.lanes = lanes;
        const rs = try a.create([W]RealRank);
        defer a.destroy(rs);
        const epoch: [16]u8 = @splat(11);
        for (rs, 0..) |*r, g| {
            r.* = .{};
            r.rx = try a.alloc(u8, @intCast(layout.nslot * layout.slot_bytes));
            r.bufs = .{ try a.alloc(u8, @intCast(bytes)), try a.alloc(u8, @intCast(4 * bytes)) };
            @memset(r.bufs[0], @intCast(g + 1));
            r.nics = try nic.Nics.open(&v, std.mem.span(dev), gid_index, @intCast(g), 4, r.rx);
        }
        defer for (rs) |*r| {
            r.nics.close();
            a.free(r.rx);
            a.free(r.bufs[0]);
            a.free(r.bufs[1]);
        };
        for (rs, 0..) |*r, g| for (0..4) |j| if (j != g) try r.nics.connect(@intCast(j), rs[j].nics.info(@intCast(g), layout.nslot, layout.slot_bytes));
        for (rs, 0..) |*r, g| {
            r.core = .{ .shared = &r.shared, .rx = r.rx, .layout = layout, .net = &r.nics, .rank = @intCast(g), .epoch = epoch, .stop = &r.stop, .exchanged = &r.exchanged, .progress_ns = 4 * std.time.ns_per_s };
            r.gpu = .{ .shared = &r.shared, .rx = r.rx, .layout = layout, .rank = @intCast(g), .calls = n, .send = r.bufs[0], .recv = r.bufs[1], .fixed = .{ .bytes = bytes, .phase = .prompt_moe, .layer = 1 }, .check = false };
        }
        var cores: [W]std.Thread = undefined;
        var gpus: [W]std.Thread = undefined;
        const t0 = wire.monotonicNs();
        for (rs, 0..) |*r, g| {
            cores[g] = try std.Thread.spawn(.{}, RealRank.runCore, .{r});
            gpus[g] = try std.Thread.spawn(.{}, FakeGpu4.run, .{&r.gpu});
        }
        gpus[0].join();
        const dt = wire.monotonicNs() - t0;
        for (gpus[1..]) |t| t.join();
        for (rs) |*r| r.stop.store(1, .release);
        for (cores) |t| t.join();
        for (rs, 0..) |*r, g| {
            try testing.expectEqual(@as(u32, 0), r.code);
            try testing.expectEqual(@as(usize, n), r.gpu.finished);
            for (0..4) |b| try testing.expect(std.mem.allEqual(u8, r.bufs[1][b * bytes ..][0..bytes], @intCast(b + 1)));
            _ = g;
        }
        ms[li] = @as(f64, @floatFromInt(dt)) / 1e6 / n;
        std.debug.print("rdma loopback world 4 prompt gather lanes={d}: {d} B a rank in {d} slot chunks, {d} calls, {d:.2} ms a call on rank 0 ({d:.2} GB/s into a rank), most turns in flight {d}\n", .{
            lanes, bytes, base.gatherChunks(bytes), n, ms[li], @as(f64, @floatFromInt(3 * bytes)) / (ms[li] * 1e6), rs[0].core.max_inflight,
        });
    }
    std.debug.print("rdma loopback world 4 prompt gather: lanes=4 takes {d:.2}x the time of lanes=1 (one HCA carries all four ranks' writes)\n", .{ms[1] / ms[0]});
}

fn us(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1000.0;
}
