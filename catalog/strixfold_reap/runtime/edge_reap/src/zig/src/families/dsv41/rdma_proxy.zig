//! TF_COLL_KIND=rdma: tcp_proxy's slot collective, each exchange one RC write with immediate (COLLECTIVES-DESIGN 3.6).
const std = @import("std");
const gpu = @import("gpu");
const coll = @import("coll.zig");
const Fan = @import("fan.zig").Fan;
const tcp_coll = @import("tcp_coll.zig");
const tcp_proxy = @import("tcp_proxy.zig");
const comm = @import("comm.zig");
pub const nic = @import("rdma_nic.zig");
pub const Core = @import("rdma_core.zig").Core;

const c = std.c;
const Verbs = nic.Verbs;
const Nic = nic.Nic;
pub const Shared = tcp_proxy.Shared;
pub const max_slots = tcp_proxy.max_slots;
pub const Error = nic.Error;
const StageK = gpu.Kernel("coll.stage", tcp_proxy.params.Stage);
const CollectK = gpu.Kernel("coll.collect", tcp_proxy.params.Collect);

/// The same slab, kernels, generation protocol and fault contract as tcp_proxy; graph capture is unchanged.
pub const RdmaProxy = struct {
    gpa: std.mem.Allocator,
    d: *const gpu.Driver,
    rank: u32,
    host: *tcp_coll.TcpColl,
    verbs: Verbs,
    nic: Nic,
    peer: nic.Info,
    epoch: [16]u8,
    slab: gpu.HostBuffer,
    shared: *Shared,
    dev: u64, // the slab's device address
    tx_off: u64,
    rx_off: u64,
    nslot: u32,
    slot_bytes: u64,
    deadline_ticks: u64,
    progress_ns: u64,
    stage: StageK,
    collect: CollectK,
    proxy_cpu: i32,
    thread: ?std.Thread = null,
    stop: std.atomic.Value(u32) = .init(0),
    /// the proxy's failure (coll.Error as a code), 0 while healthy
    failed: std.atomic.Value(u32) = .init(0),
    /// gathers the proxy has completed (diagnostics)
    exchanged: std.atomic.Value(u64) = .init(0),
    begun: ?Begun = null,

    /// `stream` is the begin's; the end must be on it (see gatherEnd)
    const Begun = struct { host: bool, recv: u64, bytes: usize, stream: gpu.Stream };
    const vtable: coll.Coll.VTable = .{ .all_gather = allGather, .exchange = exchangeColl, .abort = abort, .close = close, .gather_begin = gatherBegin, .gather_end = gatherEnd, .check = check, .counts = counts };

    /// REAP's partials are one exchange and do not fit the slot gather. The host link moves them.
    pub fn providesExchange() bool {
        return vtable.exchange != null;
    }

    fn counts(ptr: *anyopaque) coll.Coll.Counts {
        const t: *RdmaProxy = @ptrCast(@alignCast(ptr));
        return .{ .staged = @atomicLoad(u64, &t.shared.seq, .acquire), .cable1 = t.exchanged.load(.acquire), .host = t.host.w.seq, .host_bytes = t.host.xchg_bytes, .host_ns = t.host.xchg_ns };
    }

    fn check(ptr: *anyopaque) coll.Error!void {
        const t: *RdmaProxy = @ptrCast(@alignCast(ptr));
        if (t.status()) |e| return e;
    }

    fn coll_(t: *RdmaProxy) coll.Coll {
        return .{ .ptr = t, .vtable = &vtable, .rank = t.rank, .world = 2, .capturable = true };
    }

    fn status(t: *RdmaProxy) ?coll.Error {
        const f = t.failed.load(.acquire);
        if (f != 0) return switch (f) {
            1 => error.PeerLost,
            2 => error.Timeout,
            3 => error.Mismatch,
            4 => error.Aborted,
            else => error.CollectiveFailed,
        };
        if (@atomicLoad(u32, &t.shared.abort, .acquire) != 0) return error.Timeout; // a device wait gave up
        return null;
    }

    fn poison(t: *RdmaProxy, code: u32) void {
        _ = t.failed.cmpxchgStrong(0, code, .acq_rel, .acquire);
        tcp_proxy.raiseAbort(t.shared);
        t.nic.toErr();
        t.host.w.abort();
    }

    fn proxyLoop(t: *RdmaProxy) void {
        tcp_proxy.TcpProxy.pin(t.proxy_cpu);
        var core: Core = .{
            .shared = t.shared,
            .tx = t.slab.bytes[t.tx_off..][0 .. t.nslot * t.slot_bytes],
            .rx = t.slab.bytes[t.rx_off..][0 .. t.nslot * t.slot_bytes],
            .nslot = t.nslot,
            .slot_bytes = t.slot_bytes,
            .nic = &t.nic,
            .peer_rx = t.peer.rx_base,
            .peer_rkey = t.peer.rkey,
            .epoch = t.epoch,
            .stop = &t.stop,
            .exchanged = &t.exchanged,
            .progress_ns = t.progress_ns,
            .peer_fd = t.host.w.fds[0],
            .rank = t.rank,
            .warn_ns = t.host.gpu_warn_ns,
        };
        const code = core.run();
        if (code != 0) t.poison(code);
    }

    /// The host link's point to point (tcp_coll.exchangeColl). A slot gather stays on the proxy.
    fn exchangeColl(ptr: *anyopaque, stream: gpu.Stream, site: coll.Site, send: u64, send_bytes: usize, recv: u64, recv_bytes: usize, peer: u32) coll.Error!void {
        const t: *RdmaProxy = @ptrCast(@alignCast(ptr));
        if (t.status()) |e| return e;
        if (t.begun != null) return t.misuse("an exchange while a begun gather is pending");
        const f = t.host.coll_().vtable.exchange orelse return t.misuse("an exchange this transport does not provide");
        return f(t.host, stream, site, send, send_bytes, recv, recv_bytes, peer);
    }

    fn allGather(ptr: *anyopaque, stream: gpu.Stream, site: coll.Site, send: u64, recv: u64, bytes: usize) coll.Error!void {
        const t: *RdmaProxy = @ptrCast(@alignCast(ptr));
        if (t.begun != null) return t.misuse("an all-gather while a begun gather is pending");
        try gatherBegin(ptr, stream, site, send, recv, bytes);
        return gatherEnd(ptr, stream);
    }

    fn misuse(t: *RdmaProxy, what: []const u8) coll.Error {
        std.log.err("collectives: {s}", .{what});
        t.poison(5);
        return error.CollectiveFailed;
    }

    /// The stage kernel (this rank's block into its slot) or, for a gather larger than a slot, tcp_host's begin.
    fn gatherBegin(ptr: *anyopaque, stream: gpu.Stream, site: coll.Site, send: u64, recv: u64, bytes: usize) coll.Error!void {
        const t: *RdmaProxy = @ptrCast(@alignCast(ptr));
        if (t.status()) |e| return e;
        if (t.begun != null) return t.misuse("a gather begun while another is pending");
        if (bytes > nic.capacity(t.slot_bytes)) {
            const capturing = stream.capturing() catch return error.CollectiveFailed;
            if (capturing) {
                std.log.err("collectives: a {d}-byte gather inside a graph exceeds what a {d}-byte slot carries (TF_COLL_SLOT_KIB)", .{ bytes, t.slot_bytes });
                t.poison(5);
                return error.CollectiveFailed;
            }
            try t.host.coll_().vtable.gather_begin.?(t.host, stream, site, send, recv, bytes);
            t.begun = .{ .host = true, .recv = recv, .bytes = bytes, .stream = stream };
            return;
        }
        t.begun = .{ .host = false, .recv = recv, .bytes = bytes, .stream = stream };
        if (bytes == 0) return;
        const site_word = (@as(u32, @intFromEnum(site.phase)) << 16) | site.layer;
        const cfg: gpu.Config = .{ .grid = .{ .x = tcp_proxy.TcpProxy.blocks(bytes), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } };
        t.stage.launch(stream, cfg, .{ .s = t.dev, .send = send, .tx = t.dev + t.tx_off, .own = recv + @as(u64, t.rank) * bytes, .bytes = bytes, .slot_bytes = t.slot_bytes, .nslot = t.nslot, .site = site_word, .deadline = t.deadline_ticks }) catch {
            t.poison(5);
            return error.CollectiveFailed;
        };
    }

    /// The collect kernel (waits for the proxy, then the peer's block into `recv`), or tcp_host's end.
    fn gatherEnd(ptr: *anyopaque, stream: gpu.Stream) coll.Error!void {
        const t: *RdmaProxy = @ptrCast(@alignCast(ptr));
        const b = t.begun orelse return t.misuse("a gather ended that was never begun");
        t.begun = null;
        // RX[s] reuse needs collect(q) ahead of stage(q + nslot - 1) in stream order (COLLECTIVES-DESIGN 3.6)
        if (b.stream.handle != stream.handle) return t.misuse("a gather ended on a different stream from its begin");
        if (b.host) return t.host.coll_().vtable.gather_end.?(t.host, stream);
        if (t.status()) |e| return e;
        if (b.bytes == 0) return;
        const peer: u64 = 1 - t.rank;
        const cfg: gpu.Config = .{ .grid = .{ .x = tcp_proxy.TcpProxy.blocks(b.bytes), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } };
        t.collect.launch(stream, cfg, .{ .s = t.dev, .rx = t.dev + t.rx_off, .peer = b.recv + peer * b.bytes, .bytes = b.bytes, .slot_bytes = t.slot_bytes, .nslot = t.nslot, .deadline = t.deadline_ticks }) catch {
            t.poison(5);
            return error.CollectiveFailed;
        };
    }

    fn abort(ptr: *anyopaque, reason: coll.Reason) void {
        _ = reason;
        const t: *RdmaProxy = @ptrCast(@alignCast(ptr));
        t.poison(4);
        t.host.coll_().vtable.abort(t.host, .cancelled);
    }

    fn close(ptr: *anyopaque) void {
        const t: *RdmaProxy = @ptrCast(@alignCast(ptr));
        t.stop.store(1, .release);
        @atomicStore(u32, &t.shared.abort, 1, .release); // ends any device wait
        t.nic.toErr(); // posted work flushes; the loop leaves
        if (t.thread) |th| th.join();
        // queued kernels see the abort word and leave the slab alone; the MRs go after they drained and the QP is gone
        _ = t.d.api.hipDeviceSynchronize();
        t.nic.close();
        t.verbs.close();
        t.slab.free();
        t.host.coll_().vtable.close(t.host);
        const gpa = t.gpa;
        gpa.destroy(t);
    }
};

/// Which rdma_proxy a world opens: the two-rank one (world 2, and anything else it refuses as before) or the
/// per-peer one (world 3 and 4: RdmaProxyN).
pub const Path = enum { two, per_peer };
pub fn pathFor(world: u32) Path {
    return if (world == 3 or world == 4) .per_peer else .two;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const gpu.Context, kit: *const gpu.Kit, fan: *const Fan, id: *const coll.Identity) anyerror!coll.Coll {
    _ = io;
    if (pathFor(id.world) == .per_peer) return openN(gpa, ctx, kit, fan, id);
    const d = ctx.d;
    const nslot: u32 = @intCast(try tcp_proxy.envU("TF_COLL_SLOTS", 8));
    if (nslot < 2 or nslot > max_slots) {
        std.log.err("collectives: TF_COLL_SLOTS={d}: kind=rdma needs 2 to {d} slots (receive-slot reuse is safe only transitively)", .{ nslot, max_slots });
        return error.BadSetting;
    }
    const slot_kib = try tcp_proxy.envU("TF_COLL_SLOT_KIB", 4352); // 4.25 MiB: a 16-row head gather (4,136,960 B) plus the trailer fits
    const slot_bytes = std.mem.alignForward(u64, slot_kib << 10, 4096);
    const proxy_cpu: i32 = @intCast(try tcp_proxy.envU("TF_COLL_PROXY_CPU", 6));
    const stripe_kib = try tcp_proxy.envU("TF_COLL_PROXY_STRIPE_KIB", 0);
    if (stripe_kib != 0) {
        std.log.err("collectives: TF_COLL_PROXY_STRIPE_KIB={d}: kind=rdma moves every slot gather over one QP (a second thread breaks the one-sequence rule)", .{stripe_kib});
        return error.BadSetting;
    }
    const gid_index = nic.parseGidIndex(if (c.getenv("TF_RDMA_GID_INDEX")) |v| std.mem.span(v) else null) catch {
        std.log.err("collectives: TF_RDMA_GID_INDEX: a decimal GID index in 0..255", .{});
        return error.BadSetting;
    };
    var vb = try Verbs.open(null);
    errdefer vb.close();
    var name_buf: [64]u8 = undefined;
    const dev_name = try nic.deviceName(&vb, &name_buf, id.rank);
    // the device name is this rank's own (the two boxes may differ), so it is logged, not compared
    var extra_buf: [128]u8 = undefined;
    const extra = try std.fmt.bufPrint(&extra_buf, "kind=rdma gid={d} tc=0 slots={d} slot_kib={d} stripe=0", .{ gid_index, nslot, slot_kib });
    const se = try tcp_coll.handshake(fan, id, extra);
    const host = try tcp_coll.openHost(gpa, d, id.rank, se, 0);
    errdefer host.coll_().vtable.close(host);
    const stage = try StageK.bind(kit, &.{});
    const collect = try CollectK.bind(kit, &.{});
    const caps = try ctx.caps();
    const tx_off: u64 = 4096;
    const rx_off: u64 = tx_off + nslot * slot_bytes;
    var slab = try gpu.HostBuffer.allocFlags(d, rx_off + nslot * slot_bytes, .{ .mapped = true, .coherent = true });
    errdefer slab.free();
    @memset(slab.bytes[0..@sizeOf(Shared)], 0);
    const t = try gpa.create(RdmaProxy);
    errdefer gpa.destroy(t);
    t.* = .{
        .gpa = gpa,
        .d = d,
        .rank = id.rank,
        .host = host,
        .verbs = vb,
        .nic = undefined,
        .peer = undefined,
        .epoch = se.epoch,
        .slab = slab,
        .shared = @ptrCast(@alignCast(slab.bytes.ptr)),
        .dev = try slab.device(),
        .tx_off = tx_off,
        .rx_off = rx_off,
        .nslot = nslot,
        .slot_bytes = slot_bytes,
        .deadline_ticks = tcp_proxy.deadlineTicks(caps.wall_clock_hz, se.settings.gpu_wait_ns),
        .progress_ns = se.settings.opt.progress_ns,
        .stage = stage,
        .collect = collect,
        .proxy_cpu = proxy_cpu,
    };
    t.nic = try Nic.open(&t.verbs, dev_name, gid_index, slab.bytes[tx_off..][0 .. nslot * slot_bytes], slab.bytes[rx_off..][0 .. nslot * slot_bytes]);
    errdefer t.nic.close();
    t.peer = try nic.exchangeInfo(fan, t.nic.info(nslot, slot_bytes));
    try t.nic.connect(t.peer);
    try nic.startBarrier(fan);
    t.thread = try std.Thread.spawn(.{}, RdmaProxy.proxyLoop, .{t});
    std.log.info("collectives: rdma_proxy on {s} gid={d} (mtu {d}, qp {d} -> peer qp {d}, rkey {x}) with {d} slots of {d} KiB (coherent slab {d} MiB, {d}-byte trailer), proxy thread on CPU {d}, progress deadline {d} ms; larger gathers through tcp_host", .{
        dev_name, gid_index, t.nic.active_mtu, t.nic.qp.qp_num, t.peer.qpn, t.peer.rkey, nslot, slot_bytes >> 10, slab.bytes.len >> 20, nic.trailer_bytes, proxy_cpu, t.progress_ns / std.time.ns_per_ms,
    });
    return t.coll_();
}

// ---- world 4 (TP4-RDMA-N) ------------------------------------------------------------------------------------------------

const rdma_core = @import("rdma_core.zig");
const Layout = rdma_core.Layout;
const Word = rdma_core.Word;

/// TF_COLL_KIND=rdma at world 4: the slot collective of RdmaProxy over one QP per peer (rdma_core.zig's world-4
/// picture), with the four-node split's exchange. The same kernels (coll.stage, coll.collect: unchanged), the same
/// slab header and fault contract; every call is device-side, so it is capturable, and nothing falls back to tcp_host
/// (a gather or exchange larger than a slot runs as consecutive slot gathers, up to TF_COLL_LANES of them in flight,
/// one a lane). The two-rank path above is untouched.
pub const RdmaProxyN = struct {
    const Core4 = rdma_core.CoreN(nic.Nics);
    gpa: std.mem.Allocator,
    d: *const gpu.Driver,
    rank: u32,
    world: u32,
    verbs: Verbs,
    nics: nic.Nics,
    epoch: [16]u8,
    slab: gpu.HostBuffer,
    /// one slab header a lane (layout.lanes), at the slab's start
    shared: [*]Shared,
    dev: u64,
    rx_off: u64,
    layout: Layout,
    deadline_ticks: u64,
    progress_ns: u64,
    warn_ns: u64,
    stage: StageK,
    collect: CollectK,
    proxy_cpu: i32,
    ctl_fds: [rdma_core.max_world - 1]c.fd_t = .{ -1, -1, -1 },
    thread: ?std.Thread = null,
    stop: std.atomic.Value(u32) = .init(0),
    failed: std.atomic.Value(u32) = .init(0),
    exchanged: std.atomic.Value(u64) = .init(0),
    begun: ?Begun = null,
    /// TF_COLL_SWAP12=1: swap12 sites are served (Word.gatherSwap12); the same on every rank (the handshake compares it)
    swap12: bool = false,
    /// TF_DS_VGATHER=1: prompt2d assembles a multi-row equal-width quarters with two copy_rows16. Not in the
    /// handshake (the all-gather word and the bytes are the ones issued with it off).
    vgather: bool = false,

    const Begun = struct { site: u32, send: u64, recv: u64, bytes: usize, stream: gpu.Stream, chunks: u64 };
    const vtable: coll.Coll.VTable = .{ .all_gather = allGather, .exchange = exchange, .abort = abort, .close = close, .gather_begin = gatherBegin, .gather_end = gatherEnd, .check = check, .counts = counts };

    pub fn providesExchange() bool {
        return vtable.exchange != null;
    }

    fn counts(ptr: *anyopaque) coll.Coll.Counts {
        const t: *RdmaProxyN = @ptrCast(@alignCast(ptr));
        var staged: u64 = 0;
        for (0..t.layout.lanes) |li| staged += @atomicLoad(u64, &t.shared[li].seq, .acquire);
        return .{ .staged = staged, .cable1 = t.exchanged.load(.acquire) };
    }

    fn check(ptr: *anyopaque) coll.Error!void {
        const t: *RdmaProxyN = @ptrCast(@alignCast(ptr));
        if (t.status()) |e| return e;
    }

    fn coll_(t: *RdmaProxyN) coll.Coll {
        return .{ .ptr = t, .vtable = &vtable, .rank = t.rank, .world = t.world, .capturable = true, .swap12 = t.swap12, .vgather = t.vgather };
    }

    fn status(t: *RdmaProxyN) ?coll.Error {
        const f = t.failed.load(.acquire);
        if (f != 0) return switch (f) {
            1 => error.PeerLost,
            2 => error.Timeout,
            3 => error.Mismatch,
            4 => error.Aborted,
            else => error.CollectiveFailed,
        };
        for (0..t.layout.lanes) |li| if (@atomicLoad(u32, &t.shared[li].abort, .acquire) != 0) return error.Timeout;
        return null;
    }

    /// The first failure stays; the abort word ends every device wait and the proxy thread, which poisons the peers
    /// and moves its QPs to error as it leaves.
    fn poison(t: *RdmaProxyN, code: u32) void {
        _ = t.failed.cmpxchgStrong(0, code, .acq_rel, .acquire);
        for (0..t.layout.lanes) |li| tcp_proxy.raiseAbort(&t.shared[li]);
    }

    /// Lane li's header and its first slot, as the kernels see them (device addresses), and its slot count.
    fn laneHeader(t: *const RdmaProxyN, li: u32) u64 {
        return t.dev + @as(u64, li) * @sizeOf(Shared);
    }
    fn laneSlots(t: *const RdmaProxyN, li: u32) u64 {
        return t.dev + t.rx_off + @as(u64, li) * t.layout.laneSlots() * t.layout.slot_bytes;
    }

    fn proxyLoop(t: *RdmaProxyN) void {
        tcp_proxy.TcpProxy.pin(t.proxy_cpu);
        var core: Core4 = .{
            .shared = t.shared,
            .rx = t.slab.bytes[t.rx_off..][0 .. t.layout.nslot * t.layout.slot_bytes],
            .layout = t.layout,
            .net = &t.nics,
            .rank = t.rank,
            .epoch = t.epoch,
            .stop = &t.stop,
            .exchanged = &t.exchanged,
            .progress_ns = t.progress_ns,
            .ctl_fds = t.ctl_fds,
            .warn_ns = t.warn_ns,
        };
        const code = core.run();
        if (code != 0) {
            t.poison(code);
            t.nics.toErr();
        }
    }

    fn misuse(t: *RdmaProxyN, what: []const u8) coll.Error {
        std.log.err("collectives: {s}", .{what});
        t.poison(5);
        return error.CollectiveFailed;
    }

    fn launchStage(t: *RdmaProxyN, stream: gpu.Stream, s: Layout.Step) coll.Error!void {
        const cfg: gpu.Config = .{ .grid = .{ .x = tcp_proxy.TcpProxy.blocks(s.bytes), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } };
        // own = send: coll.stage writes the block into its place in the slot only; coll.collect brings it to recv
        t.stage.launch(stream, cfg, .{ .s = t.laneHeader(s.lane), .send = s.send, .tx = t.laneSlots(s.lane) + s.tx_off, .own = s.send, .bytes = s.bytes, .slot_bytes = t.layout.slot_bytes, .nslot = t.layout.laneSlots(), .site = s.site, .deadline = t.deadline_ticks }) catch {
            t.poison(5);
            return error.CollectiveFailed;
        };
    }

    fn launchCollects(t: *RdmaProxyN, stream: gpu.Stream, s: Layout.Step) coll.Error!void {
        for (s.collects[0..s.ncollect]) |cl| {
            const cfg: gpu.Config = .{ .grid = .{ .x = tcp_proxy.TcpProxy.blocks(cl.bytes), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } };
            t.collect.launch(stream, cfg, .{ .s = t.laneHeader(s.lane), .rx = t.laneSlots(s.lane) + cl.rx_off, .peer = cl.dst, .bytes = cl.bytes, .slot_bytes = t.layout.slot_bytes, .nslot = t.layout.laneSlots(), .deadline = t.deadline_ticks }) catch {
                t.poison(5);
                return error.CollectiveFailed;
            };
        }
    }

    fn allGather(ptr: *anyopaque, stream: gpu.Stream, site: coll.Site, send: u64, recv: u64, bytes: usize) coll.Error!void {
        const t: *RdmaProxyN = @ptrCast(@alignCast(ptr));
        if (t.begun != null) return t.misuse("an all-gather while a begun gather is pending");
        try gatherBegin(ptr, stream, site, send, recv, bytes);
        return gatherEnd(ptr, stream);
    }

    /// The first depth(chunks) stages (one for a gather that fits a slot); the rest go with gatherEnd's collects.
    fn gatherBegin(ptr: *anyopaque, stream: gpu.Stream, site: coll.Site, send: u64, recv: u64, bytes: usize) coll.Error!void {
        const t: *RdmaProxyN = @ptrCast(@alignCast(ptr));
        if (t.status()) |e| return e;
        if (t.begun != null) return t.misuse("a gather begun while another is pending");
        if (site.swap12 and !t.swap12) return t.misuse("a swap12 gather without TF_COLL_SWAP12=1");
        const word = if (site.swap12) Word.gatherSwap12(@intFromEnum(site.phase), site.layer) else Word.gather(@intFromEnum(site.phase), site.layer);
        const k = t.layout.gatherChunks(bytes);
        t.begun = .{ .site = word, .send = send, .recv = recv, .bytes = bytes, .stream = stream, .chunks = k };
        for (0..t.layout.depth(k)) |ci| try t.launchStage(stream, t.layout.gatherStep(t.rank, send, recv, bytes, word, ci));
    }

    fn gatherEnd(ptr: *anyopaque, stream: gpu.Stream) coll.Error!void {
        const t: *RdmaProxyN = @ptrCast(@alignCast(ptr));
        const b = t.begun orelse return t.misuse("a gather ended that was never begun");
        t.begun = null;
        if (b.stream.handle != stream.handle) return t.misuse("a gather ended on a different stream from its begin");
        if (t.status()) |e| return e;
        const w = t.layout.depth(b.chunks);
        for (0..b.chunks) |ci| {
            try t.launchCollects(stream, t.layout.gatherStep(t.rank, b.send, b.recv, b.bytes, b.site, ci));
            if (ci + w < b.chunks) try t.launchStage(stream, t.layout.gatherStep(t.rank, b.send, b.recv, b.bytes, b.site, ci + w));
        }
    }

    fn exchange(ptr: *anyopaque, stream: gpu.Stream, site: coll.Site, send: u64, send_bytes: usize, recv: u64, recv_bytes: usize, peer: u32) coll.Error!void {
        const t: *RdmaProxyN = @ptrCast(@alignCast(ptr));
        if (t.status()) |e| return e;
        if (t.begun != null) return t.misuse("an exchange while a begun gather is pending");
        if (peer >= t.world or peer == t.rank) return t.misuse("an exchange with no other rank");
        const l = t.layout;
        const k = l.exchangeChunks(send_bytes, recv_bytes);
        const w = l.depth(k);
        const ph = @intFromEnum(site.phase);
        for (0..w) |ci| try t.launchStage(stream, l.exchangeStep(t.rank, peer, send, send_bytes, recv, recv_bytes, ph, site.layer, ci));
        for (0..k) |ci| {
            try t.launchCollects(stream, l.exchangeStep(t.rank, peer, send, send_bytes, recv, recv_bytes, ph, site.layer, ci));
            if (ci + w < k) try t.launchStage(stream, l.exchangeStep(t.rank, peer, send, send_bytes, recv, recv_bytes, ph, site.layer, ci + w));
        }
    }

    fn abort(ptr: *anyopaque, reason: coll.Reason) void {
        _ = reason;
        const t: *RdmaProxyN = @ptrCast(@alignCast(ptr));
        t.poison(4);
    }

    fn close(ptr: *anyopaque) void {
        const t: *RdmaProxyN = @ptrCast(@alignCast(ptr));
        // the abort word ends any device wait and the proxy thread, which poisons the peers as it leaves (a peer that
        // closes is PeerLost to the others at once, as at world 2)
        for (0..t.layout.lanes) |li| @atomicStore(u32, &t.shared[li].abort, 1, .release);
        if (t.thread) |th| th.join();
        t.stop.store(1, .release);
        t.nics.toErr();
        _ = t.d.api.hipDeviceSynchronize();
        t.nics.close();
        t.verbs.close();
        t.slab.free();
        const gpa = t.gpa;
        gpa.destroy(t);
    }
};

/// TF_COLL_SLOT_KIB's default at world 4: 8.5 MiB, so a 16-row head quarter (16 x 32,384 fp32 = 2,072,576 B a rank)
/// fits one slot gather of four blocks.
pub const default_slot_kib_n = 8704;
/// TF_COLL_LANES's default at world 4 (TP4-RDMA-PIPE): 4 lanes of the 8 default slots (2 each), so a prompt gather's
/// chunks keep up to 4 slot gathers on the wire; a call that fits one slot (decode) always runs on lane 0. 1 is the
/// one-in-flight protocol of TP4-RDMA-N.
pub const default_lanes_n = 4;

fn openN(gpa: std.mem.Allocator, ctx: *const gpu.Context, kit: *const gpu.Kit, fan: *const Fan, id: *const coll.Identity) anyerror!coll.Coll {
    const d = ctx.d;
    if (id.world > rdma_core.max_world or id.world < 3) return error.UnsupportedWorld;
    const nslot: u32 = @intCast(try tcp_proxy.envU("TF_COLL_SLOTS", 8));
    if (nslot < 2 or nslot > max_slots) {
        std.log.err("collectives: TF_COLL_SLOTS={d}: kind=rdma needs 2 to {d} slots", .{ nslot, max_slots });
        return error.BadSetting;
    }
    const slot_kib = try tcp_proxy.envU("TF_COLL_SLOT_KIB", default_slot_kib_n);
    const slot_bytes = std.mem.alignForward(u64, slot_kib << 10, 4096);
    const lanes: u32 = @intCast(@min(try tcp_proxy.envU("TF_COLL_LANES", default_lanes_n), 1 << 16));
    if (!Layout.lanesFit(nslot, lanes)) {
        std.log.err("collectives: TF_COLL_LANES={d} with TF_COLL_SLOTS={d}: lanes 1 to {d}, dividing the slots, 2 slots a lane at least", .{ lanes, nslot, rdma_core.max_lanes });
        return error.BadSetting;
    }
    const layout: Layout = .{ .world = id.world, .slot_bytes = slot_bytes, .nslot = nslot, .lanes = lanes };
    if (slot_bytes <= id.world * nic.trailer_bytes or layout.gatherChunk() == 0) return error.BadSetting;
    const proxy_cpu: i32 = @intCast(try tcp_proxy.envU("TF_COLL_PROXY_CPU", 6));
    if (try tcp_proxy.envU("TF_COLL_PROXY_STRIPE_KIB", 0) != 0) {
        std.log.err("collectives: TF_COLL_PROXY_STRIPE_KIB: kind=rdma moves every slot gather over one QP a peer", .{});
        return error.BadSetting;
    }
    const gid_index = nic.parseGidIndex(if (c.getenv("TF_RDMA_GID_INDEX")) |v| std.mem.span(v) else null) catch {
        std.log.err("collectives: TF_RDMA_GID_INDEX: a decimal GID index in 0..255", .{});
        return error.BadSetting;
    };
    var vb = try Verbs.open(null);
    errdefer vb.close();
    var name_buf: [64]u8 = undefined;
    const dev_name = try nic.deviceName(&vb, &name_buf, id.rank);
    const swap12 = if (c.getenv("TF_COLL_SWAP12")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
    const vgather = coll.vgatherFromEnv();
    var extra_buf: [128]u8 = undefined;
    // swap12 only when set, so the handshake text of a rank without it is the one it always was
    const extra = try std.fmt.bufPrint(&extra_buf, "kind=rdma gid={d} tc=0 slots={d} slot_kib={d} stripe=0 lanes={d}{s}", .{ gid_index, nslot, slot_kib, lanes, if (swap12) " swap12=1" else "" });
    const se = try tcp_coll.handshakeStar(fan, id, extra);
    const stage = try StageK.bind(kit, &.{});
    const collect = try CollectK.bind(kit, &.{});
    const caps = try ctx.caps();
    const rx_off: u64 = std.mem.alignForward(u64, @as(u64, lanes) * @sizeOf(Shared), 4096);
    var slab = try gpu.HostBuffer.allocFlags(d, rx_off + nslot * slot_bytes, .{ .mapped = true, .coherent = true });
    errdefer slab.free();
    @memset(slab.bytes[0..rx_off], 0);
    const t = try gpa.create(RdmaProxyN);
    errdefer gpa.destroy(t);
    t.* = .{
        .gpa = gpa,
        .d = d,
        .rank = id.rank,
        .world = id.world,
        .verbs = vb,
        .nics = undefined,
        .epoch = se.epoch,
        .slab = slab,
        .shared = @ptrCast(@alignCast(slab.bytes.ptr)),
        .dev = try slab.device(),
        .rx_off = rx_off,
        .layout = layout,
        .deadline_ticks = tcp_proxy.deadlineTicks(caps.wall_clock_hz, se.settings.gpu_wait_ns),
        .progress_ns = se.settings.opt.progress_ns,
        .warn_ns = se.settings.gpu_warn_ns,
        .stage = stage,
        .collect = collect,
        .proxy_cpu = proxy_cpu,
        .swap12 = swap12,
        .vgather = vgather,
    };
    if (swap12) std.log.info("collectives: rdma_proxy world {d} rank {d}: TF_COLL_SWAP12=1 (one-row quarters gathered straight into place)", .{ id.world, id.rank });
    if (vgather) std.log.info("collectives: rdma_proxy world {d} rank {d}: TF_DS_VGATHER=1 (multi-row equal-width quarters: two copy_rows16, the same all-gather)", .{ id.world, id.rank });
    t.nics = try nic.Nics.open(&t.verbs, dev_name, gid_index, id.rank, id.world, slab.bytes[rx_off..][0 .. nslot * slot_bytes]);
    errdefer t.nics.close();
    var mine: [rdma_core.max_world]nic.Info = std.mem.zeroes([rdma_core.max_world]nic.Info);
    for (0..id.world) |j| if (j != id.rank) {
        mine[j] = t.nics.info(@intCast(j), nslot, slot_bytes);
    };
    const theirs = try nic.exchangeInfoStar(fan, id.rank, id.world, mine);
    for (0..id.world) |j| if (j != id.rank) try t.nics.connect(@intCast(j), theirs[j]);
    try nic.startBarrierStar(fan, id.rank, id.world);
    if (id.rank == 0) {
        for (1..id.world) |g| if (fan.links[g - 1]) |l| {
            t.ctl_fds[g - 1] = l.fd;
        };
    } else if (fan.links[0]) |l| t.ctl_fds[0] = l.fd;
    t.thread = try std.Thread.spawn(.{}, RdmaProxyN.proxyLoop, .{t});
    std.log.info("collectives: rdma_proxy world {d} rank {d} on {s} gid={d} (mtu {d}, {d} QPs) with {d} slots of {d} KiB in {d} lanes (a gather block up to {d} B, an exchange side up to {d} B; larger ones in chunks, up to {d} in flight), proxy thread on CPU {d}, progress deadline {d} ms", .{
        id.world, id.rank, dev_name, gid_index, t.nics.active_mtu, id.world - 1, nslot, slot_bytes >> 10, lanes, layout.gatherChunk(), layout.exchangeHalf(), lanes, proxy_cpu, t.progress_ns / std.time.ns_per_ms,
    });
    return t.coll_();
}

test "every collective a HIP serve opens provides exchange" {
    // tcp_coll and tcp_proxy at world 2, this file's two-rank proxy (TP2, TF_COLL_KIND=rdma),
    // RdmaProxyN at world 4, and Comm (RCCL). World 2 has no fake-link harness: rdma_proxy_test
    // is a real HCA loopback, so the byte move is the host TcpColl this exchange delegates to.
    try std.testing.expect(tcp_coll.TcpColl.providesExchange());
    try std.testing.expect(tcp_proxy.TcpProxy.providesExchange());
    try std.testing.expect(RdmaProxy.providesExchange());
    try std.testing.expect(RdmaProxyN.providesExchange());
    try std.testing.expect(comm.Comm.providesExchange());
}

test {
    _ = @import("rdma_proxy_test.zig");
    _ = @import("rdma_proxy4_test.zig");
}
