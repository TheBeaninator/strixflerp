//! rdma_proxy's verbs side: one RC queue pair per rank over the slab's slots, the trailer, the peer Info exchange.
const std = @import("std");
const verbs = @import("verbs");
const Fan = @import("fan.zig").Fan;

const c = std.c;
pub const vabi = verbs.abi;
pub const Verbs = verbs.Verbs;

pub const Error = error{ VerbsFailed, PostFailed, BadMtu, BadInfo, BadSetting, NoRdmaDevice, NoControlLink } || verbs.Error;

const recv_depth = 256; // receive requests kept posted: a write with immediate consumes one, one is re-posted per completion
const send_depth = 64; // one write in flight per gather; room for the flushes of a close
const cq_depth = 512;
const recv_sge_bytes = 64; // mlx5: a small SGE keeps a posted receive to one ring slot
pub const wc_recv_rdma_with_imm = 129; // IBV_WC_RECV_RDMA_WITH_IMM
const wc_wr_flush_err = 5; // IBV_WC_WR_FLUSH_ERR: this rank's QP left RTS (abort, close)
const wc_rem_inv_req_err = 9; // IBV_WC_REM_INV_REQ_ERR: the peer's QP is gone or in error
const wc_rem_op_err = 11;
const wc_retry_exc_err = 12; // IBV_WC_RETRY_EXC_ERR: no acknowledgement within the QP's retries
const wc_rnr_retry_exc_err = 13;
const wc_rem_abort_err = 22;
pub const default_gid_index: u8 = 3;

pub const trailer_bytes = 64;
pub const trailer_magic: u64 = 0x3141_4d44_525f_4654; // "TF_RDMA1"

/// The Mismatch check the TCP header carried (tcp_wire.zig Header), written after the payload at roundup(bytes, 64).
pub const Trailer = extern struct {
    magic: u64,
    epoch: [16]u8,
    q: u64,
    bytes: u64,
    site: u32,
    rank: u32,
    pad: [16]u8 = @splat(0),

    comptime {
        std.debug.assert(@sizeOf(Trailer) == trailer_bytes);
    }
};

/// Where a gather's trailer sits in its slot.
pub fn trailerAt(bytes: u64) u64 {
    return std.mem.alignForward(u64, bytes, 64);
}
/// How many bytes one write moves: the payload rounded up, then the trailer.
pub fn wireLen(bytes: u64) u64 {
    return trailerAt(bytes) + trailer_bytes;
}
/// The largest gather a slot of `slot_bytes` carries over the QP (larger ones go to tcp_host).
pub fn capacity(slot_bytes: u64) u64 {
    return slot_bytes - trailer_bytes;
}

/// What the peer's QP toward this rank needs and where this rank's receive slots are; sent over the control link.
pub const Info = extern struct {
    magic: u32 = info_magic,
    qpn: u32,
    psn: u32,
    rkey: u32,
    gid: [16]u8,
    active_mtu: c_int,
    nslot: u32,
    rx_base: u64,
    rx_len: u64,
    slot_bytes: u64,
};
const info_magic: u32 = 0x3152_4651; // "QFR1"

/// TF_RDMA_QP_TIMEOUT (default 14): the RC local ACK timeout exponent, 4.096 us x 2^n per attempt, 8 attempts.
pub fn qpTimeout() error{BadSetting}!u8 {
    const v = c.getenv("TF_RDMA_QP_TIMEOUT") orelse return 14;
    const n = std.fmt.parseInt(u8, std.mem.span(v), 10) catch return error.BadSetting;
    return if (n > 31) error.BadSetting else n;
}

pub fn pathMtu(local: c_int, remote: c_int) error{BadMtu}!c_int {
    if (local < 1 or local > 5 or remote < 1 or remote > 5) return error.BadMtu;
    return @min(local, remote);
}

pub fn parseGidIndex(text: ?[]const u8) error{BadSetting}!u8 {
    const value = text orelse return default_gid_index;
    if (value.len == 0) return error.BadSetting;
    for (value) |ch| if (ch < '0' or ch > '9') return error.BadSetting;
    return std.fmt.parseInt(u8, value, 10) catch error.BadSetting;
}

/// The immediate a write carries: the gather's low 32 bits in network order (the CUDA ring's convention).
pub fn immediate(q: u64) u32 {
    return std.mem.nativeToBig(u32, @truncate(q));
}

/// A completion status as the proxy's failure code: 1 PeerLost for the statuses a gone or flushed peer gives, else 5.
pub fn statusCode(status: c_int) u32 {
    return switch (status) {
        wc_wr_flush_err, wc_rem_inv_req_err, wc_rem_op_err, wc_retry_exc_err, wc_rnr_retry_exc_err, wc_rem_abort_err => 1,
        else => 5,
    };
}

/// One RC queue pair on one device over a slab's TX and RX slots (rdma.zig's setup, one peer, two CQs).
pub const Nic = struct {
    /// the verbs entry points by value (the Verbs that resolved them outlives the Nic; this copy outlives any frame)
    api: verbs.Api,
    ctx: *vabi.Context,
    pd: *vabi.Pd,
    cq_send: *vabi.Cq,
    cq_recv: *vabi.Cq,
    mr_tx: *vabi.Mr,
    mr_rx: *vabi.Mr,
    qp: *vabi.Qp,
    gid: [16]u8,
    gid_index: u8,
    active_mtu: c_int,
    psn: u32,
    rx_base: u64,
    rx_len: u64,
    name: [64]u8 = @splat(0),

    /// Context, PD, two CQs, two MRs (TX local write; RX local and remote write), the QP in INIT, receives posted.
    pub fn open(v: *const Verbs, name: []const u8, gid_index: u8, tx: []u8, rx: []u8) Error!Nic {
        const api = v.api;
        const ctx = try v.openDevice(name);
        errdefer _ = api.ibv_close_device(ctx);
        const port = try v.port(ctx, 1);
        if (port.state != vabi.port_active) {
            std.log.err("collectives: rdma device {s} port 1 is not active (state {d})", .{ name, port.state });
            return error.VerbsFailed;
        }
        const active_mtu = try pathMtu(port.active_mtu, port.active_mtu);
        const pd = api.ibv_alloc_pd(ctx) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_dealloc_pd(pd);
        const cq_send = api.ibv_create_cq(ctx, cq_depth, null, null, 0) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_destroy_cq(cq_send);
        const cq_recv = api.ibv_create_cq(ctx, cq_depth, null, null, 0) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_destroy_cq(cq_recv);
        const mr_tx = api.ibv_reg_mr(pd, tx.ptr, tx.len, vabi.access_local_write) orelse {
            std.log.err("collectives: ibv_reg_mr over the {d}-byte TX slots failed (errno {d}; memlock?)", .{ tx.len, std.c._errno().* });
            return error.VerbsFailed;
        };
        errdefer _ = api.ibv_dereg_mr(mr_tx);
        const mr_rx = api.ibv_reg_mr(pd, rx.ptr, rx.len, vabi.access_local_write | vabi.access_remote_write) orelse {
            std.log.err("collectives: ibv_reg_mr over the {d}-byte RX slots failed (errno {d}; memlock?)", .{ rx.len, std.c._errno().* });
            return error.VerbsFailed;
        };
        errdefer _ = api.ibv_dereg_mr(mr_rx);
        const gid = try v.gid(ctx, 1, gid_index);
        var init: vabi.QpInitAttr = .{ .send_cq = cq_send, .recv_cq = cq_recv, .cap = .{ .max_send_wr = send_depth, .max_recv_wr = recv_depth, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 0 }, .qp_type = vabi.qpt_rc };
        const qp = api.ibv_create_qp(pd, &init) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_destroy_qp(qp);
        var a: vabi.QpAttr = .{ .qp_state = vabi.qps_init, .port_num = 1, .qp_access_flags = vabi.access_local_write | vabi.access_remote_write };
        const m = vabi.mask;
        if (api.ibv_modify_qp(qp, &a, m.state | m.pkey_index | m.port | m.access_flags) != 0) return error.VerbsFailed;
        var n: Nic = .{ .api = api, .ctx = ctx, .pd = pd, .cq_send = cq_send, .cq_recv = cq_recv, .mr_tx = mr_tx, .mr_rx = mr_rx, .qp = qp, .gid = gid.raw, .gid_index = gid_index, .active_mtu = active_mtu, .psn = (qp.qp_num *% 2654435761) & 0xFF_FFFF, .rx_base = @intFromPtr(rx.ptr), .rx_len = rx.len };
        @memcpy(n.name[0..@min(name.len, 63)], name[0..@min(name.len, 63)]);
        for (0..recv_depth) |_| try n.postRecv();
        return n;
    }

    pub fn close(n: *Nic) void {
        const api = n.api;
        _ = api.ibv_destroy_qp(n.qp);
        _ = api.ibv_dereg_mr(n.mr_rx);
        _ = api.ibv_dereg_mr(n.mr_tx);
        _ = api.ibv_destroy_cq(n.cq_recv);
        _ = api.ibv_destroy_cq(n.cq_send);
        _ = api.ibv_dealloc_pd(n.pd);
        _ = api.ibv_close_device(n.ctx);
    }

    pub fn info(n: *const Nic, nslot: u32, slot_bytes: u64) Info {
        return .{ .qpn = n.qp.qp_num, .psn = n.psn, .rkey = n.mr_rx.rkey, .gid = n.gid, .active_mtu = n.active_mtu, .nslot = nslot, .rx_base = n.rx_base, .rx_len = n.rx_len, .slot_bytes = slot_bytes };
    }

    /// INIT -> RTR -> RTS toward the peer's QP: the smaller path MTU, GRH traffic class 0, retry_cnt 7, rnr_retry 6.
    pub fn connect(n: *Nic, peer: Info) Error!void {
        const api = n.api;
        const m = vabi.mask;
        var a: vabi.QpAttr = .{ .qp_state = vabi.qps_rtr, .path_mtu = try pathMtu(n.active_mtu, peer.active_mtu), .dest_qp_num = peer.qpn, .rq_psn = peer.psn, .max_dest_rd_atomic = 1, .min_rnr_timer = 12 };
        a.ah_attr = .{ .grh = .{ .dgid = .{ .raw = peer.gid }, .sgid_index = n.gid_index, .hop_limit = 64, .traffic_class = 0 }, .is_global = 1, .port_num = 1 };
        if (api.ibv_modify_qp(n.qp, &a, m.state | m.av | m.path_mtu | m.dest_qpn | m.rq_psn | m.max_dest_rd_atomic | m.min_rnr_timer) != 0) return error.VerbsFailed;
        a = .{ .qp_state = vabi.qps_rts, .timeout = try qpTimeout(), .retry_cnt = 7, .rnr_retry = 6, .sq_psn = n.psn, .max_rd_atomic = 1 };
        if (api.ibv_modify_qp(n.qp, &a, m.state | m.timeout | m.retry_cnt | m.rnr_retry | m.sq_psn | m.max_qp_rd_atomic) != 0) return error.VerbsFailed;
    }

    /// Every posted work request completes with a flush error and nothing new is accepted (abort, close).
    pub fn toErr(n: *Nic) void {
        var a: vabi.QpAttr = .{ .qp_state = vabi.qps_err };
        _ = n.api.ibv_modify_qp(n.qp, &a, vabi.mask.state);
    }

    pub fn statusText(n: *const Nic, status: c_int) []const u8 {
        return std.mem.span(n.api.ibv_wc_status_str(status));
    }

    pub fn postRecv(n: *Nic) Error!void {
        var sge: vabi.Sge = .{ .addr = n.rx_base, .length = recv_sge_bytes, .lkey = n.mr_rx.lkey };
        var wr: vabi.RecvWr = .{ .sg_list = @ptrCast(&sge), .num_sge = 1 };
        var bad: ?*vabi.RecvWr = null;
        if (n.ctx.ops.post_recv(n.qp, &wr, &bad) != 0) return error.PostFailed;
    }

    /// One signaled write with immediate of `local` (in the TX MR) to `remote` under `rkey`; wr_id and imm carry q.
    pub fn postWrite(n: *Nic, local: []const u8, remote: u64, rkey: u32, q: u64) Error!void {
        var sge: vabi.Sge = .{ .addr = @intFromPtr(local.ptr), .length = @intCast(local.len), .lkey = n.mr_tx.lkey };
        var wr: vabi.SendWr = .{ .wr_id = q, .sg_list = @ptrCast(&sge), .num_sge = 1, .opcode = vabi.wr_rdma_write_imm, .send_flags = vabi.send_signaled, .imm_data = immediate(q), .remote_addr = remote, .rkey = rkey };
        var bad: ?*vabi.SendWr = null;
        if (n.ctx.ops.post_send(n.qp, &wr, &bad) != 0) return error.PostFailed;
    }

    pub fn poll(n: *Nic, cq: *vabi.Cq, wc: []vabi.Wc) c_int {
        return n.ctx.ops.poll_cq(cq, @intCast(wc.len), wc.ptr);
    }
};

extern "c" fn access(path: [*:0]const u8, mode: c_int) c_int;

/// This rank's device: TF_RDMA_DEVICES (one name, or one per rank), else TF_RDMA_IFNAME's ibdev, else the only one.
pub fn deviceName(v: *const Verbs, buf: *[64]u8, rank: u32) Error![]const u8 {
    if (c.getenv("TF_RDMA_DEVICES")) |val| {
        const text = std.mem.span(val);
        var it = std.mem.tokenizeAny(u8, text, ",+ ");
        var first: ?[]const u8 = null;
        var pick: ?[]const u8 = null;
        var n: usize = 0;
        while (it.next()) |p| : (n += 1) {
            if (n == 0) first = p;
            if (n == rank) pick = p;
        }
        const name = (if (n == 1) first else pick) orelse {
            std.log.err("collectives: TF_RDMA_DEVICES={s} names no device for rank {d}", .{ text, rank });
            return error.BadSetting;
        };
        if (name.len >= buf.len) return error.BadSetting;
        @memcpy(buf[0..name.len], name);
        return buf[0..name.len];
    }
    var names: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
    defer names.deinit();
    _ = v.deviceNames(&names.writer) catch return error.NoRdmaDevice;
    const ifname: ?[]const u8 = if (c.getenv("TF_RDMA_IFNAME")) |val| std.mem.span(val) else null;
    var it = std.mem.tokenizeScalar(u8, names.written(), '\n');
    var only: ?[]const u8 = null;
    var count: usize = 0;
    while (it.next()) |name| {
        count += 1;
        only = name;
        if (ifname) |ifn| {
            var path: [256]u8 = undefined;
            const p = std.fmt.bufPrint(&path, "/sys/class/infiniband/{s}/device/net/{s}\x00", .{ name, ifn }) catch continue;
            if (access(@ptrCast(p.ptr), 0) == 0) {
                if (name.len >= buf.len) return error.BadSetting;
                @memcpy(buf[0..name.len], name);
                return buf[0..name.len];
            }
        }
    }
    if (ifname) |ifn| {
        std.log.err("collectives: no rdma device carries interface {s} (TF_RDMA_IFNAME) among: {s}", .{ ifn, names.written() });
        return error.NoRdmaDevice;
    }
    if (count == 1) {
        const name = only.?;
        if (name.len >= buf.len) return error.BadSetting;
        @memcpy(buf[0..name.len], name);
        return buf[0..name.len];
    }
    std.log.err("collectives: {d} rdma devices and neither TF_RDMA_DEVICES nor TF_RDMA_IFNAME set: {s}", .{ count, names.written() });
    return error.NoRdmaDevice;
}

/// The two ranks' Infos over the control link (both send first), checked for magic, size, slot count, size and length.
pub fn exchangeInfo(fan: *const Fan, mine: Info) !Info {
    const link = fan.links[0] orelse return error.NoControlLink;
    try link.send(std.mem.asBytes(&mine));
    var buf: [256]u8 = undefined;
    const got = try link.recv(&buf);
    if (got.len != @sizeOf(Info)) return error.BadInfo;
    var theirs: Info = undefined;
    @memcpy(std.mem.asBytes(&theirs), got);
    if (theirs.magic != info_magic) return error.BadInfo;
    if (theirs.nslot != mine.nslot or theirs.slot_bytes != mine.slot_bytes or theirs.rx_len != mine.rx_len) {
        std.log.err("collectives handshake refused: this rank has {d} slots of {d} bytes ({d}-byte receive region), the peer {d} of {d} ({d})", .{ mine.nslot, mine.slot_bytes, mine.rx_len, theirs.nslot, theirs.slot_bytes, theirs.rx_len });
        return error.Mismatch;
    }
    _ = try pathMtu(mine.active_mtu, theirs.active_mtu);
    return theirs;
}

/// A second small frame each once both QPs are in RTS: neither rank writes before the peer can receive.
pub fn startBarrier(fan: *const Fan) !void {
    const link = fan.links[0] orelse return error.NoControlLink;
    try link.send("rts");
    var buf: [16]u8 = undefined;
    const got = try link.recv(&buf);
    if (!std.mem.eql(u8, got, "rts")) return error.BadInfo;
}


// ---- world 4 (TP4-RDMA-N): one QP per peer over one receive region ----------------------------------------------------

/// A completion as rdma_core.CoreN reads it from its Net (this file's Nics, or the tests' fake fabric).
pub const Cqe = struct {
    status: c_int = 0,
    wr_id: u64 = 0,
    peer: u32 = 0,
    /// the immediate, host order (a trailer's: q's low 32 bits; a poison's: the failure code)
    imm: u32 = 0,
    byte_len: u32 = 0,
    with_imm: bool = false,
};
/// One local range written at remote_off of the peer's receive region.
pub const Seg = struct { local: []const u8, remote_off: u64 };

/// One device context, PD, a send CQ and a receive CQ shared by every QP, one MR over the receive region (the slots:
/// the peers write into it and this rank writes its own blocks out of it), and one RC QP toward each other rank.
/// It is rdma_core.CoreN's Net: post, postPoison, pollSend, pollRecv, statusText.
pub const Nics = struct {
    pub const max_world = 4;
    api: verbs.Api,
    ctx: *vabi.Context,
    pd: *vabi.Pd,
    cq_send: *vabi.Cq,
    cq_recv: *vabi.Cq,
    mr: *vabi.Mr,
    qps: [max_world]?*vabi.Qp = @splat(null),
    /// the peers' receive regions (from their Infos), by rank
    remote_base: [max_world]u64 = @splat(0),
    remote_rkey: [max_world]u32 = @splat(0),
    rank: u32,
    world: u32,
    gid: [16]u8,
    gid_index: u8,
    active_mtu: c_int,
    psn: u32,
    rx_base: u64,
    rx_len: u64,

    /// Everything but the connections: QPs in INIT with their receives posted.
    pub fn open(v: *const Verbs, name: []const u8, gid_index: u8, rank: u32, world: u32, rx: []u8) Error!Nics {
        if (world < 3 or world > max_world or rank >= world) return error.BadSetting;
        const api = v.api;
        const ctx = try v.openDevice(name);
        errdefer _ = api.ibv_close_device(ctx);
        const port = try v.port(ctx, 1);
        if (port.state != vabi.port_active) {
            std.log.err("collectives: rdma device {s} port 1 is not active (state {d})", .{ name, port.state });
            return error.VerbsFailed;
        }
        const active_mtu = try pathMtu(port.active_mtu, port.active_mtu);
        const pd = api.ibv_alloc_pd(ctx) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_dealloc_pd(pd);
        // every posted receive of every QP flushes into the receive CQ at close
        const cq_send = api.ibv_create_cq(ctx, cq_depth, null, null, 0) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_destroy_cq(cq_send);
        const cq_recv = api.ibv_create_cq(ctx, @intCast((max_world - 1) * recv_depth + 64), null, null, 0) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_destroy_cq(cq_recv);
        const mr = api.ibv_reg_mr(pd, rx.ptr, rx.len, vabi.access_local_write | vabi.access_remote_write) orelse {
            std.log.err("collectives: ibv_reg_mr over the {d}-byte slots failed (errno {d}; memlock?)", .{ rx.len, std.c._errno().* });
            return error.VerbsFailed;
        };
        errdefer _ = api.ibv_dereg_mr(mr);
        const gid = try v.gid(ctx, 1, gid_index);
        var n: Nics = .{ .api = api, .ctx = ctx, .pd = pd, .cq_send = cq_send, .cq_recv = cq_recv, .mr = mr, .rank = rank, .world = world, .gid = gid.raw, .gid_index = gid_index, .active_mtu = active_mtu, .psn = 0, .rx_base = @intFromPtr(rx.ptr), .rx_len = rx.len };
        var seed: [4]u8 = undefined;
        if (std.os.linux.getrandom(&seed, seed.len, 0) != seed.len) return error.VerbsFailed;
        n.psn = std.mem.readInt(u32, &seed, .little) & 0xffffff;
        errdefer n.destroyQps();
        for (0..world) |j| {
            if (j == rank) continue;
            // two send WRs a gather (data, then the trailer with immediate), one more for a poison
            var init: vabi.QpInitAttr = .{ .send_cq = cq_send, .recv_cq = cq_recv, .cap = .{ .max_send_wr = send_depth, .max_recv_wr = recv_depth, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 0 }, .qp_type = vabi.qpt_rc };
            const qp = api.ibv_create_qp(pd, &init) orelse return error.VerbsFailed;
            n.qps[j] = qp;
            var a: vabi.QpAttr = .{ .qp_state = vabi.qps_init, .port_num = 1, .qp_access_flags = vabi.access_local_write | vabi.access_remote_write };
            const m = vabi.mask;
            if (api.ibv_modify_qp(qp, &a, m.state | m.pkey_index | m.port | m.access_flags) != 0) return error.VerbsFailed;
            for (0..recv_depth) |_| try n.postRecvOn(@intCast(j));
        }
        return n;
    }

    fn destroyQps(n: *Nics) void {
        for (&n.qps) |*q| if (q.*) |qp| {
            _ = n.api.ibv_destroy_qp(qp);
            q.* = null;
        };
    }

    pub fn close(n: *Nics) void {
        const api = n.api;
        n.destroyQps();
        _ = api.ibv_dereg_mr(n.mr);
        _ = api.ibv_destroy_cq(n.cq_recv);
        _ = api.ibv_destroy_cq(n.cq_send);
        _ = api.ibv_dealloc_pd(n.pd);
        _ = api.ibv_close_device(n.ctx);
    }

    /// This rank's QP toward rank j, as rank j's QP toward this rank needs it.
    pub fn info(n: *const Nics, j: u32, nslot: u32, slot_bytes: u64) Info {
        const qp = n.qps[j] orelse return std.mem.zeroes(Info);
        return .{ .qpn = qp.qp_num, .psn = n.psn, .rkey = n.mr.rkey, .gid = n.gid, .active_mtu = n.active_mtu, .nslot = nslot, .rx_base = n.rx_base, .rx_len = n.rx_len, .slot_bytes = slot_bytes };
    }

    /// INIT -> RTR -> RTS of the QP toward rank j, as Nic.connect does it for the one peer.
    pub fn connect(n: *Nics, j: u32, peer: Info) Error!void {
        const qp = n.qps[j] orelse return error.BadSetting;
        const api = n.api;
        const m = vabi.mask;
        var a: vabi.QpAttr = .{ .qp_state = vabi.qps_rtr, .path_mtu = try pathMtu(n.active_mtu, peer.active_mtu), .dest_qp_num = peer.qpn, .rq_psn = peer.psn, .max_dest_rd_atomic = 1, .min_rnr_timer = 12 };
        a.ah_attr = .{ .grh = .{ .dgid = .{ .raw = peer.gid }, .sgid_index = n.gid_index, .hop_limit = 64, .traffic_class = 0 }, .is_global = 1, .port_num = 1 };
        if (api.ibv_modify_qp(qp, &a, m.state | m.av | m.path_mtu | m.dest_qpn | m.rq_psn | m.max_dest_rd_atomic | m.min_rnr_timer) != 0) return error.VerbsFailed;
        a = .{ .qp_state = vabi.qps_rts, .timeout = try qpTimeout(), .retry_cnt = 7, .rnr_retry = 6, .sq_psn = n.psn, .max_rd_atomic = 1 };
        if (api.ibv_modify_qp(qp, &a, m.state | m.timeout | m.retry_cnt | m.rnr_retry | m.sq_psn | m.max_qp_rd_atomic) != 0) return error.VerbsFailed;
        n.remote_base[j] = peer.rx_base;
        n.remote_rkey[j] = peer.rkey;
    }

    pub fn toErr(n: *Nics) void {
        for (n.qps) |q| if (q) |qp| {
            var a: vabi.QpAttr = .{ .qp_state = vabi.qps_err };
            _ = n.api.ibv_modify_qp(qp, &a, vabi.mask.state);
        };
    }

    pub fn statusText(n: *const Nics, status: c_int) []const u8 {
        return std.mem.span(n.api.ibv_wc_status_str(status));
    }

    fn postRecvOn(n: *Nics, j: u32) Error!void {
        const qp = n.qps[j] orelse return error.PostFailed;
        var sge: vabi.Sge = .{ .addr = n.rx_base, .length = recv_sge_bytes, .lkey = n.mr.lkey };
        var wr: vabi.RecvWr = .{ .wr_id = j, .sg_list = @ptrCast(&sge), .num_sge = 1 };
        var bad: ?*vabi.RecvWr = null;
        if (n.ctx.ops.post_recv(qp, &wr, &bad) != 0) return error.PostFailed;
    }

    /// The data (an unsignaled write, when there is one) and the trailer (a signaled write with immediate) to rank j,
    /// in one post: RC delivers them in order, so the trailer's completion at the peer means both are placed.
    pub fn post(n: *Nics, j: u32, data: ?Seg, trailer: Seg, wr_id: u64, imm: u32) Error!void {
        const qp = n.qps[j] orelse return error.PostFailed;
        var sge_t: vabi.Sge = .{ .addr = @intFromPtr(trailer.local.ptr), .length = @intCast(trailer.local.len), .lkey = n.mr.lkey };
        var wr_t: vabi.SendWr = .{ .wr_id = wr_id, .sg_list = @ptrCast(&sge_t), .num_sge = 1, .opcode = vabi.wr_rdma_write_imm, .send_flags = vabi.send_signaled, .imm_data = std.mem.nativeToBig(u32, imm), .remote_addr = n.remote_base[j] + trailer.remote_off, .rkey = n.remote_rkey[j] };
        var sge_d: vabi.Sge = undefined;
        var wr_d: vabi.SendWr = undefined;
        var first: *vabi.SendWr = &wr_t;
        if (data) |d| {
            sge_d = .{ .addr = @intFromPtr(d.local.ptr), .length = @intCast(d.local.len), .lkey = n.mr.lkey };
            wr_d = .{ .wr_id = wr_id, .next = &wr_t, .sg_list = @ptrCast(&sge_d), .num_sge = 1, .opcode = vabi.wr_rdma_write, .send_flags = 0, .remote_addr = n.remote_base[j] + d.remote_off, .rkey = n.remote_rkey[j] };
            first = &wr_d;
        }
        var bad: ?*vabi.SendWr = null;
        if (n.ctx.ops.post_send(qp, first, &bad) != 0) return error.PostFailed;
    }

    /// A zero-byte signaled write with immediate (the failure code) to rank j: its collective has failed.
    pub fn postPoison(n: *Nics, j: u32, code: u32, wr_id: u64) Error!void {
        const qp = n.qps[j] orelse return error.PostFailed;
        var wr: vabi.SendWr = .{ .wr_id = wr_id, .sg_list = undefined, .num_sge = 0, .opcode = vabi.wr_rdma_write_imm, .send_flags = vabi.send_signaled, .imm_data = std.mem.nativeToBig(u32, code), .remote_addr = n.remote_base[j], .rkey = n.remote_rkey[j] };
        var bad: ?*vabi.SendWr = null;
        if (n.ctx.ops.post_send(qp, &wr, &bad) != 0) return error.PostFailed;
    }

    fn peerOf(n: *const Nics, qp_num: u32) u32 {
        for (n.qps, 0..) |q, j| if (q) |qp| if (qp.qp_num == qp_num) return @intCast(j);
        return max_world; // not one of ours: CoreN fails the gather with Mismatch
    }

    pub fn pollSend(n: *Nics, out: []Cqe) isize {
        var wc: [8]vabi.Wc = undefined;
        const got = n.ctx.ops.poll_cq(n.cq_send, @intCast(@min(out.len, wc.len)), &wc);
        if (got < 0) return -1;
        for (wc[0..@intCast(got)], out[0..@intCast(got)]) |w, *o| o.* = .{ .status = w.status, .wr_id = w.wr_id, .peer = @intCast(w.wr_id & 7) };
        return got;
    }

    /// Receive completions, each receive re-posted on its QP (a failed re-post: -1).
    pub fn pollRecv(n: *Nics, out: []Cqe) isize {
        var wc: [8]vabi.Wc = undefined;
        const got = n.ctx.ops.poll_cq(n.cq_recv, @intCast(@min(out.len, wc.len)), &wc);
        if (got < 0) return -1;
        for (wc[0..@intCast(got)], out[0..@intCast(got)]) |w, *o| {
            const j = n.peerOf(w.qp_num);
            o.* = .{ .status = w.status, .wr_id = w.wr_id, .peer = j, .imm = std.mem.bigToNative(u32, w.imm_data), .byte_len = w.byte_len, .with_imm = w.opcode == wc_recv_rdma_with_imm and (w.wc_flags & vabi.wc_with_imm) != 0 };
            if (w.status == vabi.wc_success and j < max_world) n.postRecvOn(j) catch return -1;
        }
        return got;
    }
};


/// Every rank's Infos over the star of control links (rank 0 links to each follower): each follower sends rank 0 its
/// QPs' Infos toward every rank; rank 0 checks they agree on the slots and sends each follower every rank's Info
/// toward it. Returns the Infos toward this rank, by rank (this rank's own entry zero).
pub fn exchangeInfoStar(fan: *const Fan, rank: u32, world: u32, mine: [Nics.max_world]Info) ![Nics.max_world]Info {
    const W = Nics.max_world;
    if (world < 3 or world > W) return error.BadSetting;
    const row_bytes = W * @sizeOf(Info);
    if (rank != 0) {
        const link = fan.links[0] orelse return error.NoControlLink;
        try link.send(std.mem.asBytes(&mine));
        var buf: [row_bytes + 16]u8 = undefined;
        const got = try link.recv(&buf);
        if (got.len == 0) return error.Mismatch; // rank 0 refused: the slots differ (logged there)
        if (got.len != row_bytes) return error.BadInfo;
        var col: [W]Info = undefined;
        @memcpy(std.mem.asBytes(&col), got);
        for (col[0..world], 0..) |inf, j| if (j != rank and inf.magic != info_magic) return error.BadInfo;
        return col;
    }
    var table: [W][W]Info = undefined; // table[i][j]: rank i's QP toward rank j
    table[0] = mine;
    for (1..world) |g| {
        const link = fan.links[g - 1] orelse return error.NoControlLink;
        var buf: [row_bytes + 16]u8 = undefined;
        const got = try link.recv(&buf);
        if (got.len != row_bytes) return error.BadInfo;
        @memcpy(std.mem.asBytes(&table[g]), got);
    }
    var agree = true;
    for (0..world) |i| for (0..world) |j| {
        if (i == j) continue;
        const t = table[i][j];
        if (t.magic != info_magic) return error.BadInfo;
        if (t.nslot != mine[1].nslot or t.slot_bytes != mine[1].slot_bytes or t.rx_len != mine[1].rx_len) {
            std.log.warn("collectives handshake refused: rank {d} has {d} slots of {d} bytes ({d}-byte receive region), rank 0 {d} of {d} ({d})", .{ i, t.nslot, t.slot_bytes, t.rx_len, mine[1].nslot, mine[1].slot_bytes, mine[1].rx_len });
            agree = false;
        }
        _ = pathMtu(t.active_mtu, mine[1].active_mtu) catch {
            agree = false;
        };
    };
    for (1..world) |g| {
        const link = fan.links[g - 1] orelse return error.NoControlLink;
        if (!agree) {
            link.send("") catch {};
            continue;
        }
        var col: [W]Info = std.mem.zeroes([W]Info);
        for (0..world) |i| col[i] = if (i == g) std.mem.zeroes(Info) else table[i][g];
        try link.send(std.mem.asBytes(&col));
    }
    if (!agree) return error.Mismatch;
    var col: [W]Info = std.mem.zeroes([W]Info);
    for (1..world) |i| col[i] = table[i][0];
    return col;
}

/// Every QP in RTS before any rank writes: each follower tells rank 0, rank 0 answers every follower once all did.
pub fn startBarrierStar(fan: *const Fan, rank: u32, world: u32) !void {
    var buf: [16]u8 = undefined;
    if (rank != 0) {
        const link = fan.links[0] orelse return error.NoControlLink;
        try link.send("rts");
        const got = try link.recv(&buf);
        if (!std.mem.eql(u8, got, "go")) return error.BadInfo;
        return;
    }
    for (1..world) |g| {
        const link = fan.links[g - 1] orelse return error.NoControlLink;
        const got = try link.recv(&buf);
        if (!std.mem.eql(u8, got, "rts")) return error.BadInfo;
    }
    for (1..world) |g| try (fan.links[g - 1] orelse return error.NoControlLink).send("go");
}
