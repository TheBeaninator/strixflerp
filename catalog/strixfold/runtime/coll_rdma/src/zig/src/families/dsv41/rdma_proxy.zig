//! rdma_proxy: tcp_proxy's slot collective with the TCP exchange replaced by one RC RDMA write-with-immediate per
//! staged slot (/data/pod/reports/2026-11/strixfold-rdma-design.md v1.2 section 3; card SFR-2). Same
//! hipHostMalloc(Mapped|Coherent) slab and tf_ds_coll_shared header, same coll.stage / coll.collect kernels, same
//! ready / done / consumed generation protocol, abort word and fault contract (coll.zig); graph capture is unchanged
//! because the proxy never sees a host call. Gathers larger than a slot keep tcp_host (phase 1: prefill stays on TCP).
//!
//! Per gather q in slot s = q mod NSLOT, the proxy thread (pinned to TF_COLL_PROXY_CPU, busy-polling, never a
//! completion channel): sees ready[s] == q; writes a 64-byte trailer (magic, session epoch, q, bytes, site, rank) into
//! TX[s] at roundup(bytes, 64) (R2: one SGE, so the trailer follows the payload contiguously); posts one signaled
//! RDMA_WRITE_WITH_IMM of roundup(bytes, 64) + 64 bytes from TX[s] into the peer's RX[s] with imm = low 32 bits of q;
//! polls its send CQ and receive CQ until both the local send completion and the peer's write of q are in; checks the
//! receive completion's byte_len and immediate and the trailer the peer wrote against its own slot words (any
//! difference: Mismatch); then stores done[s] = q. TX ownership is tcp_proxy's (consumed gates the stage kernel, done
//! waits for the send completion); RX ownership is transitive and holds for NSLOT >= 2 (R1), so NSLOT < 2 and the
//! stripe thread (TF_COLL_PROXY_STRIPE_KIB > 0) are refused. Two MRs cover the TX slots and the RX slots only, never
//! the slab header (R3); the handshake compares slot count, slot size, GID index and the peer's advertised RX length.
//!
//! mlx5 notes carried from SFR-1's probe: receives are posted with a 64-byte SGE (a write with immediate lands at
//! remote_addr, the SGE only reserves a ring slot; a large SGE costs several striding-RQ strides and overruns the
//! ring), the send and receive CQs are separate, and every send is signaled (mlx5 reaps unsignaled SQ WQEs only when
//! a signaled completion is polled).
//!
//! Failure: a completion with an error status poisons the collective (RETRY_EXC / RNR_RETRY_EXC / FLUSH: PeerLost,
//! others CollectiveFailed); no completion for TF_COLL_PROGRESS_MS: Timeout (the proxy's deadline owns Timeout, the
//! NIC's retry attributes fail an unanswered write after ~0.5 s, inside it); the peer's tcp_host socket closing:
//! PeerLost within milliseconds. abort and close move the QP to ERR so posted work flushes and the loop leaves.
//! Settings (environment): tcp_proxy's TF_COLL_SLOTS (8), TF_COLL_SLOT_KIB (4352), TF_COLL_PROXY_CPU (6);
//! TF_RDMA_DEVICES (one ibdev name for both ranks, or one per rank separated by ',' or '+': F is rocep195s0, A is
//! rocep194s0; unset: the ibdev whose device/net holds TF_RDMA_IFNAME, or the box's only ibdev);
//! TF_RDMA_GID_INDEX (3: RoCE v2 of the rail address); plus tcp_coll.zig's (TF_COLL_PROGRESS_MS is the deadline).
const std = @import("std");
const gpu = @import("gpu");
const verbs = @import("verbs");
const coll = @import("coll.zig");
const Fan = @import("fan.zig").Fan;
const wire = @import("tcp_wire.zig");
const tcp_coll = @import("tcp_coll.zig");
const tcp_proxy = @import("tcp_proxy.zig");

const c = std.c;
const vabi = verbs.abi;
const Verbs = verbs.Verbs;
pub const Shared = tcp_proxy.Shared;
pub const max_slots = tcp_proxy.max_slots;
const StageK = gpu.Kernel("coll.stage", tcp_proxy.params.Stage);
const CollectK = gpu.Kernel("coll.collect", tcp_proxy.params.Collect);

pub const Error = error{ VerbsFailed, PostFailed, BadMtu, BadInfo, BadSetting, NoRdmaDevice, NoControlLink } || verbs.Error;

const recv_depth = 256; // receive requests kept posted (a write with immediate consumes one; one is re-posted per completion)
const send_depth = 64; // one write in flight per gather; room for the flushes of a close
const cq_depth = 512;
const recv_sge_bytes = 64; // mlx5: a small SGE keeps a posted receive to one ring slot
const wc_recv_rdma_with_imm = 129; // IBV_WC_RECV_RDMA_WITH_IMM
const wc_wr_flush_err = 5; // IBV_WC_WR_FLUSH_ERR: this rank's QP left RTS (abort, close)
const wc_rem_inv_req_err = 9; // IBV_WC_REM_INV_REQ_ERR: the peer's QP is gone or in error
const wc_rem_op_err = 11;
const wc_retry_exc_err = 12; // IBV_WC_RETRY_EXC_ERR: no acknowledgement for ~0.5 s (timeout 14, retry_cnt 7)
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

/// Where a gather's trailer sits in its slot, and how many bytes one write moves.
pub fn trailerAt(bytes: u64) u64 {
    return std.mem.alignForward(u64, bytes, 64);
}
pub fn wireLen(bytes: u64) u64 {
    return trailerAt(bytes) + trailer_bytes;
}
/// The largest gather a slot of `slot_bytes` carries over the QP (larger ones go to tcp_host).
pub fn capacity(slot_bytes: u64) u64 {
    return slot_bytes - trailer_bytes;
}

/// What the peer's QP toward this rank needs and where this rank's receive slots are; crosses the control link as
/// bytes right after the handshake, so both sides check its magic and size.
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

/// TF_RDMA_QP_TIMEOUT (default 14): the RC local ACK timeout exponent, 4.096 us x 2^n per attempt, 8 attempts
/// (retry_cnt 7). Measured on CX-6 Dx fw 22.43.4100: 14 fails an unanswered write after ~3.7 s, not the nominal 0.5 s.
pub fn qpTimeout() error{BadSetting}!u8 {
    const v = c.getenv("TF_RDMA_QP_TIMEOUT") orelse return 14;
    const n = std.fmt.parseInt(u8, std.mem.span(v), 10) catch return error.BadSetting;
    return if (n > 31) error.BadSetting else n;
}

fn pathMtu(local: c_int, remote: c_int) error{BadMtu}!c_int {
    if (local < 1 or local > 5 or remote < 1 or remote > 5) return error.BadMtu;
    return @min(local, remote);
}

pub fn parseGidIndex(text: ?[]const u8) error{BadSetting}!u8 {
    const value = text orelse return default_gid_index;
    if (value.len == 0) return error.BadSetting;
    for (value) |ch| if (ch < '0' or ch > '9') return error.BadSetting;
    return std.fmt.parseInt(u8, value, 10) catch error.BadSetting;
}

/// One RC queue pair on one device over a slab's TX and RX slots (rdma.zig:224-323's setup, one peer, two CQs).
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

    /// Context, PD, the two CQs, the two MRs (TX: local write; RX: local + remote write), the QP in INIT with every
    /// receive posted.
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

    /// INIT -> RTR -> RTS toward the peer's QP: path MTU the smaller active one, GRH with traffic class 0 (the
    /// lossless priority on this fabric, R5) and hop limit 64, min_rnr_timer 12; timeout 14 and retry_cnt 7 (an
    /// unanswered write fails after ~0.5 s, R4), rnr_retry 6 (bounded: a peer with no posted receive fails at the NIC).
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

    fn poll(n: *Nic, cq: *vabi.Cq, wc: []vabi.Wc) c_int {
        return n.ctx.ops.poll_cq(cq, @intCast(wc.len), wc.ptr);
    }
};

/// The immediate a write carries: the gather's low 32 bits in network order (the CUDA ring's convention).
pub fn immediate(q: u64) u32 {
    return std.mem.nativeToBig(u32, @truncate(q));
}

/// The proxy's loop on a slab, GPU-free (the host test drives it with threads standing in for the kernels): wait for
/// slot q's ready, write its transmit half with the trailer into the peer's receive half, wait for both completions,
/// check the peer's write, publish done, next q. Returns the failure code (0: stopped cleanly; 1 PeerLost, 2 Timeout,
/// 3 Mismatch, 5 the verbs failed) after setting the abort word on any failure.
pub const Core = struct {
    shared: *Shared,
    tx: []u8, // nslot * slot_bytes
    rx: []u8,
    nslot: u32,
    slot_bytes: u64,
    nic: *Nic,
    peer_rx: u64,
    peer_rkey: u32,
    epoch: [16]u8,
    stop: *std.atomic.Value(u32),
    exchanged: *std.atomic.Value(u64),
    /// no completion for this long after a write was posted: Timeout
    progress_ns: u64 = 2 * std.time.ns_per_s,
    /// a connected socket to the peer to peek at while waiting (-1: none): its end of stream is PeerLost at once
    peer_fd: c.fd_t = -1,
    /// for the log lines: this rank, and an exchange longer than this is logged (0: never)
    rank: u32 = 0,
    warn_ns: u64 = 0,
    /// diagnostics: completions seen
    sends: u64 = 0,
    recvs: u64 = 0,

    pub fn run(k: *Core) u32 {
        var q: u64 = 1;
        var idle_since: u64 = 0;
        // one completion per poll: the peer may already have written gather q + 1 into its slot while this rank is
        // still waiting for q (it needs only this rank's write of q, posted before the wait), and that completion
        // belongs to the next turn of the loop
        var wc: [1]vabi.Wc = undefined;
        while (k.stop.load(.acquire) == 0) {
            const slot: usize = @intCast(q % k.nslot);
            const r = @atomicLoad(u64, &k.shared.slot[slot].ready, .acquire);
            if (r != q) {
                if (r > q) { // a generation ahead of the one expected: stale or duplicated
                    std.log.warn("collectives: rank {d} proxy expected gather {d} in slot {d} and found gather {d}", .{ k.rank, q, slot, r });
                    return k.fail(3);
                }
                if (@atomicLoad(u32, &k.shared.abort, .acquire) != 0) {
                    tcp_proxy.deviceGaveUp(k.shared, k.rank, q, k.nslot, "waiting for its GPU to stage gather");
                    return k.fail(2);
                }
                if (idle_since == 0) idle_since = wire.monotonicNs();
                if (wire.monotonicNs() - idle_since > 20 * std.time.ns_per_ms) std.Thread.yield() catch {} else std.atomic.spinLoopHint();
                continue;
            }
            idle_since = 0;
            const bytes: u64 = k.shared.slot[slot].bytes;
            const site: u32 = @truncate(k.shared.slot[slot].site);
            if (wireLen(bytes) > k.slot_bytes) {
                std.log.warn("collectives: rank {d} proxy: gather {d} says {d} bytes, more than a slot carries ({d})", .{ k.rank, q, bytes, capacity(k.slot_bytes) });
                return k.fail(3);
            }
            const base: usize = slot * @as(usize, @intCast(k.slot_bytes));
            const at: usize = @intCast(trailerAt(bytes));
            const tr: *Trailer = @ptrCast(@alignCast(k.tx[base + at ..].ptr));
            tr.* = .{ .magic = trailer_magic, .epoch = k.epoch, .q = q, .bytes = bytes, .site = site, .rank = k.rank };
            const len: usize = @intCast(wireLen(bytes));
            const t0 = wire.monotonicNs();
            k.nic.postWrite(k.tx[base..][0..len], k.peer_rx + base, k.peer_rkey, q) catch {
                k.logExchange(q, site, bytes, t0, "ibv_post_send failed");
                return k.fail(5);
            };
            // both completions: the local send (TX[s] may be re-staged) and the peer's write of q (RX[s] is complete)
            var have_send = false;
            var have_recv = false;
            var last = t0;
            var spins: u32 = 0;
            while (!(have_send and have_recv)) {
                if (!have_send) {
                    const n = k.nic.poll(k.nic.cq_send, &wc);
                    if (n < 0) return k.fail(5);
                    for (wc[0..@intCast(n)]) |w| {
                        if (w.status != vabi.wc_success) {
                            k.logExchange(q, site, bytes, t0, k.nic.statusText(w.status));
                            return k.fail(statusCode(w.status));
                        }
                        k.sends += 1;
                        if (w.wr_id == q) have_send = true;
                    }
                    if (n > 0) last = wire.monotonicNs();
                }
                if (!have_recv) {
                    const n = k.nic.poll(k.nic.cq_recv, &wc);
                    if (n < 0) return k.fail(5);
                    for (wc[0..@intCast(n)]) |w| {
                        if (w.status != vabi.wc_success) {
                            k.logExchange(q, site, bytes, t0, k.nic.statusText(w.status));
                            return k.fail(statusCode(w.status));
                        }
                        k.recvs += 1;
                        k.nic.postRecv() catch {
                            k.logExchange(q, site, bytes, t0, "ibv_post_recv failed");
                            return k.fail(5);
                        };
                        if (w.opcode != wc_recv_rdma_with_imm or (w.wc_flags & vabi.wc_with_imm) == 0 or w.imm_data != immediate(q) or w.byte_len != len) {
                            std.log.warn("collectives: rank {d} proxy: the peer's write for gather {d} ({s} layer {d}, {d} bytes) carries imm {d} and {d} bytes (expected {d} and {d})", .{
                                k.rank, q, tcp_proxy.phaseName(site), site & 0xffff, bytes, std.mem.bigToNative(u32, w.imm_data), w.byte_len, @as(u32, @truncate(q)), len,
                            });
                            return k.fail(3);
                        }
                        have_recv = true;
                    }
                    if (n > 0) last = wire.monotonicNs();
                }
                if (have_send and have_recv) break;
                if (k.stop.load(.acquire) != 0) return 0;
                if (@atomicLoad(u32, &k.shared.abort, .acquire) != 0) {
                    tcp_proxy.deviceGaveUp(k.shared, k.rank, q, k.nslot, "waiting for the completions of gather");
                    return k.fail(2);
                }
                spins +%= 1;
                if (spins % 256 == 0) {
                    const now = wire.monotonicNs();
                    if (now -| last > k.progress_ns) {
                        k.logExchange(q, site, bytes, t0, if (have_send) "no write from the peer within the progress deadline (Timeout)" else "no send completion within the progress deadline (Timeout)");
                        return k.fail(2);
                    }
                    if (k.peer_fd >= 0 and spins % 4096 == 0 and peerGone(k.peer_fd)) {
                        k.logExchange(q, site, bytes, t0, "the peer closed its control socket (PeerLost)");
                        return k.fail(1);
                    }
                }
                std.atomic.spinLoopHint();
            }
            const got: *const Trailer = @ptrCast(@alignCast(k.rx[base + at ..].ptr));
            if (got.magic != trailer_magic or !std.mem.eql(u8, &got.epoch, &k.epoch) or got.q != q or got.bytes != bytes or got.site != site or got.rank != 1 - k.rank) {
                std.log.warn("collectives: rank {d} proxy trailer mismatch at gather {d}: this rank's site {d}/{d} bytes {d}, the peer's (rank {d}) gather {d} site {d}/{d} bytes {d}{s}", .{
                    k.rank, q, site >> 16, site & 0xffff, bytes, got.rank, got.q, got.site >> 16, got.site & 0xffff, got.bytes,
                    if (got.magic != trailer_magic or !std.mem.eql(u8, &got.epoch, &k.epoch)) "; magic or epoch differs" else "",
                });
                return k.fail(3);
            }
            if (k.warn_ns > 0 and wire.monotonicNs() -| t0 > k.warn_ns) k.logExchange(q, site, bytes, t0, "done (this rank's coll.collect waited for it)");
            @atomicStore(u64, &k.shared.slot[slot].done, q, .release);
            _ = k.exchanged.fetchAdd(1, .monotonic);
            q += 1;
        }
        return 0;
    }

    fn fail(k: *Core, code: u32) u32 {
        tcp_proxy.raiseAbort(k.shared);
        return code;
    }

    fn logExchange(k: *const Core, q: u64, site: u32, bytes: u64, t0: u64, outcome: []const u8) void {
        std.log.warn("collectives: rank {d} rdma proxy exchange of gather {d} ({s} layer {d}, {d} bytes) after {d} ms: {s}", .{
            k.rank, q, tcp_proxy.phaseName(site), site & 0xffff, bytes, (wire.monotonicNs() -| t0) / std.time.ns_per_ms, outcome,
        });
    }
};

fn statusCode(status: c_int) u32 {
    return switch (status) {
        wc_wr_flush_err, wc_rem_inv_req_err, wc_rem_op_err, wc_retry_exc_err, wc_rnr_retry_exc_err, wc_rem_abort_err => 1,
        else => 5,
    };
}

/// True when the peer's end of `fd` is closed (a read would return end of stream); data or nothing to read: alive.
fn peerGone(fd: c.fd_t) bool {
    var b: [1]u8 = undefined;
    const r = c.recv(fd, &b, 1, c.MSG.PEEK | c.MSG.DONTWAIT);
    if (r > 0) return false;
    if (r == 0) return true;
    return switch (c.errno(r)) {
        .AGAIN, .INTR => false,
        else => true,
    };
}

extern "c" fn access(path: [*:0]const u8, mode: c_int) c_int;

/// This rank's device: TF_RDMA_DEVICES (one name, or one per rank), else the device whose device/net holds
/// TF_RDMA_IFNAME, else the only device there is.
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

/// The two ranks' Infos over the control link (both send first, small frames), checked for magic, size, slot count,
/// slot size and receive length; then a second frame each as the start barrier once both QPs are in RTS.
fn exchangeInfo(fan: *const Fan, mine: Info) !Info {
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

fn startBarrier(fan: *const Fan) !void {
    const link = fan.links[0] orelse return error.NoControlLink;
    try link.send("rts");
    var buf: [16]u8 = undefined;
    const got = try link.recv(&buf);
    if (!std.mem.eql(u8, got, "rts")) return error.BadInfo;
}

pub const RdmaProxy = struct {
    gpa: std.mem.Allocator,
    d: *const gpu.Driver,
    rank: u32,
    host: *tcp_coll.TcpColl,
    verbs: Verbs,
    nic: Nic,
    peer: Info,
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

    const Begun = struct { host: bool, recv: u64, bytes: usize };
    const vtable: coll.Coll.VTable = .{ .all_gather = allGather, .abort = abort, .close = close, .gather_begin = gatherBegin, .gather_end = gatherEnd, .check = check, .counts = counts };

    fn counts(ptr: *anyopaque) coll.Coll.Counts {
        const t: *RdmaProxy = @ptrCast(@alignCast(ptr));
        return .{ .staged = @atomicLoad(u64, &t.shared.seq, .acquire), .cable1 = t.exchanged.load(.acquire), .host = t.host.w.seq };
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

    /// The stage kernel (this rank's block into its slot; the proxy writes it as soon as it lands) or, for a gather
    /// larger than a slot carries, tcp_host's begin.
    fn gatherBegin(ptr: *anyopaque, stream: gpu.Stream, site: coll.Site, send: u64, recv: u64, bytes: usize) coll.Error!void {
        const t: *RdmaProxy = @ptrCast(@alignCast(ptr));
        if (t.status()) |e| return e;
        if (t.begun != null) return t.misuse("a gather begun while another is pending");
        if (bytes > capacity(t.slot_bytes)) {
            const capturing = stream.capturing() catch return error.CollectiveFailed;
            if (capturing) {
                std.log.err("collectives: a {d}-byte gather inside a graph exceeds what a {d}-byte slot carries (TF_COLL_SLOT_KIB)", .{ bytes, t.slot_bytes });
                t.poison(5);
                return error.CollectiveFailed;
            }
            try t.host.coll_().vtable.gather_begin.?(t.host, stream, site, send, recv, bytes);
            t.begun = .{ .host = true, .recv = recv, .bytes = bytes };
            return;
        }
        t.begun = .{ .host = false, .recv = recv, .bytes = bytes };
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
        // kernels still queued see the abort word and return without touching the slab's data; the MRs go only after
        // the kernels drained and the QP is gone
        _ = t.d.api.hipDeviceSynchronize();
        t.nic.close();
        t.verbs.close();
        t.slab.free();
        t.host.coll_().vtable.close(t.host);
        const gpa = t.gpa;
        gpa.destroy(t);
    }
};

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const gpu.Context, kit: *const gpu.Kit, fan: *const Fan, id: *const coll.Identity) anyerror!coll.Coll {
    _ = io;
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
    const gid_index = parseGidIndex(if (c.getenv("TF_RDMA_GID_INDEX")) |v| std.mem.span(v) else null) catch {
        std.log.err("collectives: TF_RDMA_GID_INDEX: a decimal GID index in 0..255", .{});
        return error.BadSetting;
    };
    var vb = try Verbs.open(null);
    errdefer vb.close();
    var name_buf: [64]u8 = undefined;
    const dev_name = try deviceName(&vb, &name_buf, id.rank);
    // the device name is this rank's own (F and A differ), so it is logged, not compared
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
    t.peer = try exchangeInfo(fan, t.nic.info(nslot, slot_bytes));
    try t.nic.connect(t.peer);
    try startBarrier(fan);
    t.thread = try std.Thread.spawn(.{}, RdmaProxy.proxyLoop, .{t});
    std.log.info("collectives: rdma_proxy on {s} gid={d} (mtu {d}, qp {d} -> peer qp {d}, rkey {x}) with {d} slots of {d} KiB (coherent slab {d} MiB, {d}-byte trailer), proxy thread on CPU {d}, progress deadline {d} ms; larger gathers through tcp_host", .{
        dev_name, gid_index, t.nic.active_mtu, t.nic.qp.qp_num, t.peer.qpn, t.peer.rkey, nslot, slot_bytes >> 10, slab.bytes.len >> 20, trailer_bytes, proxy_cpu, t.progress_ns / std.time.ns_per_ms,
    });
    return t.coll_();
}

// ---------------------------------------------------------------------------------------------------------------- tests
// The slab protocol without a GPU: two ranks' Cores over two QPs on one HCA (loopback), with host threads doing what
// coll.stage and coll.collect do. They need a device: TF_RDMA_TEST_DEVICE=<ibdev> (and TF_RDMA_GID_INDEX, default 3).

const testing = std.testing;

test "trailer placement and the slot capacity" {
    try testing.expectEqual(@as(u64, 0), trailerAt(0));
    try testing.expectEqual(@as(u64, 64), trailerAt(1));
    try testing.expectEqual(@as(u64, 20480), trailerAt(20480));
    try testing.expectEqual(@as(u64, 20544), wireLen(20480));
    try testing.expectEqual(@as(u64, 4_456_448 - 64), capacity(4_456_448));
    try testing.expect(wireLen(16 * 64640 * 4) <= 4_456_448); // the 16-row head gather fits the default slot
    try testing.expect(wireLen(capacity(4_456_448)) == 4_456_448);
}

test "the GID index defaults to 3 and refuses malformed settings" {
    try testing.expectEqual(@as(u8, 3), try parseGidIndex(null));
    try testing.expectEqual(@as(u8, 5), try parseGidIndex("5"));
    for ([_][]const u8{ "", "-1", "+3", " 3", "0x3", "256" }) |bad| try testing.expectError(error.BadSetting, parseGidIndex(bad));
}

test "TF_RDMA_DEVICES: one name for both ranks, or one per rank" {
    const env = struct {
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };
    defer _ = env.unsetenv("TF_RDMA_DEVICES");
    var v: Verbs = undefined;
    var buf: [64]u8 = undefined;
    _ = env.setenv("TF_RDMA_DEVICES", "rocep195s0,rocep194s0", 1);
    try testing.expectEqualStrings("rocep195s0", try deviceName(&v, &buf, 0));
    try testing.expectEqualStrings("rocep194s0", try deviceName(&v, &buf, 1));
    _ = env.setenv("TF_RDMA_DEVICES", "rocep196s0", 1);
    try testing.expectEqualStrings("rocep196s0", try deviceName(&v, &buf, 1));
}

test "the Info a peer reads is a plain extern struct" {
    try testing.expectEqual(@as(usize, 64), @sizeOf(Info));
    try testing.expectEqual(@as(usize, 40), @offsetOf(Info, "rx_base"));
}

const FakeGpu = struct {
    shared: *Shared,
    tx: []u8,
    rx: []u8,
    nslot: u32,
    slot_bytes: u64,
    rank: u32,
    gathers: usize,
    bad: usize = 0,
    /// stage only, never collect (the peer's slots then fill up and its stage must wait)
    hold_collect_from: usize = std.math.maxInt(usize),
    /// a slow collector: this long between done and consumed
    collect_delay_ns: u64 = 0,
    /// at this gather, a different site (or bytes) from the peer's
    wrong_site_at: usize = 0,
    wrong_bytes_at: usize = 0,
    fixed_bytes: ?usize = null,
    timed_out: bool = false,
    /// ready -> done per gather (ns), for the timing test
    rtt: ?[]u64 = null,

    fn pattern(q: u64, rank: u32, i: usize) u8 {
        return @truncate((q *% 131) ^ (@as(u64, rank) << 7) ^ i);
    }

    fn bytesOf(f: *const FakeGpu, q: u64) usize {
        return f.fixed_bytes orelse 1000 + (q % 7) * 4096;
    }

    fn run(f: *FakeGpu) void {
        for (1..f.gathers + 1) |qi| {
            const q: u64 = qi;
            const slot: usize = @intCast(q % f.nslot);
            const bytes: usize = f.bytesOf(q) + @as(usize, if (qi == f.wrong_bytes_at) 64 else 0);
            const t0 = wire.monotonicNs();
            while (q > f.nslot and @atomicLoad(u64, &f.shared.slot[slot].consumed, .acquire) + f.nslot < q) {
                if (@atomicLoad(u32, &f.shared.abort, .acquire) != 0 or wire.monotonicNs() - t0 > 2 * std.time.ns_per_s) {
                    f.timed_out = true;
                    return;
                }
                std.atomic.spinLoopHint();
            }
            const tx = f.tx[slot * f.slot_bytes ..][0..bytes];
            for (tx, 0..) |*b, i| b.* = pattern(q, f.rank, i);
            f.shared.slot[slot].bytes = bytes;
            f.shared.slot[slot].site = if (qi == f.wrong_site_at) 9 else 7;
            const t1 = wire.monotonicNs();
            @atomicStore(u64, &f.shared.slot[slot].ready, q, .release);
            if (qi >= f.hold_collect_from) continue;
            while (@atomicLoad(u64, &f.shared.slot[slot].done, .acquire) != q) {
                if (@atomicLoad(u32, &f.shared.abort, .acquire) != 0) return;
                std.atomic.spinLoopHint();
            }
            if (f.rtt) |r| r[qi - 1] = wire.monotonicNs() - t1;
            if (f.collect_delay_ns > 0) {
                const t2 = wire.monotonicNs();
                while (wire.monotonicNs() - t2 < f.collect_delay_ns) std.atomic.spinLoopHint();
            }
            const rx = f.rx[slot * f.slot_bytes ..][0..bytes];
            for (rx, 0..) |b, i| f.bad += @intFromBool(b != pattern(q, 1 - f.rank, i));
            @atomicStore(u64, &f.shared.slot[slot].consumed, q, .release);
        }
    }
};

const TestRank = struct {
    shared: Shared = std.mem.zeroes(Shared),
    tx: []u8,
    rx: []u8,
    nic: Nic,
    core: Core = undefined,
    stop: std.atomic.Value(u32) = .init(0),
    exchanged: std.atomic.Value(u64) = .init(0),
    code: u32 = 0,
    alive: bool = true,

    fn runCore(r: *TestRank) void {
        r.code = r.core.run();
    }

    fn kill(r: *TestRank) void {
        if (r.alive) r.nic.close();
        r.alive = false;
    }
};

const Pair = struct {
    v: Verbs,
    rk: [2]*TestRank,
    nslot: u32,
    slot_bytes: u64,

    fn close(p: *Pair) void {
        for (p.rk) |r| {
            r.kill();
            std.heap.page_allocator.free(r.tx);
            std.heap.page_allocator.free(r.rx);
            testing.allocator.destroy(r);
        }
        p.v.close();
    }

    fn gpus(p: *Pair, gathers: usize) [2]FakeGpu {
        var g: [2]FakeGpu = undefined;
        for (&g, p.rk, 0..) |*f, r, i| f.* = .{ .shared = &r.shared, .tx = r.tx, .rx = r.rx, .nslot = p.nslot, .slot_bytes = p.slot_bytes, .rank = @intCast(i), .gathers = gathers };
        return g;
    }
};

/// Two ranks' slabs (page memory) and QPs on the test device, connected to each other; null without a device.
fn testPair(nslot: u32, slot_bytes: u64, progress_ns: u64) !?Pair {
    const dev = c.getenv("TF_RDMA_TEST_DEVICE") orelse return null;
    const gid_index = try parseGidIndex(if (c.getenv("TF_RDMA_GID_INDEX")) |v| std.mem.span(v) else null);
    var p: Pair = .{ .v = try Verbs.open(null), .rk = undefined, .nslot = nslot, .slot_bytes = slot_bytes };
    errdefer p.v.close();
    const epoch: [16]u8 = @splat(3);
    for (&p.rk) |*slot| {
        const r = try testing.allocator.create(TestRank);
        const n: usize = @intCast(nslot * slot_bytes);
        r.* = .{ .tx = try std.heap.page_allocator.alloc(u8, n), .rx = try std.heap.page_allocator.alloc(u8, n), .nic = undefined };
        @memset(r.rx, 0);
        r.nic = try Nic.open(&p.v, std.mem.span(dev), gid_index, r.tx, r.rx);
        slot.* = r;
    }
    for (p.rk, 0..) |r, i| {
        const peer = p.rk[1 - i];
        try r.nic.connect(peer.nic.info(nslot, slot_bytes));
        r.core = .{ .shared = &r.shared, .tx = r.tx, .rx = r.rx, .nslot = nslot, .slot_bytes = slot_bytes, .nic = &r.nic, .peer_rx = peer.nic.rx_base, .peer_rkey = peer.nic.mr_rx.rkey, .epoch = epoch, .stop = &r.stop, .exchanged = &r.exchanged, .progress_ns = progress_ns, .rank = @intCast(i) };
    }
    return p;
}

fn runBoth(p: *Pair, g: *[2]FakeGpu) !void {
    const c0 = try std.Thread.spawn(.{}, TestRank.runCore, .{p.rk[0]});
    const c1 = try std.Thread.spawn(.{}, TestRank.runCore, .{p.rk[1]});
    const g1 = try std.Thread.spawn(.{}, FakeGpu.run, .{&g[1]});
    g[0].run();
    g1.join();
    for (p.rk) |r| r.stop.store(1, .release);
    for (p.rk) |r| r.nic.toErr(); // a core waiting for a completion that never comes leaves at once
    c0.join();
    c1.join();
}

test "rdma loopback: the slab protocol moves every gather through reused slots, byte-verified, and stops cleanly" {
    var p = (try testPair(4, 32768, 2 * std.time.ns_per_s)) orelse return error.SkipZigTest;
    defer p.close();
    var g = p.gpus(2000);
    try runBoth(&p, &g);
    try testing.expectEqual(@as(usize, 0), g[0].bad + g[1].bad);
    for (p.rk) |r| {
        try testing.expectEqual(@as(u32, 0), r.code);
        try testing.expectEqual(@as(u64, 2000), r.exchanged.load(.acquire));
        // TX reuse: a slot was re-staged only after its send completion (one per gather, all seen)
        try testing.expectEqual(@as(u64, 2000), r.core.sends);
        try testing.expectEqual(@as(u64, 2000), r.core.recvs);
    }
}

test "rdma loopback: RX reuse with NSLOT=2, a slow collector on one rank and a fast peer: nothing is overwritten" {
    var p = (try testPair(2, 32768, 2 * std.time.ns_per_s)) orelse return error.SkipZigTest;
    defer p.close();
    var g = p.gpus(400);
    g[1].collect_delay_ns = 300 * std.time.ns_per_us;
    try runBoth(&p, &g);
    try testing.expectEqual(@as(usize, 0), g[0].bad + g[1].bad);
    for (p.rk) |r| {
        try testing.expectEqual(@as(u32, 0), r.code);
        try testing.expectEqual(@as(u64, 400), r.exchanged.load(.acquire));
    }
}

test "rdma loopback: a slot is not reused before it was consumed: a stalled collect stops the stage, not the data" {
    var p = (try testPair(4, 32768, 2 * std.time.ns_per_s)) orelse return error.SkipZigTest;
    defer p.close();
    var g = p.gpus(50);
    g[1].hold_collect_from = 10; // rank 1 stops collecting at gather 10: its stage must stall at gather 14
    const c0 = try std.Thread.spawn(.{}, TestRank.runCore, .{p.rk[0]});
    const c1 = try std.Thread.spawn(.{}, TestRank.runCore, .{p.rk[1]});
    const g0 = try std.Thread.spawn(.{}, FakeGpu.run, .{&g[0]});
    g[1].run();
    try testing.expect(g[1].timed_out); // waited for a slot that was never released
    for (p.rk) |r| r.stop.store(1, .release);
    @atomicStore(u32, &p.rk[0].shared.abort, 1, .release);
    for (p.rk) |r| r.nic.toErr();
    g0.join();
    c0.join();
    c1.join();
    try testing.expectEqual(@as(usize, 0), g[0].bad + g[1].bad); // nothing overwritten while still owned
    try testing.expect(p.rk[1].exchanged.load(.acquire) <= 13);
}

test "rdma loopback: a different site or size from the peer fails both ranks with Mismatch" {
    for ([_]u8{ 0, 1 }) |which| {
        var p = (try testPair(4, 32768, 2 * std.time.ns_per_s)) orelse return error.SkipZigTest;
        defer p.close();
        var g = p.gpus(30);
        if (which == 0) g[0].wrong_site_at = 12 else g[0].wrong_bytes_at = 12;
        try runBoth(&p, &g);
        try testing.expectEqual(@as(u32, 3), p.rk[0].code);
        try testing.expectEqual(@as(u32, 3), p.rk[1].code);
        try testing.expect(p.rk[0].exchanged.load(.acquire) == 11 and p.rk[1].exchanged.load(.acquire) == 11);
        try testing.expect(@atomicLoad(u32, &p.rk[0].shared.abort, .acquire) != 0);
    }
}

test "rdma loopback: a peer that stops staging fails the proxy at the progress deadline (Timeout), after its own send completed" {
    var p = (try testPair(4, 32768, 500 * std.time.ns_per_ms)) orelse return error.SkipZigTest;
    defer p.close();
    // rank 0 stages gather 1; rank 1 never does
    p.rk[0].shared.slot[1].bytes = 4096;
    p.rk[0].shared.slot[1].site = (@as(u32, @intFromEnum(coll.Phase.prompt_engram)) << 16) | 3;
    @atomicStore(u64, &p.rk[0].shared.slot[1].ready, 1, .release);
    const t0 = wire.monotonicNs();
    const code = p.rk[0].core.run();
    const waited = wire.monotonicNs() - t0;
    try testing.expectEqual(@as(u32, 2), code);
    try testing.expectEqual(@as(u64, 1), p.rk[0].core.sends);
    try testing.expect(waited >= 500 * std.time.ns_per_ms and waited < 5 * std.time.ns_per_s);
    try testing.expectEqual(@as(u32, 1), @atomicLoad(u32, &p.rk[0].shared.abort, .acquire));
}

test "rdma loopback: a dead peer (its QP gone) fails the write with RETRY_EXC: PeerLost from the NIC (measured, see qpTimeout)" {
    var p = (try testPair(4, 32768, 8 * std.time.ns_per_s)) orelse return error.SkipZigTest;
    defer p.close();
    p.rk[1].kill();
    p.rk[0].shared.slot[1].bytes = 20480;
    p.rk[0].shared.slot[1].site = (@as(u32, @intFromEnum(coll.Phase.round_attn)) << 16) | 3;
    @atomicStore(u64, &p.rk[0].shared.slot[1].ready, 1, .release);
    const t0 = wire.monotonicNs();
    const code = p.rk[0].core.run();
    const waited = wire.monotonicNs() - t0;
    std.debug.print("rdma loopback: dead peer surfaced after {d} ms (qp timeout {d})\n", .{ waited / std.time.ns_per_ms, try qpTimeout() });
    try testing.expectEqual(@as(u32, 1), code);
    try testing.expect(waited < 6 * std.time.ns_per_s);
}

test "rdma loopback: a stalled peer whose control socket closes is PeerLost within milliseconds" {
    var p = (try testPair(4, 32768, 4 * std.time.ns_per_s)) orelse return error.SkipZigTest;
    defer p.close();
    var sv: [2]c.fd_t = undefined;
    if (c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &sv) != 0) return error.SocketFailed;
    defer _ = c.close(sv[0]);
    p.rk[0].core.peer_fd = sv[0];
    p.rk[0].shared.slot[1].bytes = 4096;
    p.rk[0].shared.slot[1].site = 7;
    @atomicStore(u64, &p.rk[0].shared.slot[1].ready, 1, .release);
    const Closer = struct {
        fn run(fd: c.fd_t) void {
            var d: c.timespec = .{ .sec = 0, .nsec = 100 * std.time.ns_per_ms };
            _ = c.nanosleep(&d, null);
            _ = c.close(fd);
        }
    };
    const th = try std.Thread.spawn(.{}, Closer.run, .{sv[1]});
    const t0 = wire.monotonicNs();
    const code = p.rk[0].core.run();
    const waited = wire.monotonicNs() - t0;
    th.join();
    try testing.expectEqual(@as(u32, 1), code);
    try testing.expect(waited >= 100 * std.time.ns_per_ms and waited < 2 * std.time.ns_per_s);
}

test "rdma loopback: 20 KiB round trips, ready to done (hairpin through one HCA: slower than two boxes)" {
    var p = (try testPair(8, 32768, 2 * std.time.ns_per_s)) orelse return error.SkipZigTest;
    defer p.close();
    const n = 3000;
    var g = p.gpus(n);
    const rtt = try testing.allocator.alloc(u64, n);
    defer testing.allocator.free(rtt);
    for (&g) |*f| f.fixed_bytes = 20480;
    g[0].rtt = rtt;
    try runBoth(&p, &g);
    try testing.expectEqual(@as(usize, 0), g[0].bad + g[1].bad);
    const warm = 500;
    const v = rtt[warm..];
    std.mem.sort(u64, v, {}, std.sort.asc(u64));
    var sum: u64 = 0;
    for (v) |x| sum += x;
    std.debug.print("rdma loopback 20480 B: n={d} p50={d:.1} p99={d:.1} min={d:.1} max={d:.1} mean={d:.1} us (ready -> done on rank 0)\n", .{
        v.len, @as(f64, @floatFromInt(v[v.len / 2])) / 1e3, @as(f64, @floatFromInt(v[v.len * 99 / 100])) / 1e3, @as(f64, @floatFromInt(v[0])) / 1e3, @as(f64, @floatFromInt(v[v.len - 1])) / 1e3, @as(f64, @floatFromInt(sum / v.len)) / 1e3,
    });
}
