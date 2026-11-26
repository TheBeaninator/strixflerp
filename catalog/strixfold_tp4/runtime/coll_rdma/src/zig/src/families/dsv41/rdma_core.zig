//! rdma_proxy's loop on the slab, GPU-free: ready, one write with immediate, both completions, trailer check, done.
const std = @import("std");
const nic = @import("rdma_nic.zig");
const wire = @import("tcp_wire.zig");
const tcp_proxy = @import("tcp_proxy.zig");

const c = std.c;
const vabi = nic.vabi;
pub const Shared = tcp_proxy.Shared;

/// run returns a failure code (0 clean stop, 1 PeerLost, 2 Timeout, 3 Mismatch, 5 verbs) after raising the abort word.
pub const Core = struct {
    shared: *Shared,
    tx: []u8, // nslot * slot_bytes
    rx: []u8,
    nslot: u32,
    slot_bytes: u64,
    nic: *nic.Nic,
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
        // one completion per poll: the peer's write of q + 1 may already be in, and that one belongs to the next turn
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
                tcp_proxy.idleWait(&idle_since);
                continue;
            }
            idle_since = 0;
            const bytes: u64 = k.shared.slot[slot].bytes;
            const site: u32 = @truncate(k.shared.slot[slot].site);
            if (nic.wireLen(bytes) > k.slot_bytes) {
                std.log.warn("collectives: rank {d} proxy: gather {d} says {d} bytes, more than a slot carries ({d})", .{ k.rank, q, bytes, nic.capacity(k.slot_bytes) });
                return k.fail(3);
            }
            const base: usize = slot * @as(usize, @intCast(k.slot_bytes));
            const at: usize = @intCast(nic.trailerAt(bytes));
            const tr: *nic.Trailer = @ptrCast(@alignCast(k.tx[base + at ..].ptr));
            tr.* = .{ .magic = nic.trailer_magic, .epoch = k.epoch, .q = q, .bytes = bytes, .site = site, .rank = k.rank };
            const len: usize = @intCast(nic.wireLen(bytes));
            const t0 = wire.monotonicNs();
            k.nic.postWrite(k.tx[base..][0..len], k.peer_rx + base, k.peer_rkey, q) catch {
                k.logExchange(q, site, bytes, t0, "ibv_post_send failed");
                return k.fail(5);
            };
            const outcome = k.completions(q, site, bytes, len, t0, &wc);
            if (outcome != 0) return outcome;
            const got: *const nic.Trailer = @ptrCast(@alignCast(k.rx[base + at ..].ptr));
            if (got.magic != nic.trailer_magic or !std.mem.eql(u8, &got.epoch, &k.epoch) or got.q != q or got.bytes != bytes or got.site != site or got.rank != 1 - k.rank) {
                std.log.warn("collectives: rank {d} proxy trailer mismatch at gather {d}: this rank's site {d}/{d} bytes {d}, the peer's (rank {d}) gather {d} site {d}/{d} bytes {d}{s}", .{
                    k.rank, q, site >> 16, site & 0xffff, bytes, got.rank, got.q, got.site >> 16, got.site & 0xffff, got.bytes,
                    if (got.magic != nic.trailer_magic or !std.mem.eql(u8, &got.epoch, &k.epoch)) "; magic or epoch differs" else "",
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

    /// Both completions of gather q: the local send (TX[s] may be re-staged) and the peer's write (RX[s] is complete).
    fn completions(k: *Core, q: u64, site: u32, bytes: u64, len: usize, t0: u64, wc: *[1]vabi.Wc) u32 {
        var have_send = false;
        var have_recv = false;
        var last = t0;
        var spins: u32 = 0;
        while (!(have_send and have_recv)) {
            if (!have_send) {
                const n = k.nic.poll(k.nic.cq_send, wc);
                if (n < 0) return k.fail(5);
                for (wc[0..@intCast(n)]) |w| {
                    if (w.status != vabi.wc_success) {
                        k.logExchange(q, site, bytes, t0, k.nic.statusText(w.status));
                        return k.fail(nic.statusCode(w.status));
                    }
                    k.sends += 1;
                    if (w.wr_id == q) have_send = true;
                }
                if (n > 0) last = wire.monotonicNs();
            }
            if (!have_recv) {
                const n = k.nic.poll(k.nic.cq_recv, wc);
                if (n < 0) return k.fail(5);
                for (wc[0..@intCast(n)]) |w| {
                    if (w.status != vabi.wc_success) {
                        k.logExchange(q, site, bytes, t0, k.nic.statusText(w.status));
                        return k.fail(nic.statusCode(w.status));
                    }
                    k.recvs += 1;
                    k.nic.postRecv() catch {
                        k.logExchange(q, site, bytes, t0, "ibv_post_recv failed");
                        return k.fail(5);
                    };
                    if (w.opcode != nic.wc_recv_rdma_with_imm or (w.wc_flags & vabi.wc_with_imm) == 0 or w.imm_data != nic.immediate(q) or w.byte_len != len) {
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


// ---- world 4 (TP4-RDMA-N): the same slab protocol over one QP per peer ------------------------------------------------
//
// At world 2 the code above is the transport, unchanged. At world 4 (the 2D split's quarters and pair exchanges) each
// rank keeps one RC queue pair toward each of the other three and one receive region the peers write into:
//
//   slot s (slot_bytes): [ data: block of rank 0 | rank 1 | rank 2 | rank 3 ... ][ trailer of rank 0 | 1 | 2 | 3 ]
//
// A gather of n bytes a rank puts rank g's block at g * n of the slot, so one coll.collect copies all of them (its own
// included: coll.stage writes the rank's block into its own place of the slot, and nothing into recv) to recv in rank
// order. An exchange puts the lower rank's block at 0 and the higher rank's at half the data area. Every slot gather,
// exchange or not, sends a 64-byte trailer (write with immediate) to every peer: each rank hears from every rank at
// every q, so the transitive slot-reuse argument of the two-rank proxy holds (a peer writes q' into this rank's slot
// only after it heard this rank's q' - 1, staged after this rank collected q' - nslot), and every rank checks every
// rank's site at every q. A rank that fails tells every peer at once with a zero-byte write with immediate (poison).
// Gathers and exchanges larger than a slot run as consecutive slot gathers (chunks) with the same site, so nothing at
// world 4 falls back to tcp_host (which is two-rank). With TF_COLL_LANES > 1 (TP4-RDMA-PIPE) the slots are split into
// lanes, each with its own slab header and gather numbers, and a chunked call's chunks go round robin over them, so
// several chunks are on the wire while the GPU stages and collects the others (CoreN's comment).

pub const max_world = 4;

/// The 32-bit site word coll.stage writes into the slot at world 4 (the proxy never sees the host call):
/// phase 4 | layer 8 | exchange 1 | pair lo 2 | pair hi 2 | size hash 15. The two ranks of an exchange compute the same
/// word (the pair and the hash of the lower rank's and the higher rank's bytes), so equal words mean the same call.
pub const Word = struct {
    pub fn phase4(phase: u16) u32 {
        return if (phase >= 15) 15 else phase;
    }

    pub fn gather(phase: u16, layer: u16) u32 {
        return (phase4(phase) << 28) | (@as(u32, layer & 0xff) << 20);
    }

    /// A gather whose blocks land in swap12 order (bit 18, zero in every other gather word): rank g's block goes to
    /// place 2 (g % 2) + g / 2 (ranks 1 and 2 trade), so prompt2d's one-row quarters, [2 r, 2 p] of equal widths, come
    /// straight out of the slot with no copies after it (TP4-QUICKWINS). Every rank of the call sets it alike:
    /// a rank that does not writes a different word, which the trailer check reports as Mismatch.
    pub fn gatherSwap12(phase: u16, layer: u16) u32 {
        return gather(phase, layer) | (1 << 18);
    }
    pub fn isSwap12(w: u32) bool {
        return !isExchange(w) and (w >> 18) & 1 != 0;
    }
    /// The place of rank g's block in the gather of word w (its offset in the slot and in recv, in blocks).
    pub fn place(w: u32, g: u32) u32 {
        return if (isSwap12(w)) 2 * (g % 2) + g / 2 else g;
    }

    pub fn exchange(phase: u16, layer: u16, rank: u32, peer: u32, send_bytes: u64, recv_bytes: u64) u32 {
        const l = @min(rank, peer);
        const h = @max(rank, peer);
        const lo_bytes = if (rank < peer) send_bytes else recv_bytes;
        const hi_bytes = if (rank < peer) recv_bytes else send_bytes;
        return gather(phase, layer) | (1 << 19) | (l << 17) | (h << 15) | sizeHash(lo_bytes, hi_bytes);
    }

    pub fn sizeHash(lo_bytes: u64, hi_bytes: u64) u32 {
        var h = std.hash.Wyhash.init(0x5446_5244);
        h.update(std.mem.asBytes(&lo_bytes));
        h.update(std.mem.asBytes(&hi_bytes));
        return @truncate(h.final() & 0x7fff);
    }

    pub fn isExchange(w: u32) bool {
        return (w >> 19) & 1 != 0;
    }
    pub fn lo(w: u32) u32 {
        return (w >> 17) & 3;
    }
    pub fn hi(w: u32) u32 {
        return (w >> 15) & 3;
    }
    /// The call (phase, layer, kind) without the pair and sizes: equal on every rank at every q.
    pub fn call(w: u32) u32 {
        return w >> 19;
    }
    /// This rank's partner in an exchange word (null: the word does not name this rank).
    pub fn partner(w: u32, rank: u32) ?u32 {
        if (lo(w) == rank and hi(w) != rank) return hi(w);
        if (hi(w) == rank and lo(w) != rank) return lo(w);
        return null;
    }
    pub fn names(w: u32, rank: u32) bool {
        return lo(w) == rank or hi(w) == rank;
    }
    pub fn phaseOf(w: u32) u32 {
        return w >> 28;
    }
    pub fn layerOf(w: u32) u32 {
        return (w >> 20) & 0xff;
    }
};

/// Where everything sits in a world-4 slot (see the picture above). Offsets are from the slot's start.
pub const Layout = struct {
    world: u32,
    slot_bytes: u64,
    nslot: u32,
    /// independent slot lanes (TF_COLL_LANES); nslot is a multiple and every lane has at least 2 slots
    lanes: u32 = 1,

    /// Slots a lane owns.
    pub fn laneSlots(l: Layout) u32 {
        return l.nslot / l.lanes;
    }
    /// The lane chunk ci of a call takes: chunk 0 (every call that fits one slot) lane 0, then round robin.
    pub fn laneOf(l: Layout, ci: u64) u32 {
        return @intCast(ci % l.lanes);
    }
    /// Gather q of lane li: its slot's index among all the slots (lane li owns laneSlots() of them, in a row).
    pub fn slotIndex(l: Layout, li: u32, q: u64) usize {
        const m = l.laneSlots();
        return @intCast(@as(u64, li) * m + q % m);
    }
    /// A lane setting the slots carry: 1 to max_lanes, dividing nslot, 2 slots a lane at least.
    pub fn lanesFit(nslot: u32, lanes: u32) bool {
        return lanes >= 1 and lanes <= max_lanes and nslot % lanes == 0 and nslot / lanes >= 2;
    }

    pub fn trailerOff(l: Layout, g: u32) u64 {
        return l.slot_bytes - @as(u64, l.world - g) * nic.trailer_bytes;
    }
    pub fn dataBytes(l: Layout) u64 {
        return l.slot_bytes - @as(u64, l.world) * nic.trailer_bytes;
    }
    /// The most a rank's block of one slot gather carries (a 64-byte multiple).
    pub fn gatherChunk(l: Layout) u64 {
        return std.mem.alignBackward(u64, l.dataBytes() / l.world, 64);
    }
    /// The most each side of one slot exchange carries (a 64-byte multiple).
    pub fn exchangeHalf(l: Layout) u64 {
        return std.mem.alignBackward(u64, l.dataBytes() / 2, 64);
    }
    pub fn exchangePos(l: Layout, g: u32, peer: u32) u64 {
        return if (g < peer) 0 else l.exchangeHalf();
    }

    /// Slot gathers a call takes (0: nothing moves; every rank of the call computes the same count).
    pub fn gatherChunks(l: Layout, bytes: u64) u64 {
        return std.math.divCeil(u64, bytes, l.gatherChunk()) catch unreachable;
    }
    pub fn exchangeChunks(l: Layout, send_bytes: u64, recv_bytes: u64) u64 {
        const h = l.exchangeHalf();
        return @max(std.math.divCeil(u64, send_bytes, h) catch unreachable, std.math.divCeil(u64, recv_bytes, h) catch unreachable);
    }
    /// How many stages run ahead of their collects: one a lane. coll.collect reads its lane header's seq (the last
    /// stage on that lane), so on each lane a slot gather's collect follows its own stage before the lane's next stage;
    /// with chunk c on lane c % lanes, the stream runs stage 0 .. stage lanes-1, then collect c and stage c + lanes,
    /// and up to `lanes` chunks are on the wire at once (the kernels are unchanged).
    pub fn depth(l: Layout, chunks: u64) u64 {
        return @min(chunks, l.lanes);
    }

    /// One slot gather's launches: coll.stage (its tx relative to the slots' start; own = send, so no own copy) and
    /// its coll.collects (rx relative to the slots' start).
    pub const Step = struct {
        send: u64,
        tx_off: u64,
        bytes: u64,
        site: u32,
        /// the lane (header and slots) it runs on
        lane: u32 = 0,
        collects: [max_world]Collect = undefined,
        ncollect: u32 = 0,
    };
    pub const Collect = struct { rx_off: u64, dst: u64, bytes: u64 };

    /// Chunk c of a gather of "bytes" a rank: one collect of every block when it fits one slot, else one a rank.
    pub fn gatherStep(l: Layout, rank: u32, send: u64, recv: u64, bytes: u64, site: u32, ci: u64) Step {
        const k = l.gatherChunks(bytes);
        const chunk = l.gatherChunk();
        const off = ci * chunk;
        const n = @min(chunk, bytes - off);
        // the block at its place (Word.place: the rank, or swap12's); the collects stay in place order, since place p of
        // the slot holds the rank whose block belongs at place p of recv
        var s: Step = .{ .send = send + off, .tx_off = @as(u64, Word.place(site, rank)) * n, .bytes = n, .site = site, .lane = l.laneOf(ci) };
        if (k == 1) {
            s.collects[0] = .{ .rx_off = 0, .dst = recv, .bytes = @as(u64, l.world) * n };
            s.ncollect = 1;
        } else {
            for (0..l.world) |g| s.collects[g] = .{ .rx_off = @as(u64, g) * n, .dst = recv + g * bytes + off, .bytes = n };
            s.ncollect = l.world;
        }
        return s;
    }

    /// Chunk c of an exchange with a peer: this rank's part of send to its place, the peer's part of recv from its.
    pub fn exchangeStep(l: Layout, rank: u32, peer: u32, send: u64, send_bytes: u64, recv: u64, recv_bytes: u64, phase: u16, layer: u16, ci: u64) Step {
        const h = l.exchangeHalf();
        const off = ci * h;
        const sn = if (send_bytes > off) @min(h, send_bytes - off) else 0;
        const rn = if (recv_bytes > off) @min(h, recv_bytes - off) else 0;
        var s: Step = .{ .send = send + off, .tx_off = l.exchangePos(rank, peer), .bytes = sn, .site = Word.exchange(phase, layer, rank, peer, sn, rn), .lane = l.laneOf(ci) };
        s.collects[0] = .{ .rx_off = l.exchangePos(peer, rank), .dst = recv + off, .bytes = rn };
        s.ncollect = 1;
        return s;
    }
};

pub const Cqe = nic.Cqe;
pub const Seg = nic.Seg;

pub const poison_bit: u64 = 1 << 62; // wr_id of a poison write

/// Lanes a slab carries at most (TF_COLL_LANES): the lane rides in the top 2 bits of a trailer's immediate.
pub const max_lanes = 4;
const imm_lane_shift = 30;
/// A trailer's immediate: the lane, then the gather's low 30 bits (one lane: q's low 30 bits).
pub fn immOf(lane: u32, q: u64) u32 {
    return (lane << imm_lane_shift) | (@as(u32, @truncate(q)) & ((1 << imm_lane_shift) - 1));
}
/// A trailer's wr_id: gather, lane, peer (q below 2^56, clear of poison_bit).
pub fn wrOf(q: u64, lane: u32, j: u32) u64 {
    return (q << 5) | (@as(u64, lane) << 3) | j;
}
fn laneOfWr(wr_id: u64) usize {
    return @intCast((wr_id >> 3) & 3);
}
fn qOfWr(wr_id: u64) u64 {
    return wr_id >> 5;
}

/// The world-4 proxy loop on the slab, generic over the Net (post, postPoison, pollSend, pollRecv, statusText),
/// GPU-free. run returns a failure code (0 clean stop, 1 PeerLost, 2 Timeout, 3 Mismatch, 5 verbs) after raising the
/// abort word and poisoning every peer.
///
/// Lanes (TF_COLL_LANES, TP4-RDMA-PIPE): the slab holds one header a lane and lane l owns slots l*m .. l*m+m-1
/// (m = nslot / lanes). Each lane is the one-in-flight protocol above with its own gather numbers, so coll.collect,
/// which reads its header's seq, still collects its own stage; a call of several chunks puts chunk c on lane c % lanes
/// and so keeps up to `lanes` chunks on the wire while the GPU copies the others. The proxy runs every lane's turn at
/// once: a write's immediate and wr_id carry the lane, a turn finishes when its sends and every peer's trailer are in.
/// A call that fits one slot always takes lane 0, so decode moves exactly as with one lane.
pub fn CoreN(comptime Net: type) type {
    return struct {
        const K = @This();
        /// one header a lane (layout.lanes of them)
        shared: [*]Shared,
        rx: []u8, // nslot * slot_bytes: the slots the peers write into (and this rank's own blocks)
        layout: Layout,
        net: *Net,
        rank: u32,
        epoch: [16]u8,
        stop: *std.atomic.Value(u32),
        exchanged: *std.atomic.Value(u64),
        progress_ns: u64 = 2 * std.time.ns_per_s,
        /// connected sockets to peek at while waiting (rank 0: each follower's control link; a follower: rank 0's)
        ctl_fds: [max_world - 1]c.fd_t = .{ -1, -1, -1 },
        warn_ns: u64 = 0,
        sends: u64 = 0,
        recvs: u64 = 0,
        /// most turns that were in flight at once (lanes busy together; 1 when nothing overlapped)
        max_inflight: u32 = 0,
        /// the peer whose poison failed this rank (diagnostics; it is not poisoned back)
        poisoned_by: ?u32 = null,
        poisoned: bool = false,

        /// One lane's gather in flight (active) or next to start (idle, q).
        const Turn = struct {
            active: bool = false,
            q: u64 = 1,
            slot: usize = 0, // its index among the rx slots
            site: u32 = 0,
            bytes: u64 = 0,
            partner: ?u32 = null,
            t0: u64 = 0,
            have_send: [max_world]bool = @splat(false),
            have_recv: [max_world]bool = @splat(false),
            /// a peer's trailer of the lane's next turn that came early (active: q + 1; idle: q); at most one a peer,
            /// since it waits for this rank's trailer of that turn before the one after
            early: [max_world]bool = @splat(false),
        };

        pub fn run(k: *K) u32 {
            const l = k.layout;
            const nl: usize = l.lanes;
            var turns: [max_lanes]Turn = @splat(.{});
            var idle_since: u64 = 0;
            var last = wire.monotonicNs();
            var spins: u32 = 0;
            var wc: [16]Cqe = undefined;
            while (true) {
                if (k.stop.load(.acquire) != 0) return 0;
                // start every lane whose next gather is staged
                var inflight: u32 = 0;
                for (turns[0..nl], 0..) |*t, li| {
                    if (!t.active) {
                        const ls: usize = @intCast(t.q % l.laneSlots());
                        const r = @atomicLoad(u64, &k.shared[li].slot[ls].ready, .acquire);
                        if (r == t.q) {
                            const code = k.start(@intCast(li), t);
                            if (code != 0) return code;
                            last = wire.monotonicNs();
                        } else if (r > t.q) {
                            std.log.warn("collectives: rank {d} proxy expected gather {d} in slot {d} of lane {d} and found gather {d}", .{ k.rank, t.q, ls, li, r });
                            return k.fail(3);
                        }
                    }
                    if (t.active) inflight += 1;
                }
                if (inflight == 0) {
                    if (k.abortedLane()) |li| {
                        tcp_proxy.deviceGaveUp(&k.shared[li], k.rank, turns[li].q, l.laneSlots(), "waiting for its GPU to stage gather");
                        return k.fail(2);
                    }
                    tcp_proxy.idleWait(&idle_since);
                    continue;
                }
                idle_since = 0;
                if (inflight > k.max_inflight) k.max_inflight = inflight;
                var n = k.net.pollSend(&wc);
                if (n < 0) return k.fail(5);
                for (wc[0..@intCast(n)]) |w| {
                    if (w.status != 0) {
                        k.logOldest(&turns, k.net.statusText(w.status));
                        return k.fail(nic.statusCode(w.status));
                    }
                    if (w.wr_id & poison_bit != 0) continue;
                    k.sends += 1;
                    const li = laneOfWr(w.wr_id);
                    if (li < nl and turns[li].active and qOfWr(w.wr_id) == turns[li].q and w.peer < l.world) turns[li].have_send[w.peer] = true;
                }
                var moved = n > 0;
                n = k.net.pollRecv(&wc);
                if (n < 0) return k.fail(5);
                for (wc[0..@intCast(n)]) |w| {
                    if (w.status != 0) {
                        k.logOldest(&turns, k.net.statusText(w.status));
                        return k.fail(nic.statusCode(w.status));
                    }
                    k.recvs += 1;
                    if (w.with_imm and w.byte_len == 0) { // a peer failed and says so
                        k.poisoned_by = w.peer;
                        std.log.warn("collectives: rank {d} proxy: rank {d}'s collective failed (code {d}) during gather {d}", .{ k.rank, w.peer, w.imm, turns[0].q });
                        return k.fail(1);
                    }
                    const j = w.peer;
                    const li: usize = w.imm >> imm_lane_shift;
                    const ok_shape = w.with_imm and w.byte_len == nic.trailer_bytes and j < l.world and j != k.rank and li < nl;
                    const t = &turns[if (li < nl) li else 0];
                    if (ok_shape and t.active and w.imm == immOf(@intCast(li), t.q) and !t.have_recv[j]) {
                        t.have_recv[j] = true;
                        if (k.check(t.q, t.slot, t.site, t.bytes, t.partner, j)) |code| return k.fail(code);
                    } else if (ok_shape and t.active and w.imm == immOf(@intCast(li), t.q + 1) and t.have_recv[j] and !t.early[j]) {
                        t.early[j] = true;
                    } else if (ok_shape and !t.active and w.imm == immOf(@intCast(li), t.q) and !t.early[j]) {
                        t.early[j] = true;
                    } else {
                        std.log.warn("collectives: rank {d} proxy: a write from rank {d} during gather {d} (lane {d}) carries imm {d} and {d} bytes", .{ k.rank, j, t.q, li, w.imm, w.byte_len });
                        return k.fail(3);
                    }
                }
                if (n > 0) moved = true;
                // finish every lane whose sends and trailers are all in
                for (turns[0..nl], 0..) |*t, li| {
                    if (!t.active or !all(t.have_send[0..l.world]) or !all(t.have_recv[0..l.world])) continue;
                    if (k.warn_ns > 0 and wire.monotonicNs() -| t.t0 > k.warn_ns) k.logTurn(t.q, t.site, t.bytes, t.t0, "done (this rank's coll.collect waited for it)");
                    const ls: usize = @intCast(t.q % l.laneSlots());
                    @atomicStore(u64, &k.shared[li].slot[ls].done, t.q, .release);
                    _ = k.exchanged.fetchAdd(1, .monotonic);
                    t.active = false;
                    t.q += 1;
                    moved = true;
                }
                if (moved) last = wire.monotonicNs();
                if (k.stop.load(.acquire) != 0) return 0;
                if (k.abortedLane()) |li| {
                    tcp_proxy.deviceGaveUp(&k.shared[li], k.rank, turns[li].q, l.laneSlots(), "waiting for the completions of gather");
                    return k.fail(2);
                }
                spins +%= 1;
                if (spins % 256 == 0) {
                    const now = wire.monotonicNs();
                    if (now -| last > k.progress_ns) {
                        var missing: [160]u8 = undefined;
                        const t = oldest(&turns, nl);
                        k.logTurn(t.q, t.site, t.bytes, t.t0, missingText(&missing, t.have_send[0..l.world], t.have_recv[0..l.world]));
                        return k.fail(2);
                    }
                    if (spins % 4096 == 0) for (k.ctl_fds) |fd| {
                        if (fd >= 0 and peerGone(fd)) {
                            k.logOldest(&turns, "a peer closed its control socket (PeerLost)");
                            return k.fail(1);
                        }
                    };
                }
                std.atomic.spinLoopHint();
            }
        }

        /// The first lane whose abort word is up (the device gave up there, or the host raised it).
        fn abortedLane(k: *const K) ?usize {
            for (0..k.layout.lanes) |li| if (@atomicLoad(u32, &k.shared[li].abort, .acquire) != 0) return li;
            return null;
        }

        /// The active turn that started first (what a timeout or a failed completion names).
        fn oldest(turns: *const [max_lanes]Turn, nl: usize) *const Turn {
            var best: ?*const Turn = null;
            for (turns[0..nl]) |*t| if (t.active and (best == null or t.t0 < best.?.t0)) {
                best = t;
            };
            return best orelse &turns[0];
        }

        fn logOldest(k: *const K, turns: *const [max_lanes]Turn, outcome: []const u8) void {
            const t = oldest(turns, k.layout.lanes);
            k.logTurn(t.q, t.site, t.bytes, t.t0, outcome);
        }

        /// Lane li's staged gather t.q: this rank's block to whom it goes, its trailer to every peer, the peers'
        /// trailers that came early checked. Returns 0 or the failure code.
        fn start(k: *K, li: u32, t: *Turn) u32 {
            const l = k.layout;
            const q = t.q;
            const hs = &k.shared[li];
            const ls: usize = @intCast(q % l.laneSlots());
            const slot = l.slotIndex(li, q);
            const bytes: u64 = hs.slot[ls].bytes;
            const site: u32 = @truncate(hs.slot[ls].site);
            const xchg = Word.isExchange(site);
            const partner: ?u32 = if (xchg) Word.partner(site, k.rank) else null;
            if (xchg and partner == null) {
                std.log.warn("collectives: rank {d} proxy: gather {d} is an exchange that does not name this rank (pair {d}-{d})", .{ k.rank, q, Word.lo(site), Word.hi(site) });
                return k.fail(3);
            }
            const pos: u64 = if (partner) |p| l.exchangePos(k.rank, p) else @as(u64, Word.place(site, k.rank)) * bytes;
            if (pos + bytes > l.dataBytes() or (xchg and bytes > l.exchangeHalf())) {
                std.log.warn("collectives: rank {d} proxy: gather {d} says {d} bytes, more than its place in a slot carries", .{ k.rank, q, bytes });
                return k.fail(3);
            }
            const base: usize = slot * @as(usize, @intCast(l.slot_bytes));
            const tr_off: usize = @intCast(l.trailerOff(k.rank));
            const tr: *nic.Trailer = @ptrCast(@alignCast(k.rx[base + tr_off ..].ptr));
            tr.* = .{ .magic = nic.trailer_magic, .epoch = k.epoch, .q = q, .bytes = bytes, .site = site, .rank = k.rank };
            const t0 = wire.monotonicNs();
            const data = k.rx[base + @as(usize, @intCast(pos)) ..][0..@intCast(bytes)];
            t.slot = slot;
            t.site = site;
            t.bytes = bytes;
            t.partner = partner;
            t.t0 = t0;
            t.have_send = @splat(false);
            t.have_recv = @splat(false);
            t.have_send[k.rank] = true;
            t.have_recv[k.rank] = true;
            t.active = true;
            for (0..l.world) |jj| {
                const j: u32 = @intCast(jj);
                if (j == k.rank) continue;
                const to_j = bytes > 0 and (if (partner) |p| p == j else true);
                const seg: ?Seg = if (to_j) .{ .local = data, .remote_off = base + pos } else null;
                k.net.post(j, seg, .{ .local = k.rx[base + tr_off ..][0..nic.trailer_bytes], .remote_off = base + tr_off }, wrOf(q, li, j), immOf(li, q)) catch {
                    k.logTurn(q, site, bytes, t0, "posting the writes failed");
                    return k.fail(5);
                };
                if (t.early[j]) { // its trailer of this q came before this turn
                    t.early[j] = false;
                    t.have_recv[j] = true;
                    if (k.check(q, slot, site, bytes, partner, j)) |code| return k.fail(code);
                }
            }
            return 0;
        }

        /// Peer j's trailer of gather q against this rank's call: null when it agrees, else the failure code.
        fn check(k: *K, q: u64, slot: usize, site: u32, bytes: u64, partner: ?u32, j: u32) ?u32 {
            const base: usize = slot * @as(usize, @intCast(k.layout.slot_bytes));
            const got: *const nic.Trailer = @ptrCast(@alignCast(k.rx[base + @as(usize, @intCast(k.layout.trailerOff(j))) ..].ptr));
            const framed = got.magic == nic.trailer_magic and std.mem.eql(u8, &got.epoch, &k.epoch) and got.q == q and got.rank == j;
            const agrees = framed and if (partner) |p|
                (if (j == p)
                    got.site == site and Word.sizeHash(if (k.rank < p) bytes else got.bytes, if (k.rank < p) got.bytes else bytes) == (site & 0x7fff)
                else
                    Word.call(got.site) == Word.call(site) and Word.isExchange(got.site) and !Word.names(got.site, k.rank))
            else
                got.site == site and got.bytes == bytes;
            if (agrees) return null;
            std.log.warn("collectives: rank {d} proxy trailer mismatch at gather {d}: this rank's word {x} ({s}phase {d} layer {d}) bytes {d}, rank {d}'s gather {d} word {x} bytes {d}{s}", .{
                k.rank,                                                                  q,     site, if (partner != null) "exchange, " else "", Word.phaseOf(site), Word.layerOf(site), bytes, j, got.q, got.site, got.bytes,
                if (!framed) "; magic, epoch, gather or rank differs" else "",
            });
            return 3;
        }

        fn fail(k: *K, code: u32) u32 {
            for (0..k.layout.lanes) |li| tcp_proxy.raiseAbort(&k.shared[li]);
            k.poison(code);
            return code;
        }

        /// A zero-byte write with immediate to every peer (once), waiting a bounded time for their completions so a
        /// close right after does not flush them.
        pub fn poison(k: *K, code: u32) void {
            if (k.poisoned) return;
            k.poisoned = true;
            var posted: u32 = 0;
            for (0..k.layout.world) |jj| {
                const j: u32 = @intCast(jj);
                if (j == k.rank or (k.poisoned_by != null and k.poisoned_by.? == j)) continue;
                k.net.postPoison(j, code, poison_bit | j) catch continue;
                posted += 1;
            }
            const t0 = wire.monotonicNs();
            var seen: u32 = 0;
            var wc: [8]Cqe = undefined;
            while (seen < posted and wire.monotonicNs() -| t0 < 50 * std.time.ns_per_ms) {
                const n = k.net.pollSend(&wc);
                if (n < 0) return;
                for (wc[0..@intCast(n)]) |w| {
                    if (w.wr_id & poison_bit != 0) seen += 1;
                }
                std.atomic.spinLoopHint();
            }
        }

        fn logTurn(k: *const K, q: u64, site: u32, bytes: u64, t0: u64, outcome: []const u8) void {
            std.log.warn("collectives: rank {d} rdma proxy (world {d}) {s} {d} (phase {d} layer {d}, {d} bytes) after {d} ms: {s}", .{
                k.rank, k.layout.world, if (Word.isExchange(site)) "exchange" else "gather", q, Word.phaseOf(site), Word.layerOf(site), bytes, (wire.monotonicNs() -| t0) / std.time.ns_per_ms, outcome,
            });
        }
    };
}

fn all(v: []const bool) bool {
    for (v) |b| if (!b) return false;
    return true;
}

fn missingText(buf: []u8, have_send: []const bool, have_recv: []const bool) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    w.writeAll("Timeout: no progress within the deadline; waiting for") catch {};
    for (have_recv, 0..) |h, j| if (!h) w.print(" rank {d}'s write", .{j}) catch {};
    for (have_send, 0..) |h, j| if (!h) w.print(" the send to rank {d}", .{j}) catch {};
    return w.buffered();
}
