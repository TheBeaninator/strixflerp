//! One TP rank of DeepSeek-V4.1-Flash loaded for serving: the weights, kernels, caches and buffers the layer gate sets
//! up (layer_main.zig), for a pool of draft.max_streams streams (MultiDecoder's slots: each stream's window rings,
//! positional stores and drafter rings in its slot, its compressed rows in its extent of one shared plane), and the
//! primitives a lane round runs on them. Both ranks run the same primitives in the same order (lanes.zig sends them
//! to rank 1): a prompt's chunks into its stream's slot and extent (fill), one round over every stream's window and
//! the windows' taps absorbed (verify), and the drafter's batched pass (pass). Every primitive is the gates' own code
//! path (prompt.zig, round.zig, draft.zig) with no checks.
//!
//! TF_DS_PREFILL_ZIP2 (every rank alike; unset or 0: off, and nothing here differs): a prompt runs in units of two
//! chunks, chunk k in m.ch and chunk k + 1 in m.ch2, zipped layer by layer by fillSplit with parts of chunk_rows rows,
//! so the routed experts run once per chunk_rows rows instead of once per half of them while every gather still moves
//! behind the other chunk's next step. Kept states stay on the chunk_rows grid: the first chunk's state is copied
//! layer by layer into m.mid before the second chunk writes over it (midSave), and snapCopy hands that copy out for a
//! keep at that chunk end. World 4 takes the same settings in fillSplit2d. Unset, fillWhole runs and the launches
//! are unchanged.
const std = @import("std");
const gpu = @import("gpu");
const Config = @import("config.zig").Config;
const rank_cache = @import("rank_cache.zig");
const weights = @import("weights.zig");
const prompt = @import("prompt.zig");
const prompt2d = @import("prompt2d.zig");
const round = @import("round.zig");
const draft = @import("draft.zig");
const engram = @import("engram.zig");
const engram_io = @import("engram_io.zig");
const engram_aio = @import("engram_aio.zig");
const exl3_prefill = @import("exl3_prefill.zig");
const exl3_linear = @import("exl3_linear.zig");
const dense = @import("dense.zig");
const edit_mod = @import("edit.zig");
const exl3_experts = @import("exl3_experts.zig");
const ops_mod = @import("ops.zig");
const exact = @import("exact.zig");
const blas = @import("blas.zig");
const comm_mod = @import("comm.zig");
const coll = @import("coll.zig");
const fan_mod = @import("fan.zig");
/// HIP builds: kernel sets from the AMD kit (CUDA builds never reference it).
const hip_load = @import("hip_load.zig");
const reap = @import("reap.zig");
const reap_dev = @import("reap_dev.zig");
const tri_aux = @import("tri_aux.zig");
const link = @import("link.zig");
const ring2d = @import("ring2d.zig");
const rdma = @import("rdma.zig");

/// What a served request may carry (ADAPTER.md D13): text, or text with images when the vision tower is opened. The
/// checkpoint's config.json is never changed: a text model keeps its vision_config and simply never opens the tower.
pub const Modalities = enum { text, text_image };
/// CUDA keeps serving images when the checkpoint has a tower; HIP serves text unless asked (TF_DS_MODALITIES).
pub const default_modalities: Modalities = if (gpu.backend == .cuda and gpu.features.vision) .text_image else .text;

/// Backend state (gpu.features): NCCL behind the collectives (RCCL on HIP, comm.chosen), the four-node split, its RDMA
/// rings (CUDA only).
const Nccl = comm_mod.Comm;
const TwoNode = if (gpu.features.world4) prompt2d.Two else void;
const Rings2d = if (gpu.features.world4 and gpu.features.verbs_rdma) ring2d.Rings else void;
const HipKit = if (gpu.backend == .hip) *gpu.Kit else void;

/// The served prompt chunk (TF_DS_PREFILL_CHUNK): a prompt runs in chunks that start at its multiples.
pub const chunk_rows = 2048;

/// The version of lanes.zig's frames, rank 0 to rank 1 (the handshake compares it): 2 since fill_chunk carries rank 0's
/// read-ahead plan; 3 since pass carries draft sampling keys; 5 since a pass may pause; 6 since an import has frames;
/// 7 since fill_begin's pictures (upstream PR 22 numbers it 6).
pub const frame_protocol = 6;

/// The prompt chunk buffers' rows: one of chunk_rows, or two halves when chunks split (TF_DS_PREFILL_SPLIT).
pub fn chunkCaps(split_min: usize) [2]usize {
    return if (split_min > 0) .{ chunk_rows / 2, chunk_rows / 2 } else .{ chunk_rows, 0 };
}

/// The most positions one stream reaches in a pool of `pool` (Options.context; 0: the pool), at most the pool.
pub fn contextCap(pool: usize, context: usize) error{BadPool}!usize {
    if (pool == 0) return error.BadPool;
    if (context == 0) return pool;
    if (context > pool) return error.BadPool;
    return context;
}

test "the context defaults to the pool and never exceeds it" {
    try std.testing.expectEqual(@as(usize, 8192), try contextCap(8192, 0));
    try std.testing.expectEqual(@as(usize, 1048527), try contextCap(2097152, 1048527));
    try std.testing.expectEqual(@as(usize, 8192), try contextCap(8192, 8192));
    try std.testing.expectError(error.BadPool, contextCap(8192, 8193));
    try std.testing.expectError(error.BadPool, contextCap(0, 0));
}
/// The pool's streams (--parallel 4).
pub const max_streams = draft.max_streams;
/// The pool's streams when unset (--parallel 4).
pub const max_streams_default = draft.default_streams;
/// The RDMA ring's slot (the served 4.25 MiB: a 16-row round's logits half fits).
pub const ring_bytes = 4456448;

pub const Options = struct {
    model_dir: []const u8, // config.json
    cache_dir: []const u8, // the lane's per-rank weight file (TF_DS_RANK_CACHE)
    kit_dir: []const u8, // a gate dir's layout: aot/, cubins/, rope-{plain,compressed}-{cos,sin}.f32, engram.json
    aot_dir: ?[]const u8 = null, // the recorded Triton set (aot.json and its cubins; null: kit_dir/aot)
    rank: u32,
    world: u32,
    master: [4]u8,
    port: u16,
    engram_dir: ?[]const u8 = null, // the Engram tables (none: a checkpoint without Engram layers)
    token_map: ?[]const u8 = null, // the compressed token map (the lane's JSON cache)
    pool: usize = 1 << 18, // positions of the streams' shared plane (their extents together)
    // one stream's most positions (--context; 0: the pool): the RoPE rows, chunk keys and context buckets
    context: usize = 0,
    drafts: bool = true, // load the DSpark drafter
    rdma_devices: ?[]const u8 = null, // the decode gathers over RDMA on these devices (comma separated), else NCCL: TP2's
    // ring (rdma.zig), or at world 4 the 2D split's decode-size exchanges over its rings (ring2d.zig)
    rdma_kernels: ?[]const u8 = null, // world 4: the rings' kernels' image (rdma_gather.cu's fatbin; null: the embedded one)
    graphs: bool = false, // the rounds' stretches as CUDA graphs (round.Graphs)
    side: bool = false, // the mixes' side work on a stream of its own (round.Round.useSide)
    prefetch: bool = false, // the paced L2 prefetch (round.Round.usePrefetch)
    engram_aio: bool = false, // a round's Engram reads by Linux AIO on O_DIRECT (else the reader pool)
    phase_timer: bool = false, // usePhaseTimer may be called (lanes' --profile 2): a timed chunk runs its halves apart,
    // so the pair handshake compares it
    arena_bytes: usize = 0, // the device arena's first block (0: it grows from 256 MiB blocks as the buffers ask)
    round_rows: usize = round.default_rows, // a round's rows at most (TF_DS_ROUND_ROWS; the four-node lane's 48)
    streams: usize = draft.default_streams, // the pool's streams (--parallel; the four-node lane's 16)
    round_ms: ?[]const f64 = null, // the lane core's round costs by rows (TF_DS_ROUND_MS; null: the served ROUND_MS)
    draft_ms: f64 = draft.draft_ms, // a drafter pass's cost, measured with round_ms on the same backend (TF_DS_DRAFT_MS)
    vision_bias: ?[]const u8 = null, // gate_bias_vl.safetensors (null: the kit's vision/): the image-span routing bias
    modalities: Modalities = default_modalities, // .text: the image routing bias is never loaded (native.zig: the tower)
    // text_image on HIP: a request's image tokens at most, the rows both ranks reserve at the open (serve.conf
    // IMAGE_TOKENS: TF_DS_IMAGE_TOKENS on rank 0, --image-tokens on rank 1); another than the default is in the plan
    image_rows: usize = Model.image_rows_reserved,
    open_coll: ?coll.Open = null, // the collectives' transport (HIP: w-link's TCP; null on CUDA: NCCL over the links)
    hip_kit: ?[]const u8 = null, // HIP: the AMD kit (manifest.json); null: kit_dir/hip
    disk: bool = false, // sf-disk on (disk.zig): the handshake's plan says " disk=1", so both ranks must agree
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    host: std.heap.ArenaAllocator, // allocations that live as long as the model
    io: std.Io,
    ctx: *const gpu.Context,
    stream: gpu.Stream,
    cfg: Config,
    rank: u32,
    world: u32,
    streams: usize, // the pool's streams (Options.streams)
    round_ms: []const f64, // the lane core's round costs by rows (Options.round_ms, else draft.round_ms)
    draft_ms: f64, // a drafter pass's cost (Options.draft_ms): with round_ms, the profile lanes and chooseK use
    pool_cap: usize,
    context_cap: usize, // the most positions one stream reaches (Options.context; at most pool_cap)
    cache_dir: std.Io.Dir,
    cache_file: std.Io.File,
    ix: rank_cache.Index,
    w: weights.Weights,
    fan: fan_mod.Fan,
    nccl: Nccl, // the communicator behind `coll` when no transport is handed in (CUDA NCCL, HIP RCCL)
    coll: coll.Coll, // the TP collectives the engine calls
    coll_timer: coll.Timer, // TF_DS_COLL_TIME on rank 0: coll.timer points here (off: coll.timer null, this unused)
    staging: gpu.Staging, // host bytes to the device (prompt.upload)
    kit: HipKit, // HIP: the AMD kit every kernel comes from
    aux_which: tri_aux.Which = tri_aux.Which.triton, // HIP: TF_DS_AUX, which Triton launches take w-tri's HIP kernels
    aux: tri_aux.Aux = .{}, // their kernels (Engine.t.aux)
    dk: ?dense.Kernels = null, // HIP: dense.hip's kernels, when the kit has them (Engine.dk)
    edit_id: []const u8 = "", // TF_DS_EDIT's file (edit.zig): its SHA-256's first 16 hex characters, "" when none
    rdma_mod: ?gpu.Module = null, // the RDMA kernels (TP2's ring or world 4's rings)
    ring: ?*prompt.Ring = null,
    set: gpu.aot.Set,
    pf: exl3_prefill.Kernels,
    lg: exl3_linear.Kernels,
    exk: exl3_experts.Kernels,
    ops: ops_mod.Ops,
    ex: exact.Exact,
    rope_buf: gpu.DeviceBuffer,
    arena: prompt.Arena,
    eng: prompt.Engine,
    ch: prompt.Chunk,
    // TF_DS_PREFILL_SPLIT: a chunk's second half (fillSplit; ch then holds the first), and the fewest rows of a first
    // half worth splitting for (0: off)
    ch2: ?prompt.Chunk = null,
    /// Edge-REAP device state. Null unless TF_DS_REAP_STATS=1 (then every rank holds one).
    reap: ?*reap_dev.State = null,
    split_min: usize = 0,
    // TF_DS_ENGRAM_ROWS=0: fillSplit's Engram phase 0 moves the fp32 projections of the peer's columns, as before the
    // rows exchange (prompt.engramPeerLoad; the same bits either way)
    engram_rows: bool = true,
    split_zip: bool = true, // TF_DS_PREFILL_SPLIT_ZIP=0: the halves one after the other (fillWhole), to compare with
    // TF_DS_PREFILL_ZIP2 (every rank alike; 0 off). 1: a prompt unit is two whole chunks zipped by fillSplit. 2: the
    // second may also be the prompt's last, partial chunk (at least max(split_min, 256) rows). Needs the split and its
    // zip.
    zip2: u8 = 0,
    // zip2: the first chunk's kept state of the unit fillSplit ran last (snapBytes long, snapCopy's layout), the
    // position it stands at (0: none) and its slot; snap_mid: the next snapCopy save takes it, not the live state
    mid: u64 = 0,
    mid_at: usize = 0,
    mid_slot: usize = 0,
    snap_mid: bool = false,
    // TF_DS_WINDOW_LOGITS=DIR (every rank alike): the replay window's teacher-forced logits of each prompt, written by
    // rank 0 (dumpWindow); the head's columns and their gather for 16 rows, and the prompts dumped so far
    window_dir: ?[]const u8 = null,
    window_l: ?gpu.DeviceBuffer = null,
    window_g: ?gpu.DeviceBuffer = null,
    window_k: usize = 0,
    caches: prompt.Caches,
    blas: blas.Blas,
    rings: []u64, // each layer's window rings, max_streams slots
    ring_view: []u64, // a filling stream's slot of them
    rd: round.Round,
    graphs: ?round.Graphs = null,
    pgraphs: ?draft.PassGraphs = null, // the drafter pass as CUDA graphs (with graphs)
    dpool: ?draft.Pool = null,
    dr: ?draft.Drafter = null,
    tables: ?engram_io.Tables = null,
    epool: ?*engram_io.Pool = null,
    aio: ?*engram_aio.Aio = null,
    paio: ?*engram_aio.Aio = null, // the prompt chunks' Engram reads, on a ring of their own
    eh: ?prompt.EngramHost = null,
    host_pos: []i64,
    ids64: []i64,
    vocab: usize,
    logits: []f32, // round.max_rows rows of logits on the host
    amax: u64, // u32 [round.max_rows]: a round's greedy tokens on the device (roundArgmax)
    amax_host: [round.max_rows]u32 = undefined,
    // the stream a fill is running: its slot and its view of the caches (its extent from base)
    fill_slot: usize = 0,
    // the prompt a fill's chunks come from (the caller's, set for the fill) and the thread reading the next chunk's
    // Engram rows during this one (fillChunk), joined before the prompt can change (joinAhead)
    fill_seq: []const i32 = &.{},
    ahead: ?std.Thread = null,
    // the chunk before this one planned this chunk's Engram rows read-ahead (set where it plans it, whether or not its
    // thread started; cleared by joinAhead and by the chunk taking it): the same on every rank, so fillSplit decides
    // the order of its collectives on it, never on the thread's progress
    ahead_planned: bool = false,
    view: prompt.Caches = undefined,
    // the filling prompt's image spans and their rows (setImages; cleared by fillBegin)
    spans: std.ArrayList(prompt.Span) = .empty,
    img_buf: ?gpu.DeviceBuffer = null,
    image_rows: usize = image_rows_reserved, // the rows img_buf holds on HIP (Options.image_rows): a request's cap
    // profiling (lanes' --profile): the last round's forward and absorb, synchronized apart (ns)
    prof: bool = false,
    timer: ?*round.PhaseTimer = null, // a profile's GPU time by phase (eager rounds)
    ptimer: ?*round.PhaseTimer = null, // the same for prompt chunks (fills), a chunk a 'round'
    phase_timer: bool = false, // Options.phase_timer: the handshake said this rank may time its phases
    t_enqueue: u64 = 0, // the host's time to issue the forward (the GPU idle before it)
    // Model.useClock: the rounds' GPU clock gaps summed (ns; gap i = clock i + 1 - clock i: the forward's start to
    // the first stretch, then each stretch, then the wait before the next), and the rounds counted
    clk_ph: [256]u64 = @splat(0), // by the phase of the clock that ends it
    clk_cnt: [256]u64 = @splat(0),
    clk_rounds: u64 = 0,
    t_forward: u64 = 0,
    t_absorb: u64 = 0,
    // sf-disk's identity (disk.zig): the handshake's plan text and the inputs beyond the bundle
    plan_text: []const u8 = "",
    kit_dir: []const u8 = "",
    engram_dir: ?[]const u8 = null,
    token_map: ?[]const u8 = null,
    // world 4: the 2D split's exchanges (prompt2d.zig, TP4's), with the RDMA rings when asked
    two: ?TwoNode = null,
    rdma_rings: ?Rings2d = null,

    /// Loads rank o.rank's weights from the lane's rank cache, links the ranks (rank 0 listens on o.port), opens NCCL
    /// and sets up every buffer. Needs `ctx` current on this thread.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const gpu.Context, o: Options) !*Model {
        if (reap.statsOn() and gpu.backend != .hip) return error.MissingKernel;
        if (o.world != 2 and o.world != 4) return error.BadWorld; // TP2, or the four-node 2D split (prompt2d.zig)
        if (o.world == 4 and !gpu.features.world4) return error.BadWorld; // a build without the 2D split (prompt2d.zig)
        if (o.rdma_devices != null and !gpu.features.verbs_rdma) return error.NoVerbsRdma;
        if (o.prefetch and gpu.backend != .hip and !gpu.features.l2_prefetch) return error.NoL2Prefetch;
        if (o.modalities == .text_image and !gpu.features.vision) return error.VisionUnsupported;
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = undefined;
        m.gpa = gpa;
        m.aux_which = tri_aux.Which.triton;
        m.aux = .{};
        m.dk = null;
        m.edit_id = ""; // (m.* is undefined: field defaults do not apply)
        if (std.c.getenv("TF_DS_REPLAY_FLOOR")) |v| prompt.replay_floor = std.mem.eql(u8, std.mem.span(v), "1"); // every rank alike
        m.spans = .empty;
        m.img_buf = null;
        m.host = std.heap.ArenaAllocator.init(gpa);
        errdefer m.host.deinit();
        const a = m.host.allocator();
        m.io = io;
        m.ctx = ctx;
        m.rank = o.rank;
        m.world = o.world;
        m.plan_text = "";
        m.kit_dir = try a.dupe(u8, o.kit_dir);
        m.engram_dir = if (o.engram_dir) |e| try a.dupe(u8, e) else null;
        m.token_map = if (o.token_map) |t| try a.dupe(u8, t) else null;
        if (o.streams == 0 or o.streams > max_streams) return error.BadStreams;
        m.streams = o.streams;
        const profile: draft.scheduler_costs.Profile = .{ .round_ms = o.round_ms orelse &draft.round_ms, .draft_ms = o.draft_ms };
        try profile.validate();
        m.round_ms = try a.dupe(f64, profile.round_ms);
        m.draft_ms = profile.draft_ms;
        m.pool_cap = o.pool;
        m.context_cap = try contextCap(o.pool, o.context);
        m.fill_slot = 0;
        m.prof = false;
        m.timer = null;
        m.ptimer = null;
        m.t_enqueue = 0;
        m.t_forward = 0;
        m.t_absorb = 0;
        m.clk_ph = @splat(0);
        m.clk_cnt = @splat(0);
        m.clk_rounds = 0;
        m.dpool = null;
        m.dr = null;
        m.tables = null;
        m.epool = null;
        m.aio = null;
        m.paio = null;
        m.eh = null;
        m.rdma_mod = null;
        m.ring = null;
        m.graphs = null;
        m.pgraphs = null;
        m.two = null;
        m.rdma_rings = null;
        m.ch2 = null;
        m.reap = null;
        // every rank alike (the plan text carries it)
        m.split_min = if (std.c.getenv("TF_DS_PREFILL_SPLIT")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch return error.BadSetting else 0;
        m.split_zip = if (std.c.getenv("TF_DS_PREFILL_SPLIT_ZIP")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
        m.engram_rows = if (std.c.getenv("TF_DS_ENGRAM_ROWS")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
        // (empty as unset; zip2 without the split and its zip is refused, here, before anything loads)
        m.zip2 = if (std.c.getenv("TF_DS_PREFILL_ZIP2")) |v| (if (std.mem.span(v).len == 0) 0 else std.fmt.parseInt(u8, std.mem.span(v), 10) catch return error.BadSetting) else 0;
        if (m.zip2 > 2 or (m.zip2 > 0 and (m.split_min == 0 or !m.split_zip))) return error.BadSetting;
        m.mid = 0;
        m.mid_at = 0;
        m.mid_slot = 0;
        m.snap_mid = false;
        m.phase_timer = o.phase_timer;
        m.window_dir = if (std.c.getenv("TF_DS_WINDOW_LOGITS")) |v| std.mem.span(v) else null;
        m.window_l = null;
        m.window_g = null;
        m.window_k = 0;
        const d = ctx.d;
        m.cfg = try Config.read(a, io, o.model_dir);
        const c = &m.cfg;
        m.stream = try gpu.Stream.init(d, true);
        errdefer m.stream.deinit();
        // 32 MiB: a split prompt half's Engram rows (n x the projection's K bf16, 6 MiB at 1,024 rows) go up in one
        // piece behind the queued GPU work; a 4 MiB ring cut them into 2 MiB pieces whose third waited for the first to be
        // copied, i.e. for the whole queued step, with the host blocked and the GPU then idle (rank 1 trace on 7aa9b40:
        // 1-8 ms before the second half's early Engram projection)
        // (zip2: a zipped whole chunk's Engram rows are 12 MiB a piece, two of them queued: 64 MiB)
        m.staging = try gpu.Staging.init(d, if (m.zip2 > 0) 64 << 20 else 32 << 20);
        errdefer m.staging.deinit();
        // HIP: the AMD kit first (ADAPTER.md 7.4: every code object checked against its manifest, every kernel this
        // configuration reaches resolved), so a refused kit fails before the weights load, and its manifest's hash is
        // there for the peers' handshake
        if (gpu.backend == .hip) {
            const caps = try ctx.caps();
            m.kit = try gpu.Kit.open(gpa, io, d, o.hip_kit orelse try std.fs.path.join(a, &.{ o.kit_dir, "hip" }), caps.archName(), .{ .drafts = o.drafts, .engram = c.engram_layers.slice().len > 0, .graphs = o.graphs, .vision = o.modalities == .text_image and o.rank == 0 });
            m.aux_which = try tri_aux.Which.fromEnv();
        }
        errdefer if (gpu.backend == .hip) m.kit.deinit();

        // the weights (and the DSpark blocks) from the lane's rank cache
        m.cache_dir = try std.Io.Dir.cwd().openDir(io, o.cache_dir, .{ .iterate = true });
        errdefer m.cache_dir.close(io);
        const cache_name = try rank_cache.find(a, io, m.cache_dir, o.rank, o.world);
        m.ix = try rank_cache.open(gpa, io, m.cache_dir, cache_name);
        errdefer m.ix.deinit();
        m.cache_file = try m.cache_dir.openFile(io, cache_name, .{});
        errdefer m.cache_file.close(io);
        const sp = try fan_mod.split(o.rank, o.world);
        // TF_DS_EDIT: an edit of blocks' wo_b and shared w2 outputs (edit.zig), loaded with the weights; not on 2D
        var edits: ?edit_mod.File = null;
        defer if (edits) |*x| x.close(io);
        if (std.c.getenv("TF_DS_EDIT")) |v| if (std.mem.span(v).len > 0) {
            if (sp.pair != null) return error.NotPortedYet;
            edits = try edit_mod.File.open(gpa, io, std.mem.span(v));
            try edits.?.check(c.layers, c.hidden);
            m.edit_id = try a.dupe(u8, edits.?.sha256[0..16]);
        };
        m.w = try weights.load(gpa, d, .{ .cache = .{ .file = m.cache_file, .index = &m.ix, .io = io } }, c.*, sp, o.drafts, if (edits) |*x| x else null);
        errdefer m.w.deinit();
        if (o.drafts and m.w.dspark == null) return error.NoDrafter;

        // the ranks' links (rank 0 listens: rank 1 retries its connect while rank 0 loads), then the collectives: a
        // transport handed in (HIP with TF_DS_COLL=tcp, the two-rank default: w-link's host-staged TCP, after the peers'
        // handshake on coll.Identity), else NCCL over the links (RCCL on HIP: comm.chosen)
        m.fan = try fan_mod.Fan.open(io, o.master, o.port, o.rank, o.world);
        errdefer m.fan.close();
        if (o.open_coll) |open_coll| {
            var plan: [512]u8 = undefined;
            const id = try m.identity(o, &plan);
            m.plan_text = try a.dupe(u8, id.plan);
            m.coll = try open_coll(gpa, io, ctx, if (gpu.backend == .hip) m.kit else null, &m.fan, &id);
        } else {
            m.nccl = try comm_mod.Comm.fromFan(&m.fan, o.rank, o.world);
            m.coll = m.nccl.coll();
        }
        errdefer m.coll.close();
        if (o.graphs and !m.coll.capturable) return error.GraphsNeedCapturableCollective;
        // TF_DS_COLL_TIME (coll.Timer): every rank checks the value, rank 0 times its calls (lanes logs them)
        if (try coll.Timer.parseMode(if (std.c.getenv("TF_DS_COLL_TIME")) |v| std.mem.span(v) else null)) |mode| {
            if (o.graphs) return error.CollTimeNeedsEagerRounds; // a replayed graph makes no host call to time
            if (o.rank == 0) {
                m.coll_timer = .{ .mode = mode };
                m.coll.timer = &m.coll_timer;
            }
        }
        // TP2: the decode gathers' RDMA ring (both ranks connect before either returns)
        if (gpu.features.verbs_rdma and sp.pair == null and o.rdma_devices != null) {
            const list = o.rdma_devices.?;
            if (!gpu.kernels.available) return error.NoKernelImages;
            m.rdma_mod = try gpu.Module.load(d, gpu.kernels.dsv41_rdma);
            var devices: std.ArrayList([]const u8) = .empty;
            var it = std.mem.splitScalar(u8, list, ',');
            while (it.next()) |dev| try devices.append(a, dev);
            m.ring = try openRing(gpa, d, try rdma.Kernels.load(m.rdma_mod.?), devices.items, o.rank, &m.coll, m.stream);
        }
        errdefer if (gpu.features.verbs_rdma) if (m.ring) |r| r.destroy(gpa);
        errdefer if (m.rdma_mod) |*x| x.unload();

        if (gpu.backend == .cuda) {
            // kernels: the recorded Triton set, the served extension cubins, the torch-op images
            m.set = try gpu.aot.Set.load(gpa, io, d, ctx.device, o.aot_dir orelse try std.fs.path.join(a, &.{ o.kit_dir, "aot" }));
            errdefer m.set.deinit();
            m.pf = try exl3_prefill.Kernels.load(d, try readKit(a, io, o.kit_dir, "linear.cubin"));
            errdefer m.pf.unload();
            m.lg = try exl3_linear.Kernels.load(d, try readKit(a, io, o.kit_dir, "linear_grouped.cubin"));
            errdefer m.lg.unload();
            m.exk = try exl3_experts.Kernels.load(d, try readKit(a, io, o.kit_dir, "experts.cubin"), try readKit(a, io, o.kit_dir, "experts_cb.cubin"), exl3_linear.codebook_mul1);
            errdefer m.exk.unload();
            if (!gpu.kernels.available) return error.NoKernelImages;
            m.ops = try ops_mod.Ops.load(d, gpu.kernels.torch_pointwise, gpu.kernels.torch_movement, gpu.kernels.dsv41_ops);
            errdefer m.ops.unload();
            m.ex = try exact.Exact.load(d, gpu.kernels.dsv41_torch);
        } else {
            // from the kit opened above: the Triton variants and the helpers, never resolved mid-request
            const caps = try ctx.caps();
            m.set = try gpu.aot.Set.fromKit(gpa, m.kit);
            errdefer m.set.deinit();
            m.ops = try hip_load.ops(m.kit, caps.wall_clock_hz);
            m.ex = try hip_load.exact(m.kit);
            m.pf = try hip_load.exl3Prefill(m.kit);
            m.lg = try hip_load.exl3Linear(m.kit);
            m.exk = try hip_load.exl3Experts(m.kit);
            m.aux = try hip_load.aux(m.kit, m.aux_which);
            var nb: [32]u8 = undefined;
            std.log.info("dsv41: w-tri's HIP kernels in place of Triton: {s} (TF_DS_AUX)", .{m.aux_which.name(&nb)});
            // every specialization these weights reach, now rather than at its first launch (the sets hold only the
            // kit's functions: the set's errdefer above is all there is to undo)
            try hip_load.coverage(&m.w, &m.pf, &m.lg, &m.exk);
            m.dk = try hip_load.dense(m.kit);
        }
        errdefer m.unloadKernels();
        // the paced L2 prefetch needs its kernel: CUDA's tf_ds_l2_prefetch_kernel, on HIP the kit's exl3.l2_prefetch
        // (a touch into the 32 MB memory-side cache); a kit without it refuses TF_DS_L2_PREFETCH here
        if (o.prefetch and m.ops.l2_prefetch == null) return error.NoL2Prefetch;
        // a checkpoint with dense projections (a fine-tuned pack's wo_b / shared w2) needs dense.hip's kernels: refused
        // here, not at its first round (a CUDA build has none)
        for (m.w.layers) |lay| if ((lay.wo_b.dense != 0 or (lay.shared != null and lay.shared.?.w2.dense != 0)) and m.dk == null) {
            std.log.err("layer {d} has a dense projection and the kit has no dense.gemv / dense.swiglu", .{lay.idx});
            return error.MissingKernel;
        };
        // an edited projection (TF_DS_EDIT) needs dense.hip's edit too (a kit built before it has none)
        var n_edits: usize = 0;
        for (m.w.layers) |lay| {
            const edited = lay.wo_b_edit != null or (if (lay.shared) |sh| sh.edit != null else false);
            n_edits += @as(usize, @intFromBool(lay.wo_b_edit != null)) + @intFromBool(if (lay.shared) |sh| sh.edit != null else false);
            if (edited and (m.dk == null or m.dk.?.edit == null)) {
                std.log.err("layer {d} is edited (TF_DS_EDIT) and the kit has no dense.edit", .{lay.idx});
                return error.MissingKernel;
            }
        }
        if (m.edit_id.len > 0) std.log.info("dsv41: TF_DS_EDIT {s}: {d} projections edited", .{ m.edit_id, n_edits });

        // the RoPE tables (the served build's bits: torch CPU pow and polar) for every position a stream reaches
        const half = c.rope_dim / 2;
        const rope_rows = m.context_cap + round.max_rows + 1;
        m.rope_buf = try gpu.DeviceBuffer.alloc(d, 4 * rope_rows * half * 4);
        errdefer m.rope_buf.free();
        {
            const part = try gpa.alloc(u8, rope_rows * half * 4);
            defer gpa.free(part);
            for ([_][]const u8{ "rope-plain-cos.f32", "rope-plain-sin.f32", "rope-compressed-cos.f32", "rope-compressed-sin.f32" }, 0..) |name, j| {
                var f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ o.kit_dir, name }), .{});
                defer f.close(io);
                // a table of fewer rows than the context
                if (try f.readPositionalAll(io, part, 0) != part.len) return error.ShortRopeTable;
                try m.rope_buf.upload(j * part.len, part);
            }
        }
        const tbl = rope_rows * half * 4;
        const rope: prompt.Rope = .{ .cos = m.rope_buf.ptr, .sin = m.rope_buf.ptr + tbl };
        const rope_c: prompt.Rope = .{ .cos = m.rope_buf.ptr + 2 * tbl, .sin = m.rope_buf.ptr + 3 * tbl };

        m.arena = try prompt.Arena.init(d, o.arena_bytes);
        errdefer m.arena.deinit();
        // world 4 (layer_main's setup): the 2D exchanges' buffers, and with rdma_devices their rings (whose infos go
        // round over NCCL world 4)
        errdefer if (gpu.features.world4 and gpu.features.verbs_rdma) if (m.rdma_rings) |*x| x.close(gpa);
        if (gpu.features.world4 and sp.pair != null) {
            m.two = try prompt2d.Two.init(c, &m.w, &m.arena, sp, chunk_rows);
            // the rings need verbs RDMA (a HIP build has none: open refused rdma_devices above, ring2d compiles out)
            if (gpu.features.verbs_rdma) if (o.rdma_devices) |devs| {
                const image: []const u8 = if (o.rdma_kernels) |path|
                    try std.Io.Dir.cwd().readFileAllocOptions(io, path, a, .limited(1 << 26), .@"16", null)
                else if (gpu.kernels.available) gpu.kernels.dsv41_rdma else return error.NoRdmaKernels;
                m.rdma_mod = try gpu.Module.load(d, image);
                var devices: std.ArrayList([]const u8) = .empty;
                var dit = std.mem.splitScalar(u8, devs, ',');
                while (dit.next()) |x| try devices.append(a, x);
                m.rdma_rings = try ring2d.Rings.open(gpa, d, try rdma.Kernels.load(m.rdma_mod.?), devices.items, o.rank, .{ .max_bytes = ring_bytes, .gid_index = try rdma.servedGidIndex() }, &m.nccl, m.stream);
                m.two.?.rings = &m.rdma_rings.?;
            };
        }
        m.eng = .{ .d = d, .s = m.stream, .t = .{ .set = &m.set, .stream = m.stream, .aux = if (gpu.backend == .hip) &m.aux else null }, .blas = undefined, .comm = &m.coll, .pf = &m.pf, .lin = &m.lg, .ex = &m.exk, .ops = &m.ops, .exact = &m.ex, .c = c, .w = &m.w, .world = sp.world, .plain = rope, .compressed = rope_c, .ring = m.ring, .two = if (gpu.features.world4) (if (m.two) |*t| t else null) else null, .staging = &m.staging, .dk = if (m.dk) |*k| k else null };
        m.eng.round_rows = o.round_rows;
        // split prefill: two halves in the space of one whole chunk. World 4 runs them in fillSplit2d (each quarters
        // gather ends before catRank and before the Engram exchange). An Engram layer past the replay cut would need
        // those Engram steps after the cut, which neither zip does.
        if (m.split_min > 0) {
            for (c.engram_layers.slice()) |l| if (l >= c.layers / 2) return error.SplitEngramPastCut;
        }
        // (the handshake's chunk_caps; zip2: two whole chunks' buffers, its unit_caps)
        const caps = if (m.zip2 > 0) [2]usize{ chunk_rows, chunk_rows } else chunkCaps(m.split_min);
        // a chunk's compressed keys are its stream's (to the context); the caches hold every stream's extent (the pool)
        m.ch = try prompt.Chunk.init(&m.eng, &m.arena, caps[0], m.context_cap);
        if (caps[1] > 0) m.ch2 = try prompt.Chunk.init(&m.eng, &m.arena, caps[1], m.context_cap);
        if (m.ch2) |*c2| c2.reap_bank = 1;
        m.caches = try prompt.Caches.init(&m.eng, &m.arena, o.pool, m.streams);
        m.blas = try blas.Blas.open(m.stream, m.ch.blas_ws);
        errdefer m.blas.close();
        m.eng.blas = &m.blas;
        m.rings = try a.alloc(u64, c.layers);
        m.ring_view = try a.alloc(u64, c.layers);
        for (m.rings) |*r| {
            r.* = try m.arena.take(m.streams * m.eng.ringBytes());
            try gpu.fill8Async(d, m.stream.handle, r.*, 0, m.streams * m.eng.ringBytes());
        }
        if (m.zip2 > 0) m.mid = try m.arena.take(m.snapBytes()); // a zipped unit's first chunk's kept state
        m.rd = try round.Round.init(&m.eng, &m.arena, a, m.context_cap); // buckets to the context
        m.amax = try m.arena.take(round.max_rows * 4);
        if (o.graphs) {
            m.graphs = round.Graphs.init(gpa);
            if (std.c.getenv("TF_DS_GRAPHS_EAGER")) |v| m.graphs.?.eager_every = std.fmt.parseInt(u64, std.mem.span(v), 10) catch return error.BadSetting;
        }
        if (o.side) try m.rd.useSide(&m.eng);
        if (o.prefetch) try m.rd.usePrefetch(&m.eng);
        if (o.drafts) {
            m.dpool = try draft.Pool.init(&m.eng, &m.arena, m.streams);
            m.dr = try draft.Drafter.init(&m.eng, &m.arena, sp);
            if (o.graphs) {
                m.pgraphs = draft.PassGraphs.init(gpa);
                m.dr.?.graphs = &m.pgraphs.?;
            }
        }
        try m.stream.synchronize();
        try ctx.synchronize(); // the setup's legacy-stream copies too

        // Engram on the host: the compressed token map, the multipliers, the tables and 64 readers
        if (o.engram_dir) |edir| {
            const map_text = try std.Io.Dir.cwd().readFileAlloc(io, o.token_map orelse return error.NoTokenMap, a, .limited(1 << 26));
            const map = try std.json.parseFromSliceLeaky([]i32, a, map_text, .{});
            const EngramJson = struct { multipliers: [][]i64 };
            const ej_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ o.kit_dir, "engram.json" }), a, .limited(1 << 26));
            const ej = try std.json.parseFromSliceLeaky(EngramJson, a, ej_text, .{ .ignore_unknown_fields = true });
            var mult: [engram.max_layers][engram.max_ngram]i64 = @splat(@splat(0));
            for (ej.multipliers, 0..) |row, l| for (row, 0..) |v, k| {
                mult[l][k] = v;
            };
            m.tables = try engram_io.Tables.open(gpa, io, edir);
            m.epool = try engram_io.Pool.init(gpa, io, 64);
            m.eh = try prompt.EngramHost.init(a, c, engram.Hasher.init(c.*, map, mult), &m.tables.?, m.epool.?, sp.rank, sp.world, m.unitCap()); // (a unit's rows are read ahead in one piece)
            if (o.engram_aio) {
                // a round's reads, 2 x 12 x its rows at most: 1,024 slots for each 16 rows of the round-row setting
                m.aio = try engram_aio.Aio.init(gpa, io, 1024 * ((o.round_rows + 15) / 16));
                m.eh.?.aio = m.aio;
                // a prompt chunk's reads (2 x 12 x 2,048 at most): a ring of 4,096 slots (32 MiB), 1,024 rows a batch
                m.paio = try engram_aio.Aio.init(gpa, io, 4096);
                m.eh.?.paio = m.paio;
            }
        } else if (c.engram_layers.slice().len > 0) return error.NoEngramTables;

        // a vision checkpoint's image routing bias (without it a prompt with images is refused; text runs as ever)
        if (gpu.features.vision and c.vision and o.modalities == .text_image) {
            const path = o.vision_bias orelse try std.fs.path.join(a, &.{ o.kit_dir, "vision", "gate_bias_vl.safetensors" });
            // HIP: both ranks need it (a rank without it would refuse the spans its peer routes): none starts without
            m.loadVisionBias(path) catch |err| if (gpu.backend == .hip or o.vision_bias != null or err != error.FileNotFound) return if (err == error.FileNotFound) error.NoVisionKit else err;
            // HIP: a request's image rows at most, from the start: an image prompt allocates nothing on either rank
            if (gpu.backend == .hip) {
                if (o.image_rows == 0 or o.image_rows > image_rows_max) return error.BadImageTokens;
                m.image_rows = o.image_rows;
                m.img_buf = try gpu.DeviceBuffer.alloc(m.ctx.d, o.image_rows * c.hidden * 2);
            }
        }

        // fillSplit's Engram rows exchange: the peer's projection of this rank's columns (both ranks alike)
        if (m.split_min > 0 and m.split_zip and m.engram_rows and m.eh != null and prompt.engramHalves(&m.eng)) try prompt.engramPeerLoad(&m.eng, &m.arena);
        m.host_pos = try a.alloc(i64, m.unitCap());
        m.ids64 = try a.alloc(i64, m.unitCap());
        m.vocab = sp.world * (if (gpu.features.world4) (if (m.two) |t| t.hw[0] + t.hw[1] else m.w.head.n) else m.w.head.n); // 2D: both pairs' parts
        m.logits = try a.alloc(f32, round.max_rows * m.vocab);
        if (reap.statsOn()) {
            if (gpu.backend != .hip) return error.MissingKernel;
            const bound = (try hip_load.reap(m.kit)) orelse return error.MissingKernel;
            m.reap = try reap_dev.open(gpa, io, d, &m.eng, .{ .pack = bound.pack, .ssq = bound.ssq, .route = bound.route, .norm = bound.norm });
            errdefer {
                reap_dev.close(m.reap.?);
                m.reap = null;
            }
            m.eng.reap = m.reap;
            m.eng.reap_picks = reap_dev.picksOf;
            m.eng.reap_down_y = reap_dev.downYOf;
            m.eng.reap_note = reap_dev.noteLayer;
        }
        return m;
    }

    pub fn close(m: *Model) void {
        m.joinAhead();
        m.stream.synchronize() catch {};
        if (m.reap) |st| {
            reap_dev.close(st);
            m.reap = null;
        }
        if (m.img_buf) |*b| b.free();
        if (m.window_l) |*b| b.free();
        if (m.window_g) |*b| b.free();
        m.spans.deinit(m.gpa);
        if (m.graphs) |*g| g.deinit();
        if (m.pgraphs) |*g| g.deinit();
        if (m.timer) |t| {
            t.deinit();
            m.gpa.destroy(t);
        }
        if (m.ptimer) |t| {
            t.deinit();
            m.gpa.destroy(t);
        }
        m.rd.dropSide(&m.eng);
        m.rd.dropPrefetch();
        if (m.aio) |x| x.deinit();
        if (m.paio) |x| x.deinit();
        if (m.epool) |p| p.deinit(m.gpa);
        if (m.tables) |*t| t.close();
        if (gpu.features.world4 and gpu.features.verbs_rdma) if (m.rdma_rings) |*r| r.close(m.gpa);
        m.blas.close();
        m.arena.deinit();
        m.rope_buf.free();
        m.unloadKernels();
        if (gpu.features.verbs_rdma) if (m.ring) |r| {
            r.stop();
            r.destroy(m.gpa);
        };
        if (m.rdma_mod) |*x| x.unload();
        m.coll.close();
        m.fan.close();
        m.w.deinit();
        m.cache_file.close(m.io);
        m.ix.deinit();
        m.cache_dir.close(m.io);
        if (gpu.backend == .hip) m.kit.deinit();
        m.staging.deinit();
        m.stream.deinit();
        m.host.deinit();
        m.gpa.destroy(m);
    }

    /// Every kernel set this rank holds, in reverse load order (on HIP the sets' functions are the kit's: their unload
    /// leaves its modules, which close() releases last).
    fn unloadKernels(m: *Model) void {
        m.ex.unload();
        m.ops.unload();
        m.exk.unload();
        m.lg.unload();
        m.pf.unload();
        m.set.deinit();
    }

    /// This rank's side of the peers' handshake (coll.Identity): the kit, the modalities, and the settings that decide
    /// which collectives it issues and their sizes, as text in `buf`.
    fn identity(m: *const Model, o: Options, buf: []u8) !coll.Identity {
        var id: coll.Identity = .{ .rank = o.rank, .world = o.world, .kit_abi = 0, .kit_sha256 = @splat('0'), .modalities = @intFromEnum(o.modalities), .plan = "" };
        if (gpu.backend == .hip) {
            id.kit_abi = gpu.kit.abi_version;
            id.kit_sha256 = m.kit.manifest_sha256;
        }
        // (not the rank caches: each rank's file name carries its own content key, rank{r}of{w}-<key>.bin, so no part
        // of it is common to the ranks; the inputs' one bundle is checked per box against its pinned manifest hash,
        // serve.conf BUNDLE_SHA256)
        var nb: [32]u8 = undefined;
        // and the served launch's identity (serve_lib.sh hands both ranks the same values; "-" off a served launch): the
        // deployed commit, the inputs bundle's manifest hash (the checkpoint's config, tokenizer and chat template, the
        // rank caches) and serve.conf's hash, each a 16-hex prefix; the KV caches' formats; strict mode
        const strict = if (gpu.backend == .hip) m.kit.strict else false;
        var ob: [128]u8 = undefined;
        var reap_buf: [32]u8 = undefined;
        var rcfg = reap.fromEnv(std.heap.page_allocator) catch reap.Config{};
        defer rcfg.deinit(std.heap.page_allocator);
        const reap_s = reap.planSuffix(&reap_buf, rcfg);
        id.plan = try planText(buf, .{
            .drafts = o.drafts,
            .engram = m.cfg.engram_layers.slice().len > 0,
            .graphs = o.graphs,
            .pool = o.pool,
            .context = m.context_cap,
            .streams = o.streams,
            .round_rows = o.round_rows,
            .replay_floor = prompt.replay_floor,
            .aux = m.aux_which.name(&nb),
            .gather = try gatherOrder(&ob, m.split_min, m.split_zip, m.phase_timer),
            .engram_rows = m.engram_rows,
            .deploy = launchId("TF_DS_DEPLOY"),
            .bundle = launchId("TF_DS_BUNDLE"),
            .conf = launchId("TF_DS_CONF"),
            .strict = strict,
            .disk = o.disk,
            .edit = m.edit_id,
            .zip2 = m.zip2,
            .image_rows = if (o.modalities == .text_image) o.image_rows else image_rows_reserved,
            .reap = reap_s,
        });
        return id;
    }

    pub const PlanParts = struct {
        drafts: bool,
        engram: bool,
        graphs: bool,
        pool: usize,
        context: usize = 0, // " context=N" after pool= only when below the pool (0: the pool)
        streams: usize,
        round_rows: usize,
        replay_floor: bool,
        aux: []const u8,
        gather: []const u8,
        engram_rows: bool,
        deploy: []const u8,
        bundle: []const u8,
        conf: []const u8,
        strict: bool,
        disk: bool, // sf-disk: " disk=1" at the end, only when on (the text is the stock one when off)
        edit: []const u8 = "", // TF_DS_EDIT's file by its hash: " edit=<16 hex>" at the end, only when set
        // TF_DS_PREFILL_ZIP2: " prefill_zip2=N unit_caps=R/R" at the end, only when on (both ranks must zip the same
        // units; off, the text is the stock one)
        zip2: u8 = 0,
        // text_image: the image rows each rank reserved, " image_rows=N" at the end only when not the default (a rank
        // with fewer would end the pair at the first request between the two; the text is the stock one by default)
        image_rows: usize = Model.image_rows_reserved,
        /// " reap=EVERY/PREFILL/BUCKETS" when REAP stats are on, else "". Peers that differ are refused.
        reap: []const u8 = "",
    };

    /// The handshake's plan text (identity), in `buf`.
    pub fn planText(buf: []u8, p: PlanParts) ![]u8 {
        var cb: [32]u8 = undefined;
        const ctx = if (p.context != 0 and p.context != p.pool) try std.fmt.bufPrint(&cb, " context={d}", .{p.context}) else "";
        var zb: [48]u8 = undefined;
        const zip = if (p.zip2 > 0) try std.fmt.bufPrint(&zb, " prefill_zip2={d} unit_caps={d}/{d}", .{ p.zip2, chunk_rows, chunk_rows }) else "";
        var ib: [32]u8 = undefined;
        const img = if (p.image_rows != Model.image_rows_reserved) try std.fmt.bufPrint(&ib, " image_rows={d}", .{p.image_rows}) else "";
        return std.fmt.bufPrint(buf, "drafts={} engram={} graphs={} pool={d}{s} streams={d} round_rows={d} chunk_rows={d} replay_floor={} aux={s} {s} engram_rows={} deploy={s} bundle={s} conf={s} kv={s} strict={}{s}{s}{s}{s}{s}{s}", .{
            p.drafts,       p.engram,                      p.graphs,                             p.pool,        ctx,      p.streams, p.round_rows, chunk_rows,
            p.replay_floor, p.aux,                         p.gather,                             p.engram_rows, p.deploy, p.bundle,  p.conf,       kv_format,
            p.strict,       if (p.disk) " disk=1" else "", if (p.edit.len > 0) " edit=" else "", p.edit,        zip,
            img,            p.reap,
        });
    }

    /// The settings that decide the order of a prompt chunk's gathers, for the handshake (two ranks that differ in them
    /// would issue the same gathers in different orders, which the wire sees as a stall or a Mismatch only later): the
    /// split, the zip, the chunk buffers' rows, the phase timer (a timed chunk runs its halves apart) and the frames'
    /// protocol.
    pub fn gatherOrder(buf: []u8, split_min: usize, split_zip: bool, phase_timer: bool) ![]u8 {
        const caps = chunkCaps(split_min);
        return std.fmt.bufPrint(buf, "prefill_split={d} prefill_zip={} chunk_caps={d}/{d} phase_timer={} frames={d}", .{ split_min, split_zip, caps[0], caps[1], phase_timer, frame_protocol });
    }

    /// The KV caches' formats (fixed in this build): the window rings bf16 after an FP8 quant-dequant, the compressed
    /// latents and the indexer's keys FP4 with scales.
    const kv_format = "win-fp8qd,cmp-fp4,idx-fp4";

    /// A launch identity from the environment (launchWord).
    fn launchId(name: [*:0]const u8) []const u8 {
        return launchWord(if (std.c.getenv(name)) |v| std.mem.span(v) else null);
    }

    /// At most 16 characters up to the first space (one key=value token of the plan), "-" when unset or empty.
    fn launchWord(v: ?[]const u8) []const u8 {
        const s = v orelse return "-";
        const end = @min(16, std.mem.indexOfScalar(u8, s, ' ') orelse s.len);
        return if (end == 0) "-" else s[0..end];
    }

    /// TP2's RDMA ring for the decode gathers: created on both ranks, the queue-pair infos exchanged over NCCL,
    /// connected, and started once both are (an all-gather as the barrier).
    pub fn openRing(gpa: std.mem.Allocator, d: *const gpu.Driver, k: rdma.Kernels, devices: []const []const u8, rank: u32, comm: *const coll.Coll, stream: gpu.Stream) !*rdma.Ring {
        if (comm.world != 2) return error.NotTwoRanks;
        const r = try rdma.Ring.create(gpa, d, k, devices, rank, 2, .{ .max_bytes = ring_bytes, .gid_index = try rdma.servedGidIndex() });
        errdefer r.destroy(gpa);
        const per = @sizeOf(rdma.Info);
        const mine = [2]rdma.Info{ r.info(0), r.info(1) };
        var dev = try gpu.DeviceBuffer.alloc(d, 3 * 2 * per);
        defer dev.free();
        try dev.upload(0, std.mem.sliceAsBytes(&mine));
        try comm.allGather(.at(.ring_setup, 0), dev.ptr, dev.ptr + 2 * per, 2 * per, .u8, stream);
        try stream.synchronize();
        var all: [2][2]rdma.Info = undefined;
        try dev.download(2 * per, std.mem.sliceAsBytes(&all));
        try r.connect(&.{ all[0][rank], all[1][rank] });
        // every queue pair ready on both ranks before either sends
        try comm.allGather(.at(.ring_setup, 1), dev.ptr, dev.ptr + 2 * per, 1, .u8, stream);
        try stream.synchronize();
        try r.start();
        return r;
    }

    fn readKit(a: std.mem.Allocator, io: std.Io, kit: []const u8, name: []const u8) ![]const u8 {
        return std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ kit, "cubins", name }), a, .limited(1 << 28), .@"16", null);
    }

    /// The link to the other rank (rank 0's to rank 1, rank 1's to rank 0).
    pub fn peer(m: *const Model) link.Link {
        return m.fan.links[0].?;
    }

    /// Rank 0's links to its followers: rank 1's, or ranks 1-3's on the four-node split.
    pub fn followers(m: *const Model) []const ?link.Link {
        return m.fan.links[0 .. m.world - 1];
    }

    pub fn drafting(m: *const Model) bool {
        return m.dr != null;
    }

    // -- the primitives (both ranks, in the same order) ----------------------------------------------------------

    /// A stream's prompt is about to fill pool slot `slot`, its compressed rows in its extent from position `base`.
    /// Wait for the next chunk's Engram read-ahead, if one runs (before the prompt it reads changes or goes).
    pub fn joinAhead(m: *Model) void {
        if (m.ahead) |th| th.join();
        m.ahead = null;
        m.ahead_planned = false;
    }

    /// Whether the chunk at `start` of n rows has its Engram rows planned by the chunk before (a split chunk then reads
    /// its first Engram layer's rows a layer early, which moves that layer's gathers). Rank 0 sends it with the chunk:
    /// a rank's own plan is reset by the frames it gets between chunks and would order the gathers differently.
    pub fn preRead(m: *const Model, start: usize, n: usize) bool {
        return m.ahead_planned and m.eh != null and m.eh.?.ahead_start == start and m.eh.?.ahead_n == n;
    }

    /// A kept prompt state's own bytes (the prompt cache's snapshots; the compressed plane stays in the stream's
    /// extent): one slot's window ring of every layer and positional stores of every ratio-2 layer.
    pub fn snapBytes(m: *const Model) usize {
        const pb = prompt.raw_rows * m.eng.c.head_dim * 4;
        var n: usize = m.rings.len * m.eng.ringBytes();
        for (m.caches.raw_kv[0..m.eng.c.layers], m.caches.raw_score[0..m.eng.c.layers]) |k, sc| {
            if (k != 0) n += pb;
            if (sc != 0) n += pb;
        }
        return n;
    }

    /// Slot `slot`'s own state into `buf` (save: true) or back from it, in the stream's order (no wait): with the
    /// extent's compressed rows, the state a prompt pass stands in at a chunk end.
    pub fn snapCopy(m: *Model, slot: usize, buf: u64, save: bool) !void {
        const rb = m.eng.ringBytes();
        const pb = prompt.raw_rows * m.eng.c.head_dim * 4;
        // zip2: a keep at a zipped unit's first chunk end (lanes.zig sets snap_mid on both ranks) takes the state
        // midSave copied before the second chunk wrote over the live one: the same bytes in the same layout
        if (m.snap_mid) {
            if (!save or m.mid == 0 or m.mid_at == 0 or m.mid_slot != slot) return error.NoMidState;
            return gpu.copyD2DAsync(m.ctx.d, m.stream.handle, buf, m.mid, m.snapBytes());
        }
        var at = buf;
        for (m.rings) |r| {
            try m.copyState(r + slot * rb, at, rb, save);
            at += rb;
        }
        for (m.caches.raw_kv[0..m.eng.c.layers], m.caches.raw_score[0..m.eng.c.layers]) |k, sc| {
            for ([_]u64{ k, sc }) |live| if (live != 0) {
                try m.copyState(live + slot * pb, at, pb, save);
                at += pb;
            };
        }
    }

    /// The compressed rows of positions [0, at) of the extent at `from` copied into the extent at `to` (a kept prompt
    /// resumed in another extent while a live stream holds its own), in the stream's order: every kv-source layer's
    /// codes and scales and its index keys, the rows view() places at those positions.
    pub fn copyExtent(m: *Model, from: usize, to: usize, at: usize) !void {
        const c = m.eng.c;
        const cs = &m.caches;
        for (m.eng.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            const r: usize = lay.ratio;
            if (from % r != 0 or to % r != 0 or at % r != 0) return error.BadView;
            const n = at / r;
            try m.copySpan(cs.comp_codes[i], from / r, to / r, n, c.head_dim / 2);
            try m.copySpan(cs.comp_scales[i], from / r, to / r, n, c.head_dim / 16);
            if (lay.idx_wk != null) {
                try m.copySpan(cs.idx_codes[i], from / r, to / r, n, c.index_head_dim / 2);
                try m.copySpan(cs.idx_scales[i], from / r, to / r, n, c.index_head_dim / 32);
            }
        }
    }

    fn copySpan(m: *Model, plane: u64, from: usize, to: usize, n: usize, row: usize) !void {
        const d = m.ctx.d;
        try gpu.copyD2DAsync(d, m.stream.handle, plane + to * row, plane + from * row, n * row);
    }

    fn copyState(m: *Model, live: u64, kept: u64, bytes: usize, save: bool) !void {
        const d = m.ctx.d;
        const dst = if (save) kept else live;
        const src = if (save) live else kept;
        try gpu.copyD2DAsync(d, m.stream.handle, dst, src, bytes);
    }

    /// Each layer's gate_bias_vl (f32 [experts], the checkpoint's extra file) on the device: Gate.forward's bias at
    /// image-span positions.
    fn loadVisionBias(m: *Model, path: []const u8) !void {
        const core = @import("core");
        var f = try core.safetensors.File.open(m.gpa, m.io, path);
        defer f.close(m.io);
        var nb: [64]u8 = undefined;
        for (m.w.layers, 0..) |*lay, i| {
            const t = f.get(try std.fmt.bufPrint(&nb, "layers.{d}.ffn.gate.bias_vl", .{i})) orelse return error.MissingVisionBias;
            if (!t.is(.f32, &.{m.cfg.experts})) return error.BadVisionBias;
            const dst = try m.arena.take(t.bytes.len);
            const tmp = try m.gpa.dupe(u8, t.bytes); // (a copy out of the file mapping first: see vit.zig)
            defer m.gpa.free(tmp);
            try gpu.copyH2D(m.ctx.d, dst, tmp);
            lay.gate_b_vl = dst;
        }
    }

    /// The image rows a text_image rank keeps on the device from its open unless the launch asks for another count
    /// (four pictures of a whole span). Options.image_rows (serve.conf IMAGE_TOKENS) sets it; each row is bf16 [hidden],
    /// 10 KiB a rank, so 32,768 rows are 320 MiB on each rank.
    pub const image_rows_reserved = 4096;
    /// The most image rows a launch may reserve (64 pictures of a whole span: 640 MiB a rank).
    pub const image_rows_max = 65536;
    /// A request's image rows at most (HIP: what its open reserved; CUDA grows its buffer instead).
    pub fn imageRowsCap(m: *const Model) ?usize {
        return if (gpu.backend == .hip) m.image_rows else null;
    }
    /// IMAGE_TOKENS as text (TF_DS_IMAGE_TOKENS, rank 1's --image-tokens): a whole number from 1 to image_rows_max.
    pub fn parseImageRows(text: []const u8) error{BadImageTokens}!usize {
        const n = std.fmt.parseInt(usize, text, 10) catch return error.BadImageTokens;
        if (n == 0 or n > image_rows_max) return error.BadImageTokens;
        return n;
    }

    /// The filling prompt's image spans (absolute positions, in order) and their rows bf16 [tokens, hidden] from the
    /// host, after fillBegin and before its chunks.
    pub fn setImages(m: *Model, spans: []const prompt.Span, rows: []const u8) !void {
        if (rows.len > 0) {
            if (m.img_buf == null or m.img_buf.?.len < rows.len) {
                if (gpu.backend == .hip) return error.TooManyImageRows; // HIP never reallocates (both ranks alike)
                if (m.img_buf) |*b| b.free();
                m.img_buf = null;
                m.img_buf = try gpu.DeviceBuffer.alloc(m.ctx.d, rows.len);
            }
            // after the work already queued (it may read the previous prompt's rows), then a blocking copy: the
            // caller's rows may go once this returns
            try m.stream.synchronize();
            try gpu.copyH2D(m.ctx.d, m.img_buf.?.ptr, rows);
        }
        m.spans.clearRetainingCapacity();
        try m.spans.appendSlice(m.gpa, spans);
        m.ch.spans = m.spans.items;
        m.ch.img_rows = if (m.img_buf) |b| b.ptr else 0;
    }

    /// The rows the unit buffers hold: a chunk, or two under zip2.
    pub fn unitCap(m: *const Model) usize {
        return if (m.zip2 > 0) 2 * chunk_rows else chunk_rows;
    }

    /// The rows of the prompt unit at start of a prompt of len tokens (replay: len - the window), which lanes.zig
    /// hands to fillChunk on rank 0 and sends to rank 1: a chunk (@min(chunk_rows, len - start)), or under zip2 two
    /// chunks when the first is whole and encoder-only and the second is whole (zip2 = 2: or the prompt's last, of at
    /// least max(split_min, 256) rows, so its gathers stay the size class the zip's begin / end has always carried). A
    /// profile's chunks run one at a time, as its halves do.
    pub fn unitRows(m: *const Model, start: usize, len: usize, replay: usize) usize {
        const one = @min(chunk_rows, len - start);
        if (m.zip2 == 0 or m.split_min == 0 or !m.split_zip or m.ptimer != null or m.ch2 == null) return one;
        if (start % chunk_rows != 0 or len - start <= chunk_rows or start + chunk_rows > replay) return one;
        const rest = len - start - chunk_rows;
        if (rest >= chunk_rows) return 2 * chunk_rows;
        if (m.zip2 >= 2 and rest >= @max(m.split_min, 256)) return chunk_rows + rest;
        return one;
    }

    /// Where layer li's positional stores sit in a kept state (snapCopy's layout): after every layer's ring, behind the
    /// live stores of the layers before it. li = the layer count: the state's end (snapBytes).
    fn storesAt(raw_kv: []const u64, raw_score: []const u64, li: usize, rings_bytes: usize, pb: usize) usize {
        var at = rings_bytes;
        for (raw_kv[0..li], raw_score[0..li]) |k, sc| {
            if (k != 0) at += pb;
            if (sc != 0) at += pb;
        }
        return at;
    }

    /// zip2: layer li's part of the filling slot's kept state (its window ring and, where the layer has them, its
    /// positional stores) into m.mid at snapCopy's offsets, in the stream's order: fillSplit calls it for a zipped unit
    /// once the first chunk has written that layer's state and before the second chunk does.
    fn midSave(m: *Model, li: usize) !void {
        const rb = m.eng.ringBytes();
        const pb = prompt.raw_rows * m.eng.c.head_dim * 4;
        const layers = m.eng.c.layers;
        try m.copyState(m.ring_view[li], m.mid + li * rb, rb, true);
        var at = m.mid + storesAt(m.caches.raw_kv[0..layers], m.caches.raw_score[0..layers], li, m.rings.len * rb, pb);
        for ([_]u64{ m.caches.raw_kv[li], m.caches.raw_score[li] }) |live| if (live != 0) {
            try m.copyState(live + m.fill_slot * pb, at, pb, true);
            at += pb;
        };
    }

    pub fn fillBegin(m: *Model, slot: usize, base: usize) !void {
        if (slot >= m.streams) return error.BadSlot;
        m.mid_at = 0;
        m.spans.clearRetainingCapacity();
        m.ch.spans = &.{};
        m.view = try m.caches.view(&m.eng, slot, base);
        for (m.rings, m.ring_view) |r, *v| v.* = r + slot * m.eng.ringBytes();
        m.fill_slot = slot;
    }

    /// The prompt's chunk of n rows from position `start` (seq: the stream's ids through the chunk's end) through every
    /// layer with the served bounded replay (replay = the prompt's length - the window: an encoder-only chunk stops at
    /// the decoder's first layer). A chunk that reaches the head leaves the logits of its last row in ch.head_g and,
    /// drafting, its DSpark taps absorbed into the slot's drafter rings. True when it reached the head. pre_read: rank
    /// 0's preRead for this chunk, which its frame carries to every rank (it orders a split chunk's Engram gathers).
    pub fn fillChunk(m: *Model, seq: []const i32, start: usize, n: usize, replay: usize, pre_read: bool) !bool {
        if (n == 0 or n > m.unitCap() or seq.len != start + n) return error.BadChunk;
        m.mid_at = 0;
        const cut = try m.cutOf(start, n, replay);
        if (cut.s == 0) return m.fillWhole(seq, start, n, replay);
        if (m.ptimer == null and m.split_zip) return m.fillSplit(seq, start, n, cut.s, replay, pre_read, cut.unit);
        // a profile times the halves one after the other, each with fillChunk's phases (and without the zip they run
        // so, for comparing its results)
        _ = try m.fillWhole(seq[0 .. start + cut.s], start, cut.s, replay);
        return m.fillWhole(seq, start + cut.s, n - cut.s, replay);
    }

    /// How fillChunk runs a unit: whole (s = 0), or as two parts, the first of s rows. unit: a zip2 unit, whose parts
    /// are its two chunks (the first in m.ch and whole, the second in m.ch2).
    const Cut = struct { s: usize = 0, unit: bool = false };

    /// fillChunk's cut of the n rows at start: a unit of more than a chunk (unitRows chose it on rank 0) is cut at its
    /// first chunk's end; a chunk is cut where the split cuts it (splitAt, by the split's own halves: under zip2 the
    /// buffers are larger and the cut is not), or runs whole.
    fn cutOf(m: *const Model, start: usize, n: usize, replay: usize) !Cut {
        if (n > chunk_rows) {
            if (m.zip2 == 0 or m.ch2 == null or m.ptimer != null or !m.split_zip or start % chunk_rows != 0 or start + chunk_rows > replay) return error.BadChunk;
            return .{ .s = chunk_rows, .unit = true };
        }
        if (m.split_min > 0) {
            const caps = chunkCaps(m.split_min);
            const s = splitAt(n, replay -| start, caps[0], caps[1]) orelse 0;
            if (s >= m.split_min or n > caps[0]) {
                if (s == 0) return error.ChunkTooLong;
                return .{ .s = s };
            }
        }
        return .{};
    }

    /// Where fillSplit cuts a chunk of n rows whose first `enc` rows are before the replay cut: near n / 2, the first
    /// half encoder-only (<= enc) and each half within its chunk's rows; on a multiple of 128 rows (the window) when
    /// one fits, else of 4 (the compressor's ratio; even, so the second half's cut keeps its parity and the positions
    /// view after it stays 16-byte aligned). Null when no cut fits.
    fn splitAt(n: usize, enc: usize, cap_a: usize, cap_b: usize) ?usize {
        const hi = @min(enc, cap_a);
        const lo = n -| cap_b;
        const mid = @min(n / 2, hi);
        if (mid / 128 * 128 >= lo and mid / 128 * 128 > 0) return mid / 128 * 128;
        if (mid / 4 * 4 >= lo and mid / 4 * 4 > 0) return mid / 4 * 4;
        const up = (lo + 3) / 4 * 4;
        return if (up > 0 and up <= hi) up else null;
    }

    /// fillChunk on one Chunk (m.ch), every gather in place.
    fn fillWhole(m: *Model, seq: []const i32, start: usize, n: usize, replay: usize) !bool {
        const e = &m.eng;
        const c = e.c;
        const ch = &m.ch;
        // an image position's id is negative in the sequence (its n-grams dead for Engram): its embedding's id is the
        // image token's, which its span row then replaces
        const image_id: i64 = if (c.image_token) |t| t else 0;
        for (0..n) |j| m.ids64[j] = if (seq[start + j] < 0) image_id else seq[start + j];
        var shared: prompt.Shared = .{};
        const pt = m.ptimer;
        if (pt) |t| try t.mark(m.stream, .start);
        try prompt.begin(e, ch, m.ids64[0..n], start, m.host_pos);
        try m.reapBegin(ch);
        if (ch.spans.len > 0) try prompt.embedSpans(e, ch);
        if (pt) |t| try t.mark(m.stream, .embed);
        const taps = m.dr != null;
        var floor: usize = 0;
        var kv_done: ?usize = null;
        // Engram rows read ahead on a thread while the GPU runs the layers before them (the served reader pool's
        // overlap): this chunk's both layers' during the chunk before it (m.ahead), else the first layer's from the
        // chunk's start and the next's once the first is applied
        var ahead: ?std.Thread = m.ahead;
        m.ahead = null;
        m.ahead_planned = false;
        defer if (ahead) |th| th.join();
        if (m.eh) |*eh| {
            if (ahead != null and (eh.ahead_start != start or eh.ahead_n != n)) {
                ahead.?.join();
                ahead = null;
            }
            if (ahead == null) {
                eh.ahead_start = start;
                eh.ahead_n = n;
                eh.ahead_ready = .{ false, false };
                eh.ahead_err = null;
                ahead = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, @as(usize, 0) }) catch null;
            }
        }
        for (0..c.layers) |li| {
            if (li == c.layers / 2) {
                if (!try prompt.replayCut(e, ch, &m.view, &shared, li, replay, m.host_pos)) {
                    try m.reapFinish(ch, li);
                    // encoder-only: no synchronize either (as at the chunk's end below), so the next chunk's host steps
                    // run while this one's last layers do; the profile alone waits
                    if (pt) |t| {
                        try m.stream.synchronize();
                        try t.finish();
                    }
                    return false;
                }
                floor = replay;
                kv_done = li;
            }
            if (e.w.layers[li].engram_wkv != null) {
                const eh = if (m.eh) |*x| x else return error.NoEngramTables;
                if (ahead) |th| {
                    th.join();
                    ahead = null;
                }
                if (eh.ahead_err) |err| return err;
                try prompt.engramApply(e, ch, eh, li, seq);
                const j = std.mem.indexOfScalar(u16, c.engram_layers.slice(), @intCast(li)).?;
                const ne = @min(2, c.engram_layers.slice().len);
                if (j + 1 < ne) {
                    if (!eh.ahead_ready[j + 1])
                        ahead = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, j + 1 }) catch null;
                } else if (m.fill_seq.len > start + n and m.fill_seq.ptr == seq.ptr) {
                    // the next chunk's rows while this one's layers run
                    const nn = m.unitRows(start + n, m.fill_seq.len, m.fill_seq.len -| c.window); // (the next unit: a chunk, or two zipped)
                    eh.ahead_start = start + n;
                    eh.ahead_n = nn;
                    eh.ahead_ready = .{ false, false };
                    eh.ahead_err = null;
                    m.ahead_planned = true;
                    m.ahead = std.Thread.spawn(.{}, prompt.engramReadAheadAll, .{ eh, c, m.fill_seq[0 .. start + n + nn], start + n, nn }) catch null;
                }
                if (pt) |t| try t.mark(m.stream, .engram);
            }
            try prompt.attnMixes(e, ch, li);
            // a DSpark tap reads the streams after the layer's Engram (_forward_k's order), once the mixes have posted
            // the pending MoE gather into them, as a round takes it: the prompt's arithmetic is then the same whether or
            // not the drafter is loaded (a tap before the mixes posted on its own, which the fused mixes do not match bit
            // for bit, U52)
            if (taps) if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| try prompt.tap(e, ch, j);
            if (pt) |t| try t.mark(m.stream, .mix_attn);
            try prompt.attention(e, ch, &m.view, &shared, li, m.ring_view[li], floor, kv_done != null and kv_done.? == li);
            if (pt) |t| try t.mark(m.stream, .attn);
            try prompt.gather(e, ch, ch.attnOut(e), ch.ga, .at(.prompt_attn, li));
            if (pt) |t| try t.mark(m.stream, .gather_a);
            try prompt.ffnMixes(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .mix_ffn);
            try prompt.moe(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .moe);
            try prompt.gather(e, ch, ch.moeOut(e), ch.gm, .at(.prompt_moe, li));
            if (pt) |t| try t.mark(m.stream, .gather_m);
            prompt.endLayer(ch);
        }
        try m.reapFinish(ch, c.layers);
        try m.dumpWindow(ch, seq);
        try prompt.head(e, ch);
        if (pt) |t| try t.mark(m.stream, .head);
        if (m.dr) |*dr| {
            // its taps (the tap layers side by side) into the slot's drafter rings
            const dw = c.hidden;
            for (0..c.dspark_taps.slice().len) |j| try e.ops.copyRows(e.s, prompt.tapRows(e, ch, j), dw * 2, dr.at + j * dw * 2, dr.taps_w * 2, dw * 2, ch.n);
            try dr.absorb(e, ch, &m.dpool.?, m.fill_slot, dr.at, ch.n, ch.start);
        }
        // no synchronize: the next chunk's host steps (its first Engram layer's hashing, table reads and decode, ~0.14 s
        // at 2,048 rows) run while this chunk's kernels do; readers of the results (the prompt's logits) synchronize.
        // The profile (--profile 2) times a chunk to its end, so it alone waits.
        if (pt) |t| {
            try m.stream.synchronize();
            try t.finish();
        }
        return true;
    }

    /// fillSplit, once both halves' Engram rows of layer li are uploaded: the next Engram layer's rows, or the next
    /// chunk's, may be read now (on a thread).
    fn engramAheadNext(m: *Model, li: usize, seq: []const i32, start: usize, n: usize, ahead: *?std.Thread) !void {
        const c = m.eng.c;
        const eh = &m.eh.?;
        const j = std.mem.indexOfScalar(u16, c.engram_layers.slice(), @intCast(li)).?;
        const ne = @min(2, c.engram_layers.slice().len);
        if (j + 1 < ne) {
            if (!eh.ahead_ready[j + 1])
                ahead.* = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, j + 1 }) catch null;
        } else if (m.fill_seq.len > start + n and m.fill_seq.ptr == seq.ptr) {
            const nn = m.unitRows(start + n, m.fill_seq.len, m.fill_seq.len -| c.window); // (the next unit: a chunk, or two zipped)
            eh.ahead_start = start + n;
            eh.ahead_n = nn;
            eh.ahead_ready = .{ false, false };
            eh.ahead_err = null;
            m.ahead_planned = true;
            m.ahead = std.Thread.spawn(.{}, prompt.engramReadAheadAll, .{ eh, c, m.fill_seq[0 .. start + n + nn], start + n, nn }) catch null;
        }
    }

    /// A chunk as two halves zipped layer by layer (TF_DS_PREFILL_SPLIT): rows [start, start + s) in m.ch, always
    /// encoder-only (s <= replay - start), and [start + s, start + n) in m.ch2. Every step of a half (its Engram rows,
    /// its gate and attention, its FFN) is followed by the other half's previous gather's end and its own gather's begin
    /// (coll.Coll.gatherBegin / gatherEnd), so each gather moves while the GPU runs the other half's next step; the
    /// steps of a layer run A then B, which keeps the caches' and rings' order. At the replay cut the first half stops
    /// and the second runs on alone as fillWhole does. The rows are fillWhole's (as two chunks of s and n - s rows).
    /// unit: the halves are a zip2 unit's two chunks (s = chunk_rows); the first chunk's state at its end is then saved
    /// as it forms, layer by layer (midSave), for a keep at that chunk end.
    fn fillSplit(m: *Model, seq: []const i32, start: usize, n: usize, s: usize, replay: usize, pre_read: bool, unit: bool) !bool {
        // world 4: each half's wo_b and expert down call catRank, so the TP2 zip (one gather left open across the
        // other half's whole step) cannot run. fillSplit2d ends every quarters gather before catRank and Engram.
        if (comptime gpu.features.world4) {
            if (m.two != null) return m.fillSplit2d(seq, start, n, s, replay, pre_read, unit);
        }
        const e = &m.eng;
        const c = e.c;
        const hs = [2]*prompt.Chunk{ &m.ch, &m.ch2.? };
        const image_id: i64 = if (c.image_token) |t| t else 0;
        for (0..n) |j| m.ids64[j] = if (seq[start + j] < 0) image_id else seq[start + j];
        hs[1].spans = hs[0].spans;
        hs[1].img_rows = hs[0].img_rows;
        try prompt.begin(e, hs[0], m.ids64[0..s], start, m.host_pos[0..s]);
        try m.reapBegin(hs[0]);
        try prompt.begin(e, hs[1], m.ids64[s..n], start + s, m.host_pos[s..n]);
        try m.reapBegin(hs[1]);
        for (hs) |ch| if (ch.spans.len > 0) try prompt.embedSpans(e, ch);
        var sh: [2]prompt.Shared = .{ .{}, .{} };
        const taps = m.dr != null;
        // the whole chunk's Engram rows read ahead as fillWhole reads them; each half takes its rows out of them
        // (pre_read, rank 0's preRead from the frame: the chunk before planned them, so they are likely read by now; it
        // moves the first Engram layer's gathers a layer early, so every rank takes rank 0's, never its own)
        m.ahead_planned = false;
        var ahead: ?std.Thread = m.ahead;
        m.ahead = null;
        defer if (ahead) |th| th.join();
        if (m.eh) |*eh| {
            if (ahead != null and (eh.ahead_start != start or eh.ahead_n != n)) {
                ahead.?.join();
                ahead = null;
            }
            if (ahead == null) {
                eh.ahead_start = start;
                eh.ahead_n = n;
                eh.ahead_ready = .{ false, false };
                eh.ahead_err = null;
                ahead = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, @as(usize, 0) }) catch null;
            }
        }
        // the one begun gather: the half queued last's
        var pend: ?coll.Coll.Pending = null;
        const Zip = struct {
            fn next(en: *const prompt.Engine, p: *?coll.Coll.Pending, site: coll.Site, src: u64, dst: u64, count: usize, dtype: coll.DType) !void {
                if (p.*) |q| try en.comm.gatherEnd(q, en.s);
                p.* = try en.comm.gatherBegin(site, src, dst, count, dtype, en.s);
            }
        };
        const cut = c.layers / 2;
        var tq: [2]u64 = .{ 0, 0 }; // a profile's Engram clocks (engramRows to engramGate)
        const halves = prompt.engramHalves(e); // the Engram exchange in two column-half gathers (TP2 on HIP)
        const EarlyEngram = struct {
            // halves: a half's Engram rows of layer li (the next layer's: they depend on the tokens only) uploaded and
            // projected, then its phase-0 gather begun (the step before's gather ends); the pending MoE post waits for
            // layer li itself
            fn rows(mm: *Model, eh: *prompt.EngramHost, ahead_: *?std.Thread, ch: *prompt.Chunk, li: usize, sq: []const i32, t: *u64, p: *?coll.Coll.Pending) !void {
                const en = &mm.eng;
                if (ahead_.*) |th| {
                    th.join();
                    ahead_.* = null;
                }
                if (eh.ahead_err) |err| return err;
                t.* = try prompt.engramRows(en, ch, eh, li, sq, true, false);
                const x = prompt.engramXchg(en, ch, li, 0);
                try Zip.next(en, p, .at(.prompt_engram, li), x.send, x.recv, x.count, x.dtype);
            }
        };
        var early_rows: usize = std.math.maxInt(usize); // the Engram layer whose rows the layer before read early
        for (0..cut) |li| {
            const lay = e.w.layers[li];
            // halves: the next layer's Engram projections and phase-0 gathers ride in this layer, A's after B's attention
            // step, B's after A's MoE step (each gather then moves while a MoE step runs); its sums and phase-1 gathers
            // open the Engram layer, so its gate waits on 2 x n x en / 2 bf16 instead of 2 x n x en fp32
            // (the first Engram layer's rows of a chunk not read ahead come from a read begun at the chunk's start,
            // ~0.1 s at 2,048 rows: that layer keeps them at its own start, where the layer before has covered more)
            const early = halves and li + 1 < cut and e.w.layers[li + 1].engram_wkv != null and
                (pre_read or li + 1 != c.engram_layers.slice()[0]);
            if (lay.engram_wkv) |ew| {
                const eh = if (m.eh) |*x| x else return error.NoEngramTables;
                if (early_rows != li) {
                    if (ahead) |th| {
                        th.join();
                        ahead = null;
                    }
                    if (eh.ahead_err) |err| return err;
                    for (hs, &tq) |ch, *t| {
                        t.* = try prompt.engramRows(e, ch, eh, li, seq, halves, true);
                        if (halves) {
                            const x = prompt.engramXchg(e, ch, li, 0);
                            try Zip.next(e, &pend, .at(.prompt_engram, li), x.send, x.recv, x.count, x.dtype);
                        } else try Zip.next(e, &pend, .at(.prompt_engram, li), ch.engramOut(e, ew.n), ch.ekg, ch.n * ew.n, .f32);
                    }
                    try m.engramAheadNext(li, seq, start, n, &ahead);
                }
                // halves: each half's bf16 sums once its phase-0 gather has ended, then their gather
                if (halves) for (hs) |ch| {
                    try prompt.engramSum(e, ch, li);
                    const x = prompt.engramXchg(e, ch, li, 1);
                    try Zip.next(e, &pend, .at(.prompt_engram, li), x.send, x.recv, x.count, x.dtype);
                };
            }
            for (hs, &sh, &tq) |ch, *shh, *t| {
                if (lay.engram_wkv != null) {
                    if (halves) try prompt.engramPost(e, ch); // (the early rows left the pending MoE post for here)
                    try prompt.engramGate(e, ch, &m.eh.?, li, if (halves) .halves else .whole, t);
                }
                try prompt.attnMixes(e, ch, li);
                if (taps) if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| try prompt.tap(e, ch, j); // (after the mixes, U52)
                try prompt.attention(e, ch, &m.view, shh, li, m.ring_view[li], 0, false);
                if (unit and ch == hs[0]) try m.midSave(li); // the layer's state as the first chunk leaves it (only attention writes it)
                try Zip.next(e, &pend, .at(.prompt_attn, li), ch.attnOut(e), ch.ga, ch.n * c.hidden, .f32);
            }
            if (early) {
                try EarlyEngram.rows(m, &m.eh.?, &ahead, hs[0], li + 1, seq, &tq[0], &pend);
                early_rows = li + 1;
            }
            for (hs, 0..) |ch, hi| {
                try prompt.ffnMixes(e, ch, li);
                try prompt.moe(e, ch, li);
                try Zip.next(e, &pend, .at(.prompt_moe, li), ch.moeOut(e), ch.gm, ch.n * c.hidden, .f32);
                prompt.endLayer(ch);
                if (early and hi == 0) {
                    try EarlyEngram.rows(m, &m.eh.?, &ahead, hs[1], li + 1, seq, &tq[1], &pend);
                    try m.engramAheadNext(li + 1, seq, start, n, &ahead);
                }
            }
        }
        // A's cut (its layer-cut mixes and kv source: A's own gathered rows only) while B's last MoE gather moves
        const a_past = try prompt.replayCut(e, hs[0], &m.view, &sh[0], cut, replay, m.host_pos);
        if (pend) |q| try e.comm.gatherEnd(q, e.s);
        if (a_past) return error.SplitPastReplay;
        try m.reapFinish(hs[0], cut);
        // zip2: the cut layer's stores as the first chunk's replayCut left them, and the decoder layers' state the
        // first chunk never touched, before the second chunk's cut and decoder write them
        if (unit) {
            for (cut..c.layers) |li| try m.midSave(li);
            m.mid_at = start + s;
            m.mid_slot = m.fill_slot;
        }
        const ch = hs[1];
        if (!try prompt.replayCut(e, ch, &m.view, &sh[1], cut, replay, m.host_pos)) {
            try m.reapFinish(hs[1], cut);
            return false;
        }
        for (cut..c.layers) |li| {
            try prompt.attnMixes(e, ch, li);
            if (taps) if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| try prompt.tap(e, ch, j); // (after the mixes, U52)
            try prompt.attention(e, ch, &m.view, &sh[1], li, m.ring_view[li], replay, li == cut);
            try prompt.gather(e, ch, ch.attnOut(e), ch.ga, .at(.prompt_attn, li));
            try prompt.ffnMixes(e, ch, li);
            try prompt.moe(e, ch, li);
            try prompt.gather(e, ch, ch.moeOut(e), ch.gm, .at(.prompt_moe, li));
            prompt.endLayer(ch);
        }
        try m.reapFinish(ch, c.layers);
        try m.dumpWindow(ch, seq);
        try prompt.head(e, ch);
        // the prompt's logits where promptLogits reads them
        try gpu.copyD2DAsync(e.d, e.s.handle, m.ch.head_g, ch.head_g, e.world * e.w.head.n * 4);
        if (m.dr) |*dr| {
            const dw = c.hidden;
            for (0..c.dspark_taps.slice().len) |j| try e.ops.copyRows(e.s, prompt.tapRows(e, ch, j), dw * 2, dr.at + j * dw * 2, dr.taps_w * 2, dw * 2, ch.n);
            try dr.absorb(e, ch, &m.dpool.?, m.fill_slot, dr.at, ch.n, ch.start);
        }
        return true;
    }

    /// World 4's fillSplit. The step list is zip2d.zig (a host test walks it). Half A is m.ch, half B is m.ch2, and
    /// each half's own math is fillWhole's. A quarters gather is the same allParts as prompt2d.quarters; it stays open
    /// only across the other half's attention prefix or its gate / up, and ends before catRank, the Engram exchange,
    /// or another gather. Past the replay cut the second half runs fillWhole's loop. Unset, fillChunk never calls this.
    fn fillSplit2d(m: *Model, seq: []const i32, start: usize, n: usize, s: usize, replay: usize, pre_read: bool, unit: bool) !bool {
        if (comptime !gpu.features.world4) return error.SplitNeedsTp2;
        return m.fillSplit2dLive(seq, start, n, s, replay, pre_read, unit);
    }

    fn fillSplit2dLive(m: *Model, seq: []const i32, start: usize, n: usize, s: usize, replay: usize, pre_read: bool, unit: bool) !bool {
        // TP2 moves an Engram phase early off this flag. 2D Engram is engramApply, and that exchange cannot sit inside
        // a begun quarters gather, so the flag is unused here. The ahead thread is the same one fillSplit starts.
        _ = pre_read;
        const e = &m.eng;
        const c = e.c;
        const t = if (m.two) |*p| p else unreachable;
        const hs = [2]*prompt.Chunk{ &m.ch, &m.ch2.? };
        const image_id: i64 = if (c.image_token) |tkn| tkn else 0;
        for (0..n) |j| m.ids64[j] = if (seq[start + j] < 0) image_id else seq[start + j];
        hs[1].spans = hs[0].spans;
        hs[1].img_rows = hs[0].img_rows;
        try prompt.begin(e, hs[0], m.ids64[0..s], start, m.host_pos[0..s]);
        try m.reapBegin(hs[0]);
        try prompt.begin(e, hs[1], m.ids64[s..n], start + s, m.host_pos[s..n]);
        try m.reapBegin(hs[1]);
        for (hs) |ch| if (ch.spans.len > 0) try prompt.embedSpans(e, ch);
        var sh: [2]prompt.Shared = .{ .{}, .{} };
        const taps = m.dr != null;
        m.ahead_planned = false;
        var ahead: ?std.Thread = m.ahead;
        m.ahead = null;
        defer if (ahead) |th| th.join();
        if (m.eh) |*eh| {
            if (ahead != null and (eh.ahead_start != start or eh.ahead_n != n)) {
                ahead.?.join();
                ahead = null;
            }
            if (ahead == null) {
                eh.ahead_start = start;
                eh.ahead_n = n;
                eh.ahead_ready = .{ false, false };
                eh.ahead_err = null;
                ahead = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, @as(usize, 0) }) catch null;
            }
        }
        var pend: ?prompt2d.Two.Begun = null;
        const Q = struct {
            fn close(tt: *const prompt2d.Two, en: *const prompt.Engine, p: *?prompt2d.Two.Begun) !void {
                if (p.*) |q| {
                    try tt.quartersEnd(en, q);
                    p.* = null;
                }
            }
        };
        const cut = c.layers / 2;
        for (0..cut) |li| {
            if (e.w.layers[li].engram_wkv != null) {
                // end_b_moe, then both halves' Engram. engramSum is an exchange, so the gather is already ended.
                try Q.close(t, e, &pend);
                const eh = if (m.eh) |*x| x else return error.NoEngramTables;
                if (ahead) |th| {
                    th.join();
                    ahead = null;
                }
                if (eh.ahead_err) |err| return err;
                try prompt.engramApply(e, hs[0], eh, li, seq);
                try prompt.engramApply(e, hs[1], eh, li, seq);
                try m.engramAheadNext(li, seq, start, n, &ahead);
            }
            // attn_mix_a, attn_prefix_a: overlaps B's MoE quarters when one is open (no Coll, and it does not touch parts)
            try prompt.attnMixes(e, hs[0], li);
            if (taps) if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| try prompt.tap(e, hs[0], j);
            try prompt.attentionPrefix(e, hs[0], &m.view, &sh[0], li, m.ring_view[li], 0, false);
            try Q.close(t, e, &pend); // end_b_moe (no-op when Engram already closed it, or on the first layer)
            try prompt.attentionSuffix(e, hs[0], li); // attn_suffix_a: catRank + wo_b
            if (unit) try m.midSave(li); // mid_save_a: the prefix has written this layer's ring
            pend = try t.quartersBegin(e, hs[0].attnOut(e), hs[0].ga, hs[0].n, t.ow, 4); // begin_attn_a
            try prompt.attnMixes(e, hs[1], li);
            if (taps) if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| try prompt.tap(e, hs[1], j);
            try prompt.attentionPrefix(e, hs[1], &m.view, &sh[1], li, m.ring_view[li], 0, false);
            try Q.close(t, e, &pend); // end_attn_a
            try prompt.attentionSuffix(e, hs[1], li);
            pend = try t.quartersBegin(e, hs[1].attnOut(e), hs[1].ga, hs[1].n, t.ow, 4); // begin_attn_b
            try prompt.ffnMixes(e, hs[0], li);
            try prompt.moeUp(e, hs[0], li); // ffn_up_a: mixes, routing, gate / up
            try Q.close(t, e, &pend); // end_attn_b
            try prompt.moeDown(e, hs[0], li); // moe_down_a: catRank + down
            pend = try t.quartersBegin(e, hs[0].moeOut(e), hs[0].gm, hs[0].n, t.ow, 4); // begin_moe_a
            try prompt.ffnMixes(e, hs[1], li);
            try prompt.moeUp(e, hs[1], li);
            try Q.close(t, e, &pend); // end_moe_a: A's gm is complete before endLayer
            prompt.endLayer(hs[0]);
            try prompt.moeDown(e, hs[1], li);
            pend = try t.quartersBegin(e, hs[1].moeOut(e), hs[1].gm, hs[1].n, t.ow, 4); // begin_moe_b
            prompt.endLayer(hs[1]);
        }
        // replay_a while B's last MoE quarters move. replayCut is mixes plus the kv source: no Coll, A's gm already copied.
        const a_past = try prompt.replayCut(e, hs[0], &m.view, &sh[0], cut, replay, m.host_pos);
        try Q.close(t, e, &pend);
        if (a_past) return error.SplitPastReplay;
        try m.reapFinish(hs[0], cut);
        if (unit) {
            for (cut..c.layers) |li| try m.midSave(li);
            m.mid_at = start + s;
            m.mid_slot = m.fill_slot;
        }
        const ch = hs[1];
        if (!try prompt.replayCut(e, ch, &m.view, &sh[1], cut, replay, m.host_pos)) {
            try m.reapFinish(hs[1], cut);
            return false;
        }
        for (cut..c.layers) |li| {
            try prompt.attnMixes(e, ch, li);
            if (taps) if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| try prompt.tap(e, ch, j);
            try prompt.attention(e, ch, &m.view, &sh[1], li, m.ring_view[li], replay, li == cut);
            try prompt.gather(e, ch, ch.attnOut(e), ch.ga, .at(.prompt_attn, li));
            try prompt.ffnMixes(e, ch, li);
            try prompt.moe(e, ch, li);
            try prompt.gather(e, ch, ch.moeOut(e), ch.gm, .at(.prompt_moe, li));
            prompt.endLayer(ch);
        }
        try m.reapFinish(ch, c.layers);
        try m.dumpWindow(ch, seq);
        try prompt.head(e, ch);
        // 2D head_g is both pairs' columns (m.vocab); TP2's copy is world * this rank's head.n
        try gpu.copyD2DAsync(e.d, e.s.handle, m.ch.head_g, ch.head_g, m.vocab * 4);
        if (m.dr) |*dr| {
            const dw = c.hidden;
            for (0..c.dspark_taps.slice().len) |j| try e.ops.copyRows(e.s, prompt.tapRows(e, ch, j), dw * 2, dr.at + j * dw * 2, dr.taps_w * 2, dw * 2, ch.n);
            try dr.absorb(e, ch, &m.dpool.?, m.fill_slot, dr.at, ch.n, ch.start);
        }
        return true;
    }

    /// TF_DS_WINDOW_LOGITS=DIR: before the head, every row the chunk took through the decoder (the replay window) gets
    /// its logits, 16 rows at a time as a round's head (a collective: every rank runs it); rank 0 writes
    /// DIR/<k>-<start>-<n>.bin: u32 n, u32 vocab, u64 start, i32 ids[n + 1] (the rows' tokens and the one after them,
    /// -1 past the prompt), f32 logits[n][vocab] in vocabulary order. For comparing two settings teacher-forced.
    fn dumpWindow(m: *Model, ch: *prompt.Chunk, seq: []const i32) !void {
        const dir = m.window_dir orelse return;
        const e = &m.eng;
        const hn = e.w.head.n;
        const world = e.world;
        const vocab = world * hn;
        const n = ch.n;
        const d = m.ctx.d;
        if (m.window_l == null) m.window_l = try gpu.DeviceBuffer.alloc(d, 16 * hn * 4);
        if (m.window_g == null) m.window_g = try gpu.DeviceBuffer.alloc(d, world * 16 * hn * 4);
        const head_bytes = 16 + (n + 1) * 4;
        const bytes = try m.gpa.alignedAlloc(u8, .@"4", head_bytes + n * vocab * 4);
        defer m.gpa.free(bytes);
        const logits = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(bytes[head_bytes..])));
        const blk = try m.gpa.alloc(f32, world * 16 * hn);
        defer m.gpa.free(blk);
        var r0: usize = 0;
        while (r0 < n) : (r0 += 16) {
            const rows = @min(16, n - r0);
            try prompt.windowLogits(e, ch, r0, rows, try m.window_l.?.at(0), try m.window_g.?.at(0));
            try m.stream.synchronize();
            try gpu.copyD2H(d, std.mem.sliceAsBytes(blk[0 .. world * rows * hn]), try m.window_g.?.at(0));
            for (0..rows) |r| for (0..world) |k| {
                @memcpy(logits[(r0 + r) * vocab + k * hn ..][0..hn], blk[(k * rows + r) * hn ..][0..hn]);
            };
        }
        m.window_k += 1;
        if (m.rank != 0) return;
        std.mem.writeInt(u32, bytes[0..4], @intCast(n), .little);
        std.mem.writeInt(u32, bytes[4..8], @intCast(vocab), .little);
        std.mem.writeInt(u64, bytes[8..16], ch.start, .little);
        for (0..n + 1) |i| {
            const p = ch.start + i;
            const id: i32 = if (p < seq.len) seq[p] else -1;
            std.mem.writeInt(i32, bytes[16 + i * 4 ..][0..4], id, .little);
        }
        var name: [256]u8 = undefined;
        const path = try std.fmt.bufPrint(&name, "{s}/{d:0>3}-{d}-{d}.bin", .{ dir, m.window_k, ch.start, n });
        try std.Io.Dir.cwd().writeFile(m.io, .{ .sub_path = path, .data = bytes });
    }

    /// The prompt's logits (its last row's, the ranks' columns in rank order) after a fill's last chunk.
    pub fn promptLogits(m: *const Model) u64 {
        return m.ch.head_g;
    }

    fn reapBegin(m: *Model, ch: *prompt.Chunk) !void {
        const st = m.reap orelse return;
        try reap_dev.beginPrompt(st, ch);
    }

    fn reapFinish(m: *Model, ch: *prompt.Chunk, layers_ran: usize) !void {
        const st = m.reap orelse return;
        try reap_dev.finishPrompt(st, ch, @intCast(layers_ran));
    }

    /// Sampled-row norms for this round, on every rank (the exchange). Routing counts wait until the draws.
    fn reapAfterDecode(m: *Model, rows: round.Rows) !void {
        const st = m.reap orelse return;
        var buckets: [round.max_rows]u8 = @splat(0);
        const n = @min(rows.ids.len, buckets.len);
        if (rows.windows) |wins| {
            for (wins) |w| {
                const b: u8 = @intCast(@min(w.reap_bucket, 255));
                var j: usize = 0;
                while (j < w.n and w.row + j < n) : (j += 1) buckets[w.row + j] = b;
            }
        }
        try reap_dev.afterDecode(st, buckets[0..n]);
    }

    /// Rank 0, before a prompt's chunks. `domain` is the domain index (text is index * 2).
    pub fn setReapDomain(m: *Model, domain: u32) void {
        const st = m.reap orelse return;
        reap_dev.setDomain(st, domain);
    }

    /// Rank 0, after the round's draws. A failure is the caller's to log: the follower does not run this.
    pub fn noteReapDecode(m: *Model, commit: []const u8, buckets: []const u8, pos: []const i32) !void {
        const st = m.reap orelse return;
        try reap_dev.noteDecodeRoute(st, commit, buckets, pos);
    }

    /// One round over every stream's window (rows.windows, each with its stream's slot and extent), then each window
    /// whose stream drafts has its rows' taps absorbed into its slot's drafter rings at its positions (the served
    /// eager absorb: rows past the ones the round keeps sit where the drafter never reads, and the next window
    /// overwrites them). absorb[i]: window i's stream drafts.
    pub fn verify(m: *Model, rows: round.Rows, absorb: []const bool) !void {
        const e = &m.eng;
        if (m.prof) try m.stream.synchronize(); // (the GPU idle, so the issue time shows whether the host keeps it fed)
        const t0 = m.now();
        try round.forward(e, &m.rd, &m.ch, &m.caches, m.rings, if (m.eh) |*x| x else null, &.{}, rows, m.context_cap, null, if (m.graphs) |*g| g else null);
        try m.reapAfterDecode(rows);
        if (m.prof) {
            m.t_enqueue = m.now() - t0;
            try m.stream.synchronize();
            m.t_forward = m.now() - t0;
            if (m.timer) |t| try t.finish();
            if (m.rd.clk != 0) {
                var h: [1 + 2 * round.Round.max_clocks]u64 = undefined;
                const d = m.ctx.d;
                try gpu.copyD2H(d, std.mem.sliceAsBytes(&h), m.rd.clk);
                const n = @min(h[0], round.Round.max_clocks);
                for (1..n) |i| {
                    const tag = h[2 + 2 * i] & 0xff;
                    m.clk_ph[tag] += h[1 + 2 * i] -% h[1 + 2 * (i - 1)];
                    m.clk_cnt[tag] += 1;
                }
                m.clk_rounds += 1;
            }
        }
        const t1 = m.now();
        if (m.dr) |*dr| {
            const wins = rows.windows orelse return error.BadRound;
            var items: [max_streams]draft.Item = undefined;
            var n: usize = 0;
            for (wins, absorb) |win, yes| {
                if (!yes) continue;
                items[n] = .{ .slot = @intCast(rows.slots.?[win.row]), .taps = m.rd.taps + win.row * dr.taps_w * 2, .n = win.n, .start = @intCast(rows.pos[win.row]) };
                n += 1;
            }
            if (n > 0) try dr.absorbMany(e, &m.ch, &m.dpool.?, items[0..n]);
        }
        try m.stream.synchronize();
        // a gather inside the round that gave up (a captured proxy gather past its deadline: its stretch ran on with
        // the peer's half missing) fails the round here, before any of its logits are read
        try m.coll.check();
        if (m.prof) m.t_absorb = m.now() - t1;
    }

    /// The scheduler's cost profile (lane admission and draft depth): the measured one when TF_DS_ROUND_MS /
    /// TF_DS_DRAFT_MS give it, else GB10's reference.
    pub fn costs(m: *const Model) draft.scheduler_costs.Profile {
        return .{ .round_ms = m.round_ms, .draft_ms = m.draft_ms };
    }

    /// The rounds' GPU clocks (a profile): the forward's start and each stretch's ends, for its gaps.
    pub fn useClock(m: *Model, marks: u8) !void {
        m.rd.clk_marks = marks == 1 or marks == 3;
        m.rd.clk_ratio = if (marks == 3) 2 else null;
        m.rd.clk_layers = marks == 2 or marks == 3;
        if (m.rd.clk == 0) m.rd.clk = try m.arena.take((1 + 2 * round.Round.max_clocks) * 8);
    }

    /// The rounds' GPU time by phase (eager rounds; --profile 2): a timer on the round.
    pub fn usePhaseTimer(m: *Model) !void {
        if (!m.phase_timer) return error.PhaseTimerNotInHandshake; // (Options.phase_timer at open)
        if (m.timer != null) return;
        const t = try m.gpa.create(round.PhaseTimer);
        errdefer m.gpa.destroy(t);
        t.* = try round.PhaseTimer.init(m.gpa, m.ctx.d);
        m.timer = t;
        const pt = try m.gpa.create(round.PhaseTimer);
        errdefer m.gpa.destroy(pt);
        pt.* = try round.PhaseTimer.init(m.gpa, m.ctx.d);
        m.ptimer = pt;
        m.rd.timer = t;
    }

    /// The monotonic clock (ns).
    pub fn now(m: *const Model) u64 {
        return @intCast(std.Io.Timestamp.now(m.io, .awake).nanoseconds);
    }

    /// The round's logits, R rows of the vocabulary, on the host (after verify).
    /// The round's greedy tokens, R rows (the device's argmax of the round's logits, as sampling.argmax), on the host.
    pub fn roundArgmax(m: *Model, R: usize) ![]const u32 {
        const d = m.ctx.d;
        try m.ops.argmaxRows(m.stream, m.rd.logits, m.vocab, m.vocab, R, m.amax);
        try m.stream.synchronize();
        try gpu.copyD2H(d, std.mem.sliceAsBytes(m.amax_host[0..R]), m.amax);
        return m.amax_host[0..R];
    }

    pub fn roundLogits(m: *Model, R: usize) ![]const f32 {
        const d = m.ctx.d;
        try m.stream.synchronize();
        try gpu.copyD2H(d, std.mem.sliceAsBytes(m.logits[0 .. R * m.vocab]), m.rd.logits);
        return m.logits[0 .. R * m.vocab];
    }

    /// Row `r` of the round's logits on the host (after verify): a constrained greedy row whose argmax its grammar refused.
    pub fn roundLogitsRow(m: *Model, r: usize) ![]const f32 {
        const d = m.ctx.d;
        try m.stream.synchronize();
        const V = m.vocab;
        try gpu.copyD2H(d, std.mem.sliceAsBytes(m.logits[r * V ..][0..V]), m.rd.logits + r * V * @sizeOf(f32));
        return m.logits[r * V ..][0..V];
    }

    /// The prompt's logits on the host (after a fill's last chunk).
    pub fn promptLogitsHost(m: *Model) ![]const f32 {
        const d = m.ctx.d;
        try m.stream.synchronize();
        try m.coll.check();
        try gpu.copyD2H(d, std.mem.sliceAsBytes(m.logits[0..m.vocab]), m.ch.head_g);
        return m.logits[0..m.vocab];
    }

    /// Whether a pass may carry the streams' sampling keys (draft.Drafter.races); both ranks run one kit.
    pub fn draftRaces(m: *const Model) bool {
        return if (m.dr) |*dr| dr.races(&m.eng) else false;
    }

    /// The drafter's batched pass for N streams: each one's pending token at position q0 in its slot (its rings hold
    /// its positions below q0), `steps` drafts each on the host; `keys` (draftRaces()) draw each stream's drafts.
    pub fn pass(m: *Model, tokens: []const i64, q0: []const i64, slots: []const i64, steps: usize, keys: ?[]const draft.SampleKey) !void {
        if (m.dr == null) return error.NoDrafter;
        const dr = &m.dr.?;
        const pool = &m.dpool.?;
        for (q0, slots) |q, s| pool.absorbed[@intCast(s)] = @intCast(q);
        try dr.pass(&m.eng, &m.ch, pool, tokens, q0, slots, steps, keys, null);
        try m.coll.check(); // (the pass read its drafts back: the stream is synchronized)
    }
};

test "a split keeps the first half encoder-only and both halves within their chunks" {
    try std.testing.expectEqual(@as(?usize, 1024), Model.splitAt(2048, 2048, 1024, 1024));
    try std.testing.expectEqual(@as(?usize, 1024), Model.splitAt(2048, 1920, 1024, 1024)); // the prompt's last chunk
    try std.testing.expectEqual(@as(?usize, 1004), Model.splitAt(2010, 1882, 1024, 1024));
    try std.testing.expectEqual(@as(?usize, 1024), Model.splitAt(2047, 1919, 1024, 1024));
    try std.testing.expectEqual(@as(?usize, 384), Model.splitAt(800, 2048, 1024, 1024));
    try std.testing.expectEqual(@as(?usize, 72), Model.splitAt(200, 72, 1024, 1024));
    try std.testing.expectEqual(@as(?usize, null), Model.splitAt(2048, 900, 1024, 1024)); // the second half cannot fit
}

// zip2: unitRows and cutOf on a model that holds only the fields they read
fn unitModel(zip2: u8, split_min: usize) Model {
    var m: Model = undefined;
    m.zip2 = zip2;
    m.split_min = split_min;
    m.split_zip = true;
    m.ptimer = null;
    m.ch2 = @as(prompt.Chunk, undefined);
    return m;
}

test "zip2 off: a unit is a chunk" {
    const m = unitModel(0, 256);
    try std.testing.expectEqual(@as(usize, 2048), m.unitRows(0, 8163, 8163 - 128));
    try std.testing.expectEqual(@as(usize, 2019), m.unitRows(6144, 8163, 8163 - 128));
    try std.testing.expectEqual(@as(usize, 5), m.unitRows(0, 5, 0));
    try std.testing.expectError(error.BadChunk, m.cutOf(0, 4096, 8163 - 128));
}

test "zip2: two whole chunks when the first is encoder-only; else a chunk" {
    const m1 = unitModel(1, 256);
    const len = 8163;
    const replay = len - 128;
    try std.testing.expectEqual(@as(usize, 4096), m1.unitRows(0, len, replay));
    try std.testing.expectEqual(@as(usize, 2048), m1.unitRows(4096, len, replay)); // the second would be partial
    try std.testing.expectEqual(@as(usize, 2019), m1.unitRows(6144, len, replay));
    try std.testing.expectEqual(@as(usize, 4096), m1.unitRows(0, 4096, 4096 - 128)); // the second is the last and whole
    try std.testing.expectEqual(@as(usize, 2048), m1.unitRows(0, 2048, 2048 - 128)); // one chunk
    try std.testing.expectEqual(@as(usize, 1818), m1.unitRows(0, 1818, 1818 - 128)); // a short prompt
    try std.testing.expectEqual(@as(usize, 2048), m1.unitRows(0, 2148, 2148 - 128)); // the first holds the replay cut
    const m2 = unitModel(2, 256);
    try std.testing.expectEqual(@as(usize, 4096), m2.unitRows(0, len, replay));
    try std.testing.expectEqual(@as(usize, 2048 + 2019), m2.unitRows(4096, len, replay)); // the last, partial chunk rides
    try std.testing.expectEqual(@as(usize, 2048), m2.unitRows(0, 2048 + 255, 2048 + 255 - 128)); // too small to ride
    try std.testing.expectEqual(@as(usize, 2048 + 256), m2.unitRows(0, 2048 + 256, 2048 + 256 - 128));
    // a unit is cut at its first chunk's end, and only where unitRows makes one
    try std.testing.expectEqual(Model.Cut{ .s = 2048, .unit = true }, try m1.cutOf(0, 4096, replay));
    try std.testing.expectEqual(Model.Cut{ .s = 2048, .unit = true }, try m2.cutOf(4096, 2048 + 2019, replay));
    try std.testing.expectError(error.BadChunk, m1.cutOf(1024, 4096, replay)); // off the chunk grid
    try std.testing.expectError(error.BadChunk, m1.cutOf(6144, 2019 + 2048, replay)); // the first chunk holds the cut
    // a chunk under zip2 is cut as the split cuts it, by the split's halves
    try std.testing.expectEqual(Model.Cut{ .s = 1024 }, try m1.cutOf(4096, 2048, replay));
    try std.testing.expectEqual(try unitModel(0, 256).cutOf(6144, 2019, replay), try m1.cutOf(6144, 2019, replay));
}

const Part = struct { at: usize, n: usize, cap: usize, first: bool };

/// A prompt's parts in the order fillChunk runs them (lanes.zig's loop: unitRows, then cutOf) with the buffer rows each
/// has, and the chunk ends lanes.zig may keep a state at: every unit's end and a zipped unit's first chunk end.
fn rowPlan(m: *const Model, len: usize, replay: usize, parts: *std.ArrayList(Part), keeps: *std.ArrayList(usize)) !void {
    const gpa = std.testing.allocator;
    const split = chunkCaps(m.split_min);
    var at: usize = 0;
    while (at < len) {
        const n = m.unitRows(at, len, replay);
        const cut = try m.cutOf(at, n, replay);
        if (cut.s > 0) {
            const caps = if (cut.unit) [2]usize{ chunk_rows, chunk_rows } else split;
            try parts.append(gpa, .{ .at = at, .n = cut.s, .cap = caps[0], .first = true });
            try parts.append(gpa, .{ .at = at + cut.s, .n = n - cut.s, .cap = caps[1], .first = false });
            if (cut.unit) try keeps.append(gpa, at + cut.s);
        } else try parts.append(gpa, .{ .at = at, .n = n, .cap = split[0], .first = false });
        at += n;
        if (at < len and at <= replay) try keeps.append(gpa, at);
    }
}

test "zip2: every row runs once at its own position as in the unzipped pass, and states are kept at the same chunk ends" {
    const gpa = std.testing.allocator;
    const plain = unitModel(0, 256);
    for ([_]usize{ 1, 127, 129, 600, 2048, 2049, 2303, 2304, 4095, 4096, 4097, 4224, 6000, 8163, 14503, 54665, 63488, 65391 }) |len| {
        const replay = len -| 128;
        var p0: std.ArrayList(Part) = .empty;
        defer p0.deinit(gpa);
        var k0: std.ArrayList(usize) = .empty;
        defer k0.deinit(gpa);
        try rowPlan(&plain, len, replay, &p0, &k0);
        for (k0.items) |k| try std.testing.expect(k % chunk_rows == 0);
        for ([_]u8{ 1, 2 }) |mode| {
            const zip = unitModel(mode, 256);
            var p1: std.ArrayList(Part) = .empty;
            defer p1.deinit(gpa);
            var k1: std.ArrayList(usize) = .empty;
            defer k1.deinit(gpa);
            try rowPlan(&zip, len, replay, &p1, &k1);
            // the parts tile the prompt in order (a row's position is its own in either pass), each within its
            // buffer, and a first part stops before the replay cut, as a split's first half does
            var at: usize = 0;
            var whole_chunks: usize = 0;
            for (p1.items) |q| {
                try std.testing.expectEqual(at, q.at);
                try std.testing.expect(q.n > 0 and q.n <= q.cap);
                if (q.first) try std.testing.expect(q.at + q.n <= replay);
                if (q.n == chunk_rows) whole_chunks += 1;
                at += q.n;
            }
            try std.testing.expectEqual(len, at);
            // the same chunk ends are offered for a kept state, in the same order
            try std.testing.expectEqualSlices(usize, k0.items, k1.items);
            // the zip engages: with two whole chunks before the replay cut, parts of a whole chunk's rows (one
            // routed-experts pass each), which the unzipped pass never has
            if (len >= 2 * chunk_rows and chunk_rows <= replay) try std.testing.expect(whole_chunks >= 2);
        }
        for (p0.items) |q| try std.testing.expect(q.n <= chunk_rows / 2);
    }
}

test "zip2: a layer's saved stores sit where snapCopy puts them" {
    const kv = [_]u64{ 0, 7, 0, 7, 7, 0 };
    const sc = [_]u64{ 0, 7, 0, 0, 7, 0 };
    const rb = 1000;
    const pb = 24;
    var at: usize = kv.len * rb; // snapCopy's walk: every ring, then each layer's live stores
    for (kv, sc, 0..) |k, s, li| {
        try std.testing.expectEqual(at, Model.storesAt(&kv, &sc, li, kv.len * rb, pb));
        if (k != 0) at += pb;
        if (s != 0) at += pb;
    }
    try std.testing.expectEqual(at, Model.storesAt(&kv, &sc, kv.len, kv.len * rb, pb));
    try std.testing.expectEqual(@as(usize, 6 * 1000 + 5 * 24), at);
}

test "a launch identity is one plan token of at most 16 characters" {
    try std.testing.expectEqualStrings("-", Model.launchWord(null));
    try std.testing.expectEqualStrings("-", Model.launchWord(""));
    try std.testing.expectEqualStrings("dcaaee0c2178", Model.launchWord("dcaaee0c2178"));
    try std.testing.expectEqualStrings("243b984d1069b190", Model.launchWord("243b984d1069b190e762b3f9eaf12c90"));
    try std.testing.expectEqualStrings("abc", Model.launchWord("abc def"));
}

test "the handshake names what orders a prompt chunk's gathers: split, zip, chunk rows, phase timer, frames" {
    var b: [128]u8 = undefined;
    const frames = std.fmt.comptimePrint("{d}", .{frame_protocol});
    try std.testing.expectEqualStrings("prefill_split=256 prefill_zip=true chunk_caps=1024/1024 phase_timer=false frames=" ++ frames, try Model.gatherOrder(&b, 256, true, false));
    try std.testing.expectEqualStrings("prefill_split=0 prefill_zip=true chunk_caps=2048/0 phase_timer=true frames=" ++ frames, try Model.gatherOrder(&b, 0, true, true));
}

test "IMAGE_TOKENS: the image rows a launch reserves, from 1 to image_rows_max, and the plan says them only when not 4096" {
    try std.testing.expectEqual(@as(usize, 32768), try Model.parseImageRows("32768"));
    try std.testing.expectEqual(@as(usize, 1), try Model.parseImageRows("1"));
    try std.testing.expectEqual(@as(usize, Model.image_rows_max), try Model.parseImageRows("65536"));
    for ([_][]const u8{ "0", "65537", "", "32k", "-1", " 4096" }) |bad| try std.testing.expectError(error.BadImageTokens, Model.parseImageRows(bad));
    var gb: [128]u8 = undefined;
    const gather = try Model.gatherOrder(&gb, 256, true, false);
    var p: Model.PlanParts = .{ .drafts = true, .engram = true, .graphs = false, .pool = 8192, .streams = 1, .round_rows = 16, .replay_floor = false, .aux = "hip", .gather = gather, .engram_rows = true, .deploy = "70cfb8c", .bundle = "c178a012", .conf = "abcd", .strict = true, .disk = false };
    var buf: [512]u8 = undefined;
    const stock = try std.testing.allocator.dupe(u8, try Model.planText(&buf, p));
    defer std.testing.allocator.free(stock);
    try std.testing.expect(std.mem.indexOf(u8, stock, "image_rows") == null);
    p.image_rows = 4096;
    try std.testing.expectEqualStrings(stock, try Model.planText(&buf, p));
    p.image_rows = 32768;
    const big = try Model.planText(&buf, p);
    try std.testing.expect(std.mem.endsWith(u8, big, " image_rows=32768"));
    try std.testing.expectEqualStrings(stock, big[0 .. big.len - " image_rows=32768".len]);
}

test "the handshake's plan text is the stock one with sf-disk off, and says disk=1 and edit= only when on" {
    var gb: [128]u8 = undefined;
    const gather = try Model.gatherOrder(&gb, 256, true, false);
    // the stock format, byte for byte (frames= is the protocol number: the disk frames do not change it)
    const stock = "drafts=true engram=true graphs=false pool=8192 streams=1 round_rows=16 chunk_rows=2048 replay_floor=false aux=hip prefill_split=256 prefill_zip=true chunk_caps=1024/1024 phase_timer=false frames=" ++
        std.fmt.comptimePrint("{d}", .{frame_protocol}) ++
        " engram_rows=true deploy=70cfb8c bundle=c178a012 conf=abcd kv=win-fp8qd,cmp-fp4,idx-fp4 strict=true";
    var p: Model.PlanParts = .{ .drafts = true, .engram = true, .graphs = false, .pool = 8192, .streams = 1, .round_rows = 16, .replay_floor = false, .aux = "hip", .gather = gather, .engram_rows = true, .deploy = "70cfb8c", .bundle = "c178a012", .conf = "abcd", .strict = true, .disk = false };
    var buf: [512]u8 = undefined;
    try std.testing.expectEqualStrings(stock, try Model.planText(&buf, p));
    p.disk = true;
    try std.testing.expectEqualStrings(stock ++ " disk=1", try Model.planText(&buf, p));
    // a context at the pool is the stock layout; below it, the ranks compare it
    p.disk = false;
    p.context = 8192;
    try std.testing.expectEqualStrings(stock, try Model.planText(&buf, p));
    p.context = 4096;
    const split = try Model.planText(&buf, p);
    try std.testing.expect(std.mem.indexOf(u8, split, "pool=8192 context=4096 streams=1 ") != null);
    // an edited launch names its edit file's hash after disk=1, so peers and sf-disk tell it from the unedited pack
    p.context = 0;
    p.disk = true;
    p.edit = "0123456789abcdef";
    try std.testing.expectEqualStrings(stock ++ " disk=1 edit=0123456789abcdef", try Model.planText(&buf, p));
    // zip2 names its mode and its unit buffers last, so a rank without it (or in the other mode) is refused
    p.zip2 = 2;
    try std.testing.expectEqualStrings(stock ++ " disk=1 edit=0123456789abcdef prefill_zip2=2 unit_caps=2048/2048", try Model.planText(&buf, p));
    p.disk = false;
    p.edit = "";
    p.zip2 = 1;
    try std.testing.expectEqualStrings(stock ++ " prefill_zip2=1 unit_caps=2048/2048", try Model.planText(&buf, p));
    p.zip2 = 0;
    p.reap = " reap=8/4/16";
    try std.testing.expectEqualStrings(stock ++ " reap=8/4/16", try Model.planText(&buf, p));
    p.reap = "";
    try std.testing.expectEqualStrings(stock, try Model.planText(&buf, p));
    try std.testing.expect(!@hasField(Options, "disk") or (Options{ .model_dir = "", .cache_dir = "", .kit_dir = "", .rank = 0, .world = 2, .master = .{ 0, 0, 0, 0 }, .port = 0 }).disk == false);
}
