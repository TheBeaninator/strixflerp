//! A request's `acts` field: per-request hidden-state dumps, picture-row injection and the first
//! answer token's logits (core/lanes/acts.zig), behind the admin key. docs/experiments/serve-acts.md is the contract.
//!
//!   "acts": {"dump": true, "taps": ["embed", 10, "head"]}    the image-span rows at those layers (replay 0)
//!   "acts": {"text": true}                                   a text-only prompt within the window, every row
//!   "acts": {"rows": true}                                   pictures' rows from ACTS_ROWS_DIR/<digest hex>.bin
//!   "acts": {"logits": true}                                 the first answer token's logits
//!
//! Order of refusals: the key first (403 when the server has none, 401 without the bearer), before any part of the
//! field is read; then the server's own switch (400 when its conf names no ACTS_DIR); then the field (400).
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const errors = @import("errors.zig");
const cache_key = @import("cache_key.zig");
const Value = json.Value;
const Cx = errors.Cx;
const acts = api.acts;
const Allocator = std.mem.Allocator;

pub const Refusal = struct { status: u16, message: []const u8 };

pub const no_key = "acts needs the admin key, and this server has none (--cache-transfer-key unset)";
pub const bad_key = "acts needs the admin key: the cache transfer key's bearer (Authorization: Bearer or x-cache-transfer-key)";
pub const off = "acts is off on this server: its conf names no ACTS_DIR";
pub const elsewhere = "acts is taken on /v1/chat/completions, /v1/completions, /v1/messages and /v1/responses only";

/// Whether a request carrying `acts` may go on (null) or its refusal. `authorization` and `header` are the request's
/// single Authorization and x-cache-transfer-key headers (null when absent or repeated).
pub fn gate(transfer: ?cache_key.Key, info: ?acts.Info, authorization: ?[]const u8, header: ?[]const u8) ?Refusal {
    const key = transfer orelse return .{ .status = 403, .message = no_key };
    if (!key.accepts(authorization, header)) return .{ .status = 401, .message = bad_key };
    if (info == null) return .{ .status = 400, .message = off };
    return null;
}

/// The field's switches, checked against the engine's facts (a 400 for anything else). The id is set by prepare.
pub fn parse(cx: *Cx, v: Value, info: acts.Info) errors.Refused!acts.Spec {
    if (v != .object) return cx.refuse("acts must be an object: {\"dump\", \"text\", \"rows\", \"logits\": booleans, \"taps\": a list}");
    var s: acts.Spec = .{};
    var taps: ?Value = null;
    for (v.object.keys(), v.object.values()) |k, x| {
        if (std.mem.eql(u8, k, "taps")) {
            taps = x;
            continue;
        }
        const slot: *bool = if (std.mem.eql(u8, k, "dump")) &s.dump else if (std.mem.eql(u8, k, "text")) &s.text else if (std.mem.eql(u8, k, "rows")) &s.rows else if (std.mem.eql(u8, k, "logits")) &s.logits else return cx.fail(.request, "acts: unknown field \"{s}\" (dump, text, rows, logits, taps)", .{k});
        if (x != .bool) return cx.fail(.request, "acts.{s} must be true or false", .{k});
        slot.* = x.bool;
    }
    if (!s.active()) return cx.refuse("acts: turn on at least one of dump, text, rows, logits");
    if (s.dump and s.text) return cx.refuse("acts: dump (a prompt's pictures) and text (a text-only prompt) exclude each other");
    if (s.rows and info.rows_dir.len == 0) return cx.refuse("acts.rows needs ACTS_ROWS_DIR in this server's conf");
    if (taps) |t| {
        if (!s.tapsOn()) return cx.refuse("acts.taps needs acts.dump or acts.text");
        if (t != .array or t.array.len == 0 or t.array.len > acts.max_taps) return cx.fail(.request, "acts.taps must be a list of 1 to {d} layers", .{acts.max_taps});
        for (t.array) |item| {
            const layer: i64 = switch (item) {
                .string => |w| if (std.mem.eql(u8, w, "embed")) acts.embed_tap else if (std.mem.eql(u8, w, "head")) @as(i64, info.layers) - 1 else return cx.fail(.request, "acts.taps: \"{s}\" is not a layer (a number, \"embed\" or \"head\")", .{w}),
                .int => item.int64() orelse return cx.fail(.request, "acts.taps: layers are -1 (embed) to {d} (head)", .{info.layers - 1}),
                else => return cx.refuse("acts.taps: each layer is a number, \"embed\" or \"head\""),
            };
            if (!acts.tapOk(layer, info.layers)) return cx.fail(.request, "acts.taps: layer {d} is not one of this model's: -1 (embed) to {d} (head)", .{ layer, info.layers - 1 });
            const l: i32 = @intCast(layer);
            if (std.mem.indexOfScalar(i32, s.taps[0..s.n_taps], l) != null) return cx.fail(.request, "acts.taps: layer {d} twice", .{l});
            s.taps[s.n_taps] = l;
            s.n_taps += 1;
        }
    } else if (s.tapsOn()) s.n_taps = acts.defaultTaps(info.layers, &s.taps);
    return s;
}

/// After the prompt is rendered: the checks that need it, the request's size against ACTS_REQUEST_MB, the directory
/// against ACTS_TOTAL_MB (507), and the rows injection (a new image list; `replaced` the pictures whose file was there).
pub fn prepare(cx: *Cx, io: std.Io, info: acts.Info, s: *const acts.Spec, prompt_len: usize, images: []const api.Image, replaced: *u32) errors.Refused![]const api.Image {
    replaced.* = 0;
    if (s.dump and images.len == 0) return cx.refuse("acts.dump needs a picture in the prompt (acts.text dumps a text-only prompt)");
    if (s.rows and images.len == 0) return cx.refuse("acts.rows needs a picture in the prompt");
    if (s.text and (images.len > 0 or prompt_len > info.window)) return cx.fail(.request, "acts.text needs a text-only prompt of at most {d} tokens (this one: {d} tokens, {d} pictures)", .{ info.window, prompt_len, images.len });
    if (s.tapsOn()) {
        var rows: u64 = 0;
        if (s.dump) {
            for (images) |img| rows += img.tokens;
        } else rows = prompt_len;
        // each chunk's header is at most a few hundred bytes more: count one a 128 rows
        const want = acts.dumpBytes(s.n_taps, @intCast(rows), info.row_bytes) + (rows / 128 + 1) * (40 + 4 * acts.max_taps);
        const cap = info.writer.limits.request_bytes;
        if (want > cap) return cx.fail(.request, "acts would write {d} MiB ({d} taps over {d} rows), over this server's ACTS_REQUEST_MB ({d})", .{ want >> 20, s.n_taps, rows, cap >> 20 });
    }
    var used: u64 = 0;
    if (info.writer.overTotal(&used)) return cx.fail(.storage, "the acts dump directory holds {d} MiB, at its ACTS_TOTAL_MB ({d}): clear it", .{ used >> 20, info.writer.limits.total_bytes >> 20 });
    if (!s.rows) return images;
    const out = try cx.a.dupe(api.Image, images);
    for (out) |*img| {
        const d = img.digest orelse api.prompt_cache.imageDigest(img.bytes);
        const want = @as(usize, img.tokens) * info.row_bytes;
        const data = (try rowsFile(cx, io, info.rows_dir, &d, want)) orelse continue;
        img.* = .{ .at = img.at, .tokens = img.tokens, .bytes = data, .rows = true, .digest = d };
        replaced.* += 1;
    }
    return out;
}

/// ACTS_ROWS_DIR/<digest hex>.bin when it exists: exactly `want` bytes (the span's rows, BF16 [tokens x hidden]), else a
/// 400; null when there is no such file (the picture keeps the tower's rows).
fn rowsFile(cx: *Cx, io: std.Io, dir: []const u8, digest: *const api.prompt_cache.Digest, want: usize) errors.Refused!?[]const u8 {
    const hex = std.fmt.bytesToHex(digest.*, .lower);
    const path = try std.fmt.allocPrint(cx.a, "{s}/{s}.bin", .{ dir, &hex });
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, cx.a, .limited(want + 1)) catch |e| switch (e) {
        error.FileNotFound => return null,
        error.StreamTooLong => return cx.fail(.request, "acts.rows: {s}.bin is longer than the picture's span ({d} bytes)", .{ &hex, want }),
        error.OutOfMemory => return error.OutOfMemory,
        else => return cx.fail(.request, "acts.rows: {s}.bin cannot be read ({s})", .{ &hex, @errorName(e) }),
    };
    if (data.len != want) return cx.fail(.request, "acts.rows: {s}.bin holds {d} bytes, the picture's span {d}", .{ &hex, data.len, want });
    return data;
}

/// The reply's tensorfold.acts: where the request's files go and what it asked for.
pub fn replyField(a: Allocator, info: acts.Info, s: *const acts.Spec, replaced: u32) !Value {
    const o = try json.newObject(a);
    try o.put(a, "id", .{ .string = s.id });
    try o.put(a, "dir", .{ .string = try std.fmt.allocPrint(a, "{s}/{s}", .{ info.writer.dir, s.id }) });
    try o.put(a, "dump", .{ .bool = s.dump });
    try o.put(a, "text", .{ .bool = s.text });
    try o.put(a, "logits", .{ .bool = s.logits });
    const taps = try a.alloc(Value, s.n_taps);
    for (s.tapList(), taps) |t, *slot| slot.* = try json.intValue(a, t);
    try o.put(a, "taps", .{ .array = taps });
    try o.put(a, "rows_replaced", if (s.rows) try json.intValue(a, replaced) else .null);
    return .{ .object = o };
}
