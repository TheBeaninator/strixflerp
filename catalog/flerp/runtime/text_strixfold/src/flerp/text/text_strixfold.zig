//! include/flerp/text.h over StrixFold's own text layer: HfText (zig/src/server/hf_text.zig), the tokenizer and
//! Jinja chat template its server renders prompts and replies with. Nothing is re-implemented here.
//!
//! flerp_text_open keys: model_dir (the checkpoint directory: tokenizer.json, tokenizer_config.json, config.json and
//! the chat template).
//!
//! Where this differs from the header's contract, because HfText has no such thing:
//!   - `plain` ranges are not honoured: HfText.encode takes text only, so a special token's text inside a user
//!     message encodes as that token;
//!   - a token's bytes are HfText's decode of the one id, which is text: a token that ends inside a UTF-8
//!     sequence comes back as HfText decodes it alone.
//! The template switches read from a request are enable_thinking (top level or in chat_template_kwargs) and
//! reasoning_effort; the serving rules of StrixFold's own server around them (zig/src/server/chat.zig) are not here.
//! Built by flerp.serve.build.zig; it has not run.
const std = @import("std");
const c = @import("flerp_text_abi");
const hf_text = @import("../../zig/src/server/hf_text.zig");
const json = @import("../../zig/src/server/json.zig");
const model_text = @import("../../zig/src/server/model_text.zig");

const OK: c_int = 0;
const E_ARGUMENT: c_int = -1;
const E_UNSUPPORTED: c_int = -3;
const E_FAILED: c_int = -5;
const E_CONFIG: c_int = -6;

const Text = struct {
    hf: *hf_text.HfText,
    text: model_text.Text,
    render_arena: std.heap.ArenaAllocator,
    describe: [:0]u8,
    plain: [2]usize = .{ 0, 0 },
};

var g_gpa: std.mem.Allocator = undefined;
var g_io: std.Io = undefined;
var g_err_buf: [640]u8 = undefined;
var g_err: [*:0]const u8 = "";

/// The root file of the binary hands the process's allocator and io over before any call.
pub fn setProcess(gpa: std.mem.Allocator, io: std.Io) void {
    g_gpa = gpa;
    g_io = io;
}

fn fail(st: c_int, comptime fmt: []const u8, args: anytype) c_int {
    const msg = std.fmt.bufPrintSentinel(&g_err_buf, fmt, args, 0) catch blk: {
        g_err_buf[g_err_buf.len - 1] = 0;
        break :blk g_err_buf[0 .. g_err_buf.len - 1 :0];
    };
    g_err = msg.ptr;
    return st;
}

fn of(p: ?*const anyopaque) *Text {
    return @constCast(@ptrCast(@alignCast(p.?)));
}

fn span(p: [*c]const u8) []const u8 {
    return if (p == null) "" else std.mem.span(@as([*:0]const u8, @ptrCast(p)));
}

export fn flerp_text_open(config: [*c]const c.flerp_kv, n_config: usize, out: ?*?*anyopaque) c_int {
    var dir: []const u8 = "";
    var i: usize = 0;
    while (i < n_config) : (i += 1) {
        const key = span(config[i].key);
        if (!std.mem.eql(u8, key, "model_dir")) return fail(E_CONFIG, "flerp_text_open (strixfold): unknown key '{s}' (model_dir)", .{key});
        dir = span(config[i].value);
    }
    if (out == null or dir.len == 0) return fail(E_CONFIG, "flerp_text_open (strixfold): model_dir (the checkpoint directory) is required", .{});
    var problem_arena = std.heap.ArenaAllocator.init(g_gpa);
    defer problem_arena.deinit();
    var problem: []const u8 = "";
    const h = hf_text.HfText.load(g_gpa, g_io, dir, problem_arena.allocator(), &problem) catch |e|
        return fail(E_FAILED, "flerp_text_open (strixfold): {s}: {s} {s}", .{ dir, @errorName(e), problem });
    const t = g_gpa.create(Text) catch return fail(E_FAILED, "flerp_text_open: out of memory", .{});
    t.* = .{ .hf = h, .text = h.text(), .render_arena = .init(g_gpa), .describe = undefined };
    t.describe = std.fmt.allocPrintSentinel(g_gpa, "strixfold HfText (tokenizer.json and the checkpoint's chat template), {d} tokens, {s}", .{ t.text.vocabSize(), dir }, 0) catch
        return fail(E_FAILED, "flerp_text_open: out of memory", .{});
    out.?.* = t;
    return OK;
}

export fn flerp_text_close(p: ?*anyopaque) void {
    const t = of(p orelse return);
    t.render_arena.deinit();
    t.hf.deinit();
    g_gpa.free(t.describe);
    g_gpa.destroy(t);
}

export fn flerp_text_last_error(p: ?*const anyopaque) [*:0]const u8 {
    _ = p;
    return g_err;
}

export fn flerp_text_describe(p: ?*const anyopaque) [*:0]const u8 {
    return of(p).describe.ptr;
}

export fn flerp_text_vocab(p: ?*const anyopaque) i64 {
    return of(p).text.vocabSize();
}

export fn flerp_text_encode(p: ?*anyopaque, text: [*c]const u8, n: usize, plain: [*c]const usize, n_plain: usize, out: [*c]i32, cap: usize) i64 {
    _ = .{ plain, n_plain }; // HfText.encode has no plain ranges (see the file comment)
    const t = of(p);
    var arena = std.heap.ArenaAllocator.init(g_gpa);
    defer arena.deinit();
    const src: []const u8 = if (n == 0) "" else text[0..n];
    const ids = t.text.encode(arena.allocator(), src, false) catch |e| return fail(E_ARGUMENT, "the text does not tokenize: {s}", .{@errorName(e)});
    if (ids.len <= cap and out != null) {
        for (ids, 0..) |id, k| out[k] = @intCast(id);
    }
    return @intCast(ids.len);
}

export fn flerp_text_token_bytes(p: ?*anyopaque, id: i32, out: [*c]u8, cap: usize) i64 {
    const t = of(p);
    if (id < 0) return fail(E_ARGUMENT, "token id {d} is negative", .{id});
    var arena = std.heap.ArenaAllocator.init(g_gpa);
    defer arena.deinit();
    const one = [_]u32{@intCast(id)};
    const bytes = t.text.decode(arena.allocator(), &one) catch return fail(E_FAILED, "out of memory", .{});
    if (bytes.len <= cap and out != null) @memcpy(out[0..bytes.len], bytes);
    return @intCast(bytes.len);
}

export fn flerp_text_end_ids(p: ?*const anyopaque, out: [*c]i32, cap: usize) i64 {
    const t = of(p);
    const eos = t.text.vtable.eos_ids(t.text.ctx);
    if (eos.len <= cap and out != null) {
        for (eos, 0..) |id, k| out[k] = @intCast(id);
    }
    return @intCast(eos.len);
}

export fn flerp_text_token_id(p: ?*const anyopaque, piece: [*c]const u8) i32 {
    const t = of(p);
    const id = t.text.tokenId(span(piece)) orelse return -1;
    return @intCast(id);
}

fn switchOf(v: json.Value, key: []const u8) ?json.Value {
    if (v.get(key)) |x| if (x != .null) return x;
    if (v.get("chat_template_kwargs")) |k| if (k.get(key)) |x| if (x != .null) return x;
    return null;
}

export fn flerp_text_render_chat(p: ?*anyopaque, request: [*c]const u8, n: usize, out: ?*c.flerp_text_chat) c_int {
    const t = of(p);
    const o = out orelse return fail(E_ARGUMENT, "flerp_text_chat is null", .{});
    if (o.size < @sizeOf(c.flerp_text_chat)) return fail(E_ARGUMENT, "flerp_text_chat is short", .{});
    _ = t.render_arena.reset(.retain_capacity);
    const a = t.render_arena.allocator();
    const src: []const u8 = if (n == 0) "" else request[0..n];
    const parsed = json.parse(a, src) catch return fail(E_FAILED, "out of memory", .{});
    const body = switch (parsed) {
        .ok => |v| v,
        .err => |why| return fail(E_ARGUMENT, "the request is not JSON: {s}", .{why}),
    };
    const messages = body.get("messages") orelse return fail(E_ARGUMENT, "'messages' is required", .{});
    if (messages != .array) return fail(E_ARGUMENT, "'messages' must be an array", .{});
    var options: model_text.RenderOptions = .{};
    if (body.get("tools")) |tools| if (tools == .array and tools.array.len > 0) {
        options.tools = tools;
    };
    if (switchOf(body, "enable_thinking")) |x| {
        if (x != .bool) return fail(E_ARGUMENT, "'enable_thinking' must be true or false", .{});
        options.enable_thinking = x.bool;
    }
    if (switchOf(body, "reasoning_effort")) |x| {
        if (x != .string) return fail(E_ARGUMENT, "'reasoning_effort' must be a string", .{});
        options.reasoning_effort = x.string;
    }
    var problem: []const u8 = "";
    const rendered = t.text.vtable.render(t.text.ctx, a, messages, options, &problem) catch |e| switch (e) {
        error.OutOfMemory => return fail(E_FAILED, "out of memory", .{}),
        error.Template => {
            if (std.mem.indexOf(u8, problem, "no chat template") != null) return fail(E_UNSUPPORTED, "{s}", .{problem});
            return fail(E_ARGUMENT, "the chat template refused the request: {s}", .{problem});
        },
    };
    o.text = rendered.ptr;
    o.n_text = rendered.len;
    o.plain = &t.plain;
    o.n_plain = 0;
    return OK;
}
