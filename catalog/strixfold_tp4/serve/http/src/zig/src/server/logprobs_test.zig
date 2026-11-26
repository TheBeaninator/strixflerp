//! Per-token logprobs on /v1/chat/completions: an engine that reports them (Info.logprobs) gets
//! the request's top_logprobs and sink, and the reply carries choices[0].logprobs.content in OpenAI's shape; a request
//! without logprobs gets the bytes it got before (no logprobs key, the engine asked for nothing); streams, legacy
//! completions, engines without them and malformed fields are refused with a 400.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const routes = @import("routes.zig");
const Conn = @import("http_conn.zig").Conn;

/// Byte tokens and a one-line prompt.
const ByteText = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const decoded = try a.alloc(u8, ids.len);
        for (ids, decoded) |id, *byte| byte.* = @intCast(id);
        return decoded;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "hi\n");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Answers "ok" at once; with logprobs asked, each token's logprob (the second NaN: one the engine could not give) and
/// `top` alternatives before the tokens, as the lane host sends them.
const Fake = struct {
    reports: bool,
    asked: ?u8 = null,
    saw_logprobs: bool = false,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(ctx: *anyopaque) api.Info {
        const e: *@This() = @ptrCast(@alignCast(ctx));
        return .{ .logprobs = e.reports };
    }

    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *@This() = @ptrCast(@alignCast(ctx));
        e.saw_logprobs = r.logprobs != null;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        if (r.logprobs) |lp| {
            e.asked = lp.top;
            var rows: [2]api.TokenLogprob = .{ .{ .token = 'o', .logprob = -0.25 }, .{ .token = 'k', .logprob = std.math.nan(f32) } };
            for (&rows) |*t| {
                t.n = lp.top;
                for (0..lp.top) |j| {
                    t.ids[j] = 'a' + @as(u32, @intCast(j));
                    t.logprobs[j] = -0.5 - @as(f32, @floatFromInt(j));
                }
            }
            lp.add(lp.ctx, &rows);
        }
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'o', 'k' } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .length } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

fn sendTo(a: std.mem.Allocator, srv: *server.Server, path: []const u8, body: []const u8) ![]u8 {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const head = try std.fmt.allocPrint(a, "POST {s} HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n", .{ path, body.len});
    for ([_][]const u8{ head, body }) |part| {
        var sent: usize = 0;
        while (sent < part.len) {
            const n = std.c.write(pair[0], part[sent..].ptr, part.len - sent);
            if (n <= 0) return error.Closed;
            sent += @intCast(n);
        }
    }
    var conn = try Conn.init(std.testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    if (try conn.readRequest(arena.allocator(), false) != .ready) return error.BadRequest;
    routes.dispatch(srv, &conn, arena.allocator());
    var got: std.ArrayList(u8) = .empty;
    var tmp: [4096]u8 = undefined;
    var pollfd = [_]std.posix.pollfd{.{ .fd = pair[0], .events = std.posix.POLL.IN, .revents = 0 }};
    while (try std.posix.poll(&pollfd, 200) > 0) {
        const n = try std.posix.read(pair[0], &tmp);
        if (n == 0) break;
        try got.appendSlice(a, tmp[0..n]);
    }
    return got.items;
}

/// The reply's status line and its body.
fn reply(a: std.mem.Allocator, srv: *server.Server, path: []const u8, body: []const u8) !struct { status: []const u8, body: []const u8 } {
    const got = try sendTo(a, srv, path, body);
    const at = std.mem.indexOf(u8, got, "\r\n\r\n") orelse return error.BadReply;
    return .{ .status = got[0..12], .body = got[at + 4 ..] };
}

/// A body with its "id" and "created" values cut out and up to its tensorfold block (they and the block's timings
/// differ between any two replies): its choices, usage and mode.
fn steady(a: std.mem.Allocator, body: []const u8) ![]const u8 {
    var out: []const u8 = body[0 .. std.mem.indexOf(u8, body, "\"tensorfold\"") orelse return error.NoBlock];
    for ([_][]const u8{ "\"id\":", "\"created\":" }) |key| {
        const i = std.mem.indexOf(u8, out, key) orelse return error.NoKey;
        var j = i + key.len;
        while (j < out.len and out[j] != ',') j += 1;
        out = try std.mem.concat(a, u8, &.{ out[0..i], out[j..] });
    }
    return out;
}

fn open(e: *Fake, text: *ByteText) !*server.Server {
    return server.Server.init(std.testing.allocator, std.testing.io, e.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false }, null);
}

const plain = "{\"model\":\"m\",\"max_tokens\":2,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]";

test "logprobs: true with top_logprobs: each content token's logprob, bytes and alternatives in OpenAI's shape" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: ByteText = .{};
    var e: Fake = .{ .reports = true };
    const srv = try open(&e, &text);
    defer srv.deinit();
    const r = try reply(a, srv, "/v1/chat/completions", plain ++ ",\"logprobs\":true,\"top_logprobs\":2}");
    try std.testing.expectEqualStrings("HTTP/1.1 200", r.status);
    try std.testing.expectEqual(@as(?u8, 2), e.asked);
    const parsed = try json.parse(a, r.body);
    const choice = parsed.ok.get("choices").?.array[0];
    const content = choice.get("logprobs").?.get("content").?.array;
    try std.testing.expectEqual(@as(usize, 2), content.len);
    try std.testing.expectEqualStrings("o", content[0].get("token").?.string);
    try std.testing.expectEqual(@as(f64, -0.25), content[0].get("logprob").?.float);
    try std.testing.expectEqual(@as(usize, 1), content[0].get("bytes").?.array.len);
    try std.testing.expectEqualStrings("111", content[0].get("bytes").?.array[0].int);
    try std.testing.expect(content[1].get("logprob").? == .null); // NaN from the engine: null, never a bare NaN
    const top = content[0].get("top_logprobs").?.array;
    try std.testing.expectEqual(@as(usize, 2), top.len);
    try std.testing.expectEqualStrings("b", top[1].get("token").?.string);
    try std.testing.expectEqual(@as(f64, -1.5), top[1].get("logprob").?.float);
    try std.testing.expect(top[1].get("top_logprobs") == null);
    // the keys in OpenAI's order
    try std.testing.expect(std.mem.indexOf(u8, r.body, "{\"token\": \"o\", \"logprob\": -0.25, \"bytes\": [111], \"top_logprobs\": [{\"token\": \"a\", \"logprob\": -0.5, \"bytes\": [97]}") != null);
    // logprobs: true alone: no alternatives
    const r0 = try reply(a, srv, "/v1/chat/completions", plain ++ ",\"logprobs\":true}");
    try std.testing.expectEqualStrings("HTTP/1.1 200", r0.status);
    try std.testing.expectEqual(@as(?u8, 0), e.asked);
    try std.testing.expect(std.mem.indexOf(u8, r0.body, "\"top_logprobs\": []") != null);
}

test "without logprobs the reply's bytes are those of an engine that reports none, and the engine is asked for none" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: ByteText = .{};
    var with: Fake = .{ .reports = true };
    var without: Fake = .{ .reports = false };
    const s1 = try open(&with, &text);
    defer s1.deinit();
    const s2 = try open(&without, &text);
    defer s2.deinit();
    for ([_][]const u8{ plain ++ "}", plain ++ ",\"logprobs\":false}", plain ++ ",\"logprobs\":null,\"top_logprobs\":null}" }) |body| {
        const r1 = try reply(a, s1, "/v1/chat/completions", body);
        const r2 = try reply(a, s2, "/v1/chat/completions", body);
        try std.testing.expectEqualStrings("HTTP/1.1 200", r1.status);
        try std.testing.expect(!with.saw_logprobs);
        try std.testing.expect(std.mem.indexOf(u8, r1.body, "logprobs") == null);
        try std.testing.expectEqualStrings(try steady(a, r2.body), try steady(a, r1.body));
    }
}

test "refused with a 400: streams, legacy completions, an engine without logprobs, malformed fields" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: ByteText = .{};
    var e: Fake = .{ .reports = true };
    var none: Fake = .{ .reports = false };
    const srv = try open(&e, &text);
    defer srv.deinit();
    const old = try open(&none, &text);
    defer old.deinit();
    const Case = struct { srv: *server.Server, path: []const u8, body: []const u8, says: []const u8 };
    const cases = [_]Case{
        .{ .srv = srv, .path = "/v1/chat/completions", .body = plain ++ ",\"logprobs\":true,\"stream\":true}", .says = "non-streamed" },
        .{ .srv = srv, .path = "/v1/completions", .body = "{\"model\":\"m\",\"max_tokens\":2,\"prompt\":\"hi\",\"logprobs\":true}", .says = "not supported" },
        .{ .srv = old, .path = "/v1/chat/completions", .body = plain ++ ",\"logprobs\":true}", .says = "not supported" },
        .{ .srv = srv, .path = "/v1/chat/completions", .body = plain ++ ",\"logprobs\":true,\"top_logprobs\":21}", .says = "between 0 and 20" },
        .{ .srv = srv, .path = "/v1/chat/completions", .body = plain ++ ",\"top_logprobs\":3}", .says = "requires logprobs" },
        .{ .srv = srv, .path = "/v1/chat/completions", .body = plain ++ ",\"logprobs\":1}", .says = "boolean" },
    };
    for (cases) |c| {
        e.saw_logprobs = false;
        const r = try reply(a, c.srv, c.path, c.body);
        try std.testing.expectEqualStrings("HTTP/1.1 400", r.status);
        try std.testing.expect(std.mem.indexOf(u8, r.body, c.says) != null);
        try std.testing.expect(!e.saw_logprobs);
    }
}
