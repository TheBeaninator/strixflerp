//! An engine's per-request telemetry (Stats.telemetry_json) reaches the reply as speculative.proposer intact: the
//! reply is written after the request's mailbox, which held the engine's bytes, has been freed.
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

/// Finishes each request at once with telemetry in its own buffer, freed as soon as the event returns (as the lane
/// host frees its job).
const Telemetry = struct {
    const sent = "{\"rounds_by_rows\": {\"1\": [2, 30.500, 465.125]}, \"drafted\": 3, \"graphs\": {\"captured\": 1, \"replayed\": 4, \"eager\": 0}}";

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{};
    }

    fn submit(_: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const own = std.testing.allocator.dupe(u8, sent) catch return error.Busy;
        defer std.testing.allocator.free(own);
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'o', 'k' } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop, .stats = .{ .rounds = 2, .drafted = 3, .telemetry_json = own } } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The whole HTTP reply to one non-streamed request.
fn post(a: std.mem.Allocator, srv: *server.Server, body: []const u8) ![]u8 {
    return send(a, srv, body);
}

fn send(a: std.mem.Allocator, srv: *server.Server, body: []const u8) ![]u8 {
    return sendTo(a, srv, "/v1/chat/completions", body);
}

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

test "an engine's telemetry reaches the reply's speculative.proposer intact after the mailbox is freed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: ByteText = .{};
    var backend: Telemetry = .{};
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false }, null);
    defer srv.deinit();
    const reply = try post(a, srv, "{\"model\":\"m\",\"max_tokens\":2,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
    try std.testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200"));
    const at = std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return error.BadReply;
    const parsed = try json.parse(a, reply[at + 4 ..]);
    try std.testing.expect(parsed == .ok);
    const proposer = parsed.ok.get("speculative").?.get("proposer") orelse return error.NoProposer;
    try std.testing.expectEqualStrings("3", proposer.get("drafted").?.int);
    try std.testing.expectEqualStrings("4", proposer.get("graphs").?.get("replayed").?.int);
    const rows = proposer.get("rounds_by_rows").?.get("1").?.array;
    try std.testing.expectEqual(@as(usize, 3), rows.len);
    try std.testing.expectEqualStrings("2", rows[0].int);
}

/// Finishes each request at once as an engine whose loop policy fired (or not): reason, loop and answer_loop as set;
/// it keeps the policy the request carried.
const Looping = struct {
    reason: api.Reason = .loop,
    loop: ?api.LoopHit = .{ .period = 3, .at_token = 1 },
    answer_loop: ?api.LoopHit = null,
    seen: ?api.LoopPolicy = null,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = Telemetry.info, .submit = submit, .cancel = Telemetry.cancel, .status = Telemetry.status, .memory = Telemetry.memory } };
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *@This() = @ptrCast(@alignCast(ctx));
        e.seen = request.loop_policy;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'o', 'k' } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = e.reason, .stats = .{ .rounds = 2, .loop = e.loop, .answer_loop = e.answer_loop } } });
    }
};

fn loopReply(a: std.mem.Allocator, policy: api.LoopPolicy, backend: *Looping, stream: bool) !struct { body: json.Value, loops: [2]u64 } {
    var text: ByteText = .{};
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false, .loop_policy = policy }, null);
    defer srv.deinit();
    const reply = try send(a, srv, if (stream) "{\"model\":\"m\",\"max_tokens\":8,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" else "{\"model\":\"m\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
    try std.testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200"));
    const loops = .{ srv.metrics.loops_thinking, srv.metrics.loops_answer };
    if (!stream) {
        const at = std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return error.BadReply;
        const parsed = try json.parse(a, reply[at + 4 ..]);
        try std.testing.expect(parsed == .ok);
        return .{ .body = parsed.ok, .loops = loops };
    }
    // the final chunk: the last data line that carries a finish_reason
    var last: ?json.Value = null;
    var lines = std.mem.splitScalar(u8, reply, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trimEnd(u8, raw, "\r");
        if (!std.mem.startsWith(u8, line, "data: {")) continue;
        const parsed = try json.parse(a, line["data: ".len..]);
        if (parsed != .ok) continue;
        const choices = parsed.ok.get("choices") orelse continue;
        if (choices != .array or choices.array.len == 0) continue;
        const finish = choices.array[0].get("finish_reason") orelse continue;
        if (finish == .string) last = parsed.ok;
    }
    return .{ .body = last orelse return error.NoFinalChunk, .loops = loops };
}

test "loop policy stop: finish_reason loop and tensorfold.loop with period and at_token, non-streamed and in the final chunk" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |stream| {
        var backend: Looping = .{};
        const got = try loopReply(a, .stop, &backend, stream);
        try std.testing.expectEqual(api.LoopPolicy.stop, backend.seen.?);
        try std.testing.expectEqualStrings("loop", got.body.get("choices").?.array[0].get("finish_reason").?.string);
        const loop = got.body.get("tensorfold").?.get("loop") orelse return error.NoLoopField;
        try std.testing.expectEqualStrings("stop", loop.get("policy").?.string);
        try std.testing.expect(loop.get("fired").?.bool);
        try std.testing.expectEqualStrings("3", loop.get("period").?.int);
        try std.testing.expectEqualStrings("1", loop.get("at_token").?.int);
        try std.testing.expectEqualStrings("thinking", loop.get("in").?.string);
        try std.testing.expect(loop.get("answer") == null);
        try std.testing.expectEqual([2]u64{ 1, 0 }, got.loops);
    }
}

test "loop policy count: tensorfold.loop reports a quiet reply, an answer loop, and both parts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var quiet: Looping = .{ .reason = .length, .loop = null };
    const q = try loopReply(a, .count, &quiet, false);
    try std.testing.expectEqualStrings("length", q.body.get("choices").?.array[0].get("finish_reason").?.string);
    const ql = q.body.get("tensorfold").?.get("loop").?;
    try std.testing.expectEqualStrings("count", ql.get("policy").?.string);
    try std.testing.expect(!ql.get("fired").?.bool);
    try std.testing.expect(ql.get("period").? == .null and ql.get("at_token").? == .null and ql.get("in").? == .null);
    try std.testing.expectEqual([2]u64{ 0, 0 }, q.loops);

    var answer: Looping = .{ .reason = .length, .loop = null, .answer_loop = .{ .period = 2, .at_token = 1 } };
    const an = try loopReply(a, .count, &answer, false);
    const al = an.body.get("tensorfold").?.get("loop").?;
    try std.testing.expectEqualStrings("answer", al.get("in").?.string);
    try std.testing.expectEqualStrings("2", al.get("period").?.int);
    try std.testing.expectEqual([2]u64{ 0, 1 }, an.loops);

    var both: Looping = .{ .reason = .length, .answer_loop = .{ .period = 2, .at_token = 1 } };
    const b = try loopReply(a, .close, &both, true);
    const bl = b.body.get("tensorfold").?.get("loop").?;
    try std.testing.expectEqualStrings("close", bl.get("policy").?.string);
    try std.testing.expectEqualStrings("thinking", bl.get("in").?.string);
    try std.testing.expectEqualStrings("2", bl.get("answer").?.get("period").?.int);
    try std.testing.expectEqual([2]u64{ 1, 1 }, b.loops);
}

test "loop policy off: the request carries off and the reply has no tensorfold.loop" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var backend: Looping = .{ .reason = .stop, .loop = null };
    const got = try loopReply(a, .off, &backend, false);
    try std.testing.expectEqual(api.LoopPolicy.off, backend.seen.?);
    try std.testing.expect(got.body.get("tensorfold").?.get("loop") == null);
}
/// One request to /v1/messages or /v1/responses against an engine whose loop outcome is set, as the client sees it:
/// the whole JSON body, or for a stream the last SSE data object whose "type" is one of `final`.
fn apiReply(a: std.mem.Allocator, backend: *Looping, path: []const u8, body: []const u8, stream: bool, final: []const []const u8) !json.Value {
    var text: ByteText = .{};
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false, .loop_policy = .stop }, null);
    defer srv.deinit();
    const reply = try sendTo(a, srv, path, body);
    try std.testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200"));
    if (!stream) {
        const at = std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return error.BadReply;
        const parsed = try json.parse(a, reply[at + 4 ..]);
        try std.testing.expect(parsed == .ok);
        return parsed.ok;
    }
    var last: ?json.Value = null;
    var lines = std.mem.splitScalar(u8, reply, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trimEnd(u8, raw, "\r");
        if (!std.mem.startsWith(u8, line, "data: {")) continue;
        const parsed = try json.parse(a, line["data: ".len..]);
        if (parsed != .ok) continue;
        const kind = parsed.ok.get("type") orelse continue;
        if (kind != .string) continue;
        for (final) |f| if (std.mem.eql(u8, f, kind.string)) {
            last = parsed.ok;
        };
    }
    return last orelse error.NoFinalEvent;
}

fn expectLoopField(tensorfold: ?json.Value) !void {
    const loop = (tensorfold orelse return error.NoTensorfold).get("loop") orelse return error.NoLoopField;
    try std.testing.expectEqualStrings("stop", loop.get("policy").?.string);
    try std.testing.expect(loop.get("fired").?.bool);
    try std.testing.expectEqualStrings("3", loop.get("period").?.int);
    try std.testing.expectEqualStrings("1", loop.get("at_token").?.int);
    try std.testing.expectEqualStrings("thinking", loop.get("in").?.string);
}

const messages_body = "{\"model\":\"m\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
const messages_stream_body = "{\"model\":\"m\",\"max_tokens\":8,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
const responses_body = "{\"model\":\"m\",\"input\":\"hi\",\"max_output_tokens\":8}";
const responses_stream_body = "{\"model\":\"m\",\"input\":\"hi\",\"max_output_tokens\":8,\"stream\":true}";

test "loop policy stop on /v1/messages: stop_reason end_turn, stop_details loop and tensorfold.loop, non-streamed and streamed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var backend: Looping = .{};
    const plain = try apiReply(a, &backend, "/v1/messages", messages_body, false, &.{});
    try std.testing.expectEqualStrings("end_turn", plain.get("stop_reason").?.string);
    const d = plain.get("stop_details") orelse return error.NoStopDetails;
    try std.testing.expectEqualStrings("loop", d.get("type").?.string);
    try std.testing.expectEqualStrings("3", d.get("period").?.int);
    try std.testing.expectEqualStrings("1", d.get("at_token").?.int);
    try std.testing.expectEqualStrings("thinking", d.get("in").?.string);
    try expectLoopField(plain.get("tensorfold"));

    const sse = try apiReply(a, &backend, "/v1/messages", messages_stream_body, true, &.{"message_delta"});
    const delta = sse.get("delta").?;
    try std.testing.expectEqualStrings("end_turn", delta.get("stop_reason").?.string);
    const sd = delta.get("stop_details") orelse return error.NoStopDetails;
    try std.testing.expectEqualStrings("loop", sd.get("type").?.string);
    try std.testing.expectEqualStrings("3", sd.get("period").?.int);
    try expectLoopField(sse.get("tensorfold"));
}

test "a Messages reply that did not loop has no stop_details; a length cut stays max_tokens" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var normal: Looping = .{ .reason = .stop, .loop = null };
    const n = try apiReply(a, &normal, "/v1/messages", messages_body, false, &.{});
    try std.testing.expectEqualStrings("end_turn", n.get("stop_reason").?.string);
    try std.testing.expect(n.get("stop_details") == null);
    try std.testing.expectEqualStrings("stop", n.get("tensorfold").?.get("loop").?.get("policy").?.string);
    try std.testing.expect(!n.get("tensorfold").?.get("loop").?.get("fired").?.bool);
    var cut: Looping = .{ .reason = .length, .loop = null };
    const c = try apiReply(a, &cut, "/v1/messages", messages_stream_body, true, &.{"message_delta"});
    try std.testing.expectEqualStrings("max_tokens", c.get("delta").?.get("stop_reason").?.string);
    try std.testing.expect(c.get("delta").?.get("stop_details") == null);
}

test "loop policy stop on /v1/responses: status incomplete, incomplete_details.reason loop and tensorfold.loop, non-streamed and streamed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var backend: Looping = .{};
    const plain = try apiReply(a, &backend, "/v1/responses", responses_body, false, &.{});
    try std.testing.expectEqualStrings("incomplete", plain.get("status").?.string);
    try std.testing.expectEqualStrings("loop", plain.get("incomplete_details").?.get("reason").?.string);
    try expectLoopField(plain.get("tensorfold"));

    const sse = try apiReply(a, &backend, "/v1/responses", responses_stream_body, true, &.{ "response.incomplete", "response.completed" });
    try std.testing.expectEqualStrings("response.incomplete", sse.get("type").?.string);
    const resp = sse.get("response").?;
    try std.testing.expectEqualStrings("incomplete", resp.get("status").?.string);
    try std.testing.expectEqualStrings("loop", resp.get("incomplete_details").?.get("reason").?.string);
    try expectLoopField(resp.get("tensorfold"));
}

test "a Responses reply that did not loop is completed; a length cut keeps max_output_tokens" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var normal: Looping = .{ .reason = .stop, .loop = null };
    const n = try apiReply(a, &normal, "/v1/responses", responses_body, false, &.{});
    try std.testing.expectEqualStrings("completed", n.get("status").?.string);
    try std.testing.expect(n.get("incomplete_details").? == .null);
    var cut: Looping = .{ .reason = .length, .loop = null };
    const c = try apiReply(a, &cut, "/v1/responses", responses_body, false, &.{});
    try std.testing.expectEqualStrings("incomplete", c.get("status").?.string);
    try std.testing.expectEqualStrings("max_output_tokens", c.get("incomplete_details").?.get("reason").?.string);
}
