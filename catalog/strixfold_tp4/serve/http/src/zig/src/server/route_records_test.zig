//! Routing record ids and route_tag on the four generation routes.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const routes = @import("routes.zig");
const DropLate = @import("late_system_test.zig").DropLate;
const Conn = @import("http_conn.zig").Conn;

const gpa = std.testing.allocator;
const chat_ok =
    \\{"model":"m","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}
;

const Box = struct {
    record: [64]u8 = undefined,
    record_len: usize = 0,
    tag: [200]u8 = undefined,
    tag_len: usize = 0,
    request: [96]u8 = undefined,
    request_len: usize = 0,
    submits: u32 = 0,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn of(ctx: *anyopaque) *@This() {
        return @ptrCast(@alignCast(ctx));
    }
    fn info(_: *anyopaque) api.Info {
        return .{};
    }
    fn keep(dst: []u8, len: *usize, src: []const u8) void {
        len.* = @min(dst.len, src.len);
        @memcpy(dst[0..len.*], src[0..len.*]);
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e = of(ctx);
        keep(&e.record, &e.record_len, request.route_record);
        keep(&e.tag, &e.tag_len, request.route_tag);
        keep(&e.request, &e.request_len, request.route_request);
        e.submits += 1;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'h', 'i' } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn recordId(e: *const @This()) []const u8 {
        return e.record[0..e.record_len];
    }
    fn tagText(e: *const @This()) []const u8 {
        return e.tag[0..e.tag_len];
    }
};

const Reply = struct { status: u16, body: []u8 };

fn exchange(srv: *server.Server, method: []const u8, path: []const u8, headers: []const u8, body: []const u8) !Reply {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const head = if (std.mem.eql(u8, method, "GET"))
        try std.fmt.allocPrint(gpa, "GET {s} HTTP/1.1\r\nHost: t\r\n{s}\r\n", .{ path, headers })
    else
        try std.fmt.allocPrint(gpa, "POST {s} HTTP/1.1\r\nHost: t\r\n{s}Content-Length: {d}\r\n\r\n", .{ path, headers, body.len });
    defer gpa.free(head);
    try writeAll(pair[0], head);
    if (!std.mem.eql(u8, method, "GET")) try writeAll(pair[0], body);
    var conn = try Conn.init(gpa, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    if (try conn.readRequest(arena.allocator(), false) != .ready) return error.BadRequest;
    routes.dispatch(srv, &conn, arena.allocator());
    const raw = try readReady(pair[0]);
    errdefer gpa.free(raw);
    const split = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse return error.BadReply;
    const status = std.fmt.parseInt(u16, raw["HTTP/1.1 ".len..][0..3], 10) catch return error.BadReply;
    const payload = try gpa.dupe(u8, raw[split + 4 ..]);
    gpa.free(raw);
    return .{ .status = status, .body = payload };
}

fn writeAll(fd: std.c.fd_t, data: []const u8) !void {
    var sent: usize = 0;
    while (sent < data.len) {
        const n = std.c.write(fd, data[sent..].ptr, data.len - sent);
        if (n <= 0) return error.Closed;
        sent += @intCast(n);
    }
}

fn readReady(fd: std.c.fd_t) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var tmp: [4096]u8 = undefined;
    while (true) {
        var pollfd = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        if (try std.posix.poll(&pollfd, 200) == 0) break;
        const n = std.posix.read(fd, &tmp) catch |e| return e;
        if (n == 0) break;
        try out.appendSlice(gpa, tmp[0..n]);
    }
    return out.toOwnedSlice(gpa);
}

fn expectHas(r: Reply, code: u16, text: []const u8) !void {
    defer gpa.free(r.body);
    try std.testing.expectEqual(code, r.status);
    try std.testing.expect(std.mem.indexOf(u8, r.body, text) != null);
}
fn expectLacks(r: Reply, code: u16, text: []const u8) !void {
    defer gpa.free(r.body);
    try std.testing.expectEqual(code, r.status);
    try std.testing.expect(std.mem.indexOf(u8, r.body, text) == null);
}

fn init(backend: *Box, text: *DropLate, on: bool) !*server.Server {
    return server.Server.init(gpa, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .route_records = on }, null);
}

test "records off: no route keys in any reply and route_tag is ignored, even an invalid one" {
    var text: DropLate = .{};
    var backend: Box = .{};
    var off = try init(&backend, &text, false);
    defer off.deinit();
    const bad =
        \\{"model":"m","max_tokens":4,"messages":[{"role":"user","content":"hi"}],"route_tag":"NOT VALID"}
    ;
    try expectLacks(try exchange(off, "POST", "/v1/chat/completions", "x-route-tag: also bad!\r\n", bad), 200, "route_");
    try std.testing.expectEqual(@as(usize, 0), backend.record_len);
    try std.testing.expectEqual(@as(usize, 0), backend.tag_len);
    try expectLacks(try exchange(off, "POST", "/v1/completions", "", "{\"model\":\"m\",\"prompt\":\"hi\",\"max_tokens\":4,\"route_tag\":1}"), 200, "route_");
    try expectLacks(try exchange(off, "POST", "/v1/messages", "", chat_ok), 200, "route_");
    try expectLacks(try exchange(off, "POST", "/v1/responses", "", "{\"model\":\"m\",\"input\":\"hi\",\"max_output_tokens\":4}"), 200, "route_");
}

test "records on: every route carries tensorfold.route_record and the engine gets the same id" {
    var text: DropLate = .{};
    var backend: Box = .{};
    var on = try init(&backend, &text, true);
    defer on.deinit();
    const cases = [_]struct { path: []const u8, body: []const u8 }{
        .{ .path = "/v1/chat/completions", .body = chat_ok },
        .{ .path = "/v1/chat/completions", .body = "{\"model\":\"m\",\"max_tokens\":4,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" },
        .{ .path = "/v1/completions", .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"max_tokens\":4}" },
        .{ .path = "/v1/messages", .body = chat_ok },
        .{ .path = "/v1/messages", .body = "{\"model\":\"m\",\"max_tokens\":4,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" },
        .{ .path = "/v1/responses", .body = "{\"model\":\"m\",\"input\":\"hi\",\"max_output_tokens\":4}" },
        .{ .path = "/v1/responses", .body = "{\"model\":\"m\",\"input\":\"hi\",\"max_output_tokens\":4,\"stream\":true}" },
    };
    var last: [64]u8 = undefined;
    var last_len: usize = 0;
    for (cases) |c| {
        const r = try exchange(on, "POST", c.path, "x-route-tag: task=t1/arm=a/draw=0/turn=1\r\n", c.body);
        defer gpa.free(r.body);
        try std.testing.expectEqual(@as(u16, 200), r.status);
        const id = backend.recordId();
        try std.testing.expect(std.mem.startsWith(u8, id, "rr-"));
        try std.testing.expect(!std.mem.eql(u8, id, last[0..last_len])); // a new id per request
        var want: [96]u8 = undefined;
        const quoted = try std.fmt.bufPrint(&want, "\"route_record\": \"{s}\"", .{id});
        if (std.mem.indexOf(u8, r.body, quoted) == null) {
            std.debug.print("{s}: no {s} in {s}\n", .{ c.path, quoted, r.body });
            return error.NoRouteRecord;
        }
        try std.testing.expect(std.mem.indexOf(u8, r.body, "\"route_tag\": \"task=t1/arm=a/draw=0/turn=1\"") != null);
        try std.testing.expectEqualStrings("task=t1/arm=a/draw=0/turn=1", backend.tagText());
        try std.testing.expect(backend.request_len > 0);
        last_len = id.len;
        @memcpy(last[0..id.len], id);
    }
    // no tag: route_tag null, the record id still there
    const r = try exchange(on, "POST", "/v1/chat/completions", "", chat_ok);
    defer gpa.free(r.body);
    try std.testing.expect(std.mem.indexOf(u8, r.body, "\"route_tag\": null") != null);
    try std.testing.expect(std.mem.indexOf(u8, r.body, "\"route_record\": \"rr-") != null);
    try std.testing.expectEqual(@as(usize, 0), backend.tag_len);
}

test "records on: the body route_tag wins over the header on all four routes, and a bad tag is a 400" {
    var text: DropLate = .{};
    var backend: Box = .{};
    var on = try init(&backend, &text, true);
    defer on.deinit();
    const bodies = [_]struct { path: []const u8, body: []const u8 }{
        .{ .path = "/v1/chat/completions", .body = "{\"model\":\"m\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"route_tag\": \"body=1\"}" },
        .{ .path = "/v1/completions", .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"max_tokens\":4,\"route_tag\": \"body=1\"}" },
        .{ .path = "/v1/messages", .body = "{\"model\":\"m\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"route_tag\": \"body=1\"}" },
        .{ .path = "/v1/responses", .body = "{\"model\":\"m\",\"input\":\"hi\",\"max_output_tokens\":4,\"route_tag\": \"body=1\"}" },
    };
    for (bodies) |c| {
        backend.tag_len = 0;
        const r = try exchange(on, "POST", c.path, "x-route-tag: header=1\r\n", c.body);
        defer gpa.free(r.body);
        try std.testing.expectEqual(@as(u16, 200), r.status);
        try std.testing.expectEqualStrings("body=1", backend.tagText());
        try std.testing.expect(std.mem.indexOf(u8, r.body, "\"route_tag\": \"body=1\"") != null);
    }
    const before = backend.submits;
    try expectHas(try exchange(on, "POST", "/v1/chat/completions", "x-route-tag: has space\r\n", chat_ok), 400, "route_tag must match");
    try expectHas(try exchange(on, "POST", "/v1/chat/completions", "", "{\"model\":\"m\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"route_tag\": \"\"}"), 400, "route_tag must match");
    try expectHas(try exchange(on, "POST", "/v1/completions", "", "{\"model\":\"m\",\"prompt\":\"hi\",\"route_tag\":7}"), 400, "route_tag must match");
    const long: [161]u8 = @splat('x');
    try expectHas(try exchange(on, "POST", "/v1/chat/completions", "x-route-tag: " ++ long ++ "\r\n", chat_ok), 400, "route_tag must match");
    try std.testing.expectEqual(before, backend.submits); // refused before the engine
    try expectHas(try exchange(on, "POST", "/v1/chat/completions", "x-route-tag: " ++ @as([160]u8, @splat('y')) ++ "\r\n", chat_ok), 200, "\"route_record\"");
}

fn field(a: std.mem.Allocator, body: []const u8, key: []const u8) ![]u8 {
    const parsed = try std.json.parseFromSlice(std.json.Value, a, body, .{});
    defer parsed.deinit();
    const v = parsed.value.object.get(key) orelse return error.NoField;
    return std.json.Stringify.valueAlloc(a, v, .{});
}

test "records on vs off: the generated reply (choices, usage) is the same bytes" {
    var text: DropLate = .{};
    var backend: Box = .{};
    var off = try init(&backend, &text, false);
    defer off.deinit();
    var on = try init(&backend, &text, true);
    defer on.deinit();
    const a = try exchange(off, "POST", "/v1/chat/completions", "x-route-tag: t=1\r\n", chat_ok);
    defer gpa.free(a.body);
    const b = try exchange(on, "POST", "/v1/chat/completions", "x-route-tag: t=1\r\n", chat_ok);
    defer gpa.free(b.body);
    for ([_][]const u8{ "choices", "usage", "model", "object" }) |k| {
        const x = try field(gpa, a.body, k);
        defer gpa.free(x);
        const y = try field(gpa, b.body, k);
        defer gpa.free(y);
        try std.testing.expectEqualStrings(x, y);
    }
}
