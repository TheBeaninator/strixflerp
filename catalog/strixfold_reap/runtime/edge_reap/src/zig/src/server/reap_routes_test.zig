//! GET /stats/reap and reap_domain on the chat, completions and messages bodies.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const cache_key = @import("cache_key.zig");
const routes = @import("routes.zig");
const DropLate = @import("late_system_test.zig").DropLate;
const Conn = @import("http_conn.zig").Conn;

const gpa = std.testing.allocator;
const key_text = "0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c\nsk-move\n";
const bearer = "Authorization: Bearer sk-move\r\n";
const chat_ok =
    \\{"model":"m","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}
;

const Box = struct {
    bucket: u32 = 0,
    saw_submit: bool = false,
    admitted: [4][32]u8 = undefined,
    alen: [4]u8 = @splat(0),
    n: u32 = 0,
    cap: u32 = 2,
    q_domain: [64]u8 = undefined,
    q_domain_len: usize = 0,
    q_mod: [16]u8 = undefined,
    q_mod_len: usize = 0,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .reap = reap, .reap_admit = admit } };
    }
    fn of(ctx: *anyopaque) *@This() {
        return @ptrCast(@alignCast(ctx));
    }
    fn info(_: *anyopaque) api.Info {
        return .{};
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e = of(ctx);
        e.bucket = request.reap_bucket;
        e.saw_submit = true;
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
    fn reap(ctx: *anyopaque, q: *api.ReapQuery) void {
        const e = of(ctx);
        e.q_domain_len = @min(q.domain.len, e.q_domain.len);
        @memcpy(e.q_domain[0..e.q_domain_len], q.domain[0..e.q_domain_len]);
        e.q_mod_len = @min(q.modality.len, e.q_mod.len);
        @memcpy(e.q_mod[0..e.q_mod_len], q.modality[0..e.q_mod_len]);
        if (std.mem.eql(u8, q.modality, "nope")) {
            q.status = 400;
            q.out = null;
            return;
        }
        if (std.mem.eql(u8, q.domain, "missing")) {
            q.status = 404;
            q.out = null;
            return;
        }
        if (std.mem.eql(u8, q.domain, "stall")) {
            q.status = 503;
            q.out = q.allocator.dupe(u8, "{\"error\":{\"message\":\"reap stats are refreshing\"}}") catch null;
            return;
        }
        const body = if (q.domain.len == 0)
            \\{"buckets":[{"domain":"default","modality":"text","tokens":3,"norm_rows":1}]}
        else
            \\{"model_id":"m","num_layers":40,"num_experts":384,"reap_saliency":[],"reap_weighted_norm_sum":[],"gate_mass_sum":[],"token_count":[],"max_l2_norm":[],"early_pos_count":[],"domain":"default","modality":"all","tokens":1,"norm_rows":1,"sampled":{"norm_count":[],"reap_weighted_norm_sum":[]},"build":"1.0.2"}
        ;
        q.status = 200;
        q.out = q.allocator.dupe(u8, body) catch null;
        if (q.out == null) q.status = 500;
    }
    fn admit(ctx: *anyopaque, domain: []const u8, index: *u32, err: []u8) usize {
        const e = of(ctx);
        var i: u32 = 0;
        while (i < e.n) : (i += 1) {
            if (std.mem.eql(u8, e.admitted[i][0..e.alen[i]], domain)) {
                index.* = i;
                return 0;
            }
        }
        if (e.n >= e.cap) {
            const m = std.fmt.bufPrint(err, "reap_domain cap {d}: refusing new domain \"{s}\"", .{ e.cap, domain }) catch err[0..0];
            return m.len;
        }
        const n = @min(domain.len, e.admitted[e.n].len);
        @memcpy(e.admitted[e.n][0..n], domain[0..n]);
        e.alen[e.n] = @intCast(n);
        index.* = e.n;
        e.n += 1;
        return 0;
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

test "stats route is absent until stats and the transfer key are both on" {
    var text: DropLate = .{};
    var backend: Box = .{};
    var off = try server.Server.init(gpa, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer off.deinit();
    try expectHas(try exchange(off, "GET", "/stats/reap", bearer, ""), 404, "unknown path");
    var problem: []const u8 = "";
    const key = try cache_key.Key.parse(key_text, &problem);
    var keyed = try server.Server.init(gpa, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .transfer = key }, null);
    defer keyed.deinit();
    try expectHas(try exchange(keyed, "GET", "/stats/reap", bearer, ""), 404, "unknown path");
    var on = try server.Server.init(gpa, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .transfer = key, .reap_stats = true }, null);
    defer on.deinit();
    try expectHas(try exchange(on, "GET", "/v1/stats/reap", "", ""), 401, "cache transfer key is required");
    try expectHas(try exchange(on, "GET", "/stats/reap", bearer, ""), 200, "\"buckets\"");
    try expectHas(try exchange(on, "GET", "/stats/reap?domain=missing", bearer, ""), 404, "unknown reap domain");
    try expectHas(try exchange(on, "GET", "/stats/reap?domain=default&modality=nope", bearer, ""), 400, "modality must be text");
    const report = try exchange(on, "GET", "/stats/reap?domain=default&modality=all", bearer, "");
    defer gpa.free(report.body);
    try std.testing.expectEqual(@as(u16, 200), report.status);
    const keys = [_][]const u8{ "\"model_id\"", "\"num_layers\"", "\"num_experts\"", "\"reap_saliency\"", "\"reap_weighted_norm_sum\"", "\"gate_mass_sum\"", "\"token_count\"", "\"max_l2_norm\"", "\"early_pos_count\"", "\"domain\"", "\"modality\"", "\"tokens\"", "\"norm_rows\"", "\"sampled\"", "\"build\"" };
    var at: usize = 0;
    for (keys) |k| {
        const i = std.mem.indexOfPos(u8, report.body, at, k) orelse return error.MissingKey;
        at = i + k.len;
    }
    try std.testing.expectEqualStrings("default", backend.q_domain[0..backend.q_domain_len]);
    try std.testing.expectEqualStrings("all", backend.q_mod[0..backend.q_mod_len]);
}

test "reap_domain is validated only while stats are on, and a new name past the cap is a 400" {
    var text: DropLate = .{};
    var backend: Box = .{};
    var problem: []const u8 = "";
    const key = try cache_key.Key.parse(key_text, &problem);
    var off = try server.Server.init(gpa, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .transfer = key }, null);
    defer off.deinit();
    const bad_off =
        \\{"model":"m","max_tokens":4,"messages":[{"role":"user","content":"hi"}],"reap_domain":"NOT VALID"}
    ;
    try expectHas(try exchange(off, "POST", "/v1/chat/completions", "", bad_off), 200, "\"choices\"");
    try std.testing.expectEqual(@as(u32, 0), backend.bucket);

    var on = try server.Server.init(gpa, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .transfer = key, .reap_stats = true }, null);
    defer on.deinit();
    try expectHas(try exchange(on, "POST", "/v1/chat/completions", "", bad_off), 400, "reap_domain must match");
    const bad_comp =
        \\{"model":"m","prompt":"hi","reap_domain":1}
    ;
    try expectHas(try exchange(on, "POST", "/v1/completions", "", bad_comp), 400, "reap_domain must match");
    try expectHas(try exchange(on, "POST", "/v1/chat/completions", "x-reap-domain: alpha\r\n", chat_ok), 200, "\"choices\"");
    try std.testing.expectEqual(@as(u32, 0), backend.bucket);
    try std.testing.expectEqualStrings("alpha", backend.admitted[0][0..backend.alen[0]]);
    const body_wins =
        \\{"model":"m","max_tokens":4,"messages":[{"role":"user","content":"hi"}],"reap_domain":"beta"}
    ;
    try expectHas(try exchange(on, "POST", "/v1/chat/completions", "x-reap-domain: alpha\r\n", body_wins), 200, "\"choices\"");
    try std.testing.expectEqual(@as(u32, 1), backend.bucket);
    try expectHas(try exchange(on, "POST", "/v1/chat/completions", "x-reap-domain: gamma\r\n", chat_ok), 400, "reap_domain cap 2");
    const msg =
        \\{"model":"m","max_tokens":4,"messages":[{"role":"user","content":"hi"}],"reap_domain":"alpha"}
    ;
    try expectHas(try exchange(on, "POST", "/v1/messages", "", msg), 200, "content");
    try std.testing.expectEqual(@as(u32, 0), backend.bucket);
}

test "x-reap-domain is honoured on completions and messages, and the body wins" {
    var text: DropLate = .{};
    var backend: Box = .{ .cap = 4 };
    var problem: []const u8 = "";
    const key = try cache_key.Key.parse(key_text, &problem);
    var on = try server.Server.init(gpa, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .transfer = key, .reap_stats = true }, null);
    defer on.deinit();
    const comp =
        \\{"model":"m","prompt":"hi","max_tokens":4}
    ;
    try expectHas(try exchange(on, "POST", "/v1/completions", "x-reap-domain: gamma\r\n", comp), 200, "\"choices\"");
    try std.testing.expectEqual(@as(u32, 0), backend.bucket);
    try std.testing.expectEqualStrings("gamma", backend.admitted[0][0..backend.alen[0]]);
    const msg =
        \\{"model":"m","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}
    ;
    try expectHas(try exchange(on, "POST", "/v1/messages", "x-reap-domain: east\r\n", msg), 200, "content");
    try std.testing.expectEqual(@as(u32, 1), backend.bucket);
    try std.testing.expectEqualStrings("east", backend.admitted[1][0..backend.alen[1]]);
    const msg_body =
        \\{"model":"m","max_tokens":4,"messages":[{"role":"user","content":"hi"}],"reap_domain":"west"}
    ;
    try expectHas(try exchange(on, "POST", "/v1/messages", "x-reap-domain: east\r\n", msg_body), 200, "content");
    try std.testing.expectEqual(@as(u32, 2), backend.bucket);
    try std.testing.expectEqualStrings("west", backend.admitted[2][0..backend.alen[2]]);
    const comp_body =
        \\{"model":"m","prompt":"hi","max_tokens":4,"reap_domain":"north"}
    ;
    try expectHas(try exchange(on, "POST", "/v1/completions", "x-reap-domain: gamma\r\n", comp_body), 200, "\"choices\"");
    try std.testing.expectEqual(@as(u32, 3), backend.bucket);
    try std.testing.expectEqualStrings("north", backend.admitted[3][0..backend.alen[3]]);
}

fn exchangeRaw(srv: *server.Server, method: []const u8, path: []const u8, headers: []const u8, body: []const u8) ![]u8 {
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
    return readReady(pair[0]);
}

test "a stalled reap query is 503 with Retry-After" {
    var text: DropLate = .{};
    var backend: Box = .{};
    var problem: []const u8 = "";
    const key = try cache_key.Key.parse(key_text, &problem);
    var on = try server.Server.init(gpa, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .transfer = key, .reap_stats = true }, null);
    defer on.deinit();
    const raw = try exchangeRaw(on, "GET", "/stats/reap?domain=stall", bearer, "");
    defer gpa.free(raw);
    try std.testing.expect(std.mem.startsWith(u8, raw, "HTTP/1.1 503 "));
    try std.testing.expect(std.mem.indexOf(u8, raw, "Retry-After: 1\r\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, raw, "reap stats are refreshing") != null);
}
