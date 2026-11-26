//! A request's `acts` on the generation routes: the admin key, the field's checks, what reaches
//! the engine, the reply's tensorfold.acts and done.json; and with the field unused, the request and the reply as before.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const routes = @import("routes.zig");
const cache_key = @import("cache_key.zig");
const acts_fields = @import("acts_fields.zig");
const errors = @import("errors.zig");
const DropLate = @import("late_system_test.zig").DropLate;
const Conn = @import("http_conn.zig").Conn;

const gpa = std.testing.allocator;
const key_text = "0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c\nsk-move\n";
const bearer = "Authorization: Bearer sk-move\r\n";
const chat_ok =
    \\{"model":"m","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}
;

fn chatWith(comptime acts: []const u8) []const u8 {
    return "{\"model\":\"m\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"acts\":" ++ acts ++ "}";
}

const Box = struct {
    acts_info: ?api.acts.Info = null,
    last: api.acts.Spec = .{},
    had_req: bool = false,
    id: [64]u8 = undefined, // last.id's bytes (the request's own die with it)
    id_len: usize = 0,
    prompt: [256]u32 = undefined,
    prompt_len: usize = 0,
    images: usize = 0,
    chunks: usize = 0,
    submits: u32 = 0,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .cache_shape = cacheShape } };
    }
    fn cacheShape(_: *anyopaque) ?api.CacheShape {
        return .{ .ns = @splat(5), .block_tokens = 2, .max_blocks = 4, .max_positions = 8, .slot_bytes = 3, .block_bytes = 4 };
    }
    fn of(ctx: *anyopaque) *@This() {
        return @ptrCast(@alignCast(ctx));
    }
    fn info(ctx: *anyopaque) api.Info {
        return .{ .acts = of(ctx).acts_info };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e = of(ctx);
        e.last = request.acts;
        e.id_len = @min(request.acts.id.len, e.id.len);
        @memcpy(e.id[0..e.id_len], request.acts.id[0..e.id_len]);
        e.last.id = "";
        e.had_req = request.acts.req != null;
        e.prompt_len = @min(request.prompt.len, e.prompt.len);
        @memcpy(e.prompt[0..e.prompt_len], request.prompt[0..e.prompt_len]);
        e.images = request.images.len;
        e.chunks = request.chunks.len;
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
};

const Reply = struct { status: u16, body: []u8 };

fn exchange(srv: *server.Server, path: []const u8, headers: []const u8, body: []const u8) !Reply {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const head = try std.fmt.allocPrint(gpa, "POST {s} HTTP/1.1\r\nHost: t\r\n{s}Content-Length: {d}\r\n\r\n", .{ path, headers, body.len });
    defer gpa.free(head);
    try writeAll(pair[0], head);
    try writeAll(pair[0], body);
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
    const code = std.fmt.parseInt(u16, raw["HTTP/1.1 ".len..][0..3], 10) catch return error.BadReply;
    const payload = try gpa.dupe(u8, raw[split + 4 ..]);
    gpa.free(raw);
    return .{ .status = code, .body = payload };
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
    if (r.status != code or std.mem.indexOf(u8, r.body, text) == null) {
        std.debug.print("want {d} with {s}, got {d}: {s}\n", .{ code, text, r.status, r.body });
        return error.UnexpectedReply;
    }
}

fn init(backend: *Box, text: *DropLate, key: bool) !*server.Server {
    var problem: []const u8 = "";
    const k: ?cache_key.Key = if (key) try cache_key.Key.parse(key_text, &problem) else null;
    return server.Server.init(gpa, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .transfer = k }, null);
}

const Dir = struct {
    tmp: std.testing.TmpDir,
    path: []u8,
    fn open() !Dir {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        errdefer tmp.cleanup();
        const path = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ std.testing.TmpDir.parent_dir_path, &tmp.sub_path });
        return .{ .tmp = tmp, .path = path };
    }
    fn close(d: *Dir) void {
        gpa.free(d.path);
        d.tmp.cleanup();
    }
};

fn actsInfo(w: *api.acts.Writer, rows_dir: []const u8) api.acts.Info {
    return .{ .layers = 43, .window = 128, .row_bytes = 6, .rows_dir = rows_dir, .writer = w };
}

test "acts unused: the engine gets the same request with the facility on or off, and the reply has no acts" {
    var text: DropLate = .{};
    var d = try Dir.open();
    defer d.close();
    const w = try api.acts.Writer.open(gpa, std.testing.io, d.path, .{});
    defer w.close();
    var plain: Box = .{};
    var off = try init(&plain, &text, true);
    defer off.deinit();
    var withacts: Box = .{ .acts_info = actsInfo(w, "") };
    var on = try init(&withacts, &text, true);
    defer on.deinit();
    const a = try exchange(off, "/v1/chat/completions", bearer, chat_ok);
    defer gpa.free(a.body);
    const b = try exchange(on, "/v1/chat/completions", bearer, chat_ok);
    defer gpa.free(b.body);
    try std.testing.expectEqual(@as(u16, 200), a.status);
    try std.testing.expectEqual(@as(u16, 200), b.status);
    try std.testing.expect(!plain.last.active() and !withacts.last.active() and !withacts.had_req);
    try std.testing.expectEqualSlices(u32, plain.prompt[0..plain.prompt_len], withacts.prompt[0..withacts.prompt_len]);
    try std.testing.expectEqual(plain.chunks, withacts.chunks);
    try std.testing.expect(std.mem.indexOf(u8, b.body, "\"acts\"") == null);
    // the reply's bytes past its id and timings: choices and usage
    for ([_][]const u8{ "\"choices\"", "\"usage\"" }) |k| {
        const x = a.body[std.mem.indexOf(u8, a.body, k).?..];
        const y = b.body[std.mem.indexOf(u8, b.body, k).?..];
        try std.testing.expectEqualStrings(x[0..@min(x.len, 60)], y[0..@min(y.len, 60)]);
    }
}

test "acts: the admin key before anything else (403 with no key on the server, 401 without its bearer), never ignored" {
    var text: DropLate = .{};
    var d = try Dir.open();
    defer d.close();
    const w = try api.acts.Writer.open(gpa, std.testing.io, d.path, .{});
    defer w.close();
    var backend: Box = .{ .acts_info = actsInfo(w, "") };
    var nokey = try init(&backend, &text, false);
    defer nokey.deinit();
    try expectHas(try exchange(nokey, "/v1/chat/completions", bearer, chatWith("{\"logits\":true}")), 403, acts_fields.no_key);
    var on = try init(&backend, &text, true);
    defer on.deinit();
    const logits = chatWith("{\"logits\":true}");
    try expectHas(try exchange(on, "/v1/chat/completions", "", logits), 401, acts_fields.bad_key);
    try expectHas(try exchange(on, "/v1/chat/completions", "Authorization: Bearer sk-api\r\n", logits), 401, acts_fields.bad_key);
    try expectHas(try exchange(on, "/v1/chat/completions", bearer ++ bearer, logits), 401, acts_fields.bad_key); // twice: not one key
    // the key first: a broken field without the bearer is a 401, not a 400
    try expectHas(try exchange(on, "/v1/chat/completions", "", chatWith("{\"nonsense\":1}")), 401, acts_fields.bad_key);
    try expectHas(try exchange(on, "/v1/completions", "", "{\"model\":\"m\",\"prompt\":\"hi\",\"max_tokens\":4,\"acts\":{\"logits\":true}}"), 401, acts_fields.bad_key);
    try expectHas(try exchange(on, "/v1/messages", "", logits), 401, "admin key");
    try expectHas(try exchange(on, "/v1/responses", "", "{\"model\":\"m\",\"input\":\"hi\",\"max_output_tokens\":4,\"acts\":{\"logits\":true}}"), 401, "admin key");
    // on a /cache route the field is refused outright, even with the bearer
    try expectHas(try exchange(on, "/cache/probe", bearer, chatWith("{\"logits\":true}")), 400, acts_fields.elsewhere);
    try std.testing.expectEqual(@as(u32, 0), backend.submits); // nothing reached the engine
    // a server whose conf names no ACTS_DIR: with the bearer, a 400 (off)
    var plain: Box = .{};
    var off = try init(&plain, &text, true);
    defer off.deinit();
    try expectHas(try exchange(off, "/v1/chat/completions", bearer, logits), 400, acts_fields.off);
    try std.testing.expectEqual(@as(u32, 0), plain.submits);
}

test "acts: the field's checks are 400s before the engine: unknown keys, taps, combinations, the prompt" {
    var text: DropLate = .{};
    var d = try Dir.open();
    defer d.close();
    const w = try api.acts.Writer.open(gpa, std.testing.io, d.path, .{});
    defer w.close();
    var backend: Box = .{ .acts_info = actsInfo(w, "") };
    var on = try init(&backend, &text, true);
    defer on.deinit();
    const P = "/v1/chat/completions";
    try expectHas(try exchange(on, P, bearer, chatWith("true")), 400, "acts must be an object");
    try expectHas(try exchange(on, P, bearer, chatWith("{}")), 400, "turn on at least one");
    try expectHas(try exchange(on, P, bearer, chatWith("{\"dump\":false}")), 400, "turn on at least one");
    try expectHas(try exchange(on, P, bearer, chatWith("{\"dumb\":true}")), 400, "unknown field \\\"dumb\\\"");
    try expectHas(try exchange(on, P, bearer, chatWith("{\"dump\":1}")), 400, "acts.dump must be true or false");
    try expectHas(try exchange(on, P, bearer, chatWith("{\"dump\":true,\"text\":true}")), 400, "exclude each other");
    try expectHas(try exchange(on, P, bearer, chatWith("{\"logits\":true,\"taps\":[1]}")), 400, "acts.taps needs acts.dump or acts.text");
    try expectHas(try exchange(on, P, bearer, chatWith("{\"text\":true,\"taps\":[43]}")), 400, "layer 43 is not one of this model's: -1 (embed) to 42 (head)");
    try expectHas(try exchange(on, P, bearer, chatWith("{\"text\":true,\"taps\":[-2]}")), 400, "layer -2 is not one");
    try expectHas(try exchange(on, P, bearer, chatWith("{\"text\":true,\"taps\":[3,3]}")), 400, "layer 3 twice");
    try expectHas(try exchange(on, P, bearer, chatWith("{\"text\":true,\"taps\":[\"top\"]}")), 400, "is not a layer");
    try expectHas(try exchange(on, P, bearer, chatWith("{\"text\":true,\"taps\":[1.5]}")), 400, "each layer is a number");
    try expectHas(try exchange(on, P, bearer, chatWith("{\"text\":true,\"taps\":[]}")), 400, "a list of 1 to 16 layers");
    try expectHas(try exchange(on, P, bearer, chatWith("{\"text\":true,\"taps\":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}")), 400, "a list of 1 to 16 layers");
    try expectHas(try exchange(on, P, bearer, chatWith("{\"dump\":true}")), 400, "acts.dump needs a picture");
    try expectHas(try exchange(on, P, bearer, chatWith("{\"rows\":true}")), 400, "acts.rows needs ACTS_ROWS_DIR");
    const long_prompt = "{\"model\":\"m\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"" ++ @as([200]u8, @splat('x')) ++ "\"}],\"acts\":{\"text\":true}}";
    try expectHas(try exchange(on, P, bearer, long_prompt), 400, "acts.text needs a text-only prompt of at most 128 tokens");
    try std.testing.expectEqual(@as(u32, 0), backend.submits);
}

fn waitDone(d: *Dir, id: []const u8) ![]u8 {
    var pb: [256]u8 = undefined;
    const sub = try std.fmt.bufPrint(&pb, "{s}/done.json", .{id});
    var tries: usize = 0;
    while (tries < 2000) : (tries += 1) {
        if (d.tmp.dir.readFileAlloc(std.testing.io, sub, gpa, .limited(4096))) |t| return t else |_| {}
        std.Io.sleep(std.testing.io, .fromMilliseconds(2), .awake) catch {};
    }
    return error.NoDoneJson;
}

test "acts admitted: the engine gets the switches, the taps and an open record; the reply names the directory; done.json" {
    var text: DropLate = .{};
    var d = try Dir.open();
    defer d.close();
    const w = try api.acts.Writer.open(gpa, std.testing.io, d.path, .{});
    defer w.close();
    var backend: Box = .{ .acts_info = actsInfo(w, "") };
    var on = try init(&backend, &text, true);
    defer on.deinit();
    const cases = [_]struct { path: []const u8, body: []const u8, hdr: []const u8 }{
        .{ .path = "/v1/chat/completions", .body = chatWith("{\"text\":true,\"taps\":[\"embed\",10,\"head\"],\"logits\":true}"), .hdr = bearer },
        .{ .path = "/v1/chat/completions", .body = "{\"model\":\"m\",\"max_tokens\":4,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"acts\":{\"text\":true,\"taps\":[\"embed\",10,\"head\"],\"logits\":true}}", .hdr = "x-cache-transfer-key: sk-move\r\n" },
        .{ .path = "/v1/completions", .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"max_tokens\":4,\"acts\":{\"text\":true,\"taps\":[\"embed\",10,\"head\"],\"logits\":true}}", .hdr = bearer },
        .{ .path = "/v1/messages", .body = chatWith("{\"text\":true,\"taps\":[\"embed\",10,\"head\"],\"logits\":true}"), .hdr = bearer },
        .{ .path = "/v1/responses", .body = "{\"model\":\"m\",\"input\":\"hi\",\"max_output_tokens\":4,\"acts\":{\"text\":true,\"taps\":[\"embed\",10,\"head\"],\"logits\":true}}", .hdr = bearer },
    };
    for (cases) |c| {
        const r = try exchange(on, c.path, c.hdr, c.body);
        defer gpa.free(r.body);
        if (r.status != 200) {
            std.debug.print("{s}: {d} {s}\n", .{ c.path, r.status, r.body });
            return error.NotAdmitted;
        }
        const s = backend.last;
        try std.testing.expect(s.text and s.logits and !s.dump and !s.rows and backend.had_req);
        try std.testing.expectEqualSlices(i32, &.{ -1, 10, 42 }, s.tapList());
        const id = backend.id[0..backend.id_len];
        try std.testing.expect(std.mem.startsWith(u8, id, "acts-"));
        const done = try waitDone(&d, id);
        defer gpa.free(done);
        try std.testing.expect(std.mem.indexOf(u8, done, "\"finish\": \"stop\", \"files\": 0") != null);
        // chat and completions carry the tensorfold block; its acts names the request's directory
        if (std.mem.indexOf(u8, c.path, "messages") == null and std.mem.indexOf(u8, c.path, "responses") == null) {
            var want: [600]u8 = undefined;
            const dir = try std.fmt.bufPrint(&want, "\"dir\": \"{s}/{s}\"", .{ d.path, id });
            if (std.mem.indexOf(u8, r.body, dir) == null) {
                std.debug.print("{s}: no {s} in {s}\n", .{ c.path, dir, r.body });
                return error.NoActsField;
            }
        }
    }
    // the default taps when none are named: four evenly spaced layers and the head input
    const r = try exchange(on, "/v1/chat/completions", bearer, chatWith("{\"text\":true}"));
    defer gpa.free(r.body);
    try std.testing.expectEqualSlices(i32, &.{ 7, 16, 24, 33, 42 }, backend.last.tapList());
}

test "acts: the request's size against ACTS_REQUEST_MB and the directory's against ACTS_TOTAL_MB (507), before the engine" {
    var text: DropLate = .{};
    var d = try Dir.open();
    defer d.close();
    try d.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "old.bin", .data = &@as([64]u8, @splat('x')) });
    const w = try api.acts.Writer.open(gpa, std.testing.io, d.path, .{ .request_bytes = 1 << 20, .total_bytes = 64 });
    defer w.close();
    var info = actsInfo(w, "");
    info.row_bytes = 1 << 20; // a row past the request's MiB
    var backend: Box = .{ .acts_info = info };
    var on = try init(&backend, &text, true);
    defer on.deinit();
    try expectHas(try exchange(on, "/v1/chat/completions", bearer, chatWith("{\"text\":true}")), 400, "over this server's ACTS_REQUEST_MB (1)");
    try expectHas(try exchange(on, "/v1/chat/completions", bearer, chatWith("{\"logits\":true}")), 507, "at its ACTS_TOTAL_MB");
    try std.testing.expectEqual(@as(u32, 0), backend.submits);
}

test "acts.rows: a picture whose digest names a file takes its rows (exact length), others keep the tower's" {
    var d = try Dir.open();
    defer d.close();
    const w = try api.acts.Writer.open(gpa, std.testing.io, d.path, .{});
    defer w.close();
    const info = actsInfo(w, d.path); // row_bytes 6
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const pa = api.prompt_cache.imageDigest("picture A");
    const pb = api.prompt_cache.imageDigest("picture B");
    const images = [_]api.Image{ .{ .at = 1, .tokens = 2, .bytes = "picture A", .digest = pa }, .{ .at = 5, .tokens = 3, .bytes = "picture B", .digest = pb } };
    var nb: [128]u8 = undefined;
    const name = try std.fmt.bufPrint(&nb, "{s}.bin", .{&std.fmt.bytesToHex(pa, .lower)});
    try d.tmp.dir.writeFile(std.testing.io, .{ .sub_path = name, .data = "rowsAArowsAA" }); // 2 rows x 6 bytes
    const s: api.acts.Spec = .{ .rows = true };
    var replaced: u32 = 0;
    const out = try acts_fields.prepare(&cx, std.testing.io, info, &s, 10, &images, &replaced);
    try std.testing.expectEqual(@as(u32, 1), replaced);
    try std.testing.expect(out[0].rows and std.mem.eql(u8, out[0].bytes, "rowsAArowsAA") and out[0].at == 1 and out[0].tokens == 2);
    try std.testing.expect(!out[1].rows and std.mem.eql(u8, out[1].bytes, "picture B"));
    try std.testing.expect(!images[0].rows); // the caller's list untouched
    // a file of another length is a 400, shorter or longer
    try d.tmp.dir.writeFile(std.testing.io, .{ .sub_path = name, .data = "short" });
    try std.testing.expectError(error.Refused, acts_fields.prepare(&cx, std.testing.io, info, &s, 10, &images, &replaced));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "holds 5 bytes, the picture's span 12") != null);
    try d.tmp.dir.writeFile(std.testing.io, .{ .sub_path = name, .data = &@as([13]u8, @splat('x')) });
    try std.testing.expectError(error.Refused, acts_fields.prepare(&cx, std.testing.io, info, &s, 10, &images, &replaced));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "longer than the picture's span (12 bytes)") != null);
    // a dump with pictures is sized by their rows
    const dump: api.acts.Spec = .{ .dump = true, .n_taps = 2 };
    _ = try acts_fields.prepare(&cx, std.testing.io, info, &dump, 10, &images, &replaced);
    try std.testing.expectEqual(@as(u32, 0), replaced);
}
