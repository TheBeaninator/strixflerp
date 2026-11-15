// SPDX-License-Identifier: MIT
// See relay.hpp.
#include "flerp/stage_relay/relay.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <tuple>

namespace flerp::stage_relay {
namespace {

using Clock = std::chrono::steady_clock;

double Since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

int RemainingMs(Clock::time_point end) {
  const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(end - Clock::now()).count();
  return left <= 0 ? 0 : static_cast<int>(std::min<long long>(left, 1 << 30));
}

class TcpLink final : public Link {
 public:
  explicit TcpLink(int fd) : fd_(fd) {}
  ~TcpLink() override { Close(); }
  void Send(const void* data, std::size_t size, std::chrono::milliseconds deadline) override {
    const auto end = Clock::now() + deadline;
    const auto* p = static_cast<const std::uint8_t*>(data);
    while (size > 0) {
      if (fd_ < 0) throw RelayError("send on a closed link");
      const ssize_t n = ::send(fd_, p, size, MSG_NOSIGNAL | MSG_DONTWAIT);
      if (n > 0) {
        p += n;
        size -= static_cast<std::size_t>(n);
        continue;
      }
      if (n < 0 && errno == EINTR) continue;
      if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        pollfd pfd{fd_, POLLOUT, 0};
        const int r = ::poll(&pfd, 1, RemainingMs(end));
        if (r == 0) throw RelayError("send: no progress within the deadline");
        if (r < 0 && errno != EINTR) throw RelayError(std::string("send: poll: ") + std::strerror(errno));
        continue;
      }
      throw RelayError(std::string("send: ") + std::strerror(errno));
    }
  }
  void Recv(void* data, std::size_t size, std::chrono::milliseconds deadline) override {
    const auto end = Clock::now() + deadline;
    auto* p = static_cast<std::uint8_t*>(data);
    while (size > 0) {
      if (fd_ < 0) throw RelayError("receive on a closed link");
      pollfd pfd{fd_, POLLIN, 0};
      const int r = ::poll(&pfd, 1, RemainingMs(end));
      if (r == 0 && Clock::now() < end) continue;   // a deadline longer than one poll's 2^30 ms
      if (r == 0) throw RelayError("receive: no progress within the deadline");
      if (r < 0) {
        if (errno == EINTR) continue;
        throw RelayError(std::string("receive: poll: ") + std::strerror(errno));
      }
      const ssize_t n = ::recv(fd_, p, size, MSG_DONTWAIT);
      if (n == 0) throw RelayError("receive: the peer closed the link");
      if (n < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
        throw RelayError(std::string("receive: ") + std::strerror(errno));
      }
      p += n;
      size -= static_cast<std::size_t>(n);
    }
  }
  void Close() noexcept override {
    if (fd_ >= 0) {
      (void)::shutdown(fd_, SHUT_RDWR);
      (void)::close(fd_);
      fd_ = -1;
    }
  }
  void Interrupt() noexcept override {
    const int fd = fd_;
    if (fd >= 0) (void)::shutdown(fd, SHUT_RDWR);   // async-signal-safe; the descriptor stays open until Close
  }

 private:
  int fd_;
};

struct Pipe {
  std::mutex mu;
  std::deque<std::uint8_t> bytes;
  bool closed{false};
};

class MemoryEnd final : public Link {
 public:
  MemoryEnd(std::shared_ptr<Pipe> pipe, bool sender) : pipe_(std::move(pipe)), sender_(sender) {}
  ~MemoryEnd() override { Close(); }
  void Send(const void* data, std::size_t size, std::chrono::milliseconds) override {
    if (!sender_) throw RelayError("send on the receiving end of a pipe");
    std::lock_guard<std::mutex> lock(pipe_->mu);
    if (pipe_->closed) throw RelayError("send: the peer closed the link");
    const auto* p = static_cast<const std::uint8_t*>(data);
    pipe_->bytes.insert(pipe_->bytes.end(), p, p + size);
  }
  void Recv(void* data, std::size_t size, std::chrono::milliseconds) override {
    if (sender_) throw RelayError("receive on the sending end of a pipe");
    std::lock_guard<std::mutex> lock(pipe_->mu);
    if (pipe_->bytes.size() < size) {
      // One thread drives every rank: what is not queued now never will be.
      throw RelayError(pipe_->closed ? "receive: the peer closed the link"
                                     : "receive: no progress (loopback: nothing was sent)");
    }
    auto* p = static_cast<std::uint8_t*>(data);
    std::copy(pipe_->bytes.begin(), pipe_->bytes.begin() + static_cast<std::ptrdiff_t>(size), p);
    pipe_->bytes.erase(pipe_->bytes.begin(), pipe_->bytes.begin() + static_cast<std::ptrdiff_t>(size));
  }
  void Close() noexcept override {
    if (pipe_) {
      std::lock_guard<std::mutex> lock(pipe_->mu);
      pipe_->closed = true;
    }
  }

 private:
  std::shared_ptr<Pipe> pipe_;
  bool sender_;
};

bool CarriesHidden(Op op) { return op == Op::kPrefillChunk || op == Op::kDecode || op == Op::kVerify; }

std::string Hex(const std::array<std::uint8_t, 32>& a) {
  static const char* digits = "0123456789abcdef";
  std::string s;
  for (const auto b : a) {
    s += digits[b >> 4U];
    s += digits[b & 15U];
  }
  return s;
}

}  // namespace

const char* OpName(Op op) noexcept {
  switch (op) {
    case Op::kHello: return "HELLO";
    case Op::kPrefillChunk: return "PREFILL_CHUNK";
    case Op::kDecode: return "DECODE";
    case Op::kVerify: return "VERIFY";
    case Op::kReset: return "RESET";
    case Op::kClose: return "CLOSE";
    case Op::kReply: return "REPLY";
  }
  return "?";
}

const char* FaultName(Fault::Kind kind) noexcept {
  switch (kind) {
    case Fault::Kind::kNone: return "none";
    case Fault::Kind::kDrop: return "drop";
    case Fault::Kind::kStaleEpoch: return "stale-epoch";
    case Fault::Kind::kReplay: return "replay";
    case Fault::Kind::kBadBytes: return "bad-bytes";
    case Fault::Kind::kKill: return "kill";
  }
  return "?";
}

Fault ParseFault(const std::string& text) {
  const auto colon = text.find(':');
  if (colon == std::string::npos) throw std::invalid_argument("fault: want kind:ordinal, got " + text);
  const std::string kind = text.substr(0, colon);
  Fault f;
  f.ordinal = std::stoull(text.substr(colon + 1));
  for (const auto k : {Fault::Kind::kDrop, Fault::Kind::kStaleEpoch, Fault::Kind::kReplay, Fault::Kind::kBadBytes,
                       Fault::Kind::kKill}) {
    if (kind == FaultName(k)) f.kind = k;
  }
  if (f.kind == Fault::Kind::kNone) throw std::invalid_argument("fault: unknown kind " + kind);
  return f;
}

std::array<std::uint8_t, 32> PlanFromHex(const std::string& hex) {
  if (hex.size() != 64) throw std::invalid_argument("plan hash: want 64 hex digits");
  std::array<std::uint8_t, 32> out{};
  for (std::size_t i = 0; i < 32; ++i) out[i] = static_cast<std::uint8_t>(std::stoul(hex.substr(2 * i, 2), nullptr, 16));
  return out;
}

std::unique_ptr<Link> MakeTcpLink(int fd) { return std::make_unique<TcpLink>(fd); }

std::pair<std::unique_ptr<Link>, std::unique_ptr<Link>> MakeMemoryPipe() {
  auto pipe = std::make_shared<Pipe>();
  return {std::make_unique<MemoryEnd>(pipe, true), std::make_unique<MemoryEnd>(pipe, false)};
}

Relay::Relay(const Config& config, std::unique_ptr<Link> prev, std::unique_ptr<Link> next)
    : config_(config), prev_(std::move(prev)), next_(std::move(next)) {}

Relay::~Relay() { Abort("closed"); }

void Relay::Abort(const std::string& reason) noexcept {
  if (!poisoned_) {
    poisoned_ = true;
    try {
      reason_ = reason;
    } catch (...) {
    }
  }
  if (prev_) prev_->Close();
  if (next_) next_->Close();
}

void Relay::Fail(const std::string& reason) {
  Abort(reason);
  throw RelayError("rank " + std::to_string(config_.rank) + ": " + reason);
}

void Relay::SetFault(const Fault& fault, std::function<void(const Fault&)> on_fire) {
  fault_ = fault;
  on_fault_ = std::move(on_fire);
}

std::uint64_t Relay::ExpectedBytes(const FrameHeader& h) const {
  const Op op = static_cast<Op>(h.op);
  if (CarriesHidden(op)) {
    if (h.rows == 0 || h.rows > config_.max_rows || (op == Op::kDecode && h.rows != 1)) {
      throw RelayError(std::string(OpName(op)) + " with " + std::to_string(h.rows) + " rows (at most " +
                       std::to_string(config_.max_rows) + ")");
    }
    const std::uint64_t ids = op == Op::kPrefillChunk && (h.flags & kWithIds) != 0U ? std::uint64_t{h.rows} * sizeof(std::uint32_t) : 0;
    return std::uint64_t{h.rows} * config_.hidden * sizeof(float) + ids;
  }
  switch (op) {
    case Op::kHello:
    case Op::kReset:
    case Op::kClose:
      return 0;
    case Op::kReply:
      if ((h.flags & kSpecReply) != 0U) {
        if (h.rows < 2 || h.rows > kMaxSpecWords || (h.flags & kFullLogits) != 0U) {
          throw RelayError("spec REPLY with " + std::to_string(h.rows) + " words");
        }
        return std::uint64_t{h.rows} * sizeof(std::uint32_t);
      }
      if (h.rows != 1) throw RelayError("REPLY with " + std::to_string(h.rows) + " rows");
      return (h.flags & kFullLogits) != 0U ? std::uint64_t{h.rows} * config_.vocab * sizeof(float) : 0;
    default:
      throw RelayError("unknown op " + std::to_string(h.op));
  }
}

void Relay::SendRaw(const FrameHeader& h, std::span<const std::uint8_t> payload) {
  next_->Send(&h, sizeof(h), config_.deadline);
  if (!payload.empty()) next_->Send(payload.data(), payload.size(), config_.deadline);
}

void Relay::Send(Op op, std::uint32_t start_pos, std::uint32_t rows, std::uint32_t flags, std::uint32_t token,
                 std::span<const std::uint8_t> payload, std::uint32_t seq_slot) {
  if (poisoned_) throw RelayError("rank " + std::to_string(config_.rank) + ": poisoned: " + reason_);
  if (!next_) Fail("send with no next link");
  const auto t0 = Clock::now();
  FrameHeader h;
  h.op = static_cast<std::uint16_t>(op);
  h.epoch = config_.epoch;
  h.ordinal = out_ordinal_++;
  h.seq_slot = seq_slot;
  h.start_pos = start_pos;
  h.rows = rows;
  h.flags = flags;
  h.token = token;
  h.bytes = payload.size();
  h.plan = config_.plan;
  try {
    if (ExpectedBytes(h) != h.bytes) {
      Fail(std::string("sending ") + OpName(op) + " with " + std::to_string(h.bytes) + " bytes, " +
           std::to_string(ExpectedBytes(h)) + " expected");
    }
    if (fault_.kind != Fault::Kind::kNone && fault_.ordinal == h.ordinal) {
      if (on_fault_) on_fault_(fault_);
      switch (fault_.kind) {
        case Fault::Kind::kDrop:
          stats_.send_s += Since(t0);
          return;                                     // the ordinal is spent: the receiver never sees it
        case Fault::Kind::kStaleEpoch:
          h.epoch = config_.epoch - 1;
          break;
        case Fault::Kind::kReplay:
          SendRaw(h, payload);                        // the same frame twice: the second has a stale ordinal
          break;
        case Fault::Kind::kBadBytes:
          h.bytes += sizeof(float);
          SendRaw(h, payload);
          {
            const float pad = 0.0F;
            next_->Send(&pad, sizeof(pad), config_.deadline);
          }
          stats_.send_s += Since(t0);
          return;
        case Fault::Kind::kKill:
          (void)::raise(SIGKILL);
          break;
        case Fault::Kind::kNone:
          break;
      }
    }
    SendRaw(h, payload);
  } catch (const RelayError& e) {
    if (!poisoned_) Fail(e.what());
    throw;
  }
  stats_.frames_sent += 1;
  stats_.bytes_sent += sizeof(h) + payload.size();
  stats_.send_s += Since(t0);
}

void Relay::Interrupt() noexcept {
  if (prev_) prev_->Interrupt();
  if (next_) next_->Interrupt();
}

Frame Relay::Recv() { return Recv(config_.deadline); }

Frame Relay::Recv(std::chrono::milliseconds header_deadline) {
  if (poisoned_) throw RelayError("rank " + std::to_string(config_.rank) + ": poisoned: " + reason_);
  if (!prev_) Fail("receive with no previous link");
  const auto t0 = Clock::now();
  Frame f;
  try {
    prev_->Recv(&f.header, sizeof(f.header), header_deadline);
  } catch (const RelayError& e) {
    stats_.recv_s += Since(t0);
    Fail(std::string("from rank ") + std::to_string(prev_rank()) + ": " + e.what());
  }
  const auto& h = f.header;
  const std::string from = "frame from rank " + std::to_string(prev_rank()) + ": ";
  if (h.magic != kMagic || h.version != kVersion) Fail(from + "bad magic or version");
  if (h.epoch != config_.epoch) {
    Fail(from + "epoch " + std::to_string(h.epoch) + ", this rank's is " + std::to_string(config_.epoch));
  }
  if (h.ordinal != in_ordinal_) {
    Fail(from + "ordinal " + std::to_string(h.ordinal) + ", expected " + std::to_string(in_ordinal_));
  }
  if (h.plan != config_.plan) Fail(from + "plan " + Hex(h.plan) + ", this rank's is " + Hex(config_.plan));
  std::uint64_t want = 0;
  try {
    want = ExpectedBytes(h);
  } catch (const RelayError& e) {
    Fail(from + e.what());
  }
  if (h.bytes != want) {
    Fail(from + OpName(static_cast<Op>(h.op)) + " carries " + std::to_string(h.bytes) + " bytes, " +
         std::to_string(want) + " expected");
  }
  ++in_ordinal_;
  f.payload.resize(h.bytes);
  if (h.bytes > 0) {
    try {
      prev_->Recv(f.payload.data(), f.payload.size(), config_.deadline);
    } catch (const RelayError& e) {
      stats_.recv_s += Since(t0);
      Fail(std::string("from rank ") + std::to_string(prev_rank()) + ": payload: " + e.what());
    }
  }
  stats_.frames_received += 1;
  stats_.bytes_received += sizeof(h) + h.bytes;
  stats_.recv_s += Since(t0);
  return f;
}

void Relay::Handshake(std::chrono::milliseconds hello_deadline) {
  Send(Op::kHello, 0, config_.world, 0, config_.rank, {});
  const Frame f = Recv(hello_deadline);
  if (f.op() != Op::kHello) Fail(std::string("first frame is ") + OpName(f.op()) + ", not HELLO");
  if (f.header.rows != config_.world || f.header.token != prev_rank()) {
    Fail("HELLO from rank " + std::to_string(f.header.token) + " of world " + std::to_string(f.header.rows) +
         ", expected rank " + std::to_string(prev_rank()) + " of " + std::to_string(config_.world));
  }
}

std::vector<std::unique_ptr<Relay>> LoopbackRing(const std::vector<Config>& configs) {
  const std::size_t n = configs.size();
  std::vector<std::unique_ptr<Link>> send_end(n), recv_end(n);   // pipe r: rank r -> rank r + 1
  for (std::size_t r = 0; r < n; ++r) std::tie(send_end[r], recv_end[r]) = MakeMemoryPipe();
  std::vector<std::unique_ptr<Relay>> relays;
  for (std::size_t r = 0; r < n; ++r) {
    if (configs[r].rank != r || configs[r].world != n) throw std::invalid_argument("loopback: configs[r] must be rank r of world n");
    relays.push_back(std::make_unique<Relay>(configs[r], std::move(recv_end[(r + n - 1) % n]), std::move(send_end[r])));
  }
  // Two phases in one thread: every HELLO out, then every HELLO checked. A rank that refuses is left poisoned and
  // the caller sees it; the others go on.
  for (auto& relay : relays) {
    try {
      relay->Send(Op::kHello, 0, relay->config().world, 0, relay->config().rank, {});
    } catch (const RelayError&) {
    }
  }
  for (auto& relay : relays) {
    if (relay->poisoned()) continue;
    try {
      const Frame f = relay->Recv();
      if (f.op() != Op::kHello || f.header.rows != relay->config().world || f.header.token != relay->prev_rank()) {
        relay->Abort("bad HELLO from rank " + std::to_string(f.header.token));
      }
    } catch (const RelayError&) {
    }
  }
  return relays;
}

std::unique_ptr<Relay> ConnectRing(const Config& config, const std::vector<std::string>& hosts, std::uint16_t port,
                                   std::chrono::milliseconds setup_deadline) {
  if (config.world < 2) return std::make_unique<Relay>(config, nullptr, nullptr);
  const auto host_of = [&](std::uint32_t r) -> const std::string& { return hosts.size() == 1 ? hosts[0] : hosts.at(r); };
  const auto end = Clock::now() + setup_deadline;
  const auto fail = [&](const std::string& what) -> RelayError {
    return RelayError("rank " + std::to_string(config.rank) + ": setup: " + what);
  };

  const int listener = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (listener < 0) throw fail(std::string("socket: ") + std::strerror(errno));
  const int one = 1;
  (void)::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in me{};
  me.sin_family = AF_INET;
  me.sin_port = htons(static_cast<std::uint16_t>(port + config.rank));
  if (::inet_pton(AF_INET, host_of(config.rank).c_str(), &me.sin_addr) != 1) {
    ::close(listener);
    throw fail("bad address " + host_of(config.rank));
  }
  if (::bind(listener, reinterpret_cast<sockaddr*>(&me), sizeof(me)) != 0 || ::listen(listener, 4) != 0) {
    const std::string e = std::strerror(errno);
    ::close(listener);
    throw fail("listen on port " + std::to_string(port + config.rank) + ": " + e);
  }

  // Connect to the next rank, retrying until it listens.
  const std::uint32_t next = (config.rank + 1) % config.world;
  sockaddr_in peer{};
  peer.sin_family = AF_INET;
  peer.sin_port = htons(static_cast<std::uint16_t>(port + next));
  if (::inet_pton(AF_INET, host_of(next).c_str(), &peer.sin_addr) != 1) {
    ::close(listener);
    throw fail("bad address " + host_of(next));
  }
  int out_fd = -1;
  while (true) {
    out_fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (out_fd >= 0 && ::connect(out_fd, reinterpret_cast<sockaddr*>(&peer), sizeof(peer)) == 0) break;
    if (out_fd >= 0) ::close(out_fd);
    out_fd = -1;
    if (Clock::now() >= end) {
      ::close(listener);
      throw fail("rank " + std::to_string(next) + " did not listen within the setup deadline");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  (void)::setsockopt(out_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

  pollfd pfd{listener, POLLIN, 0};
  const int r = ::poll(&pfd, 1, RemainingMs(end));
  if (r <= 0) {
    ::close(listener);
    ::close(out_fd);
    throw fail("rank " + std::to_string((config.rank + config.world - 1) % config.world) +
               " did not connect within the setup deadline");
  }
  const int in_fd = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
  ::close(listener);
  if (in_fd < 0) {
    ::close(out_fd);
    throw fail(std::string("accept: ") + std::strerror(errno));
  }
  (void)::setsockopt(in_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  auto relay = std::make_unique<Relay>(config, MakeTcpLink(in_fd), MakeTcpLink(out_fd));
  relay->Handshake(std::max(config.deadline, std::chrono::milliseconds(RemainingMs(end))));
  return relay;
}

}  // namespace flerp::stage_relay
