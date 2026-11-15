// SPDX-License-Identifier: MIT
// runtime.flerp.stage_relay: the framed point-to-point relay between the stages of a model split by blocks
// (docs/PHASE5-FOURNODE-DESIGN.md section 3.3). Bytes only: it knows nothing of HIP or of the executor.
//
// Topology: a ring. Rank r sends to rank (r + 1) mod world and receives from rank (r - 1) mod world, so the data
// chain 0 -> 1 -> ... -> N-1 and the reply link N-1 -> 0 are the same mechanism. Control calls (RESET, CLOSE) go
// down the chain in order with the data, each rank forwarding a control frame before acting on it: every rank sees
// the same call sequence (the contract of runtime.strixfold.coll, coll.zig:7-15). First version: there is no
// separate control link from rank 0 to every rank; the chain carries it.
//
// Frame: an 88-byte header {magic, version, op, epoch, ordinal, seq slot, start_pos, rows, flags, token, bytes,
// plan hash} and `bytes` of payload. Every received frame is checked: magic and version; epoch equal to this
// rank's; ordinal equal to the next expected on that link (each link counts its own frames from 0, the HELLO);
// the plan hash; an op that may travel; rows within [1, max_rows] for row-carrying ops; and bytes equal to what
// the op says (rows x hidden x 4 for hidden rows, plus rows x 4 with kWithIds; rows x vocab x 4 for a full-logits
// reply; rows x 4 for a kSpecReply, rows in [2, kMaxSpecWords]; else 0).
//
// Failure: the first failed send, receive or check poisons the relay - it records the reason and closes both
// links, so the neighbours' blocked calls return with an end of stream, and so on round the ring. A poisoned relay
// refuses every later call. Abort is idempotent. A receive that sees no progress within the deadline fails.
//
// Transports: TCP (rank r listens on port + r, connects to rank r + 1; the HELLO frames are checked both ways) and
// an in-process loopback (N relays in one process, one thread: a receive with nothing queued fails at once, since
// nothing else could queue it).
#ifndef FLERP_STAGE_RELAY_RELAY_HPP_
#define FLERP_STAGE_RELAY_RELAY_HPP_

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace flerp::stage_relay {

inline constexpr std::uint32_t kMagic = 0x31525346U;   // "FSR1"
inline constexpr std::uint16_t kVersion = 1;

enum class Op : std::uint16_t {
  kHello = 1,          // rank in `token`, world in `rows`; first frame on every link
  kPrefillChunk = 2,   // rows x hidden FP32
  kDecode = 3,         // 1 x hidden FP32
  kVerify = 4,         // rows x hidden FP32 (reserved: not issued by the first version)
  kReset = 5,          // control, no payload
  kClose = 6,          // control, no payload
  kReply = 7,          // last rank -> rank 0: token id in `token`; with kFullLogits, rows x vocab FP32
};
[[nodiscard]] const char* OpName(Op op) noexcept;

enum Flags : std::uint32_t {
  kWantLogits = 1U,    // the call asks for the logits of its last row (a token id at least)
  kFullLogits = 2U,    // ... and the full row, returned in the reply
  kGraphOff = 4U,      // with kReset: run decode steps eagerly until the next reset
  // Speculative rounds (ORNITH-O3 step two, gate/ornith-o3-spec.toml; the last stage drafts with the MTP head):
  kWithIds = 8U,       // kPrefillChunk: the payload's rows are followed by the rows' token ids (rows x uint32)
  kSpecRound = 16U,    // kDecode: frame (flags >> 8) & 0xff of a round of ((flags >> 16) & 0xff) + 1 frames, its input id
                       // in `token`; kReset: drafts on for the session, at most `token` a round (value-aware with
                       // kSpecValue)
  kSpecReply = 32U,    // kReply: `rows` uint32 words: [kept, drafts, frames, kept ids.., drafts.., each frame's drawn id
                       // (0xffffffff: skipped)]; `token` the last kept id
  kSpecValue = 64U,    // kReset with kSpecRound: the value-aware draft depth (draft_round.hpp)
  kSpecAllRows = 128U, // kReset with kSpecRound: the last stage runs the target on every frame (no skip after the
                       // kept chain ends), so every frame's drawn id comes back (the ABI's verify returns every row)
  kSpecLookup = 256U,  // kReset with kSpecRound: prompt-lookup drafts first, MTP at most (flags >> 16) & 0xff a round
                       // (ORNITH-CRUCIBLE item 4, draft_round.hpp)
};
inline constexpr std::uint32_t kSkippedId = 0xffffffffU;
inline constexpr std::uint32_t SpecFrameFlags(std::uint32_t j, std::uint32_t frames) noexcept {
  return kSpecRound | ((j & 0xffU) << 8) | (((frames - 1) & 0xffU) << 16);
}
inline constexpr std::uint32_t SpecFrameIndex(std::uint32_t flags) noexcept { return (flags >> 8) & 0xffU; }
inline constexpr std::uint32_t SpecFrameCount(std::uint32_t flags) noexcept { return ((flags >> 16) & 0xffU) + 1; }
inline constexpr std::uint32_t kMaxSpecWords = 1024;   // a spec reply's words, at most

#pragma pack(push, 1)
struct FrameHeader {
  std::uint32_t magic{kMagic};
  std::uint16_t version{kVersion};
  std::uint16_t op{0};
  std::uint64_t epoch{0};
  std::uint64_t ordinal{0};
  std::uint32_t seq_slot{0};
  std::uint32_t start_pos{0};
  std::uint32_t rows{0};
  std::uint32_t flags{0};
  std::uint32_t token{0};
  std::uint32_t reserved{0};
  std::uint64_t bytes{0};
  std::array<std::uint8_t, 32> plan{};
};
#pragma pack(pop)
static_assert(sizeof(FrameHeader) == 88, "the wire header is 88 bytes");

struct Frame {
  FrameHeader header;
  std::vector<std::uint8_t> payload;
  [[nodiscard]] Op op() const noexcept { return static_cast<Op>(header.op); }
};

class RelayError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// One direction of bytes. Send and Recv throw RelayError; Close never throws and may be called twice.
class Link {
 public:
  virtual ~Link() = default;
  virtual void Send(const void* data, std::size_t size, std::chrono::milliseconds deadline) = 0;
  virtual void Recv(void* data, std::size_t size, std::chrono::milliseconds deadline) = 0;
  virtual void Close() noexcept = 0;
  /// Makes a blocked or later Send / Recv fail as if the peer had closed the link, without releasing anything.
  /// Async-signal-safe (a TCP link shuts its socket down); the link is still closed by Close. Default: nothing.
  virtual void Interrupt() noexcept {}
};

/// A connected stream socket; the link owns it.
[[nodiscard]] std::unique_ptr<Link> MakeTcpLink(int fd);
/// An in-process pipe: {sending end, receiving end}. Closing either end makes the other fail.
[[nodiscard]] std::pair<std::unique_ptr<Link>, std::unique_ptr<Link>> MakeMemoryPipe();

struct Config {
  std::uint32_t world{1};
  std::uint32_t rank{0};
  std::uint64_t epoch{1};
  std::uint32_t hidden{0};          // floats a hidden row
  std::uint32_t vocab{0};           // floats a logits row
  std::uint32_t max_rows{0};        // rows a frame may carry (the arena's prompt batch)
  std::array<std::uint8_t, 32> plan{};
  std::chrono::milliseconds deadline{60000};   // no progress on a receive or send within this fails
};

/// Fault injection for the gates: applied to this rank's outgoing frame of the given ordinal.
struct Fault {
  enum class Kind { kNone, kDrop, kStaleEpoch, kReplay, kBadBytes, kKill };
  Kind kind{Kind::kNone};
  std::uint64_t ordinal{0};
};
[[nodiscard]] Fault ParseFault(const std::string& text);   // "drop:12", "stale-epoch:12", "replay:12", "bad-bytes:12", "kill:12"
[[nodiscard]] const char* FaultName(Fault::Kind kind) noexcept;

struct RelayStats {
  std::uint64_t frames_sent{0};
  std::uint64_t frames_received{0};
  std::uint64_t bytes_sent{0};
  std::uint64_t bytes_received{0};
  double send_s{0.0};
  double recv_s{0.0};        // time inside Recv, waiting included
};

class Relay {
 public:
  Relay(const Config& config, std::unique_ptr<Link> prev, std::unique_ptr<Link> next);
  ~Relay();
  Relay(const Relay&) = delete;
  Relay& operator=(const Relay&) = delete;

  /// Sends one frame to the next rank. Throws RelayError (and poisons) on failure or when poisoned.
  void Send(Op op, std::uint32_t start_pos, std::uint32_t rows, std::uint32_t flags, std::uint32_t token,
            std::span<const std::uint8_t> payload, std::uint32_t seq_slot = 0);
  /// Receives and checks one frame from the previous rank. Throws RelayError (and poisons).
  [[nodiscard]] Frame Recv();
  /// The same, waiting at most `header_deadline` for the frame to start (the payload keeps the config's deadline).
  /// A stage that waits between its caller's calls (a server's idle time) gives a long one here.
  [[nodiscard]] Frame Recv(std::chrono::milliseconds header_deadline);

  /// Makes this rank's blocked or next Send / Recv fail (and so poison), from a signal handler: async-signal-safe,
  /// it shuts the links down and changes nothing else. The failing call records the reason as usual.
  void Interrupt() noexcept;

  /// The HELLO exchange: send ours to the next rank, then receive and check the previous rank's. The previous rank
  /// sends its HELLO only once its own ring links are up, which in a ring of sequentially loaded stages is when the
  /// last rank has loaded: the wait for it is part of the setup, so it gets `hello_deadline` (ConnectRing passes what
  /// is left of the setup deadline), not the config's per-frame deadline.
  void Handshake(std::chrono::milliseconds hello_deadline);
  void Handshake() { Handshake(config_.deadline); }

  /// Poison: record the first reason, close both links. Idempotent; never throws.
  void Abort(const std::string& reason) noexcept;
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] const std::string& reason() const noexcept { return reason_; }

  void SetFault(const Fault& fault, std::function<void(const Fault&)> on_fire = {});
  [[nodiscard]] const Config& config() const noexcept { return config_; }
  [[nodiscard]] const RelayStats& stats() const noexcept { return stats_; }
  [[nodiscard]] std::uint32_t prev_rank() const noexcept { return (config_.rank + config_.world - 1) % config_.world; }
  [[nodiscard]] std::uint32_t next_rank() const noexcept { return (config_.rank + 1) % config_.world; }

  /// The payload bytes a header must carry; throws RelayError for an op or row count that may not travel.
  [[nodiscard]] std::uint64_t ExpectedBytes(const FrameHeader& h) const;

 private:
  [[noreturn]] void Fail(const std::string& reason);
  void SendRaw(const FrameHeader& h, std::span<const std::uint8_t> payload);

  Config config_;
  std::unique_ptr<Link> prev_;
  std::unique_ptr<Link> next_;
  std::uint64_t out_ordinal_{0};
  std::uint64_t in_ordinal_{0};
  bool poisoned_{false};
  std::string reason_;
  Fault fault_;
  std::function<void(const Fault&)> on_fault_;
  RelayStats stats_;
};

/// TCP ring on one host or several: rank r listens on hosts[r]:port + r and connects to rank r + 1; then the HELLO
/// exchange. `hosts` has world entries (or one, used for every rank). World 1 has no links.
[[nodiscard]] std::unique_ptr<Relay> ConnectRing(const Config& config, const std::vector<std::string>& hosts,
                                                  std::uint16_t port, std::chrono::milliseconds setup_deadline);

/// N relays in one process joined in a ring of memory pipes, HELLO exchanged; configs[r].rank must be r.
[[nodiscard]] std::vector<std::unique_ptr<Relay>> LoopbackRing(const std::vector<Config>& configs);

/// 64 hex digits to 32 bytes; throws std::invalid_argument.
[[nodiscard]] std::array<std::uint8_t, 32> PlanFromHex(const std::string& hex);

}  // namespace flerp::stage_relay

#endif  // FLERP_STAGE_RELAY_RELAY_HPP_
