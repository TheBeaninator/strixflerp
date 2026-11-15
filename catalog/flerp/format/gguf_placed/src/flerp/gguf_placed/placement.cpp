// SPDX-License-Identifier: MIT
// See placement.hpp.
#include "flerp/gguf_placed/placement.hpp"

#include <algorithm>
#include <limits>
#include <string_view>

#include "src/core/crypto/sha256.hpp"

namespace flerp::gguf_placed {
namespace {

enum class Kind : std::uint8_t { kBlock, kEmbd, kHead, kPast, kOther };

struct Where {
  Kind kind{Kind::kOther};
  std::uint32_t block{0};
};

Where Classify(std::string_view name, std::uint32_t blocks) {
  if (name.starts_with("blk.")) {
    std::size_t i = 4;
    std::uint64_t n = 0;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9' && n < 1000000) {
      n = n * 10 + static_cast<std::uint64_t>(name[i] - '0');
      ++i;
    }
    if (i > 4 && i < name.size() && name[i] == '.') {
      return n < blocks ? Where{Kind::kBlock, static_cast<std::uint32_t>(n)}
                        : Where{Kind::kPast, static_cast<std::uint32_t>(n)};
    }
    return {};
  }
  if (name == "token_embd.weight") return {Kind::kEmbd, 0};
  if (name == "output_norm.weight" || name == "output.weight") return {Kind::kHead, 0};
  return {};
}

constexpr std::uint64_t Aligned(std::uint64_t n) { return (n + kPlanAlign - 1) / kPlanAlign * kPlanAlign; }

bool Fail(std::string* error, std::string message) {
  if (error != nullptr) *error = std::move(message);
  return false;
}

}  // namespace

std::vector<std::uint32_t> BestCuts(std::span<const std::uint64_t> block_cost, std::uint64_t first_extra,
                                    std::uint64_t last_extra, std::uint32_t world) {
  const std::size_t n = block_cost.size();
  if (world == 0 || world > n) return {};
  std::vector<std::uint64_t> pre(n + 1, 0);
  for (std::size_t i = 0; i < n; ++i) pre[i + 1] = pre[i] + block_cost[i];
  const auto cost = [&](std::uint32_t r, std::size_t a, std::size_t b) {
    return pre[b] - pre[a] + (r == 0 ? first_extra : 0) + (r + 1 == world ? last_extra : 0);
  };
  constexpr std::uint64_t kInf = std::numeric_limits<std::uint64_t>::max();
  // 1. the smallest possible largest stage
  std::vector<std::vector<std::uint64_t>> best(world + 1, std::vector<std::uint64_t>(n + 1, kInf));
  best[0][0] = 0;
  for (std::uint32_t r = 0; r < world; ++r) {
    for (std::size_t a = 0; a < n; ++a) {
      if (best[r][a] == kInf) continue;
      for (std::size_t b = a + 1; b <= n; ++b) {
        const std::uint64_t v = std::max(best[r][a], cost(r, a, b));
        if (v < best[r + 1][b]) best[r + 1][b] = v;
      }
    }
  }
  const std::uint64_t limit = best[world][n];
  // 2. under that limit the smallest sum of squares; a tie goes to the earliest cut
  using Wide = unsigned __int128;
  struct Cell {
    bool set{false};
    Wide sum{0};
    std::size_t next{0};
  };
  std::vector<std::vector<Cell>> sq(world + 1, std::vector<Cell>(n + 1));
  sq[world][n].set = true;
  for (std::uint32_t r = world; r-- > 0;) {
    for (std::size_t a = n; a-- > 0;) {
      for (std::size_t b = a + 1; b <= n; ++b) {
        if (!sq[r + 1][b].set) continue;
        const std::uint64_t c = cost(r, a, b);
        if (c > limit) continue;
        const Wide v = sq[r + 1][b].sum + static_cast<Wide>(c) * c;
        if (!sq[r][a].set || v < sq[r][a].sum) sq[r][a] = {true, v, b};
      }
    }
  }
  std::vector<std::uint32_t> cuts{0};
  std::size_t a = 0;
  for (std::uint32_t r = 0; r < world; ++r) {
    if (!sq[r][a].set) return {};
    a = sq[r][a].next;
    cuts.push_back(static_cast<std::uint32_t>(a));
  }
  return cuts;
}

std::optional<Plan> StagePlan(const gufo::core::GgufReader& reader, std::uint32_t world,
                              std::optional<BlockRange> range, std::string* error) {
  const auto regions = reader.GetMappedRegions();
  if (regions.size() != 1 || regions[0].data == nullptr) {
    Fail(error, "the placed loader takes a single-file GGUF (" + std::to_string(regions.size()) + " mapped regions)");
    return std::nullopt;
  }
  const auto base = reinterpret_cast<std::uintptr_t>(regions[0].data);
  const std::string arch(reader.GetMetadataString("general.architecture").value_or(""));
  const auto block_count = reader.GetMetadataUint32(arch + ".block_count");
  const std::uint32_t nextn = reader.GetMetadataUint32(arch + ".nextn_predict_layers").value_or(0);
  if (!block_count || *block_count <= nextn) {
    Fail(error, "the header has no usable " + arch + ".block_count");
    return std::nullopt;
  }
  Plan plan;
  plan.world = world;
  plan.blocks = *block_count - nextn;
  plan.file_bytes = regions[0].size;
  const std::uint32_t n = plan.blocks;

  struct Item {
    const gufo::core::GgufTensorInfo* tensor;
    Where where;
    std::uint64_t file_offset;
  };
  std::vector<Item> items;
  std::vector<std::uint64_t> block_cost(n, 0);
  std::uint64_t embd_cost = 0;
  std::uint64_t head_cost = 0;
  bool has_output = false;
  for (const auto& tensor : reader.GetTensors()) {
    const auto address = reinterpret_cast<std::uintptr_t>(tensor.data);
    if (tensor.data == nullptr || address < base || address - base >= regions[0].size ||
        tensor.size_bytes > regions[0].size - (address - base)) {
      Fail(error, "tensor " + std::string(tensor.name) + " lies outside the mapped file");
      return std::nullopt;
    }
    const Where where = Classify(tensor.name, n);
    items.push_back({&tensor, where, address - base});
    const std::uint64_t padded = Aligned(tensor.size_bytes);
    if (where.kind == Kind::kBlock) block_cost[where.block] += padded;
    if (where.kind == Kind::kEmbd) embd_cost += padded;
    if (where.kind == Kind::kHead) head_cost += padded;
    has_output = has_output || tensor.name == "output.weight";
  }
  std::sort(items.begin(), items.end(), [](const Item& x, const Item& y) { return x.file_offset < y.file_offset; });
  plan.tied_head = !has_output;

  if (range.has_value()) {
    if (world != 1) {
      Fail(error, "a block range is a world-1 plan");
      return std::nullopt;
    }
    if (range->begin >= range->end || range->end > n) {
      Fail(error, "block range " + std::to_string(range->begin) + ":" + std::to_string(range->end) +
                      " is outside 0:" + std::to_string(n));
      return std::nullopt;
    }
    plan.cuts = {range->begin, range->end};
  } else {
    plan.cuts = BestCuts(block_cost, embd_cost, head_cost + (plan.tied_head && world > 1 ? embd_cost : 0), world);
    if (plan.cuts.empty()) {
      Fail(error, "world " + std::to_string(world) + " does not fit " + std::to_string(n) + " blocks");
      return std::nullopt;
    }
  }

  std::vector<bool> loaded(items.size(), false);
  for (std::uint32_t r = 0; r < world; ++r) {
    RankPlan rank;
    rank.rank = r;
    rank.block_begin = plan.cuts[r];
    rank.block_end = plan.cuts[r + 1];
    std::uint64_t offset = 0;
    for (std::size_t i = 0; i < items.size(); ++i) {
      const auto& item = items[i];
      bool held = false;
      switch (item.where.kind) {
        case Kind::kBlock:
          held = item.where.block >= rank.block_begin && item.where.block < rank.block_end;
          break;
        case Kind::kEmbd:
          held = rank.block_begin == 0 || (plan.tied_head && rank.block_end == n);
          break;
        case Kind::kHead:
          held = rank.block_end == n;
          break;
        default:
          break;
      }
      if (!held) continue;
      loaded[i] = true;
      rank.entries.push_back({std::string(item.tensor->name), static_cast<std::uint32_t>(item.tensor->type),
                              item.tensor->size_bytes, item.file_offset, offset, item.tensor->data});
      rank.tensor_bytes += item.tensor->size_bytes;
      offset += Aligned(item.tensor->size_bytes);
    }
    rank.device_bytes = offset + kPlanTail;
    plan.ranks.push_back(std::move(rank));
  }
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (!loaded[i]) {
      ++plan.unloaded_tensors;
      plan.unloaded_bytes += items[i].tensor->size_bytes;
    }
  }
  return plan;
}

std::string Plan::Text() const {
  std::string out = "flerp-placement v1\nmode stage\nworld " + std::to_string(world) + "\nfile_bytes " +
                    std::to_string(file_bytes) + "\nblocks " + std::to_string(blocks) + "\ncuts";
  for (const auto cut : cuts) out += " " + std::to_string(cut);
  out += "\n";
  for (const auto& rank : ranks) {
    out += "rank " + std::to_string(rank.rank) + " blocks " + std::to_string(rank.block_begin) + " " +
           std::to_string(rank.block_end) + " tensors " + std::to_string(rank.entries.size()) + " device_bytes " +
           std::to_string(rank.device_bytes) + "\n";
    for (const auto& e : rank.entries) {
      out += e.name + "\t" + std::to_string(e.type_id) + "\t" + std::to_string(e.bytes) + "\t" +
             std::to_string(e.file_offset) + "\t" + std::to_string(e.device_offset) + "\n";
    }
  }
  return out;
}

std::string Plan::Sha256() const {
  const std::string text = Text();
  return gufo::crypto::Sha256Hex(
      std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
}

}  // namespace flerp::gguf_placed
