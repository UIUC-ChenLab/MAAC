#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace e2e {
enum OutputKind : unsigned { Samples = 0, Rows = 1, Cols = 2, Edges = 3 };
inline constexpr std::array<const char*, 4> kOutputNames{"samples", "rows", "cols", "edges"};
inline constexpr std::array<uint32_t, 4> kCapacities{220000, 450000, 450000, 450000};
inline constexpr uint64_t kGraphBase = 0x60000000000ULL;
inline constexpr uint64_t kRowOffset = 0x958000;
inline constexpr size_t kMaxColptr = 2449030, kMaxRows = 123718280;
inline constexpr std::array<uint64_t, 3> kSeedAddresses{0x4080000000ULL, 0x42C0000000ULL, 0x4500000000ULL};
inline constexpr std::array<uint64_t, 3> kFanoutAddresses{0x40C0000000ULL, 0x4300000000ULL, 0x4540000000ULL};
inline constexpr std::array<std::array<uint64_t, 4>, 3> kOutputAddresses{{
    {0x4100000000ULL, 0x4140000000ULL, 0x4180000000ULL, 0x41C0000000ULL},
    {0x4340000000ULL, 0x4380000000ULL, 0x43C0000000ULL, 0x4400000000ULL},
    {0x4580000000ULL, 0x45C0000000ULL, 0x4600000000ULL, 0x4640000000ULL}}};
inline constexpr std::array<uint64_t, 3> kCountAddresses{0x4200000000ULL, 0x4440000000ULL, 0x4680000000ULL};
using PayloadCounts = std::array<std::array<uint32_t, 4>, 3>;
using Outputs = std::array<std::vector<int32_t>, 4>;
} // namespace e2e
