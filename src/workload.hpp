#pragma once
#include "layout.hpp"
#include <filesystem>
#include <stdexcept>
#include <string>

namespace e2e {
inline constexpr unsigned kSlrs = 3;
inline constexpr unsigned kSeedsPerMinibatch = 1024;

// The graph in CSC form, as stored in the int32 files written by prepare_data.py.
struct Graph {
    std::string dataset;
    std::filesystem::path colptr_path, row_path;
    std::vector<int32_t> colptr, rows;
};
// What the three SLRs sample in one iteration, and the result the CPU golden
// model predicts for each of them.
struct Batch {
    std::array<std::vector<int32_t>, kSlrs> seeds;
    std::array<Outputs, kSlrs> expected;
};
// Three different mini-batches of 1,024 seed nodes. Iteration g gives mini-batch
// (slr + g) % 3 to each SLR, so batches[r] is rotation r of the same three.
struct Workload {
    std::vector<int32_t> fanouts;
    std::array<uint64_t, kSlrs> minibatch_seeds{};  // random seed of each mini-batch
    std::vector<uint64_t> skipped_seeds;            // candidates that did not fit the kernel buffers
    std::array<Batch, kSlrs> batches;
};
struct Counts {
    uint32_t samples_count, rows_count, cols_count, edges_count;
    std::array<uint32_t, 4> payload_order() const;
};
// Thrown by sample_reference when a mini-batch needs more than kCapacities.
struct CapacityExceeded : std::runtime_error {
    using std::runtime_error::runtime_error;
};

Graph load_graph(const std::filesystem::path& root, const std::string& dataset);
void validate_graph(const Graph& graph);
// CPU golden model of the FPGA kernel: same random numbers, same hash table.
Outputs sample_reference(const Graph&, const std::vector<int32_t>& seeds, const std::vector<int32_t>& fanouts);
Counts decode_counts(const std::array<uint8_t, 16>& packed);
// Mini-batches are drawn with random seeds first_seed, first_seed + 1, ...; a
// candidate whose sampled sub-graph does not fit the kernel's output buffers is skipped.
Workload make_workload(const Graph&, uint64_t first_seed, const std::vector<int32_t>& fanouts);
} // namespace e2e
