#include "workload.hpp"
#include <algorithm>
#include <fstream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>

namespace e2e {
namespace {
std::vector<int32_t> load_binary(const std::filesystem::path& path, size_t maximum) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open graph file: " + path.string());
    const auto bytes = file.tellg();
    if (bytes < 0 || bytes % 4 || static_cast<uint64_t>(bytes) / 4 > maximum)
        throw std::runtime_error("Invalid size/capacity: " + path.string());
    std::vector<int32_t> values(static_cast<size_t>(bytes) / 4);
    file.seekg(0);
    if (bytes && !file.read(reinterpret_cast<char*>(values.data()), bytes))
        throw std::runtime_error("Short graph read: " + path.string());
    for (auto& value : values) {
        const auto* p = reinterpret_cast<const uint8_t*>(&value);
        const uint32_t u = uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
        value = static_cast<int32_t>(u);
    }
    return values;
}
uint64_t hash_bucket(uint64_t a) {
    a = (~a) + (a << 21); a ^= a >> 24;
    a = (a + (a << 3)) + (a << 8); a ^= a >> 14;
    a = (a + (a << 2)) + (a << 4); a ^= a >> 28;
    return (a + (a << 31)) & ((1u << 19) - 1);
}
uint32_t random_below(uint32_t maximum, uint32_t& lfsr) {
    const uint32_t bit = ((lfsr >> 0) ^ (lfsr >> 1) ^ (lfsr >> 2) ^ (lfsr >> 22)) & 1;
    lfsr = (lfsr >> 1) | (bit << 31);
    return (uint64_t(lfsr) * maximum) >> 32;
}
std::vector<int32_t> generate_seeds(int32_t maximum, unsigned count, uint64_t seed) {
    if (maximum < 0 || count > uint64_t(maximum) + 1) throw std::runtime_error("More unique seeds than graph nodes");
    std::mt19937_64 generator(seed);
    std::uniform_int_distribution<int32_t> distribution(0, maximum);
    std::set<int32_t> used;
    std::vector<int32_t> nodes;
    while (nodes.size() < count) {
        const auto node = distribution(generator);
        if (used.insert(node).second) nodes.push_back(node);
    }
    std::shuffle(nodes.begin(), nodes.end(), generator);
    return nodes;
}
} // namespace

Graph load_graph(const std::filesystem::path& root, const std::string& dataset) {
    const std::string name = dataset == "products" ? "ogb_product" : dataset == "arxiv" ? "ogb_arxiv" : "reddit";
    if (dataset != "products" && dataset != "arxiv" && dataset != "reddit")
        throw std::invalid_argument("Dataset must be products, arxiv, or reddit");
    Graph graph{dataset, root / name / ("colptr_" + name + "_csc.bin"), root / name / ("row_" + name + "_csc.bin"), {}, {}};
    graph.colptr = load_binary(graph.colptr_path, kMaxColptr);
    graph.rows = load_binary(graph.row_path, kMaxRows);
    validate_graph(graph);
    return graph;
}
void validate_graph(const Graph& graph) {
    if (graph.colptr.size() < 2 || graph.colptr.size() > kMaxColptr || graph.rows.size() > kMaxRows ||
        graph.colptr.front() != 0 || graph.colptr.back() != static_cast<int32_t>(graph.rows.size()) ||
        !std::is_sorted(graph.colptr.begin(), graph.colptr.end()))
        throw std::runtime_error("Graph CSC offsets or capacities are invalid");
    for (int32_t row : graph.rows)
        if (row < 0 || static_cast<size_t>(row) >= graph.colptr.size() - 1)
            throw std::runtime_error("Graph contains an out-of-range node");
}
Outputs sample_reference(const Graph& graph, const std::vector<int32_t>& seeds, const std::vector<int32_t>& fanouts) {
    if (seeds.size() > 1024 || fanouts.empty() || fanouts.size() > 3)
        throw std::invalid_argument("Kernel supports at most 1024 seeds and 1..3 fanouts");
    for (auto f : fanouts) if (f < 0 || f > 25) throw std::invalid_argument("Supported fanouts are 0..25");
    std::vector<int32_t> table(1u << 19, 0);
    Outputs out;
    out[Samples] = seeds;
    for (size_t i = 0; i < seeds.size(); ++i) {
        if (seeds[i] < 0 || static_cast<size_t>(seeds[i]) >= graph.colptr.size() - 1)
            throw std::invalid_argument("Seed node is out of range");
        auto& bucket = table[hash_bucket(static_cast<uint64_t>(seeds[i]))];
        if (!bucket) bucket = static_cast<int32_t>(i);
    }
    // Match the HLS bucket-only table, including the zero-index empty sentinel.
    // Its collision behavior is part of the existing algorithm.
    uint32_t lfsr = 0xACE1;
    size_t begin = 0, end = seeds.size();
    for (int32_t fanout : fanouts) {
        for (size_t i = begin; i < end; ++i) {
            const auto node = out[Samples][i];
            const int32_t start = graph.colptr[node], degree = graph.colptr[node + 1] - start;
            const bool all = fanout >= degree;
            const int32_t n = std::min(fanout, degree);
            for (int32_t j = 0; j < n; ++j) {
                const int32_t offset = start + (all ? j : random_below(degree - fanout + j, lfsr));
                const int32_t value = graph.rows[offset];
                auto& bucket = table[hash_bucket(static_cast<uint64_t>(value))];
                int32_t row = bucket;
                if (!bucket) {
                    if (out[Samples].size() >= kCapacities[Samples])
                        throw CapacityExceeded("mini-batch exceeds 220000 sampled nodes");
                    row = bucket = static_cast<int32_t>(out[Samples].size());
                    out[Samples].push_back(value);
                }
                if (out[Edges].size() >= kCapacities[Edges])
                    throw CapacityExceeded("mini-batch exceeds 450000 sampled edges");
                out[Rows].push_back(row); out[Cols].push_back(static_cast<int32_t>(i)); out[Edges].push_back(offset);
            }
        }
        begin = end; end = out[Samples].size();
    }
    return out;
}
std::array<uint32_t, 4> Counts::payload_order() const { return {samples_count, rows_count, cols_count, edges_count}; }
Counts decode_counts(const std::array<uint8_t, 16>& bytes) {
    std::array<uint32_t, 4> words{};
    for (size_t i = 0; i < 4; ++i)
        for (unsigned b = 0; b < 4; ++b) words[i] |= uint32_t(bytes[i * 4 + b]) << (8 * b);
    Counts counts{words[0], words[2], words[1], words[3]};
    const auto ordered = counts.payload_order();
    for (unsigned i = 0; i < 4; ++i)
        if (ordered[i] > kCapacities[i]) // Rejects negative signed int32 values too.
            throw std::runtime_error(std::string("Invalid FPGA count: ") + kOutputNames[i] + "=" + std::to_string(ordered[i]));
    return counts;
}
Workload make_workload(const Graph& graph, uint64_t first_seed, const std::vector<int32_t>& fanouts) {
    constexpr unsigned kMaxCandidates = 64;
    // Seed nodes are drawn from [0, maximum]. Products and Reddit keep the range
    // of the paper's runs, so that the first mini-batch is the one measured there.
    const int32_t maximum = graph.dataset == "products" ? 2449027 : graph.dataset == "reddit" ? 232964 :
                            static_cast<int32_t>(graph.colptr.size() - 2);
    Workload result;
    result.fanouts = fanouts;

    // The three mini-batches and their golden-model results.
    std::array<std::vector<int32_t>, kSlrs> seeds;
    std::array<Outputs, kSlrs> expected;
    unsigned found = 0;
    for (unsigned candidate = 0; found < kSlrs; ++candidate) {
        if (candidate == kMaxCandidates)
            throw std::runtime_error("Fewer than three of the first 64 random mini-batches fit the kernel's "
                                     "output buffers (220000 nodes, 450000 edges)");
        const uint64_t seed = first_seed + candidate;
        auto nodes = generate_seeds(maximum, kSeedsPerMinibatch, seed);
        try {
            expected[found] = sample_reference(graph, nodes, fanouts);
        } catch (const CapacityExceeded&) {
            result.skipped_seeds.push_back(seed);
            continue;
        }
        seeds[found] = std::move(nodes);
        result.minibatch_seeds[found++] = seed;
    }

    // Rotation r: SLR s samples mini-batch (s + r) % 3.
    for (unsigned rotation = 0; rotation < kSlrs; ++rotation)
        for (unsigned slr = 0; slr < kSlrs; ++slr) {
            result.batches[rotation].seeds[slr] = seeds[(slr + rotation) % kSlrs];
            result.batches[rotation].expected[slr] = expected[(slr + rotation) % kSlrs];
        }

    return result;
}
} // namespace e2e
