#pragma once

#include "fpga_host.hpp"
#include "nvme_loader.hpp"
#include "p2p_batch.hpp"
#include "p2p_registration.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace e2e {

// This owner is held both by the pipeline and by every DLPack capsule/tensor.
// Tensor leases retain HIP allocations after pipeline close; exports close only
// after GPU consumers drain and all persistent registrations are removed.
struct TrainingStorage;
struct TrainingView {
    std::shared_ptr<TrainingStorage> owner;
    void* pointer = nullptr;
    std::int64_t length = 0;
    int device = 0;
};
struct PipelineInfo {
    std::string dataset;
    unsigned gpu = 0, slot_count = 0;
    std::size_t num_nodes = 0, num_edges = 0;
    std::vector<std::int32_t> fanouts;
    // The three mini-batches: their random seeds, the candidates that were skipped
    // because they do not fit the kernel buffers, and their sampled sizes
    // (samples, rows, cols, edges) according to the CPU golden model.
    std::array<std::uint64_t, kSlrs> minibatch_seeds{};
    std::vector<std::uint64_t> skipped_seeds;
    PayloadCounts minibatch_counts{};
    NvmeLoadResult nvme;
    P2pRegistrationReport registration;
};
struct TrainingBatch {
    unsigned slot = 0;
    std::uint64_t generation = 0;
    std::size_t group = 0;
    unsigned rotation = 0;  // SLR s sampled mini-batch (s + rotation) % 3
    PayloadCounts counts{};
    SamplingTiming sampling;
    P2pBatchTiming p2p;
};

class Pipeline {
public:
    // Does all one-time work: golden model, GPU buffers and their registration,
    // NVMe -> FPGA graph loading (writes the graph to the SSD first), seed upload.
    Pipeline(const std::string& graph_root, const std::string& dataset, unsigned gpu = 0,
             bool diagnostic = false);
    ~Pipeline();
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    PipelineInfo info() const;
    TrainingView view(unsigned slot, unsigned slr, unsigned kind) const;
    Outputs reference(unsigned minibatch) const;  // CPU golden-model result of mini-batch 0, 1 or 2
    void start(std::size_t groups);
    TrainingBatch acquire();
    void begin_consume(unsigned slot, std::uint64_t generation, std::uintptr_t stream);
    void release(unsigned slot, std::uint64_t generation, std::uintptr_t stream);
    void finish();
    void close();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace e2e
