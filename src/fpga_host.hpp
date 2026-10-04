#pragma once
#include "workload.hpp"
#include <memory>
#include <string>

namespace e2e {
struct SamplingTiming {
    std::array<double, 3> slr_ms{}; // Each start write through its observed ap_done.
    double all_slrs_ms = 0;        // First start through the last observed ap_done.
};
class FpgaHost {
public:
    FpgaHost(const std::string& qdma_device, const std::string& control_bdf);
    ~FpgaHost();
    FpgaHost(const FpgaHost&) = delete;
    FpgaHost& operator=(const FpgaHost&) = delete;
    void preload(const Workload&);
    void prepare_batch(size_t batch, const Workload&);
    SamplingTiming run_batch(); // Host-observed times in milliseconds.
    PayloadCounts read_counts(); // three exact 16-byte host reads, never payload readback
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace e2e
