#pragma once
#include "p2p_batch.hpp"
#include <qdma_persistent_uapi.h>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace e2e {
struct GpuPciIdentity {
    std::uint32_t domain = 0, bus = 0, device = 0, function = 0;
};
GpuPciIdentity parse_gpu_pci_identity(const std::string& bdf);
struct P2pRegistrationRecord {
    unsigned slot = 0, queue = 0;
    int export_fd = -1;
    std::uint64_t handle = 0, length = 0, registration_ns = 0, initial_mapping_generation = 0;
};
struct P2pRegistrationReport {
    std::string mode = "transient", gpu_pci, cleanup_status = "not_started";
    std::vector<std::string> caps_build_ids, errors;
    std::vector<P2pRegistrationRecord> buffers;
    std::uint64_t expected_registrations = 0, registrations_completed = 0;
    double setup_ms = 0;
    std::vector<qdma_persistent_stats> queue_stats, queue_stats_before_close, queue_stats_after_close;
};
// Borrows queue descriptors and DMA-BUFs. Must close successfully before the
// caller frees allocations. A failed close retains all still-live handles.
class P2pRegistrations {
public:
    using ExportSlot = std::array<int, kP2pPayloads>;
    P2pRegistrations(const P2pQueueFds& queues, const std::vector<ExportSlot>& slots,
                     std::uint64_t allocation_bytes, const std::string& gpu_bdf);
    ~P2pRegistrations();
    P2pRegistrations(const P2pRegistrations&) = delete;
    P2pRegistrations& operator=(const P2pRegistrations&) = delete;
    std::uint64_t handle(unsigned slot, unsigned payload) const;
    P2pRegistrationReport snapshot();
    bool close() noexcept;
    const P2pRegistrationReport& report() const noexcept { return report_; }
private:
    std::vector<qdma_persistent_stats> read_stats();
    P2pQueueFds queues_;
    std::vector<std::array<std::uint64_t, kP2pPayloads>> handles_;
    P2pRegistrationReport report_;
    bool closed_ = false;
};
} // namespace e2e
