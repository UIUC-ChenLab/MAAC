#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace e2e {
inline constexpr unsigned kP2pPayloads = 3 * 4;
inline constexpr unsigned kP2pQueues = kP2pPayloads;
using P2pQueueFds = std::array<int, kP2pQueues>;
using P2pQueueDevices = std::array<std::string, kP2pQueues>;
// From the control/readback MM-0 path, derive GPU queues 0..11 on the same PF.
// Queue index = 4 * SLR + kind; kind order is samples, rows, cols, edges.
P2pQueueDevices p2p_queue_devices(const std::string& mm0_device);
struct P2pTransfer {
    std::uint64_t source = 0;
    int dmabuf_fd = -1;
    std::uint32_t bytes = 0;
    std::uint64_t registration_handle = 0;
};
struct P2pBatchTiming {
    double setup_ms = 0;  // Prepare this batch for the existing workers; excludes startup.
    double wall_ms = 0;   // Shared release through the last completed ioctl.
    std::array<double, kP2pPayloads> ioctl_ms{};
    // Driver ktime_get_ns around blocking QDMA submission. Includes queue
    // waiting and completion handling; excludes outer DMA-BUF mapping/cleanup.
    // These per-request durations overlap and are not a shared batch interval.
    std::array<double, kP2pPayloads> kernel_ms{};
    std::array<std::uint64_t, kP2pPayloads> mapping_generation{};
    unsigned requests = 0;
    unsigned peak_inflight = 0;  // Overlapping host ioctl intervals, not DMA engines.
};
// Twelve lifetime workers: payload i has its own worker and queue i.
// Borrows queue descriptors; destroy the pool before closing queues or DMA-BUFs.
// Each transfer waits for every worker before returning/throwing. Zero-byte
// payloads are skipped. A submitted transfer failure makes the pool unusable.
class P2pWorkerPool {
public:
    explicit P2pWorkerPool(const P2pQueueFds& qdma_fds, bool persistent = false);
    ~P2pWorkerPool();
    P2pWorkerPool(const P2pWorkerPool&) = delete;
    P2pWorkerPool& operator=(const P2pWorkerPool&) = delete;
    P2pWorkerPool(P2pWorkerPool&&) = delete;
    P2pWorkerPool& operator=(P2pWorkerPool&&) = delete;

    double startup_ms() const noexcept;
    P2pBatchTiming transfer(const std::array<P2pTransfer, kP2pPayloads>& transfers);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace e2e
