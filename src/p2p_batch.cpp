#include "p2p_batch.hpp"
#include <qdma_persistent_uapi.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <thread>

namespace e2e {
namespace {
using Clock = std::chrono::steady_clock;
// Wire format of the driver's per-transfer DMA-BUF ioctl; no export-offset field.
struct QdmaP2pRequest {
    unsigned long long dev_addr;
    int dmabuf_fd;
    unsigned int num_bytes;
    int direction;
    struct {
        unsigned int bytes_transferred;
        unsigned long long transfer_duration_ns;
    } response;
};
static_assert(sizeof(QdmaP2pRequest) == 40, "Unsupported QDMA ioctl ABI");
static_assert(offsetof(QdmaP2pRequest, response) == 24, "Unsupported QDMA ioctl ABI");
constexpr auto kP2pIoctl = _IOWR('J', 1, QdmaP2pRequest);
double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
} // namespace

P2pQueueDevices p2p_queue_devices(const std::string& mm0_device) {
    const std::string suffix = "-MM-0";
    if (mm0_device.size() <= suffix.size() ||
        mm0_device.compare(mm0_device.size() - suffix.size(), suffix.size(), suffix) != 0)
        throw std::invalid_argument("The twelve-queue baseline requires a QDMA MM-0 device path");
    P2pQueueDevices devices;
    const auto prefix = mm0_device.substr(0, mm0_device.size() - 1);
    for (unsigned queue = 0; queue < kP2pQueues; ++queue)
        devices[queue] = prefix + std::to_string(queue);
    return devices;
}

struct P2pWorkerPool::Impl {
    const P2pQueueFds qdma_fds;
    const bool persistent;
    std::mutex submit_mutex, mutex;
    // Workers wait only on start; only the submitting thread waits on status.
    std::condition_variable start_condition, status_condition;
    std::array<std::thread, kP2pPayloads> workers;
    unsigned ready = 0, completed = 0;
    std::uint64_t generation = 0;
    bool stopping = false, failed = false;
    double startup = 0;
    std::array<P2pTransfer, kP2pPayloads> transfers{};
    std::array<std::exception_ptr, kP2pPayloads> errors{};
    std::array<Clock::time_point, kP2pPayloads> starts{}, finishes{};
    P2pBatchTiming timing;

    explicit Impl(const P2pQueueFds& fds, bool use_persistent) : qdma_fds(fds), persistent(use_persistent) {
        for (unsigned queue = 0; queue < kP2pQueues; ++queue) {
            if (qdma_fds[queue] < 0)
                throw std::invalid_argument("All twelve QDMA queue descriptors must be open before transfer");
            for (unsigned previous = 0; previous < queue; ++previous)
                if (qdma_fds[queue] == qdma_fds[previous])
                    throw std::invalid_argument("P2P requires twelve distinct QDMA queue descriptors");
        }
        const auto begin = Clock::now();
        try {
            for (unsigned i = 0; i < workers.size(); ++i)
                workers[i] = std::thread([this, i] { run(i); });
            std::unique_lock<std::mutex> lock(mutex);
            status_condition.wait(lock, [this] { return ready == workers.size(); });
            startup = milliseconds(begin, Clock::now());
        } catch (...) {
            // No generation has been released: partially created workers did no I/O.
            stop();
            throw;
        }
    }

    ~Impl() { stop(); }

    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        start_condition.notify_all();
        for (auto& worker : workers) if (worker.joinable()) worker.join();
    }

    void run(unsigned i) {
        std::uint64_t observed_generation = 0;
        std::unique_lock<std::mutex> lock(mutex);
        ++ready;
        status_condition.notify_one();
        for (;;) {
            start_condition.wait(lock, [&] { return stopping || generation != observed_generation; });
            if (stopping) return;
            observed_generation = generation;
            const auto transfer = transfers[i];
            lock.unlock();
            if (transfer.bytes) {
                starts[i] = Clock::now();
                try {
                    std::uint64_t transferred = 0, duration = 0;
                    if (persistent) {
                        qdma_persistent_transfer request{};
                        request.size = sizeof(request);
                        request.version = QDMA_PERSISTENT_ABI_VERSION;
                        request.handle = transfer.registration_handle;
                        request.device_address = transfer.source;
                        request.length = transfer.bytes;
                        if (ioctl(qdma_fds[i], QDMA_CDEV_IOCTL_TRANSFER_REGISTERED, &request) < 0) {
                            const int error = errno;
                            throw std::runtime_error("Registered P2P payload " + std::to_string(i) + ": " + std::strerror(error));
                        }
                        transferred = request.bytes_transferred;
                        duration = request.transfer_duration_ns;
                        timing.mapping_generation[i] = request.mapping_generation;
                    } else {
                        QdmaP2pRequest request{};
                        request.dev_addr = transfer.source;
                        request.dmabuf_fd = transfer.dmabuf_fd;
                        request.num_bytes = transfer.bytes;
                        request.direction = 0;
                        if (ioctl(qdma_fds[i], kP2pIoctl, &request) < 0) {
                            const int error = errno;
                            throw std::runtime_error("P2P payload " + std::to_string(i) + ": " + std::strerror(error));
                        }
                        transferred = request.response.bytes_transferred;
                        duration = request.response.transfer_duration_ns;
                    }
                    if (transferred != transfer.bytes)
                        throw std::runtime_error("P2P payload " + std::to_string(i) +
                            " byte-count mismatch: requested " + std::to_string(transfer.bytes) +
                            ", returned " + std::to_string(transferred));
                    timing.kernel_ms[i] = static_cast<double>(duration) / 1e6;
                } catch (...) {
                    errors[i] = std::current_exception();
                }
                // Keep the existing timer endpoint: after ioctl/checks, before
                // notifying the coordinator or waiting for another generation.
                finishes[i] = Clock::now();
            }
            lock.lock();
            if (++completed == workers.size()) status_condition.notify_one();
        }
    }

    P2pBatchTiming transfer(const std::array<P2pTransfer, kP2pPayloads>& requests) {
        std::lock_guard<std::mutex> submission(submit_mutex);
        if (failed) throw std::logic_error("P2P worker pool cannot continue after a failed transfer");
        unsigned request_count = 0;
        for (const auto& request : requests) {
            if (!request.bytes) continue;
            if ((persistent ? request.registration_handle == 0 : request.dmabuf_fd < 0) ||
                request.source > std::numeric_limits<std::uint64_t>::max() - request.bytes)
                throw std::invalid_argument("Invalid P2P descriptor or FPGA source range");
            ++request_count;
        }
        if (!request_count) return {};

        const auto setup_start = Clock::now();
        std::unique_lock<std::mutex> lock(mutex);
        transfers = requests;
        timing = {};
        timing.requests = request_count;
        errors.fill(nullptr);
        starts.fill({});
        finishes.fill({});
        completed = 0;
        const auto start = Clock::now();
        ++generation;
        lock.unlock();
        start_condition.notify_all();
        lock.lock();
        // Do not release/reuse DMA-BUFs even when one ioctl fails early.
        status_condition.wait(lock, [this] { return completed == workers.size(); });
        for (const auto& error : errors) if (error) {
            failed = true;
            std::rethrow_exception(error);
        }
        timing.setup_ms = milliseconds(setup_start, start);
        timing.wall_ms = milliseconds(start, *std::max_element(finishes.begin(), finishes.end()));
        for (unsigned i = 0; i < transfers.size(); ++i) {
            if (!transfers[i].bytes) continue;
            timing.ioctl_ms[i] = milliseconds(starts[i], finishes[i]);
            unsigned inflight = 0;
            for (unsigned j = 0; j < transfers.size(); ++j)
                if (transfers[j].bytes && starts[j] <= starts[i] && finishes[j] > starts[i]) ++inflight;
            timing.peak_inflight = std::max(timing.peak_inflight, inflight);
        }
        return timing;
    }
};

P2pWorkerPool::P2pWorkerPool(const P2pQueueFds& fds, bool persistent) : impl_(std::make_unique<Impl>(fds, persistent)) {}
P2pWorkerPool::~P2pWorkerPool() = default;
double P2pWorkerPool::startup_ms() const noexcept { return impl_->startup; }
P2pBatchTiming P2pWorkerPool::transfer(const std::array<P2pTransfer, kP2pPayloads>& transfers) {
    return impl_->transfer(transfers);
}
} // namespace e2e
