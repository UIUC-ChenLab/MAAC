#include "pipeline.hpp"
#include "diagnostics.hpp"

#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <utility>

namespace e2e {
namespace {
constexpr std::size_t kAllocationBytes = 2 * 1024 * 1024;
constexpr const char* kQdmaDevice = "/dev/qdma21000-MM-0";
constexpr const char* kControlBdf = "0000:21:00.0";

void hip_check(hipError_t rc, const char* operation) {
    if (rc != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(rc));
}
struct DeviceScope {
    int previous = -1;
    explicit DeviceScope(unsigned gpu) {
        if (gpu > static_cast<unsigned>(std::numeric_limits<int>::max()))
            throw std::out_of_range("GPU index is too large");
        hip_check(hipGetDevice(&previous), "hipGetDevice");
        if (previous != static_cast<int>(gpu)) hip_check(hipSetDevice(gpu), "hipSetDevice");
    }
    ~DeviceScope() { if (previous >= 0) (void)hipSetDevice(previous); }
};
// The counts reported by the FPGA must be exactly those of the CPU golden model.
void validate_training_counts(const PayloadCounts& counts, const Batch& expected) {
    for (unsigned slr = 0; slr < kSlrs; ++slr) {
        for (unsigned kind = 0; kind < 4; ++kind) {
            if (counts[slr][kind] > kCapacities[kind])
                throw std::runtime_error("FPGA output count exceeds allocation capacity");
            if (counts[slr][kind] != expected.expected[slr][kind].size())
                throw std::runtime_error("SLR " + std::to_string(slr) + " " + kOutputNames[kind] + " count " +
                    std::to_string(counts[slr][kind]) + " differs from the CPU golden model (" +
                    std::to_string(expected.expected[slr][kind].size()) + ")");
        }
        if (counts[slr][Samples] < kSeedsPerMinibatch)
            throw std::runtime_error("FPGA sample count does not contain all 1024 seed nodes");
        if (counts[slr][Rows] != counts[slr][Cols] || counts[slr][Rows] != counts[slr][Edges])
            throw std::runtime_error("FPGA rows, cols and edges counts must agree");
    }
}
} // namespace

struct TrainingStorage {
    struct Buffer { void* pointer = nullptr; int export_fd = -1; };
    unsigned gpu;
    std::vector<std::array<Buffer, kP2pPayloads>> slots;
    TrainingStorage(unsigned device, unsigned count) : gpu(device), slots(count) {}
    ~TrainingStorage() {
        // Last DLPack owner can disappear on any Python or PyTorch thread.
        int previous = -1;
        const bool restore = hipGetDevice(&previous) == hipSuccess;
        (void)hipSetDevice(gpu);
        for (auto& slot : slots) for (auto& buffer : slot) {
            if (buffer.export_fd >= 0) (void)::close(buffer.export_fd);
            if (buffer.pointer) (void)hipFree(buffer.pointer);
        }
        if (restore) (void)hipSetDevice(previous);
    }
    void close_exports() noexcept {
        for (auto& slot : slots) for (auto& buffer : slot) {
            if (buffer.export_fd >= 0) (void)::close(buffer.export_fd);
            buffer.export_fd = -1;
        }
    }
    // Allocate every GPU buffer and export it as a DMA-BUF for the QDMA driver.
    void initialize(hipStream_t stream) {
        DeviceScope device(gpu);
        std::size_t available = 0, total = 0;
        hip_check(hipMemGetInfo(&available, &total), "hipMemGetInfo");
        if (available < slots.size() * kP2pPayloads * kAllocationBytes)
            throw std::runtime_error("Insufficient GPU memory for training transport slots");
        for (auto& slot : slots) for (auto& buffer : slot) {
            hip_check(hipMalloc(&buffer.pointer, kAllocationBytes), "Allocate GPU payload");
            hip_check(hipMemsetAsync(buffer.pointer, 0xa5, kAllocationBytes, stream),
                      "Initialize GPU payload");
            std::uint64_t offset = 0;
            const auto rc = hsa_amd_portable_export_dmabuf(buffer.pointer, kAllocationBytes,
                                                          &buffer.export_fd, &offset);
            if (rc != HSA_STATUS_SUCCESS || buffer.export_fd < 0)
                throw std::runtime_error("GPU DMA-BUF export failed, HSA status " +
                                         std::to_string(static_cast<unsigned>(rc)));
            if (offset != 0)
                throw std::runtime_error("Nonzero DMA-BUF offset is unsupported by the installed QDMA ABI");
            const auto size = lseek(buffer.export_fd, 0, SEEK_END);
            if (size < 0 || static_cast<std::uint64_t>(size) < kAllocationBytes)
                throw std::runtime_error("GPU DMA-BUF does not cover the complete payload allocation");
            if (lseek(buffer.export_fd, 0, SEEK_SET) < 0 ||
                fcntl(buffer.export_fd, F_SETFD, FD_CLOEXEC) < 0)
                throw std::runtime_error("Configure GPU DMA-BUF: " + std::string(std::strerror(errno)));
        }
        hip_check(hipStreamSynchronize(stream), "Wait for GPU payload initialization");
    }
};

struct Pipeline::Impl {
    enum class State { Free, Writing, Ready, Acquired, Consuming, Releasing };
    struct Slot {
        State state = State::Free;
        std::uint64_t generation = 0;
        hipEvent_t handoff = nullptr, released = nullptr;
        hipStream_t consumer = nullptr;
        TrainingBatch batch;
    };
    PipelineInfo information;
    bool diagnostic;
    Workload workload;
    std::shared_ptr<TrainingStorage> storage;
    std::unique_ptr<FpgaHost> fpga;
    std::unique_ptr<P2pWorkerPool> workers;
    std::unique_ptr<P2pRegistrations> registrations;
    P2pQueueFds queue_fds;
    hipStream_t transport_stream = nullptr;
    std::vector<Slot> slots;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::thread producer;
    std::deque<unsigned> ready;
    std::set<hipStream_t> consumer_streams;
    std::exception_ptr error;
    bool closed = false, stopping = false, phase_active = false, producer_done = true;
    std::size_t requested = 0, acquired = 0, released = 0;
    std::uint64_t next_generation = 0;

    Impl(const std::string& graph_root, const std::string& dataset, unsigned gpu, bool debug)
        : diagnostic(debug) {
        queue_fds.fill(-1);
        information.registration.mode = "persistent";
        information.dataset = dataset;
        information.gpu = gpu;
        // Two sets of GPU buffers: the FPGA fills one while the GPU trains on the other.
        information.slot_count = 2;
        // Random seed of the first mini-batch candidate (the mini-batch of the paper's runs).
        const std::uint64_t first_seed = dataset == "products" ? 12346 : 12345;
        information.fanouts = dataset == "reddit" ? std::vector<std::int32_t>{25, 10} :
                                                     std::vector<std::int32_t>{15, 10, 5};
        diagnostic_log(diagnostic, "Graph file load BEGIN");
        Graph graph = load_graph(graph_root, dataset);
        information.num_nodes = graph.colptr.size() - 1;
        information.num_edges = graph.rows.size();
        diagnostic_log(diagnostic, "CPU capacity preflight BEGIN");
        workload = make_workload(graph, first_seed, information.fanouts);
        information.minibatch_seeds = workload.minibatch_seeds;
        information.skipped_seeds = workload.skipped_seeds;
        for (unsigned minibatch = 0; minibatch < kSlrs; ++minibatch)
            for (unsigned kind = 0; kind < 4; ++kind)
                information.minibatch_counts[minibatch][kind] = workload.batches[0].expected[minibatch][kind].size();
        diagnostic_log(diagnostic, "CPU capacity preflight DONE; device setup BEGIN");
        try {
            // This exclusive run lock is acquired before any graph staging or DMA.
            fpga = std::make_unique<FpgaHost>(kQdmaDevice, kControlBdf);
            const auto devices = p2p_queue_devices(kQdmaDevice);
            for (unsigned q = 0; q < kP2pQueues; ++q) {
                queue_fds[q] = open(devices[q].c_str(), O_RDWR | O_CLOEXEC);
                if (queue_fds[q] < 0)
                    throw std::runtime_error("Open " + devices[q] + ": " + std::strerror(errno) +
                                             ". Run create_qdma_queue.sh first.");
            }
            DeviceScope device(gpu);
            hip_check(hipStreamCreateWithFlags(&transport_stream, hipStreamNonBlocking),
                      "Create transport stream");
            storage = std::make_shared<TrainingStorage>(gpu, information.slot_count);
            storage->initialize(transport_stream);
            slots.resize(information.slot_count);
            for (auto& slot : slots) {
                // Default HIP events retain acquire-system fencing. Explicit release
                // scope is system as well; disabling system fences is forbidden here.
                hip_check(hipEventCreateWithFlags(&slot.handoff,
                          hipEventDisableTiming | hipEventReleaseToSystem), "Create GPU handoff event");
                hip_check(hipEventCreateWithFlags(&slot.released, hipEventDisableTiming),
                          "Create GPU release event");
            }
            // Register every GPU buffer with the QDMA driver once; transfers reuse the mapping.
            char pci_bus[32]{};
            hip_check(hipDeviceGetPCIBusId(pci_bus, sizeof(pci_bus), gpu), "Read selected GPU PCI identity");
            std::vector<P2pRegistrations::ExportSlot> exports(storage->slots.size());
            for (unsigned slot = 0; slot < exports.size(); ++slot)
                for (unsigned q = 0; q < kP2pPayloads; ++q)
                    exports[slot][q] = storage->slots[slot][q].export_fd;
            registrations = std::make_unique<P2pRegistrations>(queue_fds, exports, kAllocationBytes, pci_bus);
            information.registration = registrations->snapshot();
            workers = std::make_unique<P2pWorkerPool>(queue_fds, true);
            NvmeLoadOptions load_options;
            load_options.diagnostic = diagnostic;
            diagnostic_log(diagnostic, "KIOXIA graph stage/load/DDR validation BEGIN");
            information.nvme = load_graph_nvme(graph.colptr, graph.rows, load_options);
            diagnostic_log(diagnostic, "KIOXIA graph load and full DDR validation DONE");
            diagnostic_log(diagnostic, "FPGA seed/fanout preload BEGIN");
            fpga->preload(workload);
            diagnostic_log(diagnostic, "FPGA seed/fanout preload DONE");
            diagnostic_log(diagnostic, "Native pipeline setup DONE");
        } catch (...) {
            cleanup_resources();
            throw;
        }
    }
    ~Impl() { close_noexcept(); }

    void progress(std::size_t group, const char* stage) const {
        if (diagnostic)
            diagnostic_log(true, "Iteration " + std::to_string(group + 1) + "/" + std::to_string(requested) + ": " + stage);
    }
    void require_open() const {
        if (error) std::rethrow_exception(error);
        if (closed || stopping) throw std::logic_error("Pipeline is closed or stopping");
    }
    void reap_locked() {
        for (auto& slot : slots) if (slot.state == State::Releasing) {
            const auto rc = hipEventQuery(slot.released);
            if (rc == hipSuccess) {
                slot.state = State::Free;
                slot.consumer = nullptr;
                changed.notify_all();
            } else if (rc != hipErrorNotReady) hip_check(rc, "Query GPU buffer release");
        }
    }
    bool wait_free(std::unique_lock<std::mutex>& lock, unsigned index) {
        for (;;) {
            if (stopping) return false;
            reap_locked();
            if (slots[index].state == State::Free) return true;
            // An event can complete without a host notification. Poll only while
            // backpressured; condition notifications wake us promptly on release.
            changed.wait_for(lock, std::chrono::microseconds(100));
        }
    }
    // Iteration g: SLR s samples mini-batch (s + g) % 3, so every GPU buffer
    // receives different data in consecutive iterations.
    std::size_t workload_index(std::size_t group) const { return group % kSlrs; }
    void produce(std::size_t groups) noexcept {
        try {
            DeviceScope device(information.gpu);
            for (std::size_t group = 0; group < groups; ++group) {
                const unsigned index = group % slots.size();
                TrainingBatch batch;
                batch.slot = index;
                batch.group = group;
                const auto wi = workload_index(group);
                batch.rotation = static_cast<unsigned>(wi);
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (stopping) break;
                }
                // Only this producer touches HBM. The prior iteration's P2P has
                // completed before a subsequent sampler launch can overwrite it.
                progress(group, "FPGA batch configuration BEGIN");
                fpga->prepare_batch(wi, workload);
                progress(group, "sampling three SLRs BEGIN");
                batch.sampling = fpga->run_batch();
                progress(group, "sampling DONE; count reads BEGIN");
                batch.counts = fpga->read_counts();
                validate_training_counts(batch.counts, workload.batches[wi]);
                progress(group, "count reads DONE");
                progress(group, "waiting for reusable GPU slot");
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    if (!wait_free(lock, index)) break;
                    auto& slot = slots[index];
                    slot.state = State::Writing;
                    slot.generation = batch.generation = ++next_generation;
                }
                progress(group, "FPGA -> GPU twelve-queue P2P BEGIN");
                // One transfer per SLR and buffer kind, each on its own QDMA queue,
                // from the kernel's output window in FPGA HBM into its GPU buffer.
                std::array<P2pTransfer, kP2pPayloads> requests{};
                for (unsigned slr = 0; slr < 3; ++slr) for (unsigned kind = 0; kind < 4; ++kind) {
                    auto& request = requests[slr * 4 + kind];
                    request.source = kOutputAddresses[slr][kind];
                    request.dmabuf_fd = storage->slots[index][slr * 4 + kind].export_fd;
                    request.bytes = batch.counts[slr][kind] * sizeof(std::int32_t);
                    request.registration_handle = registrations->handle(index, slr * 4 + kind);
                }
                batch.p2p = workers->transfer(requests);
                progress(group, "GPU payload transfer DONE; publishing group");
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (stopping) break;
                    auto& slot = slots[index];
                    slot.batch = batch;
                    slot.state = State::Ready;
                    ready.push_back(index);
                    changed.notify_all();
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            error = std::current_exception();
            stopping = true;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            producer_done = true;
            changed.notify_all();
        }
    }
    void cleanup_resources() noexcept {
        // Called only after the producer exits, including all blocking ioctls.
        diagnostic_log(diagnostic, "Cleanup: draining transfer workers and GPU streams");
        workers.reset();
        int previous = -1;
        const bool restore = hipGetDevice(&previous) == hipSuccess;
        (void)hipSetDevice(information.gpu);
        bool consumers_drained = true;
        if (transport_stream && hipStreamSynchronize(transport_stream) != hipSuccess) consumers_drained = false;
        for (const auto stream : consumer_streams)
            if (hipStreamSynchronize(stream) != hipSuccess) consumers_drained = false;
        bool registrations_drained = true;
        if (registrations) {
            registrations_drained = registrations->close();
            information.registration = registrations->report();
        } else information.registration.cleanup_status = "complete";
        if (!consumers_drained || !registrations_drained) {
            information.registration.cleanup_status = "failed";
            if (!consumers_drained) information.registration.errors.push_back("GPU consumer synchronization failed");
            // A failed ioctl can outlive its userspace wait. Retain HIP buffers,
            // exports, queues and the exclusive hardware lock until process exit;
            // never hand their memory back to HIP while DMA safety is uncertain.
            (void)new std::shared_ptr<TrainingStorage>(std::move(storage));
            (void)registrations.release();
            (void)fpga.release();
            if (restore) (void)hipSetDevice(previous);
            return;
        }
        registrations.reset();
        if (storage) storage->close_exports();
        for (auto& slot : slots) {
            if (slot.handoff) (void)hipEventDestroy(slot.handoff);
            if (slot.released) (void)hipEventDestroy(slot.released);
            slot.handoff = slot.released = nullptr;
        }
        if (transport_stream) (void)hipStreamDestroy(transport_stream);
        transport_stream = nullptr;
        for (auto& fd : queue_fds) {
            if (fd >= 0) (void)::close(fd);
            fd = -1;
        }
        // DLPack tensors retain their own owner after the pipeline is closed.
        storage.reset();
        fpga.reset();
        if (restore) (void)hipSetDevice(previous);
    }
    void close_noexcept() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (closed) return;
            stopping = true;
            changed.notify_all();
        }
        if (producer.joinable()) producer.join();
        cleanup_resources();
        std::lock_guard<std::mutex> lock(mutex);
        closed = true;
        phase_active = false;
        ready.clear();
    }
};

Pipeline::Pipeline(const std::string& root, const std::string& dataset, unsigned gpu, bool diagnostic)
    : impl_(std::make_unique<Impl>(root, dataset, gpu, diagnostic)) {}
Pipeline::~Pipeline() = default;
PipelineInfo Pipeline::info() const {
    return impl_->information;
}
TrainingView Pipeline::view(unsigned slot, unsigned slr, unsigned kind) const {
    auto& state = *impl_;
    std::lock_guard<std::mutex> lock(state.mutex);
    state.require_open();
    if (slot >= state.slots.size() || slr >= 3 || kind >= 4)
        throw std::out_of_range("GPU slot, SLR or payload index");
    return {state.storage, state.storage->slots[slot][slr * 4 + kind].pointer,
            kCapacities[kind], static_cast<int>(state.information.gpu)};
}
Outputs Pipeline::reference(unsigned minibatch) const {
    if (minibatch >= kSlrs) throw std::out_of_range("Mini-batch index must be 0, 1 or 2");
    return impl_->workload.batches[0].expected[minibatch];  // rotation 0: SLR s holds mini-batch s
}
void Pipeline::start(std::size_t groups) {
    auto& state = *impl_;
    std::lock_guard<std::mutex> lock(state.mutex);
    state.require_open();
    if (!groups) throw std::invalid_argument("A producer phase requires at least one group");
    if (state.phase_active || state.producer.joinable())
        throw std::logic_error("Finish the previous phase before starting another");
    for (const auto& slot : state.slots)
        if (slot.state != Impl::State::Free) throw std::logic_error("GPU slot was not released");
    state.ready.clear();
    state.requested = groups;
    state.acquired = state.released = 0;
    state.phase_active = true;
    state.producer_done = false;
    try { state.producer = std::thread([&state, groups] { state.produce(groups); }); }
    catch (...) { state.phase_active = false; state.producer_done = true; throw; }
}
TrainingBatch Pipeline::acquire() {
    auto& state = *impl_;
    std::unique_lock<std::mutex> lock(state.mutex);
    state.require_open();
    if (!state.phase_active) throw std::logic_error("Start a producer phase before acquiring");
    state.changed.wait(lock, [&] { return !state.ready.empty() || state.producer_done || state.stopping; });
    state.require_open();
    if (state.ready.empty()) throw std::out_of_range("No unconsumed groups remain in this phase");
    const auto index = state.ready.front();
    state.ready.pop_front();
    auto& slot = state.slots[index];
    if (slot.state != Impl::State::Ready) throw std::logic_error("Internal GPU slot state mismatch");
    slot.state = Impl::State::Acquired;
    ++state.acquired;
    return slot.batch;
}
void Pipeline::begin_consume(unsigned index, std::uint64_t generation, std::uintptr_t stream_value) {
    auto& state = *impl_;
    DeviceScope device(state.information.gpu);
    std::lock_guard<std::mutex> lock(state.mutex);
    state.require_open();
    if (index >= state.slots.size()) throw std::out_of_range("GPU slot index");
    auto& slot = state.slots[index];
    if (slot.generation != generation || slot.state != Impl::State::Acquired)
        throw std::logic_error("Stale generation or GPU group was not acquired exactly once");
    auto stream = reinterpret_cast<hipStream_t>(stream_value);
    state.consumer_streams.insert(stream);
    slot.consumer = stream;
    hip_check(hipEventRecord(slot.handoff, stream), "Record system-fence GPU handoff");
    slot.state = Impl::State::Consuming;
}
void Pipeline::release(unsigned index, std::uint64_t generation, std::uintptr_t stream_value) {
    auto& state = *impl_;
    DeviceScope device(state.information.gpu);
    std::lock_guard<std::mutex> lock(state.mutex);
    state.require_open();
    if (index >= state.slots.size()) throw std::out_of_range("GPU slot index");
    auto& slot = state.slots[index];
    const auto stream = reinterpret_cast<hipStream_t>(stream_value);
    if (slot.generation != generation || slot.state != Impl::State::Consuming || slot.consumer != stream)
        throw std::logic_error("Stale/duplicate release or a different GPU consumer stream");
    hip_check(hipEventRecord(slot.released, stream), "Record GPU slot release");
    slot.state = Impl::State::Releasing;
    ++state.released;
    state.changed.notify_all();
}
void Pipeline::finish() {
    auto& state = *impl_;
    DeviceScope device(state.information.gpu);
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.phase_active) throw std::logic_error("No producer phase is active");
        if (!state.error && (state.acquired != state.requested || state.released != state.requested))
            throw std::logic_error("Acquire and release every configured group before finish");
    }
    if (state.producer.joinable()) state.producer.join();
    std::unique_lock<std::mutex> lock(state.mutex);
    if (state.error) std::rethrow_exception(state.error);
    for (;;) {
        state.reap_locked();
        if (std::all_of(state.slots.begin(), state.slots.end(), [](const auto& slot) {
                return slot.state == Impl::State::Free; })) break;
        if (state.stopping) throw std::logic_error("Pipeline stopped while waiting for GPU release");
        state.changed.wait_for(lock, std::chrono::microseconds(100));
    }
    state.phase_active = false;
}
void Pipeline::close() {
    impl_->close_noexcept();
    if (impl_->information.registration.cleanup_status == "failed")
        throw std::runtime_error("Pipeline cleanup retained unsafe DMA/GPU resources; inspect registration cleanup errors");
}
} // namespace e2e
