#include "p2p_registration.hpp"
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <sys/ioctl.h>

namespace e2e {
namespace {
template<class T> T request() {
    T value{};
    value.size = sizeof(value);
    value.version = QDMA_PERSISTENT_ABI_VERSION;
    return value;
}
void call(int fd, unsigned long command, void* value, const std::string& operation) {
    if (ioctl(fd, command, value) < 0)
        throw std::runtime_error(operation + ": " + std::strerror(errno));
}
}
GpuPciIdentity parse_gpu_pci_identity(const std::string& bdf) {
    GpuPciIdentity value;
    int end = 0;
    if (std::sscanf(bdf.c_str(), "%x:%x:%x.%x%n", &value.domain, &value.bus,
                    &value.device, &value.function, &end) != 4 ||
        end != static_cast<int>(bdf.size()) || value.domain > 0xffff ||
        value.bus > 0xff || value.device > 31 || value.function > 7)
        throw std::invalid_argument("Invalid GPU PCI identity: " + bdf);
    return value;
}
P2pRegistrations::P2pRegistrations(const P2pQueueFds& queues,
        const std::vector<ExportSlot>& slots, std::uint64_t allocation_bytes,
        const std::string& gpu_bdf) : queues_(queues), handles_(slots.size()) {
    const auto begin = std::chrono::steady_clock::now();
    if (slots.empty() || !allocation_bytes) throw std::invalid_argument("Empty persistent allocation set");
    const auto gpu = parse_gpu_pci_identity(gpu_bdf);
    report_.mode = "persistent";
    report_.gpu_pci = gpu_bdf;
    report_.expected_registrations = slots.size() * kP2pPayloads;
    report_.cleanup_status = "open";
    // Preflight every queue and export before creating any registration.
    for (unsigned q = 0; q < queues.size(); ++q) {
        if (queues[q] < 0) throw std::invalid_argument("Invalid registration queue");
        for (unsigned previous = 0; previous < q; ++previous)
            if (queues[q] == queues[previous]) throw std::invalid_argument("Duplicate registration queue");
        for (const auto& slot : slots)
            if (slot[q] < 0) throw std::invalid_argument("Invalid export descriptor");
        auto caps = request<qdma_persistent_caps>();
        call(queues[q], QDMA_CDEV_IOCTL_CAPS, &caps, "Persistent QDMA capabilities queue " + std::to_string(q));
        if (caps.version != QDMA_PERSISTENT_ABI_VERSION || caps.size != sizeof(caps) ||
            (caps.features & QDMA_PERSISTENT_FEATURES) != QDMA_PERSISTENT_FEATURES ||
            caps.max_transfer_bytes < allocation_bytes ||
            std::string(caps.build_id, strnlen(caps.build_id, sizeof(caps.build_id))) != QDMA_PERSISTENT_BUILD_ID)
            throw std::runtime_error("Persistent QDMA capability/build mismatch on queue " + std::to_string(q));
        report_.caps_build_ids.emplace_back(caps.build_id);
    }
    try {
        for (unsigned slot = 0; slot < slots.size(); ++slot) {
            for (unsigned q = 0; q < queues.size(); ++q) {
                auto reg = request<qdma_persistent_register>();
                reg.dmabuf_fd = slots[slot][q];
                reg.direction = 0;
                reg.length = allocation_bytes;
                reg.gpu_domain = gpu.domain; reg.gpu_bus = gpu.bus;
                reg.gpu_device = gpu.device; reg.gpu_function = gpu.function;
                call(queues[q], QDMA_CDEV_IOCTL_REGISTER, &reg,
                     "Register GPU slot " + std::to_string(slot) + " queue " + std::to_string(q));
                handles_[slot][q] = reg.handle;
                if (!reg.handle || !reg.mapping_generation)
                    throw std::runtime_error("Registration returned an invalid handle or unmapped allocation");
                report_.buffers.push_back({slot, q, reg.dmabuf_fd, reg.handle, reg.length,
                                           reg.registration_ns, reg.mapping_generation});
                ++report_.registrations_completed;
            }
        }
        report_.queue_stats = read_stats();
        report_.setup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-begin).count();
    } catch (...) {
        // No transfer can exist until construction returns, so this unwind is
        // safe without consulting a GPU stream or releasing any allocations.
        close();
        throw;
    }
}
P2pRegistrations::~P2pRegistrations() { close(); }
std::uint64_t P2pRegistrations::handle(unsigned slot, unsigned payload) const {
    if (closed_ || slot >= handles_.size() || payload >= kP2pPayloads || !handles_[slot][payload])
        throw std::logic_error("Closed or invalid persistent GPU buffer handle");
    return handles_[slot][payload];
}
std::vector<qdma_persistent_stats> P2pRegistrations::read_stats() {
    std::vector<qdma_persistent_stats> result;
    for (unsigned q = 0; q < queues_.size(); ++q) {
        auto stats = request<qdma_persistent_stats>();
        call(queues_[q], QDMA_CDEV_IOCTL_STATS, &stats, "Read persistent stats queue " + std::to_string(q));
        result.push_back(stats);
    }
    return result;
}
P2pRegistrationReport P2pRegistrations::snapshot() {
    if (!closed_) report_.queue_stats = read_stats();
    return report_;
}
bool P2pRegistrations::close() noexcept {
    if (closed_) return report_.cleanup_status == "complete";
    bool okay = true;
    try { report_.queue_stats_before_close = read_stats(); }
    catch (const std::exception& error) { report_.errors.push_back(error.what()); okay = false; }
    for (const auto& stats : report_.queue_stats_before_close)
        if (stats.active_requests || stats.quarantined_requests) {
            report_.errors.push_back("DMA remains active or quarantined; allocations and descriptors must stay live");
            report_.cleanup_status = "failed";
            return false;
        }
    for (auto& slot : handles_) for (unsigned q = 0; q < queues_.size(); ++q) {
        if (!slot[q]) continue;
        auto reg = request<qdma_persistent_unregister>();
        reg.handle = slot[q];
        if (ioctl(queues_[q], QDMA_CDEV_IOCTL_UNREGISTER, &reg) < 0) {
            report_.errors.push_back("Unregister queue " + std::to_string(q) + ": " + std::strerror(errno));
            okay = false;
        } else slot[q] = 0;
    }
    try {
        report_.queue_stats_after_close = read_stats();
        for (const auto& stats : report_.queue_stats_after_close)
            if (stats.live_registrations || stats.active_requests || stats.quarantined_requests) okay = false;
    } catch (const std::exception& error) { report_.errors.push_back(error.what()); okay = false; }
    report_.cleanup_status = okay ? "complete" : "failed";
    closed_ = okay;
    return okay;
}
} // namespace e2e
