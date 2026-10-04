#include "fpga_host.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <endian.h>
#include <fcntl.h>
#include <regex>
#include <stdexcept>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

namespace e2e {
namespace {
constexpr size_t kMapLength = 0x21000;
constexpr uint64_t kSeedStride = 4096;
void fail(const std::string& operation) { throw std::runtime_error(operation + ": " + std::strerror(errno)); }
struct Fd {
    int value = -1;
    ~Fd() { if (value >= 0) close(value); }
};
struct Aligned {
    void* data = nullptr;
    explicit Aligned(size_t bytes) {
        const int rc = posix_memalign(&data, 4096, std::max(size_t(4096), bytes));
        if (rc) throw std::runtime_error("Cannot allocate aligned DMA buffer");
        std::memset(data, 0, std::max(size_t(4096), bytes));
    }
    ~Aligned() { free(data); }
};
void exact_dma(int fd, uint64_t address, void* buffer, size_t bytes, bool write_direction) {
    if (!bytes) return;
    if (lseek(fd, static_cast<off_t>(address), SEEK_SET) != static_cast<off_t>(address)) fail("QDMA seek");
    const auto rc = write_direction ? ::write(fd, buffer, bytes) : ::read(fd, buffer, bytes);
    if (rc < 0) fail(write_direction ? "QDMA seed/control write" : "QDMA count read");
    if (static_cast<size_t>(rc) != bytes) throw std::runtime_error("Short QDMA transfer");
}
} // namespace
struct FpgaHost::Impl {
    Fd lock, dma, bar;
    // Allocate aligned count scratch once; QDMA only transfers the first 16 bytes.
    Aligned count_poison{16};
    Aligned count_readback{16};
    volatile uint32_t* regs = nullptr;
    Impl(const std::string& qdma, const std::string& bdf) {
        std::memset(count_poison.data, 0xff, 16);
        if (!std::regex_match(bdf, std::regex("[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\\.[0-7]")))
            throw std::invalid_argument("Malformed FPGA control BDF");
        lock.value = open(("/tmp/e2e-sampling-" + bdf + ".lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0666);
        if (lock.value < 0 || flock(lock.value, LOCK_EX | LOCK_NB)) fail("Acquire exclusive pipeline run lock");
        dma.value = open(qdma.c_str(), O_RDWR | O_CLOEXEC);
        if (dma.value < 0) fail("Open " + qdma);
        namespace fs = std::filesystem;
        struct stat information {};
        if (fstat(dma.value, &information) || !S_ISCHR(information.st_mode))
            throw std::runtime_error("QDMA must be a character device");
        const auto device_number = std::to_string(major(information.st_rdev)) + ":" +
                                   std::to_string(minor(information.st_rdev));
        const auto pci_path = fs::canonical(fs::path("/sys/bus/pci/devices") / bdf);
        if (fs::canonical(fs::path("/sys/dev/char") / device_number / "device") != pci_path ||
            fs::canonical(pci_path / "driver").filename() != "qdma-pf" ||
            fs::canonical(qdma).filename().string().find("-MM-") == std::string::npos)
            throw std::runtime_error("QDMA MM device and sampler BAR4 must belong to the same FPGA function");
        const auto path = "/sys/bus/pci/devices/" + bdf + "/resource4";
        bar.value = open(path.c_str(), O_RDWR | O_SYNC | O_CLOEXEC);
        if (bar.value < 0) fail("Open " + path);
        void* mapping = mmap(nullptr, kMapLength, PROT_READ | PROT_WRITE, MAP_SHARED, bar.value, 0);
        if (mapping == MAP_FAILED) fail("Map FPGA BAR4");
        regs = static_cast<volatile uint32_t*>(mapping);
        try { require_idle(); } catch (...) { munmap(mapping, kMapLength); regs = nullptr; throw; }
    }
    ~Impl() { if (regs) munmap(const_cast<uint32_t*>(regs), kMapLength); }
    uint32_t read(unsigned slr, unsigned offset) {
        std::atomic_signal_fence(std::memory_order_seq_cst);
        const auto value = le32toh(regs[(slr * 0x10000 + offset) / 4]);
        std::atomic_signal_fence(std::memory_order_seq_cst);
        return value;
    }
    void write(unsigned slr, unsigned offset, uint32_t value) {
        std::atomic_signal_fence(std::memory_order_seq_cst);
        regs[(slr * 0x10000 + offset) / 4] = htole32(value);
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    void pointer(unsigned slr, unsigned offset, uint64_t address) {
        write(slr, offset, static_cast<uint32_t>(address));
        write(slr, offset + 4, static_cast<uint32_t>(address >> 32));
    }
    void require_idle() {
        for (unsigned slr = 0; slr < 3; ++slr) {
            const uint32_t status = read(slr, 0); // Clears any previous ap_done.
            if (!(status & 4) || (status & 0x81))
                throw std::runtime_error("SLR " + std::to_string(slr) + " is active or auto-restarting");
        }
    }
};
FpgaHost::FpgaHost(const std::string& qdma, const std::string& bdf) : impl_(std::make_unique<Impl>(qdma, bdf)) {}
FpgaHost::~FpgaHost() = default;
void FpgaHost::preload(const Workload& workload) {
    impl_->require_idle();
    if (workload.batches.empty() || workload.batches.size() > 64) throw std::invalid_argument("Invalid batch count");
    for (unsigned slr = 0; slr < 3; ++slr) {
        for (size_t batch = 0; batch < workload.batches.size(); ++batch) {
            Aligned seeds(4096);
            const auto& source = workload.batches[batch].seeds[slr];
            if (source.size() != 1024) throw std::invalid_argument("Every baseline kernel call requires exactly 1024 initial nodes");
            auto* values = static_cast<uint32_t*>(seeds.data);
            for (size_t i = 0; i < source.size(); ++i) values[i] = htole32(source[i]);
            // Full seed slot includes padding needed for 256-bit HLS reads.
            exact_dma(impl_->dma.value, kSeedAddresses[slr] + batch * kSeedStride, seeds.data, 4096, true);
        }
        Aligned fanouts(4096);
        auto* values = static_cast<uint32_t*>(fanouts.data);
        for (size_t i = 0; i < workload.fanouts.size(); ++i) values[i] = htole32(workload.fanouts[i]);
        exact_dma(impl_->dma.value, kFanoutAddresses[slr], fanouts.data, 32, true);
        impl_->pointer(slr, 0x10, kGraphBase);
        impl_->pointer(slr, 0x1c, kGraphBase + kRowOffset);
        impl_->pointer(slr, 0x8c, kGraphBase + kRowOffset);
        impl_->pointer(slr, 0x98, kGraphBase + kRowOffset);
        impl_->pointer(slr, 0x3c, kFanoutAddresses[slr]);
        impl_->write(slr, 0x48, workload.fanouts.size());
        constexpr unsigned offsets[] = {0x50, 0x5c, 0x68, 0x74};
        for (unsigned kind = 0; kind < 4; ++kind) impl_->pointer(slr, offsets[kind], kOutputAddresses[slr][kind]);
        impl_->pointer(slr, 0x80, kCountAddresses[slr]);
        if (impl_->read(slr, 0x14) != uint32_t(kGraphBase >> 32))
            throw std::runtime_error("Graph pointer register readback mismatch");
    }
}
void FpgaHost::prepare_batch(size_t batch, const Workload& workload) {
    impl_->require_idle();
    if (batch >= workload.batches.size()) throw std::out_of_range("Batch index");
    for (const auto& seeds : workload.batches[batch].seeds)
        if (seeds.size() != 1024) throw std::runtime_error("Every baseline kernel call requires exactly 1024 initial nodes");
    for (unsigned slr = 0; slr < 3; ++slr) {
        exact_dma(impl_->dma.value, kCountAddresses[slr], impl_->count_poison.data, 16, true);
        impl_->pointer(slr, 0x28, kSeedAddresses[slr] + batch * kSeedStride);
        impl_->write(slr, 0x34, workload.batches[batch].seeds[slr].size());
        if (impl_->read(slr, 0x34) != workload.batches[batch].seeds[slr].size())
            throw std::runtime_error("Seed count register readback mismatch");
    }
}
SamplingTiming FpgaHost::run_batch() {
    using Clock = std::chrono::steady_clock;
    impl_->require_idle();
    SamplingTiming timing;
    std::array<bool, 3> done{};
    std::array<Clock::time_point, 3> starts{}, finishes{};
    for (int slr = 2; slr >= 0; --slr) {
        starts[slr] = Clock::now();
        impl_->write(slr, 0, 1);
    }
    const auto first_start = starts[2];
    for (;;) {
        for (unsigned slr = 0; slr < 3; ++slr) {
            // ap_done is clear-on-read. Capture each completion exactly once.
            if (!done[slr] && (impl_->read(slr, 0) & 2)) {
                finishes[slr] = Clock::now();
                done[slr] = true;
                timing.slr_ms[slr] = std::chrono::duration<double, std::milli>(finishes[slr] - starts[slr]).count();
            }
        }
        if (done[0] && done[1] && done[2]) break;
        if (Clock::now() - first_start >= std::chrono::seconds(20))
            throw std::runtime_error("Sampling timed out; stop other runs and inspect/reset the FPGA before retrying");
    }
    timing.all_slrs_ms = std::chrono::duration<double, std::milli>(
        *std::max_element(finishes.begin(), finishes.end()) - first_start).count();
    return timing;
}
PayloadCounts FpgaHost::read_counts() {
    PayloadCounts result{};
    for (unsigned slr = 0; slr < 3; ++slr) {
        // A no-op read must not reuse a previous SLR's or batch's valid counts.
        std::memset(impl_->count_readback.data, 0xff, 16);
        exact_dma(impl_->dma.value, kCountAddresses[slr], impl_->count_readback.data, 16, false);
        std::array<uint8_t, 16> packed;
        std::memcpy(packed.data(), impl_->count_readback.data, packed.size());
        result[slr] = decode_counts(packed).payload_order();
    }
    return result;
}
} // namespace e2e
