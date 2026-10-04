#include "nvme_loader_internal.hpp"
#include "nvme_drive_guard.h"
#include "diagnostics.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <linux/fs.h>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace e2e {
namespace {

using Clock = std::chrono::steady_clock;
using nvme_detail::kAlignment;
using nvme_detail::kDdrBase;
using nvme_detail::kReadbackChunk;
using nvme_detail::kRowsOffset;

[[noreturn]] void system_error(const std::string &operation, int error = errno) {
  throw std::runtime_error(operation + ": " + std::strerror(error));
}

double elapsed_ms(Clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

void require_full(ssize_t actual, size_t expected, const std::string &operation) {
  if (actual < 0) system_error(operation);
  if (static_cast<size_t>(actual) != expected)
    throw std::runtime_error(operation + ": short transfer (" +
                             std::to_string(actual) + "/" +
                             std::to_string(expected) + " bytes)");
}

class File {
public:
  const int fd;
  File(const std::string &path, int flags) : fd(::open(path.c_str(), flags)) {
    if (fd < 0) system_error("open(" + path + ")");
  }
  ~File() { ::close(fd); }
  File(const File &) = delete;
  File &operator=(const File &) = delete;
};

class HostBuffer {
public:
  void *data = nullptr;
  HostBuffer(size_t bytes, size_t alignment = kAlignment) {
    const int error = ::posix_memalign(&data, alignment, bytes);
    if (error) system_error("aligned host allocation", error);
  }
  ~HostBuffer() { std::free(data); }
  HostBuffer(const HostBuffer &) = delete;
  HostBuffer &operator=(const HostBuffer &) = delete;
};

uint64_t decimal_attribute(const std::filesystem::path &path) {
  const auto value = nvme_guard::read_attribute(path);
  if (value.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error("Invalid numeric device metadata: " + path.string());
  size_t used = 0;
  const auto result = std::stoull(value, &used);
  if (used != value.size())
    throw std::runtime_error("Invalid numeric device metadata: " + path.string());
  return result;
}

struct Provider {
  std::filesystem::path path;
  size_t bytes;
};

Provider inspect_provider(const std::string &bdf, uint64_t needed_bytes,
                          size_t page_size) {
  namespace fs = std::filesystem;
  const auto path = fs::canonical(fs::path("/sys/bus/pci/devices") / bdf);
  if (fs::canonical(path / "driver").filename() != "qdma-pf")
    throw std::runtime_error("P2P provider must use qdma-pf: " + bdf);
  const auto pool = path / "p2pmem";
  const auto bytes = decimal_attribute(pool / "size");
  if (nvme_guard::read_attribute(pool / "published") != "1" || !bytes ||
      decimal_attribute(pool / "available") != bytes)
    throw std::runtime_error("P2P provider must be published and fully free: " + bdf);
  if (bytes < needed_bytes || bytes % page_size ||
      bytes > std::numeric_limits<size_t>::max() ||
      bytes > static_cast<uint64_t>(std::numeric_limits<off_t>::max()))
    throw std::runtime_error("Graph destination does not fit P2P provider: " + bdf);

  std::ifstream resource(path / "resource");
  std::string line;
  for (int bar = 0; bar <= 2; ++bar)
    if (!std::getline(resource, line))
      throw std::runtime_error("Cannot inspect BAR2 resource: " + bdf);
  uint64_t first = 0, last = 0, flags = 0;
  std::istringstream fields(line);
  if (!(fields >> std::hex >> first >> last >> flags) || !(flags & 0x200) ||
      last < first || last - first != bytes - 1)
    throw std::runtime_error("P2P pool must cover the entire QDMA BAR2: " + bdf);
  return {path, static_cast<size_t>(bytes)};
}

struct ReadbackDevice {
  std::string path;
  dev_t device;
};

ReadbackDevice inspect_readback(const std::string &requested,
                                const Provider &provider) {
  namespace fs = std::filesystem;
  const auto path = fs::canonical(requested);
  struct stat information {};
  if (::stat(path.c_str(), &information) || !S_ISCHR(information.st_mode) ||
      path.filename().string().find("-MM-") == std::string::npos)
    throw std::runtime_error("Readback requires a QDMA MM character device");
  const auto sys_char = fs::path("/sys/dev/char") /
                        nvme_guard::device_number(information.st_rdev);
  if (fs::canonical(sys_char / "device") != provider.path ||
      fs::canonical(sys_char / "subsystem").filename() != "qdma-pf")
    throw std::runtime_error("QDMA readback device must belong to row_provider");
  return {path.string(), information.st_rdev};
}

void check_opened_device(int fd, dev_t expected, bool block) {
  struct stat information {};
  if (::fstat(fd, &information)) system_error("fstat(opened device)");
  if (information.st_rdev != expected ||
      (block ? !S_ISBLK(information.st_mode) : !S_ISCHR(information.st_mode)))
    throw std::runtime_error("Device changed since metadata verification");
}

class P2pMapping {
  File allocation_;
public:
  void *data = MAP_FAILED;
  const size_t bytes;
  explicit P2pMapping(const Provider &provider)
      : allocation_((provider.path / "p2pmem/allocate").string(),
                    O_RDWR | O_SYNC | O_CLOEXEC), bytes(provider.bytes) {
    // Reserve the whole sole BAR2 pool: mmap offset zero alone would not force
    // a partial allocation to begin at BAR2 offset zero.
    if (::ftruncate(allocation_.fd, static_cast<off_t>(bytes)))
      system_error("ftruncate(P2P allocation)");
    data = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                  allocation_.fd, 0);
    if (data == MAP_FAILED) system_error("mmap(full P2P pool)");
  }
  ~P2pMapping() { if (data != MAP_FAILED) ::munmap(data, bytes); }
  P2pMapping(const P2pMapping &) = delete;
  P2pMapping &operator=(const P2pMapping &) = delete;
};

class DeviceIo final : public nvme_detail::NvmeIo {
  const int nvme_fd_;
  const int fpga_fd_;
public:
  DeviceIo(int nvme_fd, int fpga_fd) : nvme_fd_(nvme_fd), fpga_fd_(fpga_fd) {}
  ssize_t write_nvme(const void *source, size_t bytes, uint64_t offset) override {
    return ::pwrite(nvme_fd_, source, bytes, static_cast<off_t>(offset));
  }
  int flush_nvme() override { return ::fsync(nvme_fd_); }
  ssize_t read_nvme(void *destination, size_t bytes, uint64_t offset) override {
    return ::pread(nvme_fd_, destination, bytes, static_cast<off_t>(offset));
  }
  ssize_t read_fpga(void *destination, size_t bytes, uint64_t address) override {
    // This installed driver's read does not advance file->f_pos.
    const auto result = ::lseek(fpga_fd_, static_cast<off_t>(address), SEEK_SET);
    if (result < 0) system_error("lseek(QDMA DDR)");
    if (static_cast<uint64_t>(result) != address)
      throw std::runtime_error("QDMA seek returned the wrong DDR address");
    return ::read(fpga_fd_, destination, bytes);
  }
};

} // namespace

namespace nvme_detail {

GraphLayout make_layout(uint64_t colptr_elements, uint64_t row_elements,
                        uint64_t disk_offset) {
  if (disk_offset % kAlignment || disk_offset > nvme_guard::kBytes)
    throw std::runtime_error("NVMe offset must be 4096-aligned and inside KIOXIA");
  GraphLayout result;
  const std::array<uint64_t, 2> elements{colptr_elements, row_elements};
  uint64_t next_disk_offset = disk_offset;
  for (size_t i = 0; i < elements.size(); ++i) {
    if (!elements[i] ||
        elements[i] > (std::numeric_limits<uint64_t>::max() - kAlignment + 1) /
                          sizeof(int32_t))
      throw std::runtime_error("Graph arrays must be nonempty and fit byte counts");
    result.source_bytes[i] = elements[i] * sizeof(int32_t);
    result.padded_bytes[i] =
        (result.source_bytes[i] + kAlignment - 1) & ~(kAlignment - 1);
    // One direct read per provider, matching the benchmark. Linux caps one
    // read/pwrite at MAX_RW_COUNT; reject larger inputs instead of truncating.
    if (result.padded_bytes[i] > 0x7ffff000ULL ||
        result.padded_bytes[i] > std::numeric_limits<size_t>::max())
      throw std::runtime_error("Graph array exceeds the single direct-I/O limit");
    if (result.padded_bytes[i] > nvme_guard::kBytes - next_disk_offset)
      throw std::runtime_error("Graph write range exceeds authorized KIOXIA capacity");
    result.disk_offsets[i] = next_disk_offset;
    next_disk_offset += result.padded_bytes[i];
  }
  if (result.padded_bytes[0] > kRowsOffset)
    throw std::runtime_error("Colptr exceeds reserved DDR space before rows at 0x958000");
  return result;
}

NvmeLoadResult transfer_graph(const GraphLayout &layout,
                              const std::array<const void *, 2> &expected,
                              const std::array<void *, 2> &peer_destinations,
                              NvmeIo &io, bool verify_ddr, bool diagnostic) {
  for (size_t i = 0; i < 2; ++i)
    if (!expected[i] || !peer_destinations[i])
      throw std::runtime_error("Graph staging and P2P destinations must be allocated");
  std::unique_ptr<HostBuffer> readback;
  if (verify_ddr) readback = std::make_unique<HostBuffer>(kReadbackChunk);
  NvmeLoadResult result;
  result.colptr_disk_offset = layout.disk_offsets[0];
  result.row_disk_offset = layout.disk_offsets[1];
  result.colptr_bytes = layout.padded_bytes[0];
  result.row_bytes = layout.padded_bytes[1];
  const std::array<const char *, 2> names{"colptr", "rows"};

  auto begin = Clock::now();
  for (size_t i = 0; i < 2; ++i) {
    if (diagnostic) diagnostic_log(true, std::string("KIOXIA staging write BEGIN: ") + names[i]);
    require_full(io.write_nvme(expected[i], layout.padded_bytes[i],
                                layout.disk_offsets[i]),
                 layout.padded_bytes[i], std::string("Host -> NVMe ") + names[i]);
    if (diagnostic) diagnostic_log(true, std::string("KIOXIA staging write DONE: ") + names[i]);
  }
  diagnostic_log(diagnostic, "KIOXIA fsync BEGIN");
  if (io.flush_nvme()) system_error("fsync(NVMe graph)");
  result.prepare_ms = elapsed_ms(begin);
  diagnostic_log(diagnostic, "KIOXIA fsync DONE");

  std::array<std::exception_ptr, 2> errors{};
  std::array<ssize_t, 2> read_results{};
  std::array<int, 2> read_errors{};
  std::array<Clock::time_point, 2> read_finishes{};
  const auto read_part = [&](size_t i) noexcept {
    try {
      if (diagnostic) diagnostic_log(true, std::string("NVMe -> FPGA read BEGIN: ") + names[i]);
      const auto bytes = io.read_nvme(peer_destinations[i], layout.padded_bytes[i],
                                     layout.disk_offsets[i]);
      const int error = errno; // Preserve the reader thread's errno before cleanup.
      read_finishes[i] = Clock::now();
      read_results[i] = bytes;
      read_errors[i] = error;
      if (diagnostic) diagnostic_log(true, std::string("NVMe -> FPGA read RETURNED: ") + names[i] +
                                    ", bytes=" + std::to_string(bytes));
    } catch (...) {
      read_finishes[i] = Clock::now();
      errors[i] = std::current_exception();
    }
  };
  const size_t large = layout.source_bytes[1] >= layout.source_bytes[0] ? 1 : 0;
  // Construct the worker and wait for readiness before starting the shared
  // transfer interval. Separate readiness/release signals avoid startup cost
  // in load_ms. Dispatch the larger array first, as in the original loader.
  std::promise<void> worker_ready, release_worker, large_started;
  auto ready = worker_ready.get_future();
  auto release = release_worker.get_future();
  auto started = large_started.get_future();
  const auto setup_begin = Clock::now();
  std::thread larger([&] {
    worker_ready.set_value();
    release.wait();
    large_started.set_value();
    read_part(large);
  });
  ready.wait();
  result.worker_setup_ms = elapsed_ms(setup_begin);
  begin = Clock::now();
  release_worker.set_value();
  started.wait();
  read_part(1 - large);
  larger.join();
  const auto joined = Clock::now();
  const auto finished = *std::max_element(read_finishes.begin(), read_finishes.end());
  result.load_ms = std::chrono::duration<double, std::milli>(finished - begin).count();
  // The join may also wait for the larger read. Only time AFTER the final read
  // returned belongs to cleanup; never count the outstanding transfer twice.
  result.worker_cleanup_ms = std::chrono::duration<double, std::milli>(joined - finished).count();
  // Both reads have stopped before checking results, propagating failures or
  // unmapping pools. Formatting/error checks are outside the transfer interval.
  for (size_t i = 0; i < 2; ++i) {
    if (errors[i]) std::rethrow_exception(errors[i]);
    if (read_results[i] >= 0 && static_cast<size_t>(read_results[i]) == layout.padded_bytes[i])
      continue;
    const auto operation = std::string("NVMe -> FPGA ") + names[i];
    if (read_results[i] < 0) system_error(operation, read_errors[i]);
    require_full(read_results[i], layout.padded_bytes[i], operation);
  }

  if (verify_ddr) {
    begin = Clock::now();
    for (size_t i = 0; i < 2; ++i) {
      if (diagnostic) diagnostic_log(true, std::string("FPGA DDR validation BEGIN: ") + names[i]);
      const auto *reference = static_cast<const unsigned char *>(expected[i]);
      auto *actual = static_cast<unsigned char *>(readback->data);
      for (uint64_t done = 0; done < layout.padded_bytes[i];) {
        const auto bytes = static_cast<size_t>(
            std::min<uint64_t>(kReadbackChunk, layout.padded_bytes[i] - done));
        for (size_t j = 0; j < bytes; ++j) actual[j] = reference[done + j] ^ 0xff;
        const auto address = kDdrBase + layout.fpga_offsets[i] + done;
        require_full(io.read_fpga(actual, bytes, address), bytes,
                     std::string("FPGA DDR -> host ") + names[i]);
        if (std::memcmp(reference + done, actual, bytes)) {
          size_t first = 0;
          while (first < bytes && reference[done + first] == actual[first]) ++first;
          std::ostringstream message;
          message << names[i] << " DDR mismatch at byte " << done + first
                  << " (FPGA 0x" << std::hex << address + first << std::dec
                  << "): expected " << static_cast<unsigned>(reference[done + first])
                  << ", got " << static_cast<unsigned>(actual[first]);
          throw std::runtime_error(message.str());
        }
        done += bytes;
        if (diagnostic && (done % (64 * 1024 * 1024) == 0 || done == layout.padded_bytes[i]))
          diagnostic_log(true, std::string("FPGA DDR validation progress: ") + names[i] +
                         " " + std::to_string(done) + "/" + std::to_string(layout.padded_bytes[i]) + " bytes");
      }
    }
    result.verify_ms = elapsed_ms(begin);
  }
  return result;
}

} // namespace nvme_detail

NvmeLoadResult load_graph_nvme(const std::vector<int32_t> &colptr,
                              const std::vector<int32_t> &rows,
                              const NvmeLoadOptions &options) {
  const auto layout = nvme_detail::make_layout(colptr.size(), rows.size(),
                                               options.disk_offset);
  const std::regex bdf_pattern("[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\\.[0-7]");
  if (!std::regex_match(options.row_provider, bdf_pattern) ||
      !std::regex_match(options.colptr_provider, bdf_pattern) ||
      options.row_provider == options.colptr_provider ||
      options.row_provider.substr(0, 10) != options.colptr_provider.substr(0, 10))
    throw std::runtime_error("Graph providers must be distinct functions of the same FPGA");

  // The existing guard checks exact model/serial/capacity, boot filesystems,
  // mounts, partitions, swap, and holders before any device is opened.
  const auto drive = nvme_guard::verify(options.device);
  const auto page_size = ::sysconf(_SC_PAGESIZE);
  if (page_size <= 0 || kAlignment % page_size || kRowsOffset % page_size ||
      kReadbackChunk % page_size)
    throw std::runtime_error("This DDR/direct-I/O layout requires 4096-byte host pages");
  const auto colptr_provider = inspect_provider(options.colptr_provider,
                                                layout.padded_bytes[0], page_size);
  const auto row_provider = inspect_provider(options.row_provider,
                                             kRowsOffset + layout.padded_bytes[1],
                                             page_size);
  std::optional<ReadbackDevice> readback_device;
  if (options.verify_ddr)
    readback_device = inspect_readback(options.qdma_device, row_provider);

  File nvme(drive.path, O_RDWR | O_DIRECT | O_EXCL | O_CLOEXEC);
  check_opened_device(nvme.fd, drive.device, true);
  uint64_t capacity = 0;
  unsigned int logical_block_bytes = 0;
  if (::ioctl(nvme.fd, BLKGETSIZE64, &capacity) ||
      ::ioctl(nvme.fd, BLKSSZGET, &logical_block_bytes))
    system_error("NVMe geometry ioctl");
  if (capacity != nvme_guard::kBytes || !logical_block_bytes ||
      (logical_block_bytes & (logical_block_bytes - 1)) ||
      kAlignment % logical_block_bytes || options.disk_offset % logical_block_bytes)
    throw std::runtime_error("NVMe capacity changed or graph layout is not LBA aligned");

  std::unique_ptr<File> fpga;
  if (readback_device) {
    fpga = std::make_unique<File>(readback_device->path, O_RDONLY | O_CLOEXEC);
    check_opened_device(fpga->fd, readback_device->device, false);
  }
  P2pMapping colptr_mapping(colptr_provider);
  P2pMapping row_mapping(row_provider);
  HostBuffer colptr_staging(layout.padded_bytes[0]);
  HostBuffer row_staging(layout.padded_bytes[1]);
  std::memset(colptr_staging.data, 0, layout.padded_bytes[0]);
  std::memset(row_staging.data, 0, layout.padded_bytes[1]);
  std::memcpy(colptr_staging.data, colptr.data(), layout.source_bytes[0]);
  std::memcpy(row_staging.data, rows.data(), layout.source_bytes[1]);

  std::cout << "Verified graph NVMe: " << drive.path << " (" << nvme_guard::kModel
            << ", serial " << nvme_guard::kSerial << ")\n";
  for (const auto &mount : drive.system_mounts)
    std::cout << "Excluded system mount: " << mount << '\n';
  std::cout << "Graph staging overwrites NVMe bytes [" << layout.disk_offsets[0]
            << ", " << layout.disk_offsets[1] + layout.padded_bytes[1] << ").\n"
            << "colptr: " << options.colptr_provider << ", " << layout.padded_bytes[0]
            << " bytes -> DDR 0x" << std::hex << kDdrBase << std::dec << '\n'
            << "rows: " << options.row_provider << ", " << layout.padded_bytes[1]
            << " bytes -> DDR 0x" << std::hex << kDdrBase + kRowsOffset << std::dec
            << std::endl;
  DeviceIo io(nvme.fd, fpga ? fpga->fd : -1);
  return nvme_detail::transfer_graph(
      layout, {colptr_staging.data, row_staging.data},
      {colptr_mapping.data, static_cast<unsigned char *>(row_mapping.data) + kRowsOffset},
      io, options.verify_ddr, options.diagnostic);
}

} // namespace e2e
