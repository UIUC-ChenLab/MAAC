#pragma once

// Internal test seam: production users should include nvme_loader.hpp only.
#include "nvme_loader.hpp"
#include "layout.hpp"

#include <array>
#include <cstddef>
#include <sys/types.h>

namespace e2e::nvme_detail {

inline constexpr uint64_t kAlignment = 4096;
inline constexpr uint64_t kDdrBase = ::e2e::kGraphBase;
inline constexpr uint64_t kRowsOffset = ::e2e::kRowOffset;
inline constexpr size_t kReadbackChunk = 1024 * 1024;

// Index 0 is colptr; index 1 is rows throughout the internal interface.
struct GraphLayout {
  std::array<uint64_t, 2> source_bytes{};
  std::array<uint64_t, 2> padded_bytes{};
  std::array<uint64_t, 2> disk_offsets{};
  std::array<uint64_t, 2> fpga_offsets{0, kRowsOffset};
};

GraphLayout make_layout(uint64_t colptr_elements, uint64_t row_elements,
                        uint64_t disk_offset);

class NvmeIo {
public:
  virtual ~NvmeIo() = default;
  virtual ssize_t write_nvme(const void *source, size_t bytes,
                             uint64_t offset) = 0;
  virtual int flush_nvme() = 0;
  // The two read_nvme calls may execute concurrently.
  virtual ssize_t read_nvme(void *destination, size_t bytes,
                            uint64_t offset) = 0;
  virtual ssize_t read_fpga(void *destination, size_t bytes,
                            uint64_t address) = 0;
};

// All expected buffers include initialized padding. Peer buffers already point
// at their distinct final destinations within the full provider mappings.
NvmeLoadResult transfer_graph(const GraphLayout &layout,
                              const std::array<const void *, 2> &expected,
                              const std::array<void *, 2> &peer_destinations,
                              NvmeIo &io, bool verify_ddr, bool diagnostic = false);

} // namespace e2e::nvme_detail
