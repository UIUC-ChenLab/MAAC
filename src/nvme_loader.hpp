#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace e2e {

struct NvmeLoadOptions {
  std::string device =
      "/dev/disk/by-id/nvme-KIOXIA_KCD8XPUG7T68_7E50A04D0V43_1";
  std::string qdma_device = "/dev/qdma21000-MM-0";
  std::string row_provider = "0000:21:00.0";
  std::string colptr_provider = "0000:21:00.1";
  uint64_t disk_offset = 0;
  bool verify_ddr = true;
  bool diagnostic = false;
};

struct NvmeLoadResult {
  double prepare_ms = 0;
  // Shared read release -> last read return. Excludes worker creation,
  // post-read cleanup, return-value validation and DDR verification.
  double load_ms = 0;
  double worker_setup_ms = 0;
  double worker_cleanup_ms = 0;
  double verify_ms = 0;
  uint64_t colptr_disk_offset = 0;
  uint64_t row_disk_offset = 0;
  // Padded byte counts actually written to NVMe and transferred to FPGA.
  uint64_t colptr_bytes = 0;
  uint64_t row_bytes = 0;
};

// Writes the graph into the authorized, unused KIOXIA namespace, flushes it,
// then concurrently reads each array into its own FPGA P2P provider.
// BAR2 offset zero on both providers must map to FPGA DDR 0x60000000000.
// Colptr uses offset zero; rows use offset 0x958000. Disk offsets are independent
// of these FPGA addresses. The selected NVMe range is overwritten permanently.
// Throws std::exception on metadata, I/O, short transfer, or comparison errors.
NvmeLoadResult load_graph_nvme(const std::vector<int32_t> &colptr,
                              const std::vector<int32_t> &rows,
                              const NvmeLoadOptions &options);

} // namespace e2e
