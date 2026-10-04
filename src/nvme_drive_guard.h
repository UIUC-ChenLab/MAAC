#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <vector>

namespace nvme_guard {

// Current KIOXIA path; device numbers can change after reboot.
// verify() must confirm the identity below before any raw drive access.
inline constexpr char kDefaultDevice[] = "/dev/nvme1n1";
inline constexpr char kModel[] = "KIOXIA KCD8XPUG7T68";
inline constexpr char kSerial[] = "7E50A04D0V43";
inline constexpr uint64_t kSectors = 15002931888ULL;
inline constexpr uint64_t kBytes = kSectors * 512ULL;

struct VerifiedDrive {
  std::string path;
  dev_t device = 0;
  std::vector<std::string> system_mounts;
};

inline std::string device_number(dev_t device) {
  return std::to_string(major(device)) + ":" + std::to_string(minor(device));
}

inline std::string read_attribute(const std::filesystem::path &path) {
  std::ifstream input(path);
  std::string value;
  if (!std::getline(input, value))
    throw std::runtime_error("Cannot read drive metadata: " + path.string());
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    throw std::runtime_error("Empty drive metadata: " + path.string());
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

inline void add_unused_device(const std::filesystem::path &sys_device,
                              std::set<std::string> &devices) {
  // Holders cover devices stacked on this disk/partition, such as LVM and RAID.
  // Missing or unreadable metadata throws, so uncertainty never enables I/O.
  if (!std::filesystem::is_empty(sys_device / "holders"))
    throw std::runtime_error("Drive or partition is in use by another block device: " +
                             sys_device.filename().string());
  devices.insert(read_attribute(sys_device / "dev"));
}

// Reads filesystem and kernel metadata only. Never opens a raw block device.
inline VerifiedDrive verify(const std::string &requested_path) {
  namespace fs = std::filesystem;
  VerifiedDrive result;
  result.path = fs::canonical(requested_path).string();
  struct stat info {};
  if (::stat(result.path.c_str(), &info) != 0 || !S_ISBLK(info.st_mode))
    throw std::runtime_error("Target must be an existing NVMe block device: " +
                             requested_path);
  result.device = info.st_rdev;
  const auto sys_device = fs::canonical(
      fs::path("/sys/dev/block") / device_number(result.device));
  if (fs::exists(sys_device / "partition"))
    throw std::runtime_error("Select the whole authorized NVMe namespace, not a partition");

  const auto model = read_attribute(sys_device / "device/model");
  const auto serial = read_attribute(sys_device / "device/serial");
  if (model != kModel || serial != kSerial)
    throw std::runtime_error("Unauthorized drive " + result.path + ": " + model +
                             ", serial " + serial + "; only " + kModel +
                             ", serial " + kSerial + " is permitted");
  const auto sectors_text = read_attribute(sys_device / "size");
  size_t parsed = 0;
  const auto sectors = std::stoull(sectors_text, &parsed);
  if (parsed != sectors_text.size() || sectors != kSectors)
    throw std::runtime_error("Authorized drive capacity must be " +
                             std::to_string(kBytes) + " bytes (7.68 TB)");

  std::set<std::string> devices;
  add_unused_device(sys_device, devices);
  for (const auto &entry : fs::directory_iterator(sys_device)) {
    if (entry.is_directory() && fs::exists(entry.path() / "partition"))
      add_unused_device(entry.path(), devices);
  }

  std::ifstream mounts("/proc/self/mountinfo");
  if (!mounts)
    throw std::runtime_error("Cannot inspect current mounts; refusing drive access");
  bool found_root = false;
  std::string line;
  while (std::getline(mounts, line)) {
    std::istringstream fields(line);
    std::string id, parent, device, root, mountpoint;
    if (!(fields >> id >> parent >> device >> root >> mountpoint))
      throw std::runtime_error("Cannot parse current mount information");
    if (devices.count(device))
      throw std::runtime_error("Authorized drive or a partition is mounted at " +
                               mountpoint + "; refusing P2P access");
    if (mountpoint == "/" || mountpoint == "/boot" || mountpoint == "/boot/efi") {
      const auto system_device = fs::canonical(fs::path("/sys/dev/block") / device);
      auto system_disk = system_device;
      if (fs::exists(system_device / "partition"))
        system_disk = system_device.parent_path();
      result.system_mounts.push_back(
          mountpoint + " -> /dev/" + system_device.filename().string() +
          " (parent /dev/" + system_disk.filename().string() + ")");
      if (mountpoint == "/")
        found_root = true;
    }
  }
  if (mounts.bad() || !found_root)
    throw std::runtime_error("Cannot identify the current system root device");

  std::ifstream swaps("/proc/swaps");
  if (!std::getline(swaps, line))
    throw std::runtime_error("Cannot inspect active swap; refusing drive access");
  while (std::getline(swaps, line)) {
    std::istringstream fields(line);
    std::string path;
    if (!(fields >> path))
      throw std::runtime_error("Cannot parse active swap information");
    struct stat swap_info {};
    if (::stat(path.c_str(), &swap_info) != 0)
      throw std::runtime_error("Cannot identify active swap: " + path);
    const auto device = S_ISBLK(swap_info.st_mode) ? swap_info.st_rdev : swap_info.st_dev;
    if (devices.count(device_number(device)))
      throw std::runtime_error("Authorized drive is used for swap; refusing P2P access");
  }
  if (swaps.bad())
    throw std::runtime_error("Cannot finish inspecting active swap");
  return result;
}

} // namespace nvme_guard
