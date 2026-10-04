#pragma once

#include <cerrno>
#include <mutex>
#include <string>
#include <string_view>
#include <unistd.h>

namespace e2e {
// Diagnostics must never interrupt DMA/GPU cleanup if the output stream fails.
inline void diagnostic_log(bool enabled, std::string_view message) noexcept {
    if (!enabled) return;
    try {
        const std::string line = "[diagnostic] " + std::string(message) + "\n";
        static std::mutex output_mutex;
        std::lock_guard<std::mutex> lock(output_mutex);
        // Emit a complete line at once: Python's consumer logs concurrently.
        std::size_t done = 0;
        while (done < line.size()) {
            const auto written = ::write(STDERR_FILENO, line.data() + done, line.size() - done);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) break;
            done += static_cast<std::size_t>(written);
        }
    } catch (...) {}
}
} // namespace e2e
