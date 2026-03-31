#pragma once
#include <string>
#include <chrono>
#include <cstdint>
#include <mutex>

namespace picluster {
namespace progress {

enum class Phase {
    INIT,
    DETECT,
    BENCHMARK,
    COMPUTE_LOCAL,
    LOCAL_MERGE,
    GLOBAL_MERGE,
    CHECKPOINT,
    FINALIZE,
    DONE
};

const char* phase_name(Phase p);

struct ProgressState {
    Phase phase = Phase::INIT;
    double fraction = 0.0;          // 0.0 - 1.0
    std::int64_t digits_target = 0;
    std::int64_t digits_done = 0;
    double throughput = 0.0;        // digits/sec or terms/sec
    double eta_seconds = -1.0;      // -1 = unknown
    std::size_t ram_used_bytes = 0;
    std::size_t scratch_used_bytes = 0;
    int mpi_rank = 0;
    int mpi_size = 1;
    int checkpoint_age_sec = -1;
    int errors = 0;
    int warnings = 0;
    std::string message;
    // Chunk-level detail
    std::int64_t chunks_total = 0;
    std::int64_t chunks_done = 0;
    std::int64_t chunks_failed = 0;
    std::int64_t chunks_restored = 0;
    // Merge-level detail
    int merge_level = 0;
    int merge_total_levels = 0;
    double merge_bytes_transferred = 0;
};

class ProgressTracker {
public:
    ProgressTracker();

    void set_target(std::int64_t digits);
    void set_phase(Phase p, const std::string& msg = "");
    void update(double fraction, std::int64_t digits_done = 0, const std::string& msg = "");
    void set_mpi(int rank, int size);
    void set_checkpoint_age(int seconds);
    void increment_errors();
    void increment_warnings();
    void set_chunk_stats(std::int64_t total, std::int64_t done, std::int64_t failed, std::int64_t restored);
    void set_merge_stats(int level, int total_levels, double bytes);

    ProgressState snapshot() const;

    // Render to terminal (clears line, redraws)
    void render_terminal() const;

    // Export current state as JSON line
    std::string to_json_line() const;

private:
    mutable std::mutex mu_;
    ProgressState state_;
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point phase_start_;
};

} // namespace progress
} // namespace picluster
