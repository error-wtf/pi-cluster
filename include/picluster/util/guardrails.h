#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <functional>

namespace picluster { namespace guardrails {

// ============================================================
// SAFE PATH POLICY
// ============================================================

struct SafePathConfig {
    std::string project_root;       // base project directory
    std::string output_root;        // where results go
    std::string checkpoint_root;    // where checkpoints go
    std::string scratch_root;       // node-local scratch
    std::vector<std::string> allowed_prefixes; // additional safe prefixes
};

// Validate that a path is safe to write to
bool is_safe_write_path(const std::string& path, const SafePathConfig& cfg);

// Validate that a path is safe to delete from (only our own dirs)
bool is_safe_delete_path(const std::string& path, const SafePathConfig& cfg);

// ============================================================
// RESOURCE GUARDRAILS
// ============================================================

struct ResourceLimits {
    double max_ram_fraction = 0.70;         // max fraction of free RAM to use
    std::uint64_t max_scratch_bytes = 0;    // 0 = no limit (use scratch_reserve instead)
    std::uint64_t max_output_bytes = 0;     // 0 = no limit
    int max_nodes = 0;                      // 0 = no limit
    int max_gpus = 0;                       // 0 = no limit
    double max_walltime_hours = 0.0;        // 0 = no limit
    double scratch_reserve_fraction = 0.15; // keep this fraction of scratch free
    double disk_abort_threshold = 0.05;     // abort if free disk drops below 5%
};

struct ResourceEstimate {
    std::size_t ram_needed_bytes = 0;
    std::uint64_t scratch_needed_bytes = 0;
    std::uint64_t output_size_bytes = 0;
    std::int64_t chunk_count = 0;
    int recommended_nodes = 1;
    double estimated_walltime_hours = 0.0;
    bool fits_in_ram = false;
    bool fits_on_scratch = false;
    std::string warning;
};

// Check if a run is feasible given current resources and limits
ResourceEstimate check_feasibility(
    std::int64_t digits,
    std::size_t free_ram,
    std::uint64_t free_scratch,
    int available_nodes,
    const ResourceLimits& limits
);

// Runtime disk watermark check — call periodically during computation
bool check_disk_watermark(const std::string& scratch_path, double abort_threshold);

// ============================================================
// DRY-RUN / PLAN MODE
// ============================================================

struct RunPlan {
    std::int64_t digits = 0;
    std::string backend;
    int nodes = 1;
    int gpus = 0;
    std::int64_t total_terms = 0;
    std::int64_t chunk_count = 0;
    std::size_t chunk_terms = 0;
    std::size_t ram_per_node = 0;
    std::uint64_t scratch_per_node = 0;
    std::uint64_t output_size = 0;
    double estimated_hours = 0.0;
    std::string scratch_path;
    std::string output_path;
    std::string checkpoint_path;
    bool gpu_available = false;
    bool mpi_active = false;
    std::vector<std::string> warnings;
    std::vector<std::string> blockers;  // if non-empty, run must not proceed
};

// Print a run plan for user review before starting
void print_run_plan(const RunPlan& plan);

// Export plan as JSON
std::string plan_to_json(const RunPlan& plan);

// ============================================================
// SIGNAL HANDLING / GRACEFUL SHUTDOWN
// ============================================================

// Install signal handlers for SIGINT, SIGTERM, Slurm SIGUSR1
void install_signal_handlers();

// Check if shutdown was requested (thread-safe)
bool shutdown_requested();

// Set a callback to run on shutdown (e.g., save last checkpoint)
void set_shutdown_callback(std::function<void()> cb);

// ============================================================
// LOGGING GUARDRAILS
// ============================================================

enum class LogLevel { DEBUG, INFO, WARN, ERROR, FATAL };

// Structured log with rate limiting to prevent stdout floods
void log_msg(LogLevel level, const char* component, const char* fmt, ...);

// Set max log lines per second (0 = unlimited)
void set_log_rate_limit(int max_per_second);

// Set log level filter
void set_log_level(LogLevel min_level);

}} // namespace picluster::guardrails
