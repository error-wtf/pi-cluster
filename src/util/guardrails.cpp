// guardrails.cpp — Failsafe, path validation, resource limits, signal handling
#include "picluster/util/guardrails.h"
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <csignal>
#include <atomic>
#include <mutex>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <iomanip>

#if defined(__linux__)
#include <sys/statvfs.h>
#endif

namespace fs = std::filesystem;

namespace picluster { namespace guardrails {

// ============================================================
// SAFE PATH POLICY
// ============================================================

static bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

bool is_safe_write_path(const std::string& path, const SafePathConfig& cfg) {
    if (path.empty()) return false;
    // Reject absolute system paths
    if (path == "/" || path == "/tmp" || path == "/home") return false;
    // Must be under one of the allowed roots
    std::vector<std::string> safe = cfg.allowed_prefixes;
    if (!cfg.project_root.empty()) safe.push_back(cfg.project_root);
    if (!cfg.output_root.empty()) safe.push_back(cfg.output_root);
    if (!cfg.checkpoint_root.empty()) safe.push_back(cfg.checkpoint_root);
    if (!cfg.scratch_root.empty()) safe.push_back(cfg.scratch_root);
    // Also allow TMPDIR-based paths
    const char* tmpdir = std::getenv("TMPDIR");
    if (tmpdir) safe.push_back(tmpdir);
    const char* slurm_tmp = std::getenv("SLURM_TMPDIR");
    if (slurm_tmp) safe.push_back(slurm_tmp);
    safe.push_back("/tmp/pi-cluster");

    for (auto& prefix : safe) {
        if (!prefix.empty() && starts_with(path, prefix)) return true;
    }
    return false;
}

bool is_safe_delete_path(const std::string& path, const SafePathConfig& cfg) {
    // Even stricter: only delete inside scratch or our own project dirs
    if (path.empty()) return false;
    if (!is_safe_write_path(path, cfg)) return false;
    // Must contain "pi-cluster" or "pi_cluster" in the path
    if (path.find("pi-cluster") != std::string::npos) return true;
    if (path.find("pi_cluster") != std::string::npos) return true;
    // Or be explicitly under scratch_root
    if (!cfg.scratch_root.empty() && starts_with(path, cfg.scratch_root)) return true;
    return false;
}

// ============================================================
// RESOURCE GUARDRAILS
// ============================================================

ResourceEstimate check_feasibility(
    std::int64_t digits, std::size_t free_ram,
    std::uint64_t free_scratch, int available_nodes,
    const ResourceLimits& limits)
{
    ResourceEstimate est;
    // ~10 bytes RAM per digit, ~4 bytes scratch per digit
    est.ram_needed_bytes = static_cast<std::size_t>(digits) * 10;
    est.scratch_needed_bytes = static_cast<std::uint64_t>(digits) * 4;
    est.output_size_bytes = static_cast<std::uint64_t>(digits) + 100;

    std::size_t usable_ram = static_cast<std::size_t>(free_ram * limits.max_ram_fraction);
    std::uint64_t usable_scratch = static_cast<std::uint64_t>(
        free_scratch * (1.0 - limits.scratch_reserve_fraction));

    est.fits_in_ram = est.ram_needed_bytes <= usable_ram;
    est.fits_on_scratch = est.scratch_needed_bytes <= usable_scratch;

    // Recommended nodes
    if (est.fits_in_ram) {
        est.recommended_nodes = 1;
    } else {
        est.recommended_nodes = static_cast<int>(
            std::ceil((double)est.ram_needed_bytes / usable_ram));
    }
    if (limits.max_nodes > 0)
        est.recommended_nodes = std::min(est.recommended_nodes, limits.max_nodes);

    // Terms and chunks
    std::int64_t terms = digits / 14 + 10;
    est.chunk_count = std::max((std::int64_t)1,
        terms / std::max((std::int64_t)1, (std::int64_t)(usable_ram / 10)));

    // Time estimate (~0.001s/term for moderate, ~0.01 for large)
    double sec_per_term = digits > 1000000 ? 0.01 : 0.001;
    est.estimated_walltime_hours = (terms * sec_per_term) / 3600.0 / est.recommended_nodes;

    if (!est.fits_in_ram && !est.fits_on_scratch)
        est.warning = "Neither RAM nor scratch sufficient for single-node. Multi-node required.";
    else if (!est.fits_in_ram)
        est.warning = "Exceeds single-node RAM. Out-of-core chunking or multi-node recommended.";

    return est;
}

bool check_disk_watermark(const std::string& scratch_path, double abort_threshold) {
#if defined(__linux__)
    struct statvfs buf;
    if (statvfs(scratch_path.c_str(), &buf) != 0) return true; // can't check, assume OK
    double free_frac = (double)buf.f_bavail / (double)buf.f_blocks;
    return free_frac > abort_threshold;
#else
    (void)scratch_path; (void)abort_threshold;
    return true;
#endif
}

// ============================================================
// DRY-RUN / PLAN MODE
// ============================================================

static std::string bytes_h(std::uint64_t b) {
    if (b >= (1ULL<<30)) return std::to_string(b/(1ULL<<30)) + " GB";
    if (b >= (1ULL<<20)) return std::to_string(b/(1ULL<<20)) + " MB";
    return std::to_string(b) + " B";
}

void print_run_plan(const RunPlan& plan) {
    printf("\n=== pi-cluster Run Plan (DRY RUN) ===\n");
    printf("  Digits:       %lld\n", (long long)plan.digits);
    printf("  Backend:      %s\n", plan.backend.c_str());
    printf("  Nodes:        %d\n", plan.nodes);
    printf("  GPUs:         %d%s\n", plan.gpus, plan.gpu_available ? "" : " (none detected)");
    printf("  Total terms:  %lld\n", (long long)plan.total_terms);
    printf("  Chunks:       %lld (x %zu terms each)\n",
           (long long)plan.chunk_count, plan.chunk_terms);
    printf("  RAM/node:     %s\n", bytes_h(plan.ram_per_node).c_str());
    printf("  Scratch/node: %s\n", bytes_h(plan.scratch_per_node).c_str());
    printf("  Output size:  %s\n", bytes_h(plan.output_size).c_str());
    printf("  Est. time:    %.1f hours\n", plan.estimated_hours);
    printf("  Scratch:      %s\n", plan.scratch_path.c_str());
    printf("  Output:       %s\n", plan.output_path.c_str());
    printf("  Checkpoint:   %s\n", plan.checkpoint_path.c_str());

    for (auto& w : plan.warnings)
        printf("  [WARN] %s\n", w.c_str());
    for (auto& b : plan.blockers)
        printf("  [BLOCK] %s\n", b.c_str());

    if (plan.blockers.empty())
        printf("\n  Status: READY — run with --confirm to start\n");
    else
        printf("\n  Status: BLOCKED — resolve issues before running\n");
    printf("=====================================\n\n");
}

std::string plan_to_json(const RunPlan& plan) {
    std::ostringstream j;
    j << "{\"digits\":" << plan.digits
      << ",\"backend\":\"" << plan.backend << "\""
      << ",\"nodes\":" << plan.nodes
      << ",\"chunks\":" << plan.chunk_count
      << ",\"chunk_terms\":" << plan.chunk_terms
      << ",\"ram_per_node\":" << plan.ram_per_node
      << ",\"scratch_per_node\":" << plan.scratch_per_node
      << ",\"output_size\":" << plan.output_size
      << ",\"estimated_hours\":" << std::fixed << std::setprecision(2) << plan.estimated_hours
      << ",\"blockers\":" << plan.blockers.size()
      << ",\"warnings\":" << plan.warnings.size()
      << "}";
    return j.str();
}

// ============================================================
// SIGNAL HANDLING
// ============================================================

static std::atomic<bool> g_shutdown{false};
static std::function<void()> g_shutdown_cb;

static void signal_handler(int sig) {
    g_shutdown.store(true);
    if (g_shutdown_cb) g_shutdown_cb();
    // Don't exit immediately — let the main loop detect and clean up
}

void install_signal_handlers() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
#if defined(__linux__)
    std::signal(SIGUSR1, signal_handler); // Slurm preemption signal
#endif
}

bool shutdown_requested() { return g_shutdown.load(); }

void set_shutdown_callback(std::function<void()> cb) { g_shutdown_cb = cb; }

// ============================================================
// LOGGING
// ============================================================

static LogLevel g_min_level = LogLevel::INFO;
static int g_rate_limit = 0; // 0 = unlimited
static std::mutex g_log_mu;
static int g_log_count_this_sec = 0;
static std::chrono::steady_clock::time_point g_log_sec_start;

void set_log_level(LogLevel min_level) { g_min_level = min_level; }
void set_log_rate_limit(int max_per_second) { g_rate_limit = max_per_second; }

static const char* level_str(LogLevel l) {
    switch (l) {
        case LogLevel::DEBUG: return "DBG";
        case LogLevel::INFO:  return "INF";
        case LogLevel::WARN:  return "WRN";
        case LogLevel::ERROR: return "ERR";
        case LogLevel::FATAL: return "FTL";
    }
    return "???";
}

void log_msg(LogLevel level, const char* component, const char* fmt, ...) {
    if (level < g_min_level) return;

    std::lock_guard<std::mutex> lk(g_log_mu);

    // Rate limiting
    if (g_rate_limit > 0) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - g_log_sec_start);
        if (elapsed.count() >= 1) {
            g_log_count_this_sec = 0;
            g_log_sec_start = now;
        }
        if (g_log_count_this_sec >= g_rate_limit) return; // suppress
        g_log_count_this_sec++;
    }

    fprintf(stderr, "[%s][%s] ", level_str(level), component);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

}} // namespace
