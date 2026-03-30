// progress.cpp — Phase-based progress tracking with terminal rendering
#include "picluster/progress/progress.h"
#include <cstdio>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace picluster { namespace progress {

const char* phase_name(Phase p) {
    switch (p) {
        case Phase::INIT:          return "init";
        case Phase::DETECT:        return "detect";
        case Phase::BENCHMARK:     return "benchmark";
        case Phase::COMPUTE_LOCAL: return "compute";
        case Phase::LOCAL_MERGE:   return "local-merge";
        case Phase::GLOBAL_MERGE:  return "global-merge";
        case Phase::CHECKPOINT:    return "checkpoint";
        case Phase::FINALIZE:      return "finalize";
        case Phase::DONE:          return "done";
    }
    return "unknown";
}

ProgressTracker::ProgressTracker()
    : start_time_(std::chrono::steady_clock::now())
    , phase_start_(start_time_) {}

void ProgressTracker::set_target(std::int64_t digits) {
    std::lock_guard<std::mutex> lk(mu_);
    state_.digits_target = digits;
}

void ProgressTracker::set_phase(Phase p, const std::string& msg) {
    std::lock_guard<std::mutex> lk(mu_);
    state_.phase = p;
    state_.message = msg;
    phase_start_ = std::chrono::steady_clock::now();
}

void ProgressTracker::update(double fraction, std::int64_t digits_done, const std::string& msg) {
    std::lock_guard<std::mutex> lk(mu_);
    state_.fraction = fraction;
    if (digits_done > 0) state_.digits_done = digits_done;
    if (!msg.empty()) state_.message = msg;

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - phase_start_).count();
    if (elapsed > 0.5 && fraction > 0.01) {
        state_.throughput = state_.digits_done / elapsed;
        double remaining_frac = 1.0 - fraction;
        state_.eta_seconds = (remaining_frac / fraction) * elapsed;
    }
}

void ProgressTracker::set_mpi(int rank, int size) {
    std::lock_guard<std::mutex> lk(mu_);
    state_.mpi_rank = rank;
    state_.mpi_size = size;
}

void ProgressTracker::set_checkpoint_age(int seconds) {
    std::lock_guard<std::mutex> lk(mu_);
    state_.checkpoint_age_sec = seconds;
}

void ProgressTracker::increment_errors() { std::lock_guard<std::mutex> lk(mu_); state_.errors++; }
void ProgressTracker::increment_warnings() { std::lock_guard<std::mutex> lk(mu_); state_.warnings++; }

ProgressState ProgressTracker::snapshot() const {
    std::lock_guard<std::mutex> lk(mu_);
    return state_;
}

static std::string format_time(double sec) {
    if (sec < 0) return "??:??";
    int h = (int)(sec / 3600);
    int m = (int)(std::fmod(sec, 3600) / 60);
    int s = (int)(std::fmod(sec, 60));
    char buf[32];
    if (h > 0) snprintf(buf, sizeof(buf), "%dh%02dm%02ds", h, m, s);
    else if (m > 0) snprintf(buf, sizeof(buf), "%dm%02ds", m, s);
    else snprintf(buf, sizeof(buf), "%ds", s);
    return buf;
}

static std::string format_digits(std::int64_t d) {
    if (d >= 1000000000LL) return std::to_string(d / 1000000000) + "B";
    if (d >= 1000000LL) return std::to_string(d / 1000000) + "M";
    if (d >= 1000LL) return std::to_string(d / 1000) + "K";
    return std::to_string(d);
}

void ProgressTracker::render_terminal() const {
    ProgressState s = snapshot();
    const int bar_w = 30;
    int filled = (int)(bar_w * s.fraction);

    // Build progress bar
    char bar[64] = {};
    for (int i = 0; i < bar_w; i++) {
        if (i < filled) bar[i] = '#';
        else if (i == filled) bar[i] = '>';
        else bar[i] = '.';
    }

    auto now = std::chrono::steady_clock::now();
    double wall = std::chrono::duration<double>(now - start_time_).count();

    fprintf(stderr,
        "\r[%s] %5.1f%%  %-10s  %s/%s digits  %.0f d/s  ETA %s  wall %s",
        bar,
        s.fraction * 100.0,
        phase_name(s.phase),
        format_digits(s.digits_done).c_str(),
        format_digits(s.digits_target).c_str(),
        s.throughput,
        format_time(s.eta_seconds).c_str(),
        format_time(wall).c_str()
    );

    // Extra info on MPI
    if (s.mpi_size > 1)
        fprintf(stderr, "  rank %d/%d", s.mpi_rank, s.mpi_size);
    if (s.errors > 0)
        fprintf(stderr, "  E:%d", s.errors);
    if (s.warnings > 0)
        fprintf(stderr, "  W:%d", s.warnings);

    fflush(stderr);
}

std::string ProgressTracker::to_json_line() const {
    ProgressState s = snapshot();
    std::ostringstream j;
    j << "{\"phase\":\"" << phase_name(s.phase) << "\""
      << ",\"fraction\":" << std::fixed << std::setprecision(4) << s.fraction
      << ",\"digits_done\":" << s.digits_done
      << ",\"digits_target\":" << s.digits_target
      << ",\"throughput\":" << std::setprecision(1) << s.throughput
      << ",\"eta_sec\":" << std::setprecision(1) << s.eta_seconds
      << ",\"mpi_rank\":" << s.mpi_rank
      << ",\"mpi_size\":" << s.mpi_size
      << ",\"errors\":" << s.errors
      << ",\"warnings\":" << s.warnings
      << ",\"message\":\"" << s.message << "\""
      << "}";
    return j.str();
}

}} // namespace
