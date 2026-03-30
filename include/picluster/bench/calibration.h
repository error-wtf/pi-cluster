#pragma once
#include <string>
#include <cstdint>

namespace picluster { namespace bench {

struct NodeProfile {
    double cpu_mflops = 0;
    double mem_bandwidth_gbps = 0;
    double scratch_write_mbps = 0;
    double scratch_read_mbps = 0;
    double mpi_latency_us = -1;     // -1 = not measured
    double mpi_bandwidth_gbps = -1;
    double gpu_h2d_gbps = -1;
    double gpu_kernel_gflops = -1;
    double bytes_per_digit_measured = 0;  // calibrated from actual run
    double seconds_per_term_measured = 0; // calibrated from actual run
    std::string hostname;
    std::string timestamp;
};

// Run calibration benchmarks and save profile
NodeProfile calibrate(const std::string& scratch_path);

// Save profile to JSON
bool save_profile(const NodeProfile& p, const std::string& path);

// Load profile from JSON
NodeProfile load_profile(const std::string& path);

// Estimate digits feasible on this node given profile
std::size_t estimate_from_profile(const NodeProfile& p, std::size_t free_ram, std::uint64_t free_scratch);

// Estimate seconds for N digits on this node
double estimate_time_from_profile(const NodeProfile& p, std::int64_t digits, int num_ranks = 1);

}} // namespace
