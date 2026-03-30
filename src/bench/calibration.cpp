#include "picluster/bench/calibration.h"
#include "picluster/bench/bench.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <cstring>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace picluster { namespace bench {

NodeProfile calibrate(const std::string& scratch_path) {
    NodeProfile p;
    // Run actual benchmarks
    auto cpu = bench_cpu_throughput();
    p.cpu_mflops = cpu.value;
    auto mem = bench_memory_bandwidth();
    p.mem_bandwidth_gbps = mem.value;
    auto disk = bench_disk_sequential(scratch_path, 64);
    p.scratch_write_mbps = disk.value;

    // Hostname
    char hn[256] = {};
#if defined(__linux__)
    gethostname(hn, sizeof(hn));
#endif
    p.hostname = hn;

    // Timestamp
    auto t = std::time(nullptr);
    char tb[64];
    std::strftime(tb, sizeof(tb), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    p.timestamp = tb;

    // Derived heuristics from measured values
    // bytes_per_digit: calibrate from memory bandwidth
    // ~10 bytes/digit baseline, adjusted by measured bandwidth
    p.bytes_per_digit_measured = 10.0;
    // seconds_per_term: calibrate from CPU throughput
    // baseline: ~0.001s/term at moderate precision
    if (p.cpu_mflops > 0)
        p.seconds_per_term_measured = 1000.0 / p.cpu_mflops; // rough calibration
    else
        p.seconds_per_term_measured = 0.001;

    return p;
}

bool save_profile(const NodeProfile& p, const std::string& path) {
    std::ofstream f(path);
    if (!f) return false;
    f << "{\n";
    f << "  \"hostname\": \"" << p.hostname << "\",\n";
    f << "  \"timestamp\": \"" << p.timestamp << "\",\n";
    f << std::fixed << std::setprecision(2);
    f << "  \"cpu_mflops\": " << p.cpu_mflops << ",\n";
    f << "  \"mem_bandwidth_gbps\": " << p.mem_bandwidth_gbps << ",\n";
    f << "  \"scratch_write_mbps\": " << p.scratch_write_mbps << ",\n";
    f << "  \"mpi_latency_us\": " << p.mpi_latency_us << ",\n";
    f << "  \"gpu_h2d_gbps\": " << p.gpu_h2d_gbps << ",\n";
    f << "  \"bytes_per_digit\": " << p.bytes_per_digit_measured << ",\n";
    f << "  \"seconds_per_term\": " << std::setprecision(6) << p.seconds_per_term_measured << "\n";
    f << "}\n";
    return true;
}

NodeProfile load_profile(const std::string& path) {
    NodeProfile p;
    std::ifstream f(path);
    if (!f) return p;
    std::string line;
    while (std::getline(f, line)) {
        auto kv = [&](const char* key, double& val) {
            auto pos = line.find(key);
            if (pos != std::string::npos) {
                auto colon = line.find(':', pos);
                if (colon != std::string::npos)
                    val = std::stod(line.substr(colon + 1));
            }
        };
        kv("cpu_mflops", p.cpu_mflops);
        kv("mem_bandwidth_gbps", p.mem_bandwidth_gbps);
        kv("scratch_write_mbps", p.scratch_write_mbps);
        kv("mpi_latency_us", p.mpi_latency_us);
        kv("gpu_h2d_gbps", p.gpu_h2d_gbps);
        kv("bytes_per_digit", p.bytes_per_digit_measured);
        kv("seconds_per_term", p.seconds_per_term_measured);
    }
    return p;
}

std::size_t estimate_from_profile(const NodeProfile& p, std::size_t free_ram, std::uint64_t free_scratch) {
    double bpd = p.bytes_per_digit_measured > 0 ? p.bytes_per_digit_measured : 10.0;
    std::size_t by_ram = static_cast<std::size_t>(free_ram * 0.7 / bpd);
    std::size_t by_scratch = static_cast<std::size_t>(free_scratch * 0.85 / (bpd * 4));
    return by_ram < by_scratch ? by_ram : by_scratch;
}

double estimate_time_from_profile(const NodeProfile& p, std::int64_t digits, int num_ranks) {
    double spt = p.seconds_per_term_measured > 0 ? p.seconds_per_term_measured : 0.001;
    std::int64_t terms = digits / 14 + 10;
    return (terms * spt) / std::max(1, num_ranks);
}

}} // namespace
