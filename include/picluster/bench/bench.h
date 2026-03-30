#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace picluster {
namespace bench {

struct BenchResult {
    std::string name;
    std::string unit;
    double value = 0.0;
    double duration_sec = 0.0;
    std::string notes;
};

struct BenchReport {
    std::vector<BenchResult> results;
    std::string hostname;
    std::string timestamp;
    std::string to_json() const;
    std::string to_text() const;
};

// Run all benchmarks
BenchReport run_all();

// Individual benchmarks
BenchResult bench_cpu_throughput();
BenchResult bench_memory_bandwidth();
BenchResult bench_disk_sequential(const std::string& path, std::size_t block_mb = 64);
BenchResult bench_disk_random(const std::string& path);

// Optional (require CUDA / MPI at compile time)
BenchResult bench_gpu_transfer();
BenchResult bench_gpu_kernel();
BenchResult bench_mpi_pingpong();
BenchResult bench_mpi_allreduce();

} // namespace bench
} // namespace picluster
