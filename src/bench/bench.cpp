// bench.cpp — Microbenchmark suite for pi-cluster
#include "picluster/bench/bench.h"
#include <chrono>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <filesystem>

namespace picluster { namespace bench {

static double now_sec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

BenchResult bench_cpu_throughput() {
    BenchResult r;
    r.name = "cpu_throughput";
    r.unit = "MFLOPS";
    // Simple FMA-like loop to measure raw CPU throughput
    const int N = 50000000;
    volatile double acc = 1.0;
    double t0 = now_sec();
    for (int i = 1; i <= N; i++) {
        acc = acc * 1.000000001 + 0.000000001;
    }
    double dt = now_sec() - t0;
    r.value = (double)N / dt / 1e6;
    r.duration_sec = dt;
    r.notes = "FMA-like scalar loop, single-thread";
    (void)acc; // prevent optimization
    return r;
}

BenchResult bench_memory_bandwidth() {
    BenchResult r;
    r.name = "memory_bandwidth";
    r.unit = "GB/s";
    const std::size_t SIZE = 256 * 1024 * 1024; // 256 MB
    std::vector<char> buf(SIZE);
    // Write pass (warm up)
    std::memset(buf.data(), 0xAA, SIZE);
    // Read pass (measure)
    volatile char sink = 0;
    double t0 = now_sec();
    for (std::size_t i = 0; i < SIZE; i += 64) {
        sink ^= buf[i];
    }
    double dt = now_sec() - t0;
    r.value = (double)SIZE / dt / 1e9;
    r.duration_sec = dt;
    r.notes = "Sequential read, 256 MB, stride=64";
    (void)sink;
    return r;
}

BenchResult bench_disk_sequential(const std::string& path, std::size_t block_mb) {
    BenchResult r;
    r.name = "disk_seq_write";
    r.unit = "MB/s";
    std::size_t block_bytes = block_mb * 1024 * 1024;
    std::vector<char> data(block_bytes, 'X');
    std::string fpath = path + "/pi_cluster_bench_tmp.bin";
    // Write
    double t0 = now_sec();
    {
        std::ofstream f(fpath, std::ios::binary);
        f.write(data.data(), block_bytes);
        f.flush();
    }
    double dt_w = now_sec() - t0;
    // Read
    double t1 = now_sec();
    {
        std::ifstream f(fpath, std::ios::binary);
        f.read(data.data(), block_bytes);
    }
    double dt_r = now_sec() - t1;
    std::filesystem::remove(fpath);
    r.value = (double)block_mb / dt_w;
    r.duration_sec = dt_w + dt_r;
    r.notes = "Write " + std::to_string(block_mb) + " MB: " +
              std::to_string((int)(block_mb / dt_w)) + " MB/s write, " +
              std::to_string((int)(block_mb / dt_r)) + " MB/s read";
    return r;
}

BenchResult bench_disk_random(const std::string& path) {
    BenchResult r;
    r.name = "disk_random_io";
    r.unit = "IOPS";
    // Create a test file, then do random 4K reads
    std::string fpath = path + "/pi_cluster_bench_rnd.bin";
    const size_t FILE_SIZE = 64 * 1024 * 1024; // 64 MB
    const int NUM_READS = 5000;
    const size_t BLOCK = 4096;
    // Write test file
    {
        std::vector<char> data(FILE_SIZE, 'R');
        std::ofstream f(fpath, std::ios::binary);
        f.write(data.data(), FILE_SIZE);
    }
    // Random 4K reads
    std::ifstream f(fpath, std::ios::binary);
    std::vector<char> buf(BLOCK);
    srand(42);
    double t0 = now_sec();
    for (int i = 0; i < NUM_READS; i++) {
        size_t offset = (rand() % (FILE_SIZE / BLOCK)) * BLOCK;
        f.seekg(offset);
        f.read(buf.data(), BLOCK);
    }
    double dt = now_sec() - t0;
    f.close();
    std::filesystem::remove(fpath);
    r.value = NUM_READS / dt;
    r.duration_sec = dt;
    r.notes = "4K random reads from 64 MB file, " + std::to_string(NUM_READS) + " reads";
    return r;
}

BenchResult bench_gpu_transfer() {
    BenchResult r;
    r.name = "gpu_h2d_transfer";
    r.unit = "GB/s";
#ifdef PICLUSTER_HAVE_CUDA
    extern "C" double gpu_bench_h2d_bandwidth_gbps();
    r.value = gpu_bench_h2d_bandwidth_gbps();
    r.notes = r.value > 0 ? "64 MB H2D, 10 iterations" : "CUDA init failed";
#else
    r.notes = "CUDA not compiled";
    r.value = 0;
#endif
    return r;
}

BenchResult bench_gpu_kernel() {
    BenchResult r;
    r.name = "gpu_kernel";
    r.unit = "GFLOPS";
#ifdef PICLUSTER_HAVE_CUDA
    extern "C" double gpu_bench_fma_gflops();
    r.value = gpu_bench_fma_gflops();
    r.notes = r.value > 0 ? "1M threads, 2000 FMA/thread, 5 reps" : "CUDA init failed";
#else
    r.notes = "CUDA not compiled";
    r.value = 0;
#endif
    return r;
}

// MPI benchmarks: real implementations in mpi_bench.cpp
// bench_mpi_pingpong() and bench_mpi_allreduce() are defined there

BenchReport run_all() {
    BenchReport rep;
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    char tbuf[64];
    std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%S", std::localtime(&tt));
    rep.timestamp = tbuf;

    char hn[256] = {};
#if defined(__linux__)
    gethostname(hn, sizeof(hn));
#endif
    rep.hostname = hn;

    rep.results.push_back(bench_cpu_throughput());
    rep.results.push_back(bench_memory_bandwidth());

    const char* tmp = std::getenv("TMPDIR");
    std::string scratch = tmp ? tmp : "/tmp";
    rep.results.push_back(bench_disk_sequential(scratch, 64));

    // Optional GPU/MPI benchmarks
    rep.results.push_back(bench_gpu_transfer());
    rep.results.push_back(bench_gpu_kernel());
    rep.results.push_back(bench_mpi_pingpong());
    rep.results.push_back(bench_mpi_allreduce());

    return rep;
}

std::string BenchReport::to_json() const {
    std::ostringstream j;
    j << "{\"hostname\":\"" << hostname << "\",\"timestamp\":\"" << timestamp << "\",\"results\":[";
    for (std::size_t i = 0; i < results.size(); i++) {
        if (i > 0) j << ",";
        j << "{\"name\":\"" << results[i].name << "\""
          << ",\"value\":" << std::fixed << std::setprecision(2) << results[i].value
          << ",\"unit\":\"" << results[i].unit << "\""
          << ",\"duration\":" << std::setprecision(4) << results[i].duration_sec
          << ",\"notes\":\"" << results[i].notes << "\"}";
    }
    j << "]}";
    return j.str();
}

std::string BenchReport::to_text() const {
    std::ostringstream t;
    t << "=== pi-cluster Benchmark Report ===\n";
    t << "Host: " << hostname << "  Time: " << timestamp << "\n";
    t << std::string(50, '-') << "\n";
    for (auto& r : results) {
        t << std::left << std::setw(22) << r.name
          << std::right << std::setw(12) << std::fixed << std::setprecision(2) << r.value
          << " " << std::left << std::setw(8) << r.unit;
        if (!r.notes.empty()) t << "  (" << r.notes << ")";
        t << "\n";
    }
    t << std::string(50, '-') << "\n";
    return t.str();
}

}} // namespace
