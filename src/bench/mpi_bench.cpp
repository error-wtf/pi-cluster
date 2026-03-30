// mpi_bench.cpp — Real MPI microbenchmarks (ping-pong latency + allreduce bandwidth)
// Only active when compiled with BUILD_MPI=ON

#include "picluster/bench/bench.h"
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdio>

#ifdef PICLUSTER_HAVE_MPI
#include <mpi.h>
#endif

namespace picluster { namespace bench {

BenchResult bench_mpi_pingpong() {
    BenchResult r;
    r.name = "mpi_pingpong";
    r.unit = "us";
#ifdef PICLUSTER_HAVE_MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        r.notes = "Need >= 2 ranks for ping-pong";
        r.value = -1;
        return r;
    }
    const int MSG_SIZE = 8; // 8 bytes
    const int ITERS = 10000;
    char buf[8] = {};
    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ITERS; i++) {
        if (rank == 0) {
            MPI_Send(buf, MSG_SIZE, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(buf, MSG_SIZE, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == 1) {
            MPI_Recv(buf, MSG_SIZE, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buf, MSG_SIZE, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    r.value = (sec / ITERS / 2.0) * 1e6; // one-way latency in microseconds
    r.duration_sec = sec;
    r.notes = "8-byte ping-pong, " + std::to_string(ITERS) + " iterations, ranks 0<->1";
#else
    r.notes = "MPI not compiled";
    r.value = -1;
#endif
    return r;
}

BenchResult bench_mpi_allreduce() {
    BenchResult r;
    r.name = "mpi_allreduce";
    r.unit = "GB/s";
#ifdef PICLUSTER_HAVE_MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        r.notes = "Need >= 2 ranks for allreduce";
        r.value = -1;
        return r;
    }
    const size_t COUNT = 1024 * 1024; // 1M doubles = 8 MB
    const int ITERS = 50;
    std::vector<double> sendbuf(COUNT, 1.0);
    std::vector<double> recvbuf(COUNT, 0.0);
    MPI_Barrier(MPI_COMM_WORLD);
    // Warm up
    MPI_Allreduce(sendbuf.data(), recvbuf.data(), COUNT, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ITERS; i++) {
        MPI_Allreduce(sendbuf.data(), recvbuf.data(), COUNT, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    double bytes = (double)COUNT * sizeof(double) * ITERS;
    r.value = bytes / sec / 1e9;
    r.duration_sec = sec;
    r.notes = "8 MB allreduce, " + std::to_string(ITERS) + " iters, " + std::to_string(size) + " ranks";
#else
    r.notes = "MPI not compiled";
    r.value = -1;
#endif
    return r;
}

}} // namespace
