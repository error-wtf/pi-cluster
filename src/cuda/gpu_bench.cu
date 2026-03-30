// gpu_bench.cu — Real GPU microbenchmarks (H2D bandwidth + FMA kernel throughput)
// Only compiled when BUILD_CUDA=ON

#ifdef PICLUSTER_HAVE_CUDA
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cstring>

namespace picluster { namespace bench {

// --- H2D / D2H bandwidth ---
extern "C" double gpu_bench_h2d_bandwidth_gbps() {
    const size_t SIZE = 64 * 1024 * 1024; // 64 MB
    void *d_buf = nullptr;
    std::vector<char> h_buf(SIZE, 0x42);

    if (cudaMalloc(&d_buf, SIZE) != cudaSuccess) return -1.0;
    cudaDeviceSynchronize();

    // Warm up
    cudaMemcpy(d_buf, h_buf.data(), SIZE, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Measure H2D
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; i++) {
        cudaMemcpy(d_buf, h_buf.data(), SIZE, cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double h2d_sec = std::chrono::duration<double>(t1 - t0).count();
    double h2d_gbps = (10.0 * SIZE) / h2d_sec / 1e9;

    cudaFree(d_buf);
    return h2d_gbps;
}

// --- FMA kernel throughput ---
__global__ void fma_kernel(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a = 1.0f + 0.0001f * idx;
        float b = a;
        // 1000 FMAs per thread
        for (int i = 0; i < 1000; i++) {
            a = a * 1.00001f + 0.00001f;
            b = b * 0.99999f + 0.00001f;
        }
        out[idx] = a + b;
    }
}

extern "C" double gpu_bench_fma_gflops() {
    const int N = 1024 * 1024; // 1M threads
    const int FMA_PER_THREAD = 2000; // 2 chains x 1000
    float *d_out = nullptr;
    if (cudaMalloc(&d_out, N * sizeof(float)) != cudaSuccess) return -1.0;
    cudaDeviceSynchronize();

    // Warm up
    fma_kernel<<<N/256, 256>>>(d_out, N);
    cudaDeviceSynchronize();

    auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < 5; rep++) {
        fma_kernel<<<N/256, 256>>>(d_out, N);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    double total_flops = 5.0 * N * FMA_PER_THREAD;
    double gflops = total_flops / sec / 1e9;

    cudaFree(d_out);
    return gflops;
}

}} // namespace
#endif // PICLUSTER_HAVE_CUDA
