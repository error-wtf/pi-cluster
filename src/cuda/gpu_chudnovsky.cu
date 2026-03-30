// gpu_chudnovsky.cu — CUDA hybrid Chudnovsky implementation
// Ported from CALCULATION_OF_NUMBER_PI/final-linux-cuda-pi-hybrid.cpp
// LIMITED TO ~700 digits due to double-precision factorial overflow.
// For higher precision, falls back to CPU path automatically.

#ifdef PICLUSTER_HAVE_CUDA

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <functional>

#ifdef PICLUSTER_HAVE_GMP
#include <gmp.h>
#include <gmpxx.h>
#endif

namespace picluster { namespace core {

using ProgressCallback = std::function<void(double, const std::string&, std::int64_t)>;

// CUDA constants for Chudnovsky
__constant__ double d_A = 13591409.0;
__constant__ double d_B = 545140134.0;
__constant__ double d_C3_24 = 10939058860032000.0; // 640320^3 / 24

__global__ void chudnovsky_terms_kernel(double* nums, double* dens, int start, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int k = start + idx;
        nums[idx] = tgamma(6.0 * k + 1.0) * (d_A + d_B * k);
        dens[idx] = tgamma(3.0 * k + 1.0) * pow(tgamma((double)k + 1.0), 3.0) * pow(-d_C3_24, (double)k);
    }
}

static inline void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(err) << "\n";
    }
}

// GPU compute: limited to ~700 digits (double precision)
std::string gpu_compute_pi(std::int64_t digits, int threads_per_block, ProgressCallback cb) {
    if (digits > 700) {
        if (cb) cb(0.0, "Digits > 700: GPU double-precision insufficient, falling back to CPU", 0);
        // Forward to CPU path
        extern std::string compute_pi_cpu(std::int64_t, int, ProgressCallback);
        return compute_pi_cpu(digits, 1000, cb);
    }

    int terms = static_cast<int>(digits / 14 + 2);
    std::size_t size = terms * sizeof(double);

    double *d_nums = nullptr, *d_dens = nullptr;
    cuda_check(cudaMalloc(&d_nums, size), "malloc nums");
    cuda_check(cudaMalloc(&d_dens, size), "malloc dens");

    int blocks = (terms + threads_per_block - 1) / threads_per_block;

    if (cb) cb(0.1, "Launching CUDA kernel", 0);
    chudnovsky_terms_kernel<<<blocks, threads_per_block>>>(d_nums, d_dens, 0, terms);
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaDeviceSynchronize(), "sync");
    if (cb) cb(0.5, "Kernel done, transferring results", 0);

    std::vector<double> h_nums(terms), h_dens(terms);
    cuda_check(cudaMemcpy(h_nums.data(), d_nums, size, cudaMemcpyDeviceToHost), "memcpy nums");
    cuda_check(cudaMemcpy(h_dens.data(), d_dens, size, cudaMemcpyDeviceToHost), "memcpy dens");
    cudaFree(d_nums);
    cudaFree(d_dens);

#ifdef PICLUSTER_HAVE_GMP
    mpf_set_default_prec(static_cast<mp_bitcnt_t>(digits * 4));
    mpf_class S = 0;
    for (int i = 0; i < terms; i++) {
        S += mpf_class(h_nums[i]) / mpf_class(h_dens[i]);
    }
    mpf_class pi = (426880 * sqrt(mpf_class(10005))) / S;

    std::size_t bufsize = static_cast<std::size_t>(digits) + 20;
    std::vector<char> buf(bufsize);
    gmp_snprintf(buf.data(), bufsize, "%.*Ff", static_cast<int>(digits), pi.get_mpf_t());
    if (cb) cb(1.0, "GPU hybrid complete", digits);
    return std::string(buf.data());
#else
    // Without GMP, sum in double
    double S = 0.0;
    for (int i = 0; i < terms; i++) S += h_nums[i] / h_dens[i];
    double pi = 426880.0 * std::sqrt(10005.0) / S;
    char buf[64];
    snprintf(buf, sizeof(buf), "%.15f", pi);
    if (cb) cb(1.0, "GPU compute done (double only)", 15);
    return std::string(buf);
#endif
}

}} // namespace

#endif // PICLUSTER_HAVE_CUDA
