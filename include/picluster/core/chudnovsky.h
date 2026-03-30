#pragma once
#include <string>
#include <functional>
#include <cstdint>

namespace picluster {
namespace core {

// Progress callback: (progress 0.0-1.0, phase name, digits_done)
using ProgressCallback = std::function<void(double, const std::string&, std::int64_t)>;

// CPU Chudnovsky — arbitrary precision via GMP
// Returns pi as decimal string with `digits` places after '3.'
std::string compute_pi_cpu(std::int64_t digits,
                           int chunk_size = 1000,
                           ProgressCallback cb = nullptr);

// GPU Chudnovsky — double precision, max ~700 digits
// Falls back to CPU if CUDA unavailable or digits > 700
std::string compute_pi_gpu(std::int64_t digits,
                           int threads_per_block = 256,
                           ProgressCallback cb = nullptr);

// BBP spot-check: returns double-precision pi (for validation)
double compute_pi_bbp(int terms);

// Estimate iterations needed for N digits (Chudnovsky: ~14.18 digits/term)
std::int64_t estimate_terms(std::int64_t digits);

// Estimate RAM bytes needed for N digits
std::size_t estimate_ram_bytes(std::int64_t digits);

} // namespace core
} // namespace picluster
