#pragma once
#include <string>
#include <functional>
#include <cstdint>
#include <vector>

namespace picluster {
namespace core {

// Progress callback: (progress 0.0-1.0, phase name, digits_done)
using ProgressCallback = std::function<void(double, const std::string&, std::int64_t)>;

// CPU Chudnovsky — arbitrary precision via GMP (full computation)
std::string compute_pi_cpu(std::int64_t digits, int chunk_size = 1000, ProgressCallback cb = nullptr);

// GPU Chudnovsky — double precision, max ~700 digits, falls back to CPU
std::string compute_pi_gpu(std::int64_t digits, int threads_per_block = 256, ProgressCallback cb = nullptr);

// BBP spot-check: returns double-precision pi (for validation)
double compute_pi_bbp(int terms);

// Estimate iterations needed for N digits
std::int64_t estimate_terms(std::int64_t digits);

// Estimate RAM bytes needed for N digits
std::size_t estimate_ram_bytes(std::int64_t digits);

#ifdef PICLUSTER_HAVE_GMP
#include <gmpxx.h>

// Compute PARTIAL Chudnovsky sum for terms [start_term, end_term)
// This is the core of chunk-based distributed computation.
void compute_partial_sum(std::int64_t start_term, std::int64_t end_term,
                         std::int64_t digits, mpf_class& out_partial_S,
                         ProgressCallback cb = nullptr);

// Finalize: given total sum S (from all chunks), compute pi = C / S
std::string finalize_pi(const mpf_class& total_S, std::int64_t digits);

// Serialize/deserialize mpf_class for MPI transfer
std::vector<char> serialize_mpf(const mpf_class& val);
mpf_class deserialize_mpf(const std::vector<char>& data, std::int64_t digits);

#endif

} // namespace core
} // namespace picluster
