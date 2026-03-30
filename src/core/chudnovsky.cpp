// chudnovsky.cpp — Pi computation via Chudnovsky algorithm
// Ported from CALCULATION_OF_NUMBER_PI with clean architecture
#include "picluster/core/chudnovsky.h"
#include <cmath>
#include <sstream>
#include <vector>
#include <iomanip>

#ifdef PICLUSTER_HAVE_GMP
#include <gmp.h>
#include <gmpxx.h>
#endif

namespace picluster { namespace core {

std::int64_t estimate_terms(std::int64_t digits) {
    // Chudnovsky converges at ~14.18 digits/term
    return digits / 14 + 10;
}

std::size_t estimate_ram_bytes(std::int64_t digits) {
    // ~10 bytes/digit for GMP overhead (conservative)
    return static_cast<std::size_t>(digits) * 10;
}

#ifdef PICLUSTER_HAVE_GMP
std::string compute_pi_cpu(std::int64_t digits, int chunk_size, ProgressCallback cb) {
    if (digits <= 0) return "";
    std::int64_t terms = estimate_terms(digits);

    // Set GMP precision: ~3.32 bits per decimal digit + guard
    mp_bitcnt_t prec = static_cast<mp_bitcnt_t>((digits + 20) * 3.4);
    mpf_set_default_prec(prec);

    mpf_class C_val(426880);
    C_val *= sqrt(mpf_class(10005));

    mpz_class M(1), K(6), X(1);
    mpf_class L(13591409), S(13591409);

    for (std::int64_t k = 1; k < terms; ++k) {
        // M *= (K^3 - 16K) / k^3
        mpz_class Kcubed;
        mpz_pow_ui(Kcubed.get_mpz_t(), K.get_mpz_t(), 3);
        mpz_class temp = Kcubed - 16 * K;
        M = (M * temp) / (mpz_class(k) * mpz_class(k) * mpz_class(k));
        X *= mpz_class("-262537412640768000");
        L += 545140134;
        S += (mpf_class(M) * L) / mpf_class(X);
        K += 12;

        if (cb && k % chunk_size == 0) {
            double frac = static_cast<double>(k) / terms;
            cb(frac, "Chudnovsky CPU", k * 14);
        }
    }

    mpf_class pi = C_val / S;

    // Convert to string
    std::size_t bufsize = static_cast<std::size_t>(digits) + 20;
    std::vector<char> buf(bufsize);
    gmp_snprintf(buf.data(), bufsize, "%.*Ff", static_cast<int>(digits), pi.get_mpf_t());

    if (cb) cb(1.0, "Done", digits);
    return std::string(buf.data());
}
#else
std::string compute_pi_cpu(std::int64_t digits, int chunk_size, ProgressCallback cb) {
    // Fallback: very basic Machin-like formula for small digits
    if (digits <= 0) return "";
    // Use standard library for up to 15 digits
    if (digits > 15) {
        if (cb) cb(1.0, "GMP not available, limited to 15 digits", 15);
        digits = 15;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(static_cast<int>(digits)) << M_PI;
    if (cb) cb(1.0, "Done (no GMP)", digits);
    return oss.str();
}
#endif

std::string compute_pi_gpu(std::int64_t digits, int threads_per_block, ProgressCallback cb) {
    // GPU path is in src/cuda/gpu_chudnovsky.cu
    // If compiled without CUDA, fall back to CPU
#ifndef PICLUSTER_HAVE_CUDA
    if (cb) cb(0.0, "No CUDA — falling back to CPU", 0);
    return compute_pi_cpu(digits, 1000, cb);
#else
    // Forward to CUDA implementation (declared in cuda module)
    extern std::string gpu_compute_pi(std::int64_t, int, ProgressCallback);
    return gpu_compute_pi(digits, threads_per_block, cb);
#endif
}

double compute_pi_bbp(int terms) {
    if (terms <= 0) return 0.0;
    long double sum = 0.0L;
    for (int k = 0; k < terms; ++k) {
        long double kld = static_cast<long double>(k);
        long double mult = 1.0L / powl(16.0L, kld);
        long double term = 4.0L / (8.0L * kld + 1.0L)
                         - 2.0L / (8.0L * kld + 4.0L)
                         - 1.0L / (8.0L * kld + 5.0L)
                         - 1.0L / (8.0L * kld + 6.0L);
        sum += mult * term;
    }
    return static_cast<double>(sum);
}

}} // namespace
