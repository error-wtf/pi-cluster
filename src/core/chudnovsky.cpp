// chudnovsky.cpp — Pi computation via Chudnovsky algorithm
// V3: Real partial sums per chunk range [start_term, end_term)
#include "picluster/core/chudnovsky.h"
#include <cmath>
#include <sstream>
#include <vector>
#include <iomanip>
#include <cstring>

#ifdef PICLUSTER_HAVE_GMP
#include <gmp.h>
#include <gmpxx.h>
#endif

namespace picluster { namespace core {

std::int64_t estimate_terms(std::int64_t digits) {
    return digits / 14 + 10;
}

std::size_t estimate_ram_bytes(std::int64_t digits) {
    return static_cast<std::size_t>(digits) * 10;
}

#ifdef PICLUSTER_HAVE_GMP

// Compute partial Chudnovsky sum for terms [start_term, end_term)
// Returns the partial sum S_partial that contributes to the total S.
// Final pi = C / S_total where S_total = sum of all partial sums.
void compute_partial_sum(std::int64_t start_term, std::int64_t end_term,
                         std::int64_t digits, mpf_class& out_partial_S,
                         ProgressCallback cb) {
    mp_bitcnt_t prec = static_cast<mp_bitcnt_t>((digits + 50) * 3.4);
    mpf_set_default_prec(prec);

    // We need to compute each term independently:
    // T_k = (-1)^k * (6k)! * (13591409 + 545140134*k) / ((3k)! * (k!)^3 * 640320^(3k+3/2))
    // But for efficiency we use the recurrence relation starting from term 0.
    // For chunk computation, we must compute the recurrence from term 0 up to start_term
    // (skipping the sum), then accumulate from start_term to end_term.

    mpz_class M(1), K(6), X(1);
    mpf_class L(13591409);
    mpf_class S(0);

    // Fast-forward recurrence to start_term (compute M, K, X, L without accumulating)
    for (std::int64_t k = 1; k < start_term; ++k) {
        mpz_class Kcubed;
        mpz_pow_ui(Kcubed.get_mpz_t(), K.get_mpz_t(), 3);
        mpz_class temp = Kcubed - 16 * K;
        M = (M * temp) / (mpz_class(k) * mpz_class(k) * mpz_class(k));
        X *= mpz_class("-262537412640768000");
        L += 545140134;
        K += 12;
    }

    // If start_term == 0, include the k=0 term
    if (start_term == 0) {
        S = mpf_class(13591409); // k=0 term: M=1, L=13591409, X=1
        // Now advance to k=1
        M = 1; K = 6; X = 1; L = 13591409;
        // Compute k=1 onward
        for (std::int64_t k = 1; k < end_term; ++k) {
            mpz_class Kcubed;
            mpz_pow_ui(Kcubed.get_mpz_t(), K.get_mpz_t(), 3);
            mpz_class temp = Kcubed - 16 * K;
            M = (M * temp) / (mpz_class(k) * mpz_class(k) * mpz_class(k));
            X *= mpz_class("-262537412640768000");
            L += 545140134;
            S += (mpf_class(M) * L) / mpf_class(X);
            K += 12;

            if (cb && k % 500 == 0) {
                double frac = static_cast<double>(k - start_term) / (end_term - start_term);
                cb(frac, "Chudnovsky partial sum", k * 14);
            }
        }
    } else {
        // Accumulate from start_term to end_term
        for (std::int64_t k = start_term; k < end_term; ++k) {
            if (k > 0) {
                mpz_class Kcubed;
                mpz_pow_ui(Kcubed.get_mpz_t(), K.get_mpz_t(), 3);
                mpz_class temp = Kcubed - 16 * K;
                M = (M * temp) / (mpz_class(k) * mpz_class(k) * mpz_class(k));
                X *= mpz_class("-262537412640768000");
                L += 545140134;
                K += 12;
            }
            S += (mpf_class(M) * L) / mpf_class(X);

            if (cb && (k - start_term) % 500 == 0) {
                double frac = static_cast<double>(k - start_term) / (end_term - start_term);
                cb(frac, "Chudnovsky partial sum", k * 14);
            }
        }
    }

    out_partial_S = S;
    if (cb) cb(1.0, "Chunk done", end_term * 14);
}

// Full pi computation: all terms in one go
std::string compute_pi_cpu(std::int64_t digits, int chunk_size, ProgressCallback cb) {
    if (digits <= 0) return "";
    std::int64_t terms = estimate_terms(digits);

    mp_bitcnt_t prec = static_cast<mp_bitcnt_t>((digits + 50) * 3.4);
    mpf_set_default_prec(prec);

    mpf_class partial_S;
    compute_partial_sum(0, terms, digits, partial_S, cb);

    mpf_class C_val(426880);
    C_val *= sqrt(mpf_class(10005));
    mpf_class pi = C_val / partial_S;

    std::size_t bufsize = static_cast<std::size_t>(digits) + 20;
    std::vector<char> buf(bufsize);
    gmp_snprintf(buf.data(), bufsize, "%.*Ff", static_cast<int>(digits), pi.get_mpf_t());
    if (cb) cb(1.0, "Done", digits);
    return std::string(buf.data());
}

// Finalize: given total sum S, compute pi = C / S
std::string finalize_pi(const mpf_class& total_S, std::int64_t digits) {
    mp_bitcnt_t prec = static_cast<mp_bitcnt_t>((digits + 50) * 3.4);
    mpf_set_default_prec(prec);
    mpf_class C_val(426880);
    C_val *= sqrt(mpf_class(10005));
    mpf_class pi = C_val / total_S;
    std::size_t bufsize = static_cast<std::size_t>(digits) + 20;
    std::vector<char> buf(bufsize);
    gmp_snprintf(buf.data(), bufsize, "%.*Ff", static_cast<int>(digits), pi.get_mpf_t());
    return std::string(buf.data());
}

// Serialize mpf_class to byte vector (for MPI transfer)
std::vector<char> serialize_mpf(const mpf_class& val) {
    // Export as string — simple, portable, correct
    mp_exp_t exponent;
    char* raw = mpf_get_str(nullptr, &exponent, 10, 0, val.get_mpf_t());
    std::string s = std::to_string(exponent) + "|" + std::string(raw);
    free(raw);
    return std::vector<char>(s.begin(), s.end());
}

// Deserialize byte vector to mpf_class
mpf_class deserialize_mpf(const std::vector<char>& data, std::int64_t digits) {
    mp_bitcnt_t prec = static_cast<mp_bitcnt_t>((digits + 50) * 3.4);
    mpf_set_default_prec(prec);
    std::string s(data.begin(), data.end());
    auto sep = s.find('|');
    if (sep == std::string::npos) return mpf_class(0);
    // Reconstruct from string representation
    std::string mantissa = s.substr(sep + 1);
    long exp_val = std::stol(s.substr(0, sep));
    // Build decimal string
    std::string dec;
    if (mantissa.empty() || mantissa == "0") return mpf_class(0);
    if (mantissa[0] == '-') {
        dec = "-0." + mantissa.substr(1);
    } else {
        dec = "0." + mantissa;
    }
    dec += "e" + std::to_string(exp_val);
    return mpf_class(dec);
}

#else
// No-GMP fallback
std::string compute_pi_cpu(std::int64_t digits, int, ProgressCallback cb) {
    if (digits <= 0) return "";
    if (digits > 15) digits = 15;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(static_cast<int>(digits)) << M_PI;
    if (cb) cb(1.0, "Done (no GMP)", digits);
    return oss.str();
}
#endif

std::string compute_pi_gpu(std::int64_t digits, int tpb, ProgressCallback cb) {
#ifndef PICLUSTER_HAVE_CUDA
    if (cb) cb(0.0, "No CUDA, falling back to CPU", 0);
    return compute_pi_cpu(digits, 1000, cb);
#else
    extern std::string gpu_compute_pi(std::int64_t, int, ProgressCallback);
    return gpu_compute_pi(digits, tpb, cb);
#endif
}

double compute_pi_bbp(int terms) {
    if (terms <= 0) return 0.0;
    long double sum = 0.0L;
    for (int k = 0; k < terms; ++k) {
        long double kld = static_cast<long double>(k);
        long double mult = 1.0L / powl(16.0L, kld);
        long double term = 4.0L / (8.0L*kld+1.0L) - 2.0L / (8.0L*kld+4.0L)
                         - 1.0L / (8.0L*kld+5.0L) - 1.0L / (8.0L*kld+6.0L);
        sum += mult * term;
    }
    return static_cast<double>(sum);
}

}} // namespace
