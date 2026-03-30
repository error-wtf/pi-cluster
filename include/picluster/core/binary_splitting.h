#pragma once
// binary_splitting.h — Binary Splitting architecture for billion+ digit Pi
// STATUS: SKELETON — architectural hooks ready, full implementation planned
//
// Binary splitting computes the Chudnovsky series as a product tree:
//   P(a,b), Q(a,b), T(a,b) where pi = Q(0,N) * C / T(0,N)
// This enables O(n log^2 n) complexity vs O(n^2) for naive summation.
// Combined with NTT-based multiplication, this is the path to trillion digits.

#include <cstdint>
#include <string>

namespace picluster { namespace core {

// Binary splitting state for a range [a, b)
struct BSState {
    // In full implementation: GMP integers P, Q, T
    std::int64_t a = 0;
    std::int64_t b = 0;
    // Placeholder for future GMP fields
};

// Compute binary splitting for range [a, b)
// SKELETON: returns empty state. Full impl needs GMP integer product tree.
BSState binary_split(std::int64_t a, std::int64_t b);

// Merge two binary splitting states (for parallel/distributed computation)
// SKELETON: architectural hook for MPI-distributed binary splitting
BSState merge_bs(const BSState& left, const BSState& right);

// Estimate: is binary splitting worth it for this digit count?
// Returns true if digits > threshold where BS outperforms naive summation
bool should_use_binary_splitting(std::int64_t digits);

}} // namespace
