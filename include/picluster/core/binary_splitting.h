#pragma once
#include <cstdint>
#include <string>
#include <functional>

namespace picluster { namespace core {

using ProgressCallback = std::function<void(double, const std::string&, std::int64_t)>;

struct BSState { std::int64_t a = 0, b = 0; };

std::string compute_pi_binary_splitting(std::int64_t digits, ProgressCallback cb = nullptr);
std::string compute_pi_binary_splitting_mpi(std::int64_t digits, int mpi_rank, int mpi_size, ProgressCallback cb = nullptr);
bool should_use_binary_splitting(std::int64_t digits);
BSState binary_split(std::int64_t a, std::int64_t b);
BSState merge_bs(const BSState& left, const BSState& right);

}} // namespace
