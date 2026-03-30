#pragma once
#include <string>
#include <cstdint>
#include <vector>

namespace picluster { namespace storage {

struct ChunkSumMeta {
    std::int64_t chunk_id = -1;
    std::int64_t range_start = 0;
    std::int64_t range_end = 0;
    std::int64_t digits = 0;
    std::int64_t precision_bits = 0;
    std::string backend;
    std::string hostname;
    int rank = 0;
    std::uint64_t checksum = 0;
    std::string format_version = "1.0";
};

// Save a chunk's partial sum + metadata to checkpoint directory
// Files: chunk_sums/chunk_<id>.sum (GMP string) + chunk_<id>.meta.json
bool save_chunk_sum(const std::string& checkpoint_dir,
                    const ChunkSumMeta& meta,
                    const std::string& sum_str);

// Load a chunk's partial sum string from checkpoint directory
// Returns empty string if not found or corrupt
std::string load_chunk_sum(const std::string& checkpoint_dir,
                           std::int64_t chunk_id,
                           ChunkSumMeta* meta_out = nullptr);

// Verify a chunk sum file's integrity (checksum match)
bool verify_chunk_sum(const std::string& checkpoint_dir, std::int64_t chunk_id);

// Save merge-level state (for hierarchical merge checkpointing)
bool save_merge_level(const std::string& checkpoint_dir,
                      int level, int pair_id,
                      const std::string& sum_str);

std::string load_merge_level(const std::string& checkpoint_dir,
                             int level, int pair_id);

}} // namespace
