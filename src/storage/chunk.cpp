// chunk.cpp — Chunk-based out-of-core computation manager
#include "picluster/storage/chunk.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cassert>

namespace picluster { namespace storage {

const char* chunk_status_name(ChunkStatus s) {
    switch (s) {
        case ChunkStatus::PENDING:      return "pending";
        case ChunkStatus::COMPUTING:    return "computing";
        case ChunkStatus::COMPUTED:     return "computed";
        case ChunkStatus::MERGED:       return "merged";
        case ChunkStatus::CHECKPOINTED: return "checkpointed";
        case ChunkStatus::FAILED:       return "failed";
        case ChunkStatus::CORRUPT:      return "corrupt";
    }
    return "unknown";
}

ChunkManager::ChunkManager(const ChunkConfig& cfg) : cfg_(cfg) {}

std::size_t ChunkManager::compute_optimal_chunk_terms(
    std::size_t free_ram_bytes,
    std::uint64_t free_scratch_bytes,
    int num_ranks,
    double ram_fraction,
    double scratch_reserve)
{
    // Each Chudnovsky term needs ~10 bytes RAM for GMP intermediate state
    // plus ~4 bytes scratch for checkpoint storage
    const std::size_t BYTES_PER_TERM_RAM = 10;
    const std::size_t BYTES_PER_TERM_SCRATCH = 4;

    std::size_t usable_ram = static_cast<std::size_t>(free_ram_bytes * ram_fraction);
    std::uint64_t usable_scratch = static_cast<std::uint64_t>(
        free_scratch_bytes * (1.0 - scratch_reserve));

    // RAM-limited chunk size (per rank)
    std::size_t ram_per_rank = usable_ram / std::max(1, num_ranks);
    std::size_t max_terms_by_ram = ram_per_rank / BYTES_PER_TERM_RAM;

    // Scratch-limited chunk size (per rank)
    std::uint64_t scratch_per_rank = usable_scratch / std::max(1, num_ranks);
    std::size_t max_terms_by_scratch = static_cast<std::size_t>(
        scratch_per_rank / BYTES_PER_TERM_SCRATCH);

    // Take the minimum
    std::size_t optimal = std::min(max_terms_by_ram, max_terms_by_scratch);

    // Clamp to reasonable range
    optimal = std::max(optimal, (std::size_t)100);
    optimal = std::min(optimal, (std::size_t)10000000);

    return optimal;
}

void ChunkManager::plan(std::int64_t total_terms, int num_ranks) {
    chunks_.clear();

    std::size_t chunk_terms;
    if (cfg_.target_chunk_bytes > 0) {
        // User specified a target chunk size in bytes
        chunk_terms = cfg_.target_chunk_bytes / 10; // ~10 bytes/term
    } else {
        // Auto-determine based on system resources
        // Use conservative defaults if we can't detect
        std::size_t free_ram = 4ULL * 1024 * 1024 * 1024; // 4 GB default
        std::uint64_t free_scratch = 10ULL * 1024 * 1024 * 1024; // 10 GB default
        chunk_terms = compute_optimal_chunk_terms(
            free_ram, free_scratch, num_ranks,
            cfg_.max_ram_fraction, cfg_.scratch_reserve_fraction);
    }

    // Clamp to configured limits
    chunk_terms = std::max(chunk_terms, cfg_.min_chunk_terms);
    chunk_terms = std::min(chunk_terms, cfg_.max_chunk_terms);

    // Create chunks
    std::int64_t offset = 0;
    std::int64_t chunk_id = 0;
    while (offset < total_terms) {
        ChunkMeta cm;
        cm.chunk_id = chunk_id++;
        cm.range_start = offset;
        cm.range_end = std::min(offset + (std::int64_t)chunk_terms, total_terms);
        cm.status = ChunkStatus::PENDING;

        // Round-robin rank assignment
        cm.owner_rank = static_cast<int>(cm.chunk_id % num_ranks);

        // Set scratch path
        if (!cfg_.scratch_root.empty()) {
            cm.scratch_path = cfg_.scratch_root + "/chunk_" +
                              std::to_string(cm.chunk_id);
        }

        chunks_.push_back(cm);
        offset = cm.range_end;
    }
}

std::vector<ChunkMeta*> ChunkManager::chunks_for_rank(int rank) {
    std::vector<ChunkMeta*> result;
    for (auto& c : chunks_) {
        if (c.owner_rank == rank) result.push_back(&c);
    }
    return result;
}

void ChunkManager::set_status(std::int64_t chunk_id, ChunkStatus status) {
    for (auto& c : chunks_) {
        if (c.chunk_id == chunk_id) { c.status = status; return; }
    }
}

void ChunkManager::set_checksum(std::int64_t chunk_id, std::uint64_t checksum) {
    for (auto& c : chunks_) {
        if (c.chunk_id == chunk_id) { c.checksum = checksum; return; }
    }
}

void ChunkManager::set_compute_time(std::int64_t chunk_id, double sec) {
    for (auto& c : chunks_) {
        if (c.chunk_id == chunk_id) { c.compute_time_sec = sec; return; }
    }
}

std::vector<std::int64_t> ChunkManager::incomplete_chunk_ids() const {
    std::vector<std::int64_t> ids;
    for (auto& c : chunks_) {
        if (c.status != ChunkStatus::MERGED &&
            c.status != ChunkStatus::CHECKPOINTED &&
            c.status != ChunkStatus::COMPUTED) {
            ids.push_back(c.chunk_id);
        }
    }
    return ids;
}

std::int64_t ChunkManager::completed_chunks() const {
    std::int64_t n = 0;
    for (auto& c : chunks_) {
        if (c.status == ChunkStatus::COMPUTED ||
            c.status == ChunkStatus::MERGED ||
            c.status == ChunkStatus::CHECKPOINTED) n++;
    }
    return n;
}

std::int64_t ChunkManager::failed_chunks() const {
    std::int64_t n = 0;
    for (auto& c : chunks_) {
        if (c.status == ChunkStatus::FAILED ||
            c.status == ChunkStatus::CORRUPT) n++;
    }
    return n;
}

double ChunkManager::completion_fraction() const {
    if (chunks_.empty()) return 0.0;
    return static_cast<double>(completed_chunks()) / chunks_.size();
}

std::string ChunkManager::to_json() const {
    std::ostringstream j;
    j << "{\"chunk_count\":" << chunks_.size() << ",\"chunks\":[";
    for (std::size_t i = 0; i < chunks_.size(); i++) {
        if (i > 0) j << ",";
        auto& c = chunks_[i];
        j << "{\"id\":" << c.chunk_id
          << ",\"start\":" << c.range_start
          << ",\"end\":" << c.range_end
          << ",\"rank\":" << c.owner_rank
          << ",\"status\":\"" << chunk_status_name(c.status) << "\""
          << ",\"checksum\":" << c.checksum
          << ",\"time\":" << std::fixed << std::setprecision(3) << c.compute_time_sec
          << "}";
    }
    j << "]}";
    return j.str();
}

}} // namespace
