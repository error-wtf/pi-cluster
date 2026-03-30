#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <functional>

namespace picluster { namespace storage {

enum class ChunkStatus {
    PENDING,        // not yet started
    COMPUTING,      // currently being computed
    COMPUTED,       // computation done, not yet merged
    MERGED,         // merged into parent
    CHECKPOINTED,   // written to disk
    FAILED,         // computation failed
    CORRUPT         // checksum mismatch on load
};

const char* chunk_status_name(ChunkStatus s);

struct ChunkMeta {
    std::int64_t chunk_id = -1;
    std::int64_t range_start = 0;   // first Chudnovsky term index
    std::int64_t range_end = 0;     // last term index (exclusive)
    int owner_rank = 0;             // MPI rank owning this chunk
    std::string backend;            // cpu / hybrid / mpi
    std::string scratch_path;       // where temp data lives
    std::uint64_t checksum = 0;     // CRC64 or similar
    ChunkStatus status = ChunkStatus::PENDING;
    double compute_time_sec = 0.0;
    std::size_t memory_peak_bytes = 0;
};

struct ChunkConfig {
    std::size_t target_chunk_bytes = 0;     // 0 = auto
    std::size_t min_chunk_terms = 100;
    std::size_t max_chunk_terms = 10000000;
    double max_ram_fraction = 0.70;         // never use more than 70% of free RAM
    double scratch_reserve_fraction = 0.15; // keep 15% scratch free
    std::string scratch_root;               // override scratch path (empty = auto)
};

class ChunkManager {
public:
    explicit ChunkManager(const ChunkConfig& cfg);

    // Plan chunks for a given total number of terms
    // Adapts chunk size based on detected RAM, scratch, node count
    void plan(std::int64_t total_terms, int num_ranks = 1);

    // Get all chunk metadata
    const std::vector<ChunkMeta>& chunks() const { return chunks_; }

    // Get chunks assigned to a specific rank
    std::vector<ChunkMeta*> chunks_for_rank(int rank);

    // Update chunk status
    void set_status(std::int64_t chunk_id, ChunkStatus status);
    void set_checksum(std::int64_t chunk_id, std::uint64_t checksum);
    void set_compute_time(std::int64_t chunk_id, double sec);

    // Resume support: find incomplete chunks
    std::vector<std::int64_t> incomplete_chunk_ids() const;

    // Stats
    std::int64_t total_chunks() const { return (std::int64_t)chunks_.size(); }
    std::int64_t completed_chunks() const;
    std::int64_t failed_chunks() const;
    double completion_fraction() const;

    // Adaptive chunk sizing based on system resources
    static std::size_t compute_optimal_chunk_terms(
        std::size_t free_ram_bytes,
        std::uint64_t free_scratch_bytes,
        int num_ranks,
        double ram_fraction = 0.70,
        double scratch_reserve = 0.15
    );

    // Serialize/deserialize chunk plan for checkpoint
    std::string to_json() const;
    static ChunkManager from_json(const std::string& json, const ChunkConfig& cfg);

private:
    ChunkConfig cfg_;
    std::vector<ChunkMeta> chunks_;
};

}} // namespace picluster::storage
