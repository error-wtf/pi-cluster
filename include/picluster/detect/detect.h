#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace picluster {
namespace detect {

struct CpuInfo {
    std::string model_name;
    int physical_cores = 0;
    int logical_cores = 0;
    int sockets = 0;
    std::string architecture;
};

struct MemInfo {
    std::size_t total_ram_bytes = 0;
    std::size_t free_ram_bytes = 0;
    std::size_t total_swap_bytes = 0;
};

struct NumaInfo {
    int num_nodes = 0;
    std::vector<std::size_t> node_memory_bytes;
};

struct GpuInfo {
    bool available = false;
    int count = 0;
    std::vector<std::string> names;
    std::vector<std::size_t> vram_bytes;
    std::string cuda_version;
    std::string driver_version;
};

struct ScratchInfo {
    std::string path;
    std::uint64_t free_bytes = 0;
    std::string fs_type;
    bool is_local = false;  // true if node-local (not shared)
};

struct SlurmInfo {
    bool in_slurm_job = false;
    std::string job_id;
    int nnodes = 0;
    int ntasks = 0;
    int cpus_per_task = 0;
    std::string nodelist;
    std::string partition;
    std::string tmpdir;
};

struct SystemProfile {
    CpuInfo cpu;
    MemInfo mem;
    NumaInfo numa;
    GpuInfo gpu;
    ScratchInfo scratch;
    SlurmInfo slurm;
    std::string hostname;
    std::string os_version;
};

// Run full system detection
SystemProfile detect_system();

// Print profile to stdout (human-readable)
void print_profile(const SystemProfile& p);

// Export profile as JSON string
std::string profile_to_json(const SystemProfile& p);

// Export profile as markdown string
std::string profile_to_markdown(const SystemProfile& p);

} // namespace detect
} // namespace picluster
