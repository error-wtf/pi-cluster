// scratch.cpp — Node-local scratch management
#include <string>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <sstream>

namespace fs = std::filesystem;

namespace picluster { namespace storage {

std::string get_scratch_dir(const std::string& job_tag) {
    // Priority: SLURM_TMPDIR > TMPDIR > /tmp/pi-cluster
    const char* slurm_tmp = std::getenv("SLURM_TMPDIR");
    const char* tmpdir = std::getenv("TMPDIR");
    std::string base;
    if (slurm_tmp && slurm_tmp[0]) base = slurm_tmp;
    else if (tmpdir && tmpdir[0]) base = tmpdir;
    else base = "/tmp";

    std::string dir = base + "/pi-cluster";
    if (!job_tag.empty()) dir += "-" + job_tag;
    fs::create_directories(dir);
    return dir;
}

std::string get_output_dir(const std::string& base_path) {
    std::string dir = base_path.empty() ? "output" : base_path;
    fs::create_directories(dir);
    return dir;
}

std::uint64_t get_dir_usage_bytes(const std::string& path) {
    std::uint64_t total = 0;
    try {
        for (auto& entry : fs::recursive_directory_iterator(path)) {
            if (entry.is_regular_file())
                total += entry.file_size();
        }
    } catch (...) {}
    return total;
}

void cleanup_scratch(const std::string& scratch_dir) {
    try {
        fs::remove_all(scratch_dir);
    } catch (...) {}
}

}} // namespace
