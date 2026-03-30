#include "picluster/storage/chunk_persist.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <functional>

namespace fs = std::filesystem;

namespace picluster { namespace storage {

static std::uint64_t simple_hash(const std::string& s) {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (char c : s) { h ^= (unsigned char)c; h *= 0x100000001b3ULL; }
    return h;
}

static std::string meta_path(const std::string& dir, std::int64_t id) {
    return dir + "/chunk_sums/chunk_" + std::to_string(id) + ".meta.json";
}

static std::string sum_path(const std::string& dir, std::int64_t id) {
    return dir + "/chunk_sums/chunk_" + std::to_string(id) + ".sum";
}

bool save_chunk_sum(const std::string& checkpoint_dir,
                    const ChunkSumMeta& meta,
                    const std::string& sum_str) {
    std::string sums_dir = checkpoint_dir + "/chunk_sums";
    fs::create_directories(sums_dir);

    // Write sum (atomic: write to .tmp then rename)
    std::string sp = sum_path(checkpoint_dir, meta.chunk_id);
    std::string tmp = sp + ".tmp";
    { std::ofstream f(tmp); if (!f) return false; f << sum_str; }
    fs::rename(tmp, sp);

    // Write metadata
    std::string mp = meta_path(checkpoint_dir, meta.chunk_id);
    std::ofstream mf(mp);
    if (!mf) return false;
    mf << "{\n";
    mf << "  \"chunk_id\": " << meta.chunk_id << ",\n";
    mf << "  \"range_start\": " << meta.range_start << ",\n";
    mf << "  \"range_end\": " << meta.range_end << ",\n";
    mf << "  \"digits\": " << meta.digits << ",\n";
    mf << "  \"precision_bits\": " << meta.precision_bits << ",\n";
    mf << "  \"backend\": \"" << meta.backend << "\",\n";
    mf << "  \"hostname\": \"" << meta.hostname << "\",\n";
    mf << "  \"rank\": " << meta.rank << ",\n";
    mf << "  \"checksum\": " << simple_hash(sum_str) << ",\n";
    mf << "  \"format_version\": \"" << meta.format_version << "\"\n";
    mf << "}\n";
    return true;
}

std::string load_chunk_sum(const std::string& checkpoint_dir,
                           std::int64_t chunk_id,
                           ChunkSumMeta* meta_out) {
    std::string sp = sum_path(checkpoint_dir, chunk_id);
    if (!fs::exists(sp)) return "";
    std::ifstream f(sp);
    std::string sum;
    std::getline(f, sum);
    if (sum.empty()) return "";

    // Optionally load metadata
    if (meta_out) {
        std::string mp = meta_path(checkpoint_dir, chunk_id);
        if (fs::exists(mp)) {
            std::ifstream mf(mp);
            std::string line;
            while (std::getline(mf, line)) {
                if (line.find("\"chunk_id\"") != std::string::npos)
                    meta_out->chunk_id = chunk_id;
                // Simple parsing for other fields as needed
            }
        }
    }
    return sum;
}

bool verify_chunk_sum(const std::string& checkpoint_dir, std::int64_t chunk_id) {
    std::string sum = load_chunk_sum(checkpoint_dir, chunk_id);
    if (sum.empty()) return false;
    // Check against stored checksum in meta
    std::string mp = meta_path(checkpoint_dir, chunk_id);
    if (!fs::exists(mp)) return false;
    std::ifstream mf(mp);
    std::string content((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
    auto pos = content.find("\"checksum\":");
    if (pos == std::string::npos) return false;
    std::uint64_t stored = std::stoull(content.substr(pos + 12, 20));
    return stored == simple_hash(sum);
}

bool save_merge_level(const std::string& checkpoint_dir,
                      int level, int pair_id,
                      const std::string& sum_str) {
    std::string dir = checkpoint_dir + "/merge_levels/level" + std::to_string(level);
    fs::create_directories(dir);
    std::string path = dir + "/pair_" + std::to_string(pair_id) + ".sum";
    std::string tmp = path + ".tmp";
    { std::ofstream f(tmp); if (!f) return false; f << sum_str; }
    fs::rename(tmp, path);
    return true;
}

std::string load_merge_level(const std::string& checkpoint_dir,
                             int level, int pair_id) {
    std::string path = checkpoint_dir + "/merge_levels/level" + std::to_string(level)
                     + "/pair_" + std::to_string(pair_id) + ".sum";
    if (!fs::exists(path)) return "";
    std::ifstream f(path);
    std::string sum;
    std::getline(f, sum);
    return sum;
}

}} // namespace
