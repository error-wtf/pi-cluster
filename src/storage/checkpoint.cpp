// checkpoint.cpp — Checkpoint save/load for long-running computations
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <cstdint>

namespace fs = std::filesystem;

namespace picluster { namespace storage {

struct CheckpointMeta {
    std::string version = "1.0";
    std::string backend;       // cpu, hybrid, mpi
    std::int64_t digits_target = 0;
    std::int64_t terms_done = 0;
    std::int64_t terms_total = 0;
    int mpi_rank = 0;
    int mpi_size = 1;
    std::string timestamp;
    std::string hostname;
};

static std::string now_iso() {
    auto t = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(t);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&tt));
    return buf;
}

bool save_checkpoint(const std::string& dir, const CheckpointMeta& meta,
                     const char* data, std::size_t data_size) {
    fs::create_directories(dir);
    // Write metadata
    std::string meta_path = dir + "/checkpoint.meta";
    std::ofstream mf(meta_path);
    if (!mf) return false;
    mf << "version=" << meta.version << "\n";
    mf << "backend=" << meta.backend << "\n";
    mf << "digits_target=" << meta.digits_target << "\n";
    mf << "terms_done=" << meta.terms_done << "\n";
    mf << "terms_total=" << meta.terms_total << "\n";
    mf << "mpi_rank=" << meta.mpi_rank << "\n";
    mf << "mpi_size=" << meta.mpi_size << "\n";
    mf << "timestamp=" << now_iso() << "\n";
    mf << "hostname=" << meta.hostname << "\n";
    mf.close();

    // Write binary state data
    if (data && data_size > 0) {
        std::string data_path = dir + "/checkpoint.bin";
        std::ofstream df(data_path, std::ios::binary);
        if (!df) return false;
        df.write(data, data_size);
    }
    return true;
}

bool load_checkpoint(const std::string& dir, CheckpointMeta& meta,
                     std::vector<char>& data) {
    std::string meta_path = dir + "/checkpoint.meta";
    if (!fs::exists(meta_path)) return false;

    std::ifstream mf(meta_path);
    std::string line;
    while (std::getline(mf, line)) {
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        if (key == "version") meta.version = val;
        else if (key == "backend") meta.backend = val;
        else if (key == "digits_target") meta.digits_target = std::stoll(val);
        else if (key == "terms_done") meta.terms_done = std::stoll(val);
        else if (key == "terms_total") meta.terms_total = std::stoll(val);
        else if (key == "mpi_rank") meta.mpi_rank = std::stoi(val);
        else if (key == "mpi_size") meta.mpi_size = std::stoi(val);
        else if (key == "timestamp") meta.timestamp = val;
        else if (key == "hostname") meta.hostname = val;
    }

    std::string data_path = dir + "/checkpoint.bin";
    if (fs::exists(data_path)) {
        auto sz = fs::file_size(data_path);
        data.resize(sz);
        std::ifstream df(data_path, std::ios::binary);
        df.read(data.data(), sz);
    }
    return true;
}

}} // namespace
