// test_chunk_persist.cpp — Verify chunk sum save/load/verify cycle
#include "picluster/storage/chunk_persist.h"
#include <iostream>
#include <cassert>
#include <filesystem>

int main() {
    std::cout << "=== test_chunk_persist ===" << std::endl;
    namespace ps = picluster::storage;

    std::string test_dir = "/tmp/pi_cluster_test_persist";
    std::filesystem::create_directories(test_dir);

    // Save
    ps::ChunkSumMeta meta;
    meta.chunk_id = 42;
    meta.range_start = 100;
    meta.range_end = 200;
    meta.digits = 10000;
    meta.precision_bits = 34000;
    meta.backend = "cpu";
    meta.hostname = "testhost";
    meta.rank = 0;

    std::string sum = "1.23456789012345678901234567890e+5";
    assert(ps::save_chunk_sum(test_dir, meta, sum));
    std::cout << "  Save OK" << std::endl;

    // Load
    ps::ChunkSumMeta loaded_meta;
    std::string loaded = ps::load_chunk_sum(test_dir, 42, &loaded_meta);
    assert(loaded == sum);
    std::cout << "  Load OK: " << loaded.substr(0, 30) << std::endl;

    // Verify checksum
    assert(ps::verify_chunk_sum(test_dir, 42));
    std::cout << "  Verify OK" << std::endl;

    // Merge level save/load
    assert(ps::save_merge_level(test_dir, 0, 1, "9.87654321e+3"));
    std::string ml = ps::load_merge_level(test_dir, 0, 1);
    assert(ml == "9.87654321e+3");
    std::cout << "  Merge level OK" << std::endl;

    // Cleanup
    std::filesystem::remove_all(test_dir);

    std::cout << "=== test_chunk_persist PASSED ===" << std::endl;
    return 0;
}
