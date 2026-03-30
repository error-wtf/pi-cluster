// test_detect.cpp — Unit test for hardware detection
#include "picluster/detect/detect.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "=== test_detect ===" << std::endl;
    auto p = picluster::detect::detect_system();

    // CPU must be detected
    assert(p.cpu.logical_cores > 0);
    std::cout << "  CPU: " << p.cpu.model_name << " (" << p.cpu.logical_cores << " logical cores) OK" << std::endl;

    // RAM must be nonzero
    assert(p.mem.total_ram_bytes > 0);
    std::cout << "  RAM: " << (p.mem.total_ram_bytes / (1024*1024)) << " MB total OK" << std::endl;

    // Scratch path must exist
    assert(!p.scratch.path.empty());
    std::cout << "  Scratch: " << p.scratch.path << " OK" << std::endl;

    // Hostname
    assert(!p.hostname.empty());
    std::cout << "  Hostname: " << p.hostname << " OK" << std::endl;

    // JSON export must not be empty
    std::string json = picluster::detect::profile_to_json(p);
    assert(!json.empty());
    assert(json.front() == '{');
    std::cout << "  JSON export OK (" << json.size() << " bytes)" << std::endl;

    std::cout << "=== test_detect PASSED ===" << std::endl;
    return 0;
}
