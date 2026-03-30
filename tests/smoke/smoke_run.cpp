// smoke_run.cpp — Quick smoke test: detect + small pi computation
#include "picluster/detect/detect.h"
#include "picluster/core/chudnovsky.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "=== smoke_run ===" << std::endl;

    // 1. Detect
    auto p = picluster::detect::detect_system();
    assert(p.cpu.logical_cores > 0);
    std::cout << "  detect: OK (" << p.hostname << ", " << p.cpu.logical_cores << " cores)" << std::endl;

    // 2. Compute 100 digits
    std::string pi = picluster::core::compute_pi_cpu(100, 50, nullptr);
    assert(pi.size() >= 100);
    assert(pi.substr(0, 6) == "3.1415");
    std::cout << "  compute(100): " << pi.substr(0, 40) << "..." << std::endl;

    // 3. BBP cross-check
    double bbp = picluster::core::compute_pi_bbp(30);
    assert(std::abs(bbp - 3.14159265) < 1e-7);
    std::cout << "  bbp check: OK" << std::endl;

    std::cout << "=== smoke_run PASSED ===" << std::endl;
    return 0;
}
