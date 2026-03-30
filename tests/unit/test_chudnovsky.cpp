// test_chudnovsky.cpp — Unit test for Chudnovsky pi computation
#include "picluster/core/chudnovsky.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <string>

int main() {
    std::cout << "=== test_chudnovsky ===" << std::endl;

    // Test BBP (double precision)
    double pi_bbp = picluster::core::compute_pi_bbp(50);
    double err = std::abs(pi_bbp - 3.14159265358979323846);
    std::cout << "  BBP(50 terms): " << pi_bbp << " (error: " << err << ")" << std::endl;
    assert(err < 1e-14);
    std::cout << "  BBP OK" << std::endl;

    // Test term estimation
    auto terms = picluster::core::estimate_terms(1000);
    assert(terms > 70 && terms < 100);
    std::cout << "  estimate_terms(1000) = " << terms << " OK" << std::endl;

    // Test RAM estimation
    auto ram = picluster::core::estimate_ram_bytes(1000000);
    assert(ram > 5000000);
    std::cout << "  estimate_ram(1M digits) = " << ram << " bytes OK" << std::endl;

    // Test CPU compute (small)
    std::string pi_str = picluster::core::compute_pi_cpu(50, 100, nullptr);
    std::cout << "  CPU pi(50): " << pi_str.substr(0, 55) << std::endl;
    assert(pi_str.size() >= 50);
    assert(pi_str.substr(0, 5) == "3.141");
    // Check known digits
    assert(pi_str.find("14159265358979") != std::string::npos);
    std::cout << "  CPU compute(50 digits) OK" << std::endl;

    std::cout << "=== test_chudnovsky PASSED ===" << std::endl;
    return 0;
}
