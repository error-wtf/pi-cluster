// test_serialize.cpp — Verify GMP serialize/deserialize roundtrip
#include "picluster/core/chudnovsky.h"
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=== test_serialize ===" << std::endl;
#ifdef PICLUSTER_HAVE_GMP
    // Create a known value
    mpf_set_default_prec(1024);
    mpf_class original("3.14159265358979323846264338327950288419716939937510");

    // Serialize
    auto bytes = picluster::core::serialize_mpf(original);
    assert(!bytes.empty());
    std::cout << "  Serialized: " << bytes.size() << " bytes" << std::endl;

    // Deserialize
    mpf_class restored = picluster::core::deserialize_mpf(bytes, 50);

    // Compare
    mpf_class diff = original - restored;
    double err = mpf_get_d(diff.get_mpf_t());
    std::cout << "  Error: " << err << std::endl;
    assert(std::abs(err) < 1e-40);
    std::cout << "  Roundtrip OK" << std::endl;
#else
    std::cout << "  Skipped (no GMP)" << std::endl;
#endif
    std::cout << "=== test_serialize PASSED ===" << std::endl;
    return 0;
}
