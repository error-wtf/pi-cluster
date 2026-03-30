// resources.cpp — Resource detection and digit estimation
// Ported from CALCULATION_OF_NUMBER_PI/resources.cpp with enhancements
#include "picluster/detect/detect.h"
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <limits>

#if defined(__linux__)
#include <sys/sysinfo.h>
#include <sys/statvfs.h>
#endif

namespace picluster { namespace detect {

std::size_t estimate_max_digits_by_ram(std::size_t free_ram, double usable_fraction) {
    if (free_ram == 0 || usable_fraction <= 0.0) return 0;
    std::size_t usable = static_cast<std::size_t>(free_ram * usable_fraction);
    // ~10 bytes RAM per digit (conservative, accounts for GMP overhead)
    return usable / 10;
}

std::size_t estimate_max_digits_by_disk(std::uint64_t free_disk,
                                         double scratch_mult,
                                         double bytes_per_digit) {
    if (free_disk == 0) return 0;
    if (scratch_mult <= 0.0) scratch_mult = 6.0;
    if (bytes_per_digit <= 0.0) bytes_per_digit = 4.0 / 9.0;
    long double per_digit = static_cast<long double>(bytes_per_digit) * (scratch_mult + 1.0L);
    if (per_digit <= 0.0L) return 0;
    long double max_d = static_cast<long double>(free_disk) / per_digit;
    if (max_d > static_cast<long double>(std::numeric_limits<std::size_t>::max()))
        return std::numeric_limits<std::size_t>::max();
    return static_cast<std::size_t>(max_d);
}

std::size_t estimate_max_digits(const SystemProfile& profile) {
    std::size_t by_ram = estimate_max_digits_by_ram(profile.mem.free_ram_bytes, 0.7);
    std::size_t by_disk = estimate_max_digits_by_disk(profile.scratch.free_bytes);
    return std::min(by_ram, by_disk);
}

}} // namespace
