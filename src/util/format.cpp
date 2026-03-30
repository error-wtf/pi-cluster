// format.cpp — Utility formatting functions
#include <string>
#include <sstream>
#include <cstdint>

namespace picluster { namespace util {

std::string bytes_human(std::uint64_t b) {
    if (b >= (1ULL << 40)) {
        std::ostringstream s; s << (b / (1ULL << 40)) << "." << ((b % (1ULL << 40)) * 10 / (1ULL << 40)) << " TB"; return s.str();
    }
    if (b >= (1ULL << 30)) {
        std::ostringstream s; s << (b / (1ULL << 30)) << "." << ((b % (1ULL << 30)) * 10 / (1ULL << 30)) << " GB"; return s.str();
    }
    if (b >= (1ULL << 20)) return std::to_string(b / (1ULL << 20)) + " MB";
    if (b >= (1ULL << 10)) return std::to_string(b / (1ULL << 10)) + " KB";
    return std::to_string(b) + " B";
}

std::string digits_human(std::int64_t d) {
    if (d >= 1000000000000LL) return std::to_string(d / 1000000000000LL) + "T";
    if (d >= 1000000000LL) return std::to_string(d / 1000000000LL) + "B";
    if (d >= 1000000LL) return std::to_string(d / 1000000LL) + "M";
    if (d >= 1000LL) return std::to_string(d / 1000LL) + "K";
    return std::to_string(d);
}

}} // namespace
