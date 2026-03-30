// gpu_detect.cpp — GPU detection helpers (non-CUDA compilation path)
// When compiled with CUDA, additional runtime queries are available.
#include <string>

namespace picluster { namespace cuda {

#ifdef PICLUSTER_HAVE_CUDA
#include <cuda_runtime.h>

int get_gpu_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count;
}

std::string get_gpu_name(int device) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return "";
    return prop.name;
}

std::size_t get_gpu_vram(int device) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 0;
    return prop.totalGlobalMem;
}

#else

int get_gpu_count() { return 0; }
std::string get_gpu_name(int) { return ""; }
std::size_t get_gpu_vram(int) { return 0; }

#endif

}} // namespace
