// gpu_bench.cu — GPU microbenchmark stubs
// Only compiled when BUILD_CUDA=ON
#ifdef PICLUSTER_HAVE_CUDA
#include <cuda_runtime.h>
#include <cstdio>

namespace picluster { namespace bench {

// TODO: Implement H2D/D2H bandwidth measurement
// TODO: Implement FMA kernel throughput measurement

}} // namespace
#endif
