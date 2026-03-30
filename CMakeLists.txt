cmake_minimum_required(VERSION 3.18)
project(pi-cluster
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "HPC Pi computation suite + cluster benchmark tool for PHYSnet"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==== Options ====
option(BUILD_CUDA "Enable CUDA/GPU support (optional)" OFF)
option(BUILD_MPI  "Enable MPI multi-node support" OFF)
option(BUILD_TESTS "Build unit and smoke tests" ON)

# ==== Dependencies ====

# Threads (always needed)
find_package(Threads REQUIRED)

# GMP (required for arbitrary precision)
find_library(GMP_LIB gmp)
find_library(GMPXX_LIB gmpxx)
find_path(GMP_INCLUDE gmp.h)
if(NOT GMP_LIB OR NOT GMP_INCLUDE)
    message(WARNING "GMP not found. Install libgmp-dev. CPU arbitrary precision will be limited.")
    set(HAVE_GMP OFF)
else()
    set(HAVE_GMP ON)
    message(STATUS "Found GMP: ${GMP_LIB}")
endif()

# Boost.Multiprecision (header-only, fallback if GMP unavailable)
find_package(Boost QUIET COMPONENTS headers)
if(Boost_FOUND)
    set(HAVE_BOOST ON)
    message(STATUS "Found Boost: ${Boost_INCLUDE_DIRS}")
else()
    set(HAVE_BOOST OFF)
    message(STATUS "Boost not found. Using GMP-only path.")
endif()

# MPI (optional)
if(BUILD_MPI)
    find_package(MPI REQUIRED)
    message(STATUS "MPI enabled: ${MPI_CXX_COMPILER}")
endif()

# CUDA (optional)
if(BUILD_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(HAVE_CUDA ON)
        message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER}")
    else()
        message(WARNING "CUDA requested but no compiler found. Disabling GPU support.")
        set(BUILD_CUDA OFF)
        set(HAVE_CUDA OFF)
    endif()
else()
    set(HAVE_CUDA OFF)
endif()

# ==== Configure Header ====
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/picluster_config.h
)

# ==== Include Paths ====
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
)

# ==== Library: picluster_core ====
add_library(picluster_core STATIC
    src/core/chudnovsky.cpp
    src/detect/detect.cpp
    src/detect/resources.cpp
    src/bench/bench.cpp
    src/progress/progress.cpp
    src/storage/scratch.cpp
    src/storage/checkpoint.cpp
    src/storage/chunk.cpp
    src/core/binary_splitting.cpp
    src/bench/mpi_bench.cpp
    src/util/format.cpp
    src/storage/chunk_persist.cpp
    src/bench/calibration.cpp
    src/util/guardrails.cpp
)
target_link_libraries(picluster_core PUBLIC Threads::Threads)
if(HAVE_GMP)
    target_link_libraries(picluster_core PUBLIC ${GMP_LIB} ${GMPXX_LIB})
    target_include_directories(picluster_core PUBLIC ${GMP_INCLUDE})
    target_compile_definitions(picluster_core PUBLIC PICLUSTER_HAVE_GMP)
endif()
if(HAVE_BOOST)
    target_link_libraries(picluster_core PUBLIC Boost::headers)
    target_compile_definitions(picluster_core PUBLIC PICLUSTER_HAVE_BOOST)
endif()

# ==== Library: picluster_mpi (optional) ====
if(BUILD_MPI)
    add_library(picluster_mpi STATIC
        src/mpi/mpi_runner.cpp
        src/mpi/mpi_merge.cpp
    )
    target_link_libraries(picluster_mpi PUBLIC picluster_core MPI::MPI_CXX)
    target_compile_definitions(picluster_mpi PUBLIC PICLUSTER_HAVE_MPI)
endif()

# ==== Library: picluster_cuda (optional) ====
if(HAVE_CUDA)
    add_library(picluster_cuda STATIC
        src/cuda/gpu_chudnovsky.cu
        src/cuda/gpu_detect.cpp
        src/cuda/gpu_bench.cu
    )
    set_target_properties(picluster_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(picluster_cuda PUBLIC picluster_core)
    target_compile_definitions(picluster_cuda PUBLIC PICLUSTER_HAVE_CUDA)
endif()

# ==== Main Executable ====
add_executable(pi-cluster src/cli/main.cpp)
target_link_libraries(pi-cluster PRIVATE picluster_core)
if(BUILD_MPI)
    target_link_libraries(pi-cluster PRIVATE picluster_mpi)
endif()
if(HAVE_CUDA)
    target_link_libraries(pi-cluster PRIVATE picluster_cuda)
endif()

# ==== Tests ====
if(BUILD_TESTS)
    enable_testing()
    add_executable(test_detect tests/unit/test_detect.cpp)
    target_link_libraries(test_detect PRIVATE picluster_core)
    add_test(NAME test_detect COMMAND test_detect)

    add_executable(test_chudnovsky tests/unit/test_chudnovsky.cpp)
    target_link_libraries(test_chudnovsky PRIVATE picluster_core)
    add_test(NAME test_chudnovsky COMMAND test_chudnovsky)

    add_executable(smoke_run tests/smoke/smoke_run.cpp)
    target_link_libraries(smoke_run PRIVATE picluster_core)
    add_test(NAME smoke_run COMMAND smoke_run)

    add_executable(test_serialize tests/unit/test_serialize.cpp)
    target_link_libraries(test_serialize PRIVATE picluster_core)
    add_test(NAME test_serialize COMMAND test_serialize)

    add_executable(test_chunk_persist tests/unit/test_chunk_persist.cpp)
    target_link_libraries(test_chunk_persist PRIVATE picluster_core)
    add_test(NAME test_chunk_persist COMMAND test_chunk_persist)
endif()

# ==== Install ====
install(TARGETS pi-cluster DESTINATION bin)
install(DIRECTORY include/picluster DESTINATION include)

message(STATUS "")
message(STATUS "pi-cluster configuration:")
message(STATUS "  GMP:  ${HAVE_GMP}")
message(STATUS "  Boost: ${HAVE_BOOST}")
message(STATUS "  MPI:  ${BUILD_MPI}")
message(STATUS "  CUDA: ${HAVE_CUDA}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "")
