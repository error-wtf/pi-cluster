# pi-cluster

**HPC Pi Computation Suite + Cluster Benchmark Tool**

*Target: PHYSnet Cluster, Universität Hamburg*

[![License: Anti-Capitalist](https://img.shields.io/badge/License-Anti--Capitalist%20v1.4-blue.svg)](LICENSE)

---

## What is pi-cluster?

**pi-cluster** is a high-performance computing tool that does two things:

1. **Computes π** using the Chudnovsky algorithm with arbitrary precision (GMP), optional CUDA GPU acceleration, and MPI multi-node distribution
2. **Benchmarks and diagnoses HPC clusters** — hardware detection, memory/disk/CPU benchmarks, Slurm awareness, and system validation

It is designed specifically for the **PHYSnet cluster** at Universität Hamburg (12,812 cores, 45.5 TB RAM, 539 nodes, 42 GPGPUs) but works on any Linux system.

### Why?

- **For researchers:** Compute π to arbitrary precision on a single node or across the cluster
- **For admins:** Validate cluster health, benchmark nodes, test scratch I/O
- **For teaching:** Demonstrate HPC concepts (MPI, CUDA, Slurm, checkpointing) with a real computation
- **For records:** Architecture designed to scale toward trillion-digit computations

---

## Quick Start

### Local build (no MPI, no CUDA)

```bash
git clone https://github.com/error-wtf/pi-cluster.git
cd pi-cluster
mkdir build && cd build
cmake .. -DBUILD_CUDA=OFF -DBUILD_MPI=OFF
make -j$(nproc)

# Detect your hardware
./pi-cluster detect

# Run benchmarks
./pi-cluster bench

# Check system readiness
./pi-cluster doctor

# Compute 10,000 digits of pi
./pi-cluster run -d 10000 -b cpu -v
```

### PHYSnet build (with MPI)

```bash
module load gcc gmp    # adjust to PHYSnet modules
mkdir build && cd build
cmake .. -DBUILD_MPI=ON -DBUILD_CUDA=OFF
make -j$(nproc)

# Submit single-node job
sbatch scripts/slurm/pi-single-node.sbatch

# Submit 8-node MPI job
sbatch scripts/slurm/pi-8nodes.sbatch
```

### Full hybrid build (MPI + CUDA)

```bash
module load gcc gmp cuda mpi
mkdir build && cd build
cmake .. -DBUILD_MPI=ON -DBUILD_CUDA=ON
make -j$(nproc)
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `pi-cluster detect` | Probe hardware: CPU, RAM, NUMA, GPU, scratch, Slurm |
| `pi-cluster bench` | Run microbenchmarks: CPU throughput, memory BW, disk I/O |
| `pi-cluster doctor` | Check system readiness (OK / WARN / FAIL) |
| `pi-cluster estimate -d N` | Estimate resources needed for N digits |
| `pi-cluster run -d N` | Compute π with N decimal digits |
| `pi-cluster resume -c path` | Resume from checkpoint |

### Run options

```
-d, --digits <N>      Number of decimal digits
-b, --backend <B>     Backend: auto, cpu, hybrid, mpi, mpi-hybrid
-o, --output <file>   Output file (default: pi.txt)
--chunk <N>           Checkpoint interval (default: 1000 terms)
--scratch <path>      Override scratch directory
--json                Machine-readable JSON output
-v, --verbose         Show progress bar and detailed output
```

---

## Architecture

```
pi-cluster/
├── src/
│   ├── core/          # Chudnovsky algorithm (GMP arbitrary precision)
│   ├── cpu/           # CPU-specific optimizations (planned: NTT)
│   ├── cuda/          # Optional CUDA hybrid (≤700 digits GPU, else CPU fallback)
│   ├── mpi/           # MPI multi-node: rank discovery, hierarchical merge
│   ├── detect/        # Hardware detection: CPU, RAM, NUMA, GPU, Scratch, Slurm
│   ├── bench/         # Microbenchmarks: CPU, memory, disk, GPU, MPI
│   ├── progress/      # Phase-based progress: ETA, throughput, JSON telemetry
│   ├── storage/       # Scratch manager, checkpoint save/load, resume
│   ├── cli/           # Main CLI with subcommands
│   └── util/          # Formatting helpers
├── include/picluster/ # Public headers
├── tests/             # Unit, integration, smoke tests
├── scripts/slurm/     # Slurm job templates (1, 8, 64 nodes)
├── configs/           # YAML configs for local and PHYSnet
└── docs/              # Architecture, PHYSnet guide, benchmarking
```

### Execution Modes

| Mode | Nodes | Backend | Use case |
|------|-------|---------|----------|
| **local-validate** | 1 (laptop) | cpu | Quick test, demo, teaching |
| **physnet-single** | 1 (cluster) | cpu or hybrid | Moderate computation, benchmarking |
| **physnet-cluster** | 8–64+ | mpi or mpi-hybrid | Large-scale computation |

### Backend Selection

| Backend | Description |
|---------|-------------|
| `auto` | Conservative: CPU on PHYSnet, hybrid only if GPU detected and ≤700 digits |
| `cpu` | GMP arbitrary precision, single-node |
| `hybrid` | GPU Chudnovsky (≤700 digits) + CPU for higher precision |
| `mpi` | Multi-node CPU via MPI |
| `mpi-hybrid` | Multi-node with optional GPU per node |

**GPU/CUDA policy:** CUDA is **optional**. The CPU path is always the primary production path. GPU acceleration is available for benchmarking and moderate-precision runs. The build never fails if CUDA is absent.

---

## Origin

This project is a complete rewrite of [CALCULATION_OF_NUMBER_PI](https://github.com/error-wtf/CALCULATION_OF_NUMBER_PI), restructured for HPC cluster deployment.

### What was kept from the original:
- Chudnovsky algorithm (CPU + GMP)
- CUDA hybrid approach (GPU kernel for term computation)
- Resource detection heuristics (RAM, disk, VRAM)
- BBP validation method

### What is new:
- Modular C++ architecture with clean separation of concerns
- CMake build system with optional CUDA/MPI
- Full hardware detection (CPU, NUMA, GPU, Scratch, Slurm)
- Microbenchmark suite
- Phase-based progress tracking with ETA and JSON telemetry
- Checkpoint/resume support
- MPI multi-node skeleton with hierarchical merge architecture
- Slurm job templates for PHYSnet
- Doctor/estimate commands for pre-flight checks

---

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Hardware detection | ✅ Production | CPU, RAM, NUMA, GPU, Scratch, Slurm |
| Chudnovsky CPU (GMP) | ✅ Production | Arbitrary precision, chunked |
| BBP validation | ✅ Production | Double-precision cross-check |
| Progress tracker | ✅ Production | Phase-based, terminal + JSON |
| Benchmarks | ✅ Production | CPU, memory, disk I/O |
| Checkpoint/resume | ✅ Functional | Save/load with metadata |
| CLI | ✅ Production | detect, bench, doctor, estimate, run |
| CUDA hybrid | ⚠️ Experimental | ≤700 digits, double precision limit |
| MPI skeleton | ⚠️ Skeleton | Architecture ready, merge not yet implemented |
| GPU benchmarks | 🔲 Stub | Needs CUDA compilation |
| MPI benchmarks | 🔲 Stub | Needs MPI compilation |
| Binary splitting | 🔲 Planned | For billion+ digit efficiency |

---

## PHYSnet Notes

- **Login:** `ssh login1.physnet.uni-hamburg.de`
- **Scheduler:** Slurm
- **Modules:** Use `module avail` to find available compilers/libraries
- **Scratch:** Use `$SLURM_TMPDIR` for node-local temporary storage
- **GPU:** 42 GPGPUs available; request via `--gres=gpu:N`
- **Support:** support@physnet.uni-hamburg.de

See [docs/physnet.md](docs/physnet.md) for detailed PHYSnet integration guide.

---

## License

[Anti-Capitalist Software License v1.4](LICENSE)

© 2024–2026 Lino Casu
