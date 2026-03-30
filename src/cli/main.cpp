// main.cpp — pi-cluster CLI with subcommands
// Usage: pi-cluster <command> [options]
// Commands: detect, bench, doctor, estimate, run, resume
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <fstream>

#include "picluster/detect/detect.h"
#include "picluster/core/chudnovsky.h"
#include "picluster/bench/bench.h"
#include "picluster/progress/progress.h"

static void print_usage() {
    std::cout << R"(pi-cluster — HPC Pi computation suite + cluster benchmark tool
Copyright (c) 2024-2026 Lino Casu — Anti-Capitalist Software License v1.4
Target: PHYSnet Cluster, Universität Hamburg

Usage: pi-cluster <command> [options]

Commands:
  detect              Probe hardware (CPU, RAM, GPU, scratch, Slurm)
  bench               Run microbenchmarks (CPU, memory, disk, optional GPU/MPI)
  doctor              Check system readiness for pi computation
  estimate -d <N>     Estimate resources needed for N digits
  run -d <N>          Compute pi with N decimal digits
  resume -c <path>    Resume from checkpoint

Run options:
  -d, --digits <N>    Number of decimal digits (required for run/estimate)
  -b, --backend <B>   Backend: auto, cpu, hybrid, mpi, mpi-hybrid (default: auto)
  -o, --output <F>    Output file (default: pi.txt)
  --chunk <N>         Checkpoint interval in terms (default: 1000)
  --scratch <path>    Override scratch directory
  --json              Output machine-readable JSON
  -v, --verbose       Verbose output

Examples:
  pi-cluster detect
  pi-cluster bench
  pi-cluster doctor
  pi-cluster estimate -d 1000000000
  pi-cluster run -d 1000000 -b cpu -o result.txt
  pi-cluster run -d 100000 -b hybrid
  pi-cluster resume -c checkpoints/run_001
)";
}

static int cmd_detect(bool json) {
    auto profile = picluster::detect::detect_system();
    if (json) {
        std::cout << picluster::detect::profile_to_json(profile) << std::endl;
    } else {
        picluster::detect::print_profile(profile);
    }
    return 0;
}

static int cmd_bench(bool json) {
    auto report = picluster::bench::run_all();
    if (json) {
        std::cout << report.to_json() << std::endl;
    } else {
        std::cout << report.to_text();
    }
    return 0;
}

static int cmd_doctor() {
    std::cout << "=== pi-cluster Doctor ===\n";
    auto p = picluster::detect::detect_system();

    auto check = [](const char* name, bool ok, const char* note) {
        printf("  [%s] %-30s %s\n", ok ? " OK " : "FAIL", name, note);
        return ok ? 0 : 1;
    };

    int fails = 0;
    fails += check("CPU detected", p.cpu.logical_cores > 0, p.cpu.model_name.c_str());
    fails += check("RAM available", p.mem.free_ram_bytes > 100*1024*1024, 
                    (std::to_string(p.mem.free_ram_bytes / (1024*1024)) + " MB free").c_str());
    fails += check("Scratch writable", p.scratch.free_bytes > 1024*1024, p.scratch.path.c_str());

#ifdef PICLUSTER_HAVE_GMP
    fails += check("GMP library", true, "compiled with GMP support");
#else
    fails += check("GMP library", false, "NOT compiled with GMP — limited to 15 digits");
#endif

#ifdef PICLUSTER_HAVE_CUDA
    fails += check("CUDA compiled", true, "GPU backend available");
#else
    check("CUDA compiled", false, "GPU backend not available (optional)");
#endif

#ifdef PICLUSTER_HAVE_MPI
    fails += check("MPI compiled", true, "multi-node backend available");
#else
    check("MPI compiled", false, "multi-node backend not available (optional)");
#endif

    if (p.gpu.available) {
        check("GPU detected", true, (std::to_string(p.gpu.count) + " GPU(s)").c_str());
    } else {
        check("GPU detected", false, "no NVIDIA GPU found (optional)");
    }

    if (p.slurm.in_slurm_job) {
        check("Slurm environment", true, ("Job " + p.slurm.job_id).c_str());
    } else {
        check("Slurm environment", false, "not in Slurm job (OK for local runs)");
    }

    printf("\n  Result: %d critical issue(s)\n", fails);
    return fails > 0 ? 1 : 0;
}

static int cmd_estimate(std::int64_t digits) {
    if (digits <= 0) { std::cerr << "Error: --digits required\n"; return 1; }
    auto profile = picluster::detect::detect_system();
    std::size_t ram_needed = picluster::core::estimate_ram_bytes(digits);
    std::int64_t terms = picluster::core::estimate_terms(digits);

    std::cout << "=== pi-cluster Estimate ===\n";
    printf("  Target digits:      %lld\n", (long long)digits);
    printf("  Chudnovsky terms:   %lld\n", (long long)terms);
    printf("  RAM needed (est):   %zu MB\n", ram_needed / (1024*1024));
    printf("  RAM available:      %zu MB\n", profile.mem.free_ram_bytes / (1024*1024));
    printf("  Scratch available:  %llu MB\n", (unsigned long long)profile.scratch.free_bytes / (1024*1024));

    bool fits_ram = ram_needed < profile.mem.free_ram_bytes * 7 / 10;
    printf("  Fits in RAM (70%%):  %s\n", fits_ram ? "YES" : "NO — needs multi-node or out-of-core");

    // Very rough time estimate: ~14 digits/term, ~0.001s/term for moderate sizes
    double est_seconds = terms * 0.001;
    if (digits > 1000000) est_seconds = terms * 0.01;  // slower for large precision
    if (digits > 100000000) est_seconds = terms * 0.1;
    printf("  Time estimate:      %.0f sec (%.1f hours) — ROUGH, single-node CPU\n",
           est_seconds, est_seconds / 3600.0);
    printf("  Confidence:         LOW (actual performance depends on hardware)\n");
    return 0;
}

static int cmd_run(std::int64_t digits, const std::string& backend,
                   const std::string& output, int chunk, bool verbose) {
    if (digits <= 0) { std::cerr << "Error: --digits required\n"; return 1; }

    picluster::progress::ProgressTracker tracker;
    tracker.set_target(digits);

    // Detect system
    tracker.set_phase(picluster::progress::Phase::DETECT, "Detecting hardware...");
    auto profile = picluster::detect::detect_system();
    if (verbose) picluster::detect::print_profile(profile);

    // Select backend
    std::string actual_backend = backend;
    if (actual_backend == "auto") {
        // Conservative: prefer CPU. Use hybrid only if GPU detected and digits <= 700
        if (profile.gpu.available && digits <= 700) actual_backend = "hybrid";
        else actual_backend = "cpu";
    }

    std::cout << "Computing pi with " << digits << " digits using backend: " << actual_backend << "\n";

    tracker.set_phase(picluster::progress::Phase::COMPUTE_LOCAL, "Computing...");

    std::string result;
    auto progress_cb = [&](double frac, const std::string& msg, std::int64_t d) {
        tracker.update(frac, d, msg);
        if (verbose) tracker.render_terminal();
    };

    if (actual_backend == "cpu" || actual_backend == "mpi") {
        result = picluster::core::compute_pi_cpu(digits, chunk, progress_cb);
    } else if (actual_backend == "hybrid" || actual_backend == "mpi-hybrid") {
        result = picluster::core::compute_pi_gpu(digits, 256, progress_cb);
    } else {
        std::cerr << "Unknown backend: " << actual_backend << "\n";
        return 1;
    }

    tracker.set_phase(picluster::progress::Phase::FINALIZE, "Writing output...");

    // Write output
    std::ofstream ofs(output);
    if (!ofs) { std::cerr << "Cannot write to " << output << "\n"; return 1; }
    ofs << result << "\n";
    ofs.close();

    tracker.set_phase(picluster::progress::Phase::DONE, "Complete");
    if (verbose) { fprintf(stderr, "\n"); tracker.render_terminal(); fprintf(stderr, "\n"); }

    std::cout << "\nWrote " << digits << " digits of pi to " << output << "\n";
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) { print_usage(); return 0; }

    std::string cmd = argv[1];
    std::int64_t digits = 0;
    std::string backend = "auto";
    std::string output = "pi.txt";
    std::string checkpoint_path;
    int chunk = 1000;
    bool verbose = false;
    bool json = false;

    // Parse remaining args
    for (int i = 2; i < argc; i++) {
        if ((std::strcmp(argv[i], "-d") == 0 || std::strcmp(argv[i], "--digits") == 0) && i+1 < argc)
            digits = std::atoll(argv[++i]);
        else if ((std::strcmp(argv[i], "-b") == 0 || std::strcmp(argv[i], "--backend") == 0) && i+1 < argc)
            backend = argv[++i];
        else if ((std::strcmp(argv[i], "-o") == 0 || std::strcmp(argv[i], "--output") == 0) && i+1 < argc)
            output = argv[++i];
        else if ((std::strcmp(argv[i], "-c") == 0 || std::strcmp(argv[i], "--checkpoint") == 0) && i+1 < argc)
            checkpoint_path = argv[++i];
        else if (std::strcmp(argv[i], "--chunk") == 0 && i+1 < argc)
            chunk = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--json") == 0)
            json = true;
        else if (std::strcmp(argv[i], "-v") == 0 || std::strcmp(argv[i], "--verbose") == 0)
            verbose = true;
    }

    if (cmd == "detect") return cmd_detect(json);
    if (cmd == "bench") return cmd_bench(json);
    if (cmd == "doctor") return cmd_doctor();
    if (cmd == "estimate") return cmd_estimate(digits);
    if (cmd == "run") return cmd_run(digits, backend, output, chunk, verbose);
    if (cmd == "resume") {
        std::cerr << "Resume not yet fully implemented. Checkpoint path: " << checkpoint_path << "\n";
        return 1;
    }

    std::cerr << "Unknown command: " << cmd << "\n";
    print_usage();
    return 1;
}
