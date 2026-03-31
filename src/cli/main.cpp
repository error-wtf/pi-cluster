// main.cpp — pi-cluster CLI with REAL wiring of MPI, Chunks, Checkpoints, Guardrails
// V2: Everything is connected. No more stubs in the run path.
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <chrono>
#include <vector>
#include <filesystem>
#include <cmath>

#include "picluster/detect/detect.h"
#include "picluster/core/chudnovsky.h"
#include "picluster/core/binary_splitting.h"
#include "picluster/bench/bench.h"
#include "picluster/progress/progress.h"
#include "picluster/storage/chunk.h"
#include "picluster/storage/chunk_persist.h"
#include "picluster/bench/calibration.h"
#include "picluster/util/guardrails.h"

#ifdef PICLUSTER_HAVE_MPI
#include <mpi.h>
#endif

// ============================================================
// USAGE
// ============================================================
static void print_usage() {
    std::cout << R"(pi-cluster v2 — HPC Pi computation suite + cluster benchmark tool
Copyright (c) 2024-2026 Lino Casu — Anti-Capitalist Software License v1.4

Usage: pi-cluster <command> [options]

Commands:
  detect              Probe hardware (CPU, RAM, GPU, scratch, Slurm)
  bench               Run microbenchmarks
  doctor              Check system readiness
  estimate -d <N>     Estimate resources for N digits
  run -d <N>          Compute pi (with chunks, checkpoints, guardrails)
  resume -c <path>    Resume from checkpoint

Run options:
  -d, --digits <N>    Number of decimal digits
  -b, --backend <B>   auto, cpu, hybrid, mpi, mpi-hybrid
  -o, --output <F>    Output file (default: pi.txt)
  --chunk <N>         Terms per chunk (0=auto)
  --scratch <path>    Override scratch directory
  --dry-run           Show plan without executing
  --confirm           Skip interactive confirmation
  --max-ram <F>       Max RAM fraction (default: 0.70)
  --json              Machine-readable JSON output
  -v, --verbose       Verbose progress
)";
}

// ============================================================
// DETECT
// ============================================================
static int cmd_detect(bool json) {
    auto p = picluster::detect::detect_system();
    if (json) std::cout << picluster::detect::profile_to_json(p) << std::endl;
    else picluster::detect::print_profile(p);
    return 0;
}

// ============================================================
// BENCH
// ============================================================
static int cmd_bench(bool json) {
    auto r = picluster::bench::run_all();
    if (json) std::cout << r.to_json() << std::endl;
    else std::cout << r.to_text();
    // Save calibrated node profile
    auto np = picluster::bench::calibrate("/tmp");
    picluster::bench::save_profile(np, "node_profile.json");
    std::cout << "Node profile saved to node_profile.json\n";
    return 0;
}

// ============================================================
// DOCTOR
// ============================================================
static int cmd_doctor() {
    std::cout << "=== pi-cluster Doctor ===\n";
    auto p = picluster::detect::detect_system();
    int fails = 0;
    auto chk = [&](const char* name, bool ok, const char* note) {
        printf("  [%s] %-30s %s\n", ok ? " OK " : "FAIL", name, note);
        if (!ok) fails++;
    };
    chk("CPU detected", p.cpu.logical_cores > 0, p.cpu.model_name.c_str());
    chk("RAM available", p.mem.free_ram_bytes > 100*1024*1024,
        (std::to_string(p.mem.free_ram_bytes/(1024*1024)) + " MB free").c_str());
    chk("Scratch writable", p.scratch.free_bytes > 1024*1024, p.scratch.path.c_str());
#ifdef PICLUSTER_HAVE_GMP
    chk("GMP library", true, "compiled with GMP");
#else
    chk("GMP library", false, "NOT compiled — limited to 15 digits"); fails++;
#endif
#ifdef PICLUSTER_HAVE_MPI
    chk("MPI compiled", true, "multi-node ready");
#else
    chk("MPI compiled", false, "single-node only (optional)");
#endif
    chk("GPU detected", p.gpu.available,
        p.gpu.available ? (std::to_string(p.gpu.count)+" GPU(s)").c_str() : "none (optional)");
    chk("Slurm env", p.slurm.in_slurm_job,
        p.slurm.in_slurm_job ? ("Job "+p.slurm.job_id).c_str() : "not in job (OK for local)");
    printf("\n  Result: %d critical issue(s)\n", fails);
    return fails > 0 ? 1 : 0;
}

// ============================================================
// ESTIMATE
// ============================================================
static int cmd_estimate(std::int64_t digits) {
    if (digits <= 0) { std::cerr << "Error: --digits required\n"; return 1; }
    auto profile = picluster::detect::detect_system();
    picluster::guardrails::ResourceLimits limits;
    auto est = picluster::guardrails::check_feasibility(
        digits, profile.mem.free_ram_bytes, profile.scratch.free_bytes, 1, limits);
    // Try calibrated estimate if profile exists
    std::string profile_path = "node_profile.json";
    auto np = picluster::bench::load_profile(profile_path);
    double cal_time = -1;
    if (np.cpu_mflops > 0) {
        cal_time = picluster::bench::estimate_time_from_profile(np, digits, 1);
    }

    std::cout << "=== pi-cluster Estimate ===\n";
    printf("  Digits:        %lld\n", (long long)digits);
    printf("  RAM needed:    %zu MB\n", est.ram_needed_bytes/(1024*1024));
    printf("  RAM available: %zu MB\n", profile.mem.free_ram_bytes/(1024*1024));
    printf("  Scratch needed:%llu MB\n", (unsigned long long)est.scratch_needed_bytes/(1024*1024));
    printf("  Fits in RAM:   %s\n", est.fits_in_ram ? "YES" : "NO");
    printf("  Rec. nodes:    %d\n", est.recommended_nodes);
    printf("  Chunks:        %lld\n", (long long)est.chunk_count);
    if (cal_time > 0) {
        printf("  Est. time:     %.1f hours (CALIBRATED from node_profile.json)\n", cal_time / 3600.0);
    } else {
        printf("  Est. time:     %.1f hours (heuristic, run 'bench' for calibrated estimate)\n", est.estimated_walltime_hours);
    }
    if (!est.warning.empty()) printf("  WARNING: %s\n", est.warning.c_str());
    return 0;
}

// ============================================================
// RUN — the real deal: chunks, checkpoints, guardrails, MPI
// ============================================================
static int cmd_run(std::int64_t digits, const std::string& backend,
                   const std::string& output, std::size_t chunk_terms_override,
                   const std::string& scratch_override, bool dry_run,
                   bool confirm, double max_ram_frac, bool verbose) {
    if (digits <= 0) { std::cerr << "Error: --digits required\n"; return 1; }

    // --- MPI init ---
    int mpi_rank = 0, mpi_size = 1;
#ifdef PICLUSTER_HAVE_MPI
    if (backend == "mpi" || backend == "mpi-hybrid") {
        // MPI_Init should be called in main() before this, but handle it here too
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            MPI_Init(nullptr, nullptr);
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        if (mpi_rank == 0)
            printf("[MPI] Running with %d ranks\n", mpi_size);
    }
#endif

    // --- Install signal handlers ---
    picluster::guardrails::install_signal_handlers();

    // --- Detect system ---
    auto profile = picluster::detect::detect_system();

    // --- Guardrails: feasibility check ---
    picluster::guardrails::ResourceLimits limits;
    limits.max_ram_fraction = max_ram_frac;
    auto est = picluster::guardrails::check_feasibility(
        digits, profile.mem.free_ram_bytes, profile.scratch.free_bytes, mpi_size, limits);

    // --- Build run plan ---
    picluster::guardrails::RunPlan plan;
    plan.digits = digits;
    plan.backend = backend;
    plan.nodes = mpi_size;
    plan.gpus = profile.gpu.available ? profile.gpu.count : 0;
    plan.total_terms = picluster::core::estimate_terms(digits);
    plan.ram_per_node = est.ram_needed_bytes / std::max(1, mpi_size);
    plan.scratch_per_node = est.scratch_needed_bytes / std::max(1, mpi_size);
    plan.output_size = est.output_size_bytes;
    plan.estimated_hours = est.estimated_walltime_hours;
    plan.scratch_path = scratch_override.empty() ? profile.scratch.path : scratch_override;
    plan.output_path = output;
    plan.checkpoint_path = plan.scratch_path + "/checkpoints";
    plan.gpu_available = profile.gpu.available;
    plan.mpi_active = mpi_size > 1;

    // --- Chunk planning ---
    picluster::storage::ChunkConfig ccfg;
    ccfg.max_ram_fraction = max_ram_frac;
    ccfg.scratch_root = plan.scratch_path;
    if (chunk_terms_override > 0) ccfg.target_chunk_bytes = chunk_terms_override * 10;

    picluster::storage::ChunkManager chunks(ccfg);
    chunks.plan(plan.total_terms, mpi_size);
    plan.chunk_count = chunks.total_chunks();
    plan.chunk_terms = chunks.chunks().empty() ? 0 : 
        (chunks.chunks()[0].range_end - chunks.chunks()[0].range_start);

    if (!est.fits_in_ram && mpi_size == 1)
        plan.warnings.push_back("RAM insufficient for single-node. Out-of-core chunking active.");
    if (!est.fits_on_scratch)
        plan.blockers.push_back("Scratch space insufficient. Increase scratch or reduce digits.");

    // --- Dry run: show plan and exit ---
    if (dry_run) {
        if (mpi_rank == 0) picluster::guardrails::print_run_plan(plan);
        return plan.blockers.empty() ? 0 : 1;
    }

    // --- Check blockers ---
    if (!plan.blockers.empty()) {
        if (mpi_rank == 0) {
            picluster::guardrails::print_run_plan(plan);
            std::cerr << "BLOCKED: Cannot proceed. Resolve issues above.\n";
        }
        return 1;
    }

    // --- Show plan ---
    if (mpi_rank == 0 && verbose) {
        picluster::guardrails::print_run_plan(plan);
    }

    // --- Progress tracker ---
    picluster::progress::ProgressTracker tracker;
    tracker.set_target(digits);
    tracker.set_mpi(mpi_rank, mpi_size);

    // --- Confirm gate ---
    if (!confirm && digits > 10000000 && !dry_run) {
        if (mpi_rank == 0) {
            picluster::guardrails::print_run_plan(plan);
            printf("Large run. Use --confirm to proceed.\n");
        }
        return 1;
    }

    // --- Real shutdown callback: saves checkpoint ---
    picluster::guardrails::set_shutdown_callback([&]() {
        std::filesystem::create_directories(plan.checkpoint_path);
        std::ofstream cf(plan.checkpoint_path + "/chunks.json");
        if (cf) cf << chunks.to_json();
    });

    // --- Backend selection ---
    std::string actual_backend = backend;
    if (actual_backend == "auto") {
        if (mpi_size > 1) actual_backend = "mpi";
        else if (profile.gpu.available && digits <= 700) actual_backend = "hybrid";
        else actual_backend = "cpu";
    }

    // --- COMPUTE: REAL partial sums per chunk ---
    tracker.set_phase(picluster::progress::Phase::COMPUTE_LOCAL, "Computing...");
    auto my_chunks = chunks.chunks_for_rank(mpi_rank);
    if (mpi_rank == 0)
        printf("Backend: %s, Chunks: %lld, Ranks: %d\n",
               actual_backend.c_str(), (long long)chunks.total_chunks(), mpi_size);

    auto progress_cb = [&](double frac, const std::string& msg, std::int64_t d) {
        tracker.update(frac, d, msg);
        if (verbose && mpi_rank == 0) tracker.render_terminal();
        // Hard disk watermark abort
        if (frac > 0.05 && !picluster::guardrails::check_disk_watermark(plan.scratch_path, 0.05)) {
            fprintf(stderr, "\nDISK CRITICAL — aborting safely.\n");
            std::filesystem::create_directories(plan.checkpoint_path);
            std::ofstream cf(plan.checkpoint_path + "/chunks.json");
            if (cf) cf << chunks.to_json();
            exit(2);
        }
    };

    std::string final_result;

    if (actual_backend == "hybrid" || actual_backend == "mpi-hybrid") {
        final_result = picluster::core::compute_pi_gpu(digits, 256, progress_cb);
        for (auto* ch : my_chunks)
            chunks.set_status(ch->chunk_id, picluster::storage::ChunkStatus::COMPUTED);
    } else if (mpi_size == 1 && picluster::core::should_use_binary_splitting(digits)) {
        // Single-node, large digits: use binary splitting (O(n log^2 n))
        if (mpi_rank == 0) printf("Using binary splitting (faster for %lld digits)\n", (long long)digits);
        final_result = picluster::core::compute_pi_binary_splitting(digits, progress_cb);
        for (auto* ch : my_chunks)
            chunks.set_status(ch->chunk_id, picluster::storage::ChunkStatus::COMPUTED);
    } else {
        // CPU / MPI: real partial sums per chunk
#ifdef PICLUSTER_HAVE_GMP
        mpf_class local_S(0);
        std::string chunk_data_dir = plan.checkpoint_path + "/chunk_sums";
        std::filesystem::create_directories(chunk_data_dir);

        for (auto* ch : my_chunks) {
            if (picluster::guardrails::shutdown_requested()) break;

            // Try loading persisted chunk sum (proper persistence API)
            std::string saved = picluster::storage::load_chunk_sum(plan.checkpoint_path, ch->chunk_id);
            if (!saved.empty() && picluster::storage::verify_chunk_sum(plan.checkpoint_path, ch->chunk_id)) {
                mpf_class restored(saved);
                local_S += restored;
                chunks.set_status(ch->chunk_id, picluster::storage::ChunkStatus::COMPUTED);
                if (verbose && mpi_rank == 0)
                    printf("  Chunk %lld: restored (verified)\n", (long long)ch->chunk_id);
                continue;
            }

            chunks.set_status(ch->chunk_id, picluster::storage::ChunkStatus::COMPUTING);
            auto t0 = std::chrono::steady_clock::now();

            mpf_class chunk_S;
            picluster::core::compute_partial_sum(
                ch->range_start, ch->range_end, digits, chunk_S, progress_cb);
            local_S += chunk_S;

            // Persist via chunk_persist API (atomic write + checksum + metadata)
            {
                std::size_t bufsize = static_cast<std::size_t>(digits) + 50;
                std::vector<char> buf(bufsize);
                gmp_snprintf(buf.data(), bufsize, "%.*Fe", (int)(digits/3 + 20), chunk_S.get_mpf_t());
                picluster::storage::ChunkSumMeta meta;
                meta.chunk_id = ch->chunk_id;
                meta.range_start = ch->range_start;
                meta.range_end = ch->range_end;
                meta.digits = digits;
                meta.precision_bits = (digits + 50) * 3.4;
                meta.backend = actual_backend;
                meta.rank = mpi_rank;
                picluster::storage::save_chunk_sum(plan.checkpoint_path, meta, std::string(buf.data()));
            }

            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
            chunks.set_compute_time(ch->chunk_id, dt);
            chunks.set_status(ch->chunk_id, picluster::storage::ChunkStatus::COMPUTED);
        }

        // --- MPI MERGE: hierarchical tree-reduce ---
        mpf_class total_S = local_S;
#ifdef PICLUSTER_HAVE_MPI
        if (mpi_size > 1) {
            tracker.set_phase(picluster::progress::Phase::GLOBAL_MERGE, "MPI tree-reduce...");
            // Pairwise tree reduction: at each step, rank pairs merge.
            // Step 0: ranks 0+1, 2+3, 4+5, ...
            // Step 1: ranks 0+2, 4+6, ...
            // Step k: ranks 0+2^k, 2^(k+1)+3*2^k, ...
            // Final result on rank 0. O(log N) steps instead of O(N).
            for (int step = 1; step < mpi_size; step *= 2) {
                if (mpi_rank % (2 * step) == 0) {
                    int partner = mpi_rank + step;
                    if (partner < mpi_size) {
                        // Receive partner's partial sum
                        int partner_sz = 0;
                        MPI_Recv(&partner_sz, 1, MPI_INT, partner, 100+step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        std::vector<char> rbuf(partner_sz);
                        MPI_Recv(rbuf.data(), partner_sz, MPI_CHAR, partner, 200+step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        total_S += picluster::core::deserialize_mpf(rbuf, digits);
                    }
                } else if (mpi_rank % (2 * step) == step) {
                    int partner = mpi_rank - step;
                    // Send our partial sum to partner
                    auto buf = picluster::core::serialize_mpf(total_S);
                    int sz = (int)buf.size();
                    MPI_Send(&sz, 1, MPI_INT, partner, 100+step, MPI_COMM_WORLD);
                    MPI_Send(buf.data(), sz, MPI_CHAR, partner, 200+step, MPI_COMM_WORLD);
                    // After sending, this rank is done with merge
                }
                MPI_Barrier(MPI_COMM_WORLD); // sync before next step
            }
            if (mpi_rank == 0 && verbose)
                printf("\n[MPI] Hierarchical tree-reduce complete (log2(%d) = %d steps)\n",
                       mpi_size, (int)ceil(log2((double)mpi_size)));
        }
#endif
        if (mpi_rank == 0)
            final_result = picluster::core::finalize_pi(total_S, digits);
#else
        final_result = picluster::core::compute_pi_cpu(digits, 1000, progress_cb);
        for (auto* ch : my_chunks)
            chunks.set_status(ch->chunk_id, picluster::storage::ChunkStatus::COMPUTED);
#endif
    }

    // --- CHECKPOINT ---
    tracker.set_phase(picluster::progress::Phase::CHECKPOINT, "Saving checkpoint...");
    std::filesystem::create_directories(plan.checkpoint_path);
    { std::ofstream cf(plan.checkpoint_path + "/chunks.json"); if (cf) cf << chunks.to_json(); }

    // --- OUTPUT ---
    if (mpi_rank == 0) {
        tracker.set_phase(picluster::progress::Phase::FINALIZE, "Writing output...");
        std::ofstream ofs(output);
        if (ofs) { ofs << final_result << "\n"; ofs.close(); }
        tracker.set_phase(picluster::progress::Phase::DONE, "Complete");
        if (verbose) { fprintf(stderr, "\n"); tracker.render_terminal(); fprintf(stderr, "\n"); }
        printf("\nWrote %lld digits to %s (%lld chunks done, %lld failed)\n",
               (long long)digits, output.c_str(),
               (long long)chunks.completed_chunks(), (long long)chunks.failed_chunks());
    }

#ifdef PICLUSTER_HAVE_MPI
    if (mpi_size > 1) MPI_Finalize();
#endif
    return 0;
}

// ============================================================
// RESUME
// ============================================================
static int cmd_resume(const std::string& checkpoint_path, const std::string& backend,
                      const std::string& output, bool verbose) {
    if (checkpoint_path.empty()) {
        std::cerr << "Error: --checkpoint path required\n"; return 1;
    }
    std::string chunks_json_path = checkpoint_path + "/chunks.json";
    if (!std::filesystem::exists(chunks_json_path)) {
        std::cerr << "No chunks.json found at " << checkpoint_path << "\n";
        return 1;
    }
    std::ifstream f(chunks_json_path);
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();
    printf("Checkpoint loaded from %s\n", checkpoint_path.c_str());

    // Parse chunk metadata from JSON
    struct ResumeChunk { std::int64_t id, start, end; std::string status; };
    std::vector<ResumeChunk> all_chunks;
    std::int64_t digits_target = 0;

    // Simple JSON parser for our known format
    std::size_t pos = 0;
    while ((pos = json.find("\"id\":", pos)) != std::string::npos) {
        ResumeChunk rc;
        rc.id = std::stoll(json.substr(pos + 5, json.find(',', pos + 5) - pos - 5));
        auto sp = json.find("\"start\":", pos);
        if (sp != std::string::npos) rc.start = std::stoll(json.substr(sp + 8, 20));
        auto ep = json.find("\"end\":", pos);
        if (ep != std::string::npos) rc.end = std::stoll(json.substr(ep + 6, 20));
        auto st = json.find("\"status\":\"", pos);
        if (st != std::string::npos) {
            auto se = json.find("\"", st + 10);
            rc.status = json.substr(st + 10, se - st - 10);
        }
        all_chunks.push_back(rc);
        pos += 10;
    }

    // Find incomplete chunks
    std::vector<ResumeChunk> incomplete;
    for (auto& c : all_chunks) {
        if (c.status != "computed" && c.status != "merged" && c.status != "checkpointed")
            incomplete.push_back(c);
    }

    printf("Chunks: %zu total, %zu completed, %zu incomplete\n",
           all_chunks.size(), all_chunks.size() - incomplete.size(), incomplete.size());

    if (incomplete.empty()) {
        printf("All chunks completed. No recomputation needed.\n");
        return 0;
    }

    // Estimate digits from chunk ranges
    if (!all_chunks.empty()) {
        std::int64_t max_end = 0;
        for (auto& c : all_chunks) if (c.end > max_end) max_end = c.end;
        digits_target = max_end * 14; // rough: 14 digits per Chudnovsky term
    }

    printf("Recomputing %zu incomplete chunks (est. %lld digits)...\n",
           incomplete.size(), (long long)digits_target);

    // Recompute only incomplete chunks
#ifdef PICLUSTER_HAVE_GMP
    mp_bitcnt_t prec = static_cast<mp_bitcnt_t>((digits_target + 50) * 3.4);
    mpf_set_default_prec(prec);
    mpf_class resumed_S(0);

    picluster::progress::ProgressTracker tracker;
    tracker.set_target(digits_target);
    tracker.set_phase(picluster::progress::Phase::COMPUTE_LOCAL, "Resuming incomplete chunks...");

    auto progress_cb = [&](double frac, const std::string& msg, std::int64_t d) {
        tracker.update(frac, d, msg);
        if (verbose) tracker.render_terminal();
    };

    for (std::size_t i = 0; i < incomplete.size(); i++) {
        auto& ic = incomplete[i];
        printf("  Recomputing chunk %lld [%lld, %lld)...\n",
               (long long)ic.id, (long long)ic.start, (long long)ic.end);
        mpf_class chunk_S;
        picluster::core::compute_partial_sum(ic.start, ic.end, digits_target, chunk_S, progress_cb);
        resumed_S += chunk_S;
    }

    // Load completed chunks' persisted sums
    mpf_class completed_S(0);
    int loaded_count = 0;
    for (auto& c : all_chunks) {
        if (c.status == "computed" || c.status == "merged" || c.status == "checkpointed") {
            std::string sum = picluster::storage::load_chunk_sum(checkpoint_path, c.id);
            if (!sum.empty()) {
                completed_S += mpf_class(sum);
                loaded_count++;
            }
        }
    }
    printf("Loaded %d completed chunk sums from checkpoint\n", loaded_count);

    // Merge all: completed (loaded) + incomplete (just recomputed)
    mpf_class total_S = completed_S + resumed_S;

    // Finalize pi = C / S
    std::string pi_result = picluster::core::finalize_pi(total_S, digits_target);
    if (!pi_result.empty() && !output.empty()) {
        std::ofstream ofs(output);
        if (ofs) { ofs << pi_result << "\n"; ofs.close(); }
        printf("Wrote %lld digits of pi to %s (resumed)\n", (long long)digits_target, output.c_str());
    } else {
        printf("Resume complete. %d loaded + %zu recomputed = %zu total chunks merged.\n",
               loaded_count, incomplete.size(), all_chunks.size());
    }

    tracker.set_phase(picluster::progress::Phase::DONE, "Resume complete");
    if (verbose) { fprintf(stderr, "\n"); tracker.render_terminal(); fprintf(stderr, "\n"); }
#else
    printf("GMP not available. Cannot recompute chunks.\n");
#endif
    return 0;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 2) { print_usage(); return 0; }

    std::string cmd = argv[1];
    std::int64_t digits = 0;
    std::string backend = "auto";
    std::string output = "pi.txt";
    std::string checkpoint_path;
    std::string scratch_override;
    std::size_t chunk_terms = 0;
    bool verbose = false, json = false, dry_run = false, confirm_flag = false;
    double max_ram = 0.70;

    for (int i = 2; i < argc; i++) {
        if ((!strcmp(argv[i],"-d") || !strcmp(argv[i],"--digits")) && i+1<argc) digits = std::atoll(argv[++i]);
        else if ((!strcmp(argv[i],"-b") || !strcmp(argv[i],"--backend")) && i+1<argc) backend = argv[++i];
        else if ((!strcmp(argv[i],"-o") || !strcmp(argv[i],"--output")) && i+1<argc) output = argv[++i];
        else if ((!strcmp(argv[i],"-c") || !strcmp(argv[i],"--checkpoint")) && i+1<argc) checkpoint_path = argv[++i];
        else if (!strcmp(argv[i],"--chunk") && i+1<argc) chunk_terms = std::atoll(argv[++i]);
        else if (!strcmp(argv[i],"--scratch") && i+1<argc) scratch_override = argv[++i];
        else if (!strcmp(argv[i],"--max-ram") && i+1<argc) max_ram = std::atof(argv[++i]);
        else if (!strcmp(argv[i],"--dry-run")) dry_run = true;
        else if (!strcmp(argv[i],"--confirm")) confirm_flag = true;
        else if (!strcmp(argv[i],"--json")) json = true;
        else if (!strcmp(argv[i],"-v") || !strcmp(argv[i],"--verbose")) verbose = true;
    }

    if (cmd == "detect") return cmd_detect(json);
    if (cmd == "bench") return cmd_bench(json);
    if (cmd == "doctor") return cmd_doctor();
    if (cmd == "estimate") return cmd_estimate(digits);
    if (cmd == "run") return cmd_run(digits, backend, output, chunk_terms, scratch_override, dry_run, confirm_flag, max_ram, verbose);
    if (cmd == "resume") return cmd_resume(checkpoint_path, backend, output, verbose);

    std::cerr << "Unknown command: " << cmd << "\n";
    print_usage();
    return 1;
}
