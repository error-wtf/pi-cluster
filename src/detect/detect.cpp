// detect.cpp — System hardware detection for pi-cluster
#include "picluster/detect/detect.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <array>
#include <memory>
#include <algorithm>

#if defined(__linux__)
#include <sys/sysinfo.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

namespace picluster { namespace detect {

static std::string exec_cmd(const char* cmd) {
    std::array<char, 256> buf;
    std::string out;
#if defined(_WIN32)
    std::unique_ptr<FILE, decltype(&_pclose)> p(_popen(cmd, "r"), _pclose);
#else
    std::unique_ptr<FILE, decltype(&pclose)> p(popen(cmd, "r"), pclose);
#endif
    if (!p) return "";
    while (fgets(buf.data(), buf.size(), p.get()))
        out += buf.data();
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r'))
        out.pop_back();
    return out;
}

static std::string get_env(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : "";
}

static std::string read_file_line(const char* path) {
    std::ifstream f(path);
    std::string line;
    if (f) std::getline(f, line);
    return line;
}

// --- CPU ---
static CpuInfo detect_cpu() {
    CpuInfo ci;
#if defined(__linux__)
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    int logical = 0;
    while (std::getline(f, line)) {
        if (line.find("model name") == 0 && ci.model_name.empty()) {
            auto pos = line.find(':');
            if (pos != std::string::npos)
                ci.model_name = line.substr(pos + 2);
        }
        if (line.find("processor") == 0) logical++;
    }
    ci.logical_cores = logical;
    // physical cores via lscpu
    std::string cores = exec_cmd("lscpu 2>/dev/null | grep '^CPU(s):' | awk '{print $2}'");
    if (!cores.empty()) ci.logical_cores = std::atoi(cores.c_str());
    std::string phys = exec_cmd("lscpu 2>/dev/null | grep 'Core(s) per socket' | awk '{print $NF}'");
    std::string socks = exec_cmd("lscpu 2>/dev/null | grep 'Socket(s)' | awk '{print $NF}'");
    if (!phys.empty() && !socks.empty()) {
        ci.physical_cores = std::atoi(phys.c_str()) * std::atoi(socks.c_str());
        ci.sockets = std::atoi(socks.c_str());
    }
    ci.architecture = exec_cmd("uname -m 2>/dev/null");
#elif defined(_WIN32)
    ci.model_name = exec_cmd("wmic cpu get name /value 2>nul");
    ci.logical_cores = std::atoi(get_env("NUMBER_OF_PROCESSORS").c_str());
    ci.physical_cores = ci.logical_cores; // approximation
    ci.architecture = "x86_64";
#endif
    return ci;
}

// --- Memory ---
static MemInfo detect_mem() {
    MemInfo mi;
#if defined(__linux__)
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        mi.total_ram_bytes = (std::size_t)si.totalram * si.mem_unit;
        mi.free_ram_bytes = (std::size_t)si.freeram * si.mem_unit;
        mi.total_swap_bytes = (std::size_t)si.totalswap * si.mem_unit;
    }
#elif defined(_WIN32)
    // simplified — full impl in resources.cpp
#endif
    return mi;
}

// --- NUMA ---
static NumaInfo detect_numa() {
    NumaInfo ni;
#if defined(__linux__)
    std::string nodes = exec_cmd("ls -d /sys/devices/system/node/node* 2>/dev/null | wc -l");
    ni.num_nodes = std::max(1, std::atoi(nodes.c_str()));
#endif
    return ni;
}

// --- GPU ---
static GpuInfo detect_gpu() {
    GpuInfo gi;
    std::string count_str = exec_cmd("nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null");
    if (count_str.empty()) { gi.available = false; return gi; }
    gi.available = true;
    gi.count = std::atoi(count_str.c_str());
    // names
    std::string names = exec_cmd("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null");
    std::istringstream ns(names);
    std::string n;
    while (std::getline(ns, n)) { if (!n.empty()) gi.names.push_back(n); }
    // vram
    std::string vram = exec_cmd("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null");
    std::istringstream vs(vram);
    std::string v;
    while (std::getline(vs, v)) {
        if (!v.empty()) gi.vram_bytes.push_back((std::size_t)std::atoll(v.c_str()) * 1024 * 1024);
    }
    gi.driver_version = exec_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null");
    gi.cuda_version = exec_cmd("nvcc --version 2>/dev/null | grep 'release' | sed 's/.*release //' | sed 's/,.*//'");
    return gi;
}

// --- Scratch ---
static ScratchInfo detect_scratch() {
    ScratchInfo si;
    // Priority: TMPDIR > SLURM_TMPDIR > /tmp
    std::string tmp = get_env("SLURM_TMPDIR");
    if (tmp.empty()) tmp = get_env("TMPDIR");
    if (tmp.empty()) tmp = "/tmp";
    si.path = tmp;
    si.is_local = true; // assume local for TMPDIR
#if defined(__linux__)
    struct statvfs buf;
    if (statvfs(tmp.c_str(), &buf) == 0) {
        si.free_bytes = (std::uint64_t)buf.f_bsize * buf.f_bavail;
    }
    // detect fs type
    si.fs_type = exec_cmd(("df -T " + tmp + " 2>/dev/null | tail -1 | awk '{print $2}'").c_str());
#endif
    return si;
}

// --- Slurm ---
static SlurmInfo detect_slurm() {
    SlurmInfo si;
    si.job_id = get_env("SLURM_JOB_ID");
    si.in_slurm_job = !si.job_id.empty();
    if (si.in_slurm_job) {
        si.nnodes = std::atoi(get_env("SLURM_NNODES").c_str());
        si.ntasks = std::atoi(get_env("SLURM_NTASKS").c_str());
        si.cpus_per_task = std::atoi(get_env("SLURM_CPUS_PER_TASK").c_str());
        si.nodelist = get_env("SLURM_JOB_NODELIST");
        si.partition = get_env("SLURM_JOB_PARTITION");
        si.tmpdir = get_env("SLURM_TMPDIR");
    }
    return si;
}

// --- Main entry ---
SystemProfile detect_system() {
    SystemProfile p;
    p.cpu = detect_cpu();
    p.mem = detect_mem();
    p.numa = detect_numa();
    p.gpu = detect_gpu();
    p.scratch = detect_scratch();
    p.slurm = detect_slurm();
#if defined(__linux__)
    char hn[256] = {};
    gethostname(hn, sizeof(hn));
    p.hostname = hn;
    p.os_version = exec_cmd("cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"'");
#elif defined(_WIN32)
    p.hostname = get_env("COMPUTERNAME");
    p.os_version = "Windows";
#endif
    return p;
}

static std::string bytes_human(std::uint64_t b) {
    if (b >= (1ULL << 30)) return std::to_string(b / (1ULL << 30)) + " GB";
    if (b >= (1ULL << 20)) return std::to_string(b / (1ULL << 20)) + " MB";
    return std::to_string(b) + " B";
}

void print_profile(const SystemProfile& p) {
    printf("=== pi-cluster System Profile ===\n");
    printf("Hostname:    %s\n", p.hostname.c_str());
    printf("OS:          %s\n", p.os_version.c_str());
    printf("CPU:         %s\n", p.cpu.model_name.c_str());
    printf("Cores:       %d physical, %d logical (%d sockets)\n",
           p.cpu.physical_cores, p.cpu.logical_cores, p.cpu.sockets);
    printf("RAM:         %s total, %s free\n",
           bytes_human(p.mem.total_ram_bytes).c_str(),
           bytes_human(p.mem.free_ram_bytes).c_str());
    printf("Swap:        %s\n", bytes_human(p.mem.total_swap_bytes).c_str());
    printf("NUMA:        %d nodes\n", p.numa.num_nodes);
    if (p.gpu.available) {
        printf("GPU:         %d device(s)\n", p.gpu.count);
        for (int i = 0; i < (int)p.gpu.names.size(); i++)
            printf("  GPU %d:     %s (%s VRAM)\n", i, p.gpu.names[i].c_str(),
                   i < (int)p.gpu.vram_bytes.size() ? bytes_human(p.gpu.vram_bytes[i]).c_str() : "?");
        if (!p.gpu.cuda_version.empty())
            printf("CUDA:        %s (driver %s)\n", p.gpu.cuda_version.c_str(), p.gpu.driver_version.c_str());
    } else {
        printf("GPU:         not detected\n");
    }
    printf("Scratch:     %s (%s free, %s, %s)\n",
           p.scratch.path.c_str(), bytes_human(p.scratch.free_bytes).c_str(),
           p.scratch.fs_type.c_str(), p.scratch.is_local ? "local" : "shared");
    if (p.slurm.in_slurm_job) {
        printf("Slurm:       job %s, %d nodes, %d tasks, partition=%s\n",
               p.slurm.job_id.c_str(), p.slurm.nnodes, p.slurm.ntasks, p.slurm.partition.c_str());
    } else {
        printf("Slurm:       not in a Slurm job\n");
    }
    printf("=================================\n");
}

std::string profile_to_json(const SystemProfile& p) {
    std::ostringstream j;
    j << "{";
    j << "\"hostname\":\"" << p.hostname << "\",";
    j << "\"cpu\":{\"model\":\"" << p.cpu.model_name << "\",\"physical\":" << p.cpu.physical_cores << ",\"logical\":" << p.cpu.logical_cores << "},";
    j << "\"ram\":{\"total\":" << p.mem.total_ram_bytes << ",\"free\":" << p.mem.free_ram_bytes << "},";
    j << "\"gpu\":{\"available\":" << (p.gpu.available ? "true" : "false") << ",\"count\":" << p.gpu.count << "},";
    j << "\"scratch\":{\"path\":\"" << p.scratch.path << "\",\"free\":" << p.scratch.free_bytes << ",\"local\":" << (p.scratch.is_local ? "true" : "false") << "},";
    j << "\"slurm\":{\"active\":" << (p.slurm.in_slurm_job ? "true" : "false") << ",\"job_id\":\"" << p.slurm.job_id << "\"}";
    j << "}";
    return j.str();
}

std::string profile_to_markdown(const SystemProfile& p) {
    std::ostringstream m;
    m << "# System Profile: " << p.hostname << "\n\n";
    m << "| Property | Value |\n|----------|-------|\n";
    m << "| CPU | " << p.cpu.model_name << " |\n";
    m << "| Cores | " << p.cpu.physical_cores << " physical / " << p.cpu.logical_cores << " logical |\n";
    m << "| RAM | " << bytes_human(p.mem.total_ram_bytes) << " total, " << bytes_human(p.mem.free_ram_bytes) << " free |\n";
    m << "| GPU | " << (p.gpu.available ? std::to_string(p.gpu.count) + " device(s)" : "none") << " |\n";
    m << "| Scratch | " << p.scratch.path << " (" << bytes_human(p.scratch.free_bytes) << " free) |\n";
    m << "| Slurm | " << (p.slurm.in_slurm_job ? "Job " + p.slurm.job_id : "inactive") << " |\n";
    return m.str();
}

}} // namespace picluster::detect
