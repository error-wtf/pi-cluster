// binary_splitting.cpp — MPI-distributed Binary Splitting for Chudnovsky Pi
// O(n log^2 n) via recursive product tree, distributable across MPI ranks
//
// Architecture for MPI distribution:
//   1. Rank 0 partitions [0, N) into equal sub-ranges, one per rank
//   2. Each rank computes bs_recursive on its sub-range → local BSNode {P,Q,T}
//   3. Pairwise tree-reduce merges BSNodes across ranks (same as Chudnovsky merge rule)
//   4. Rank 0 gets final BSNode, computes pi = Q*C / (T + A*Q)
//
// The merge rule P(a,b)=P(a,m)*P(m,b), Q(a,b)=Q(a,m)*Q(m,b),
// T(a,b)=T(a,m)*Q(m,b)+P(a,m)*T(m,b) is associative, so tree-reduce works.

#include "picluster/core/binary_splitting.h"
#include <cmath>
#include <string>
#include <vector>
#include <cstring>

#ifdef PICLUSTER_HAVE_GMP
#include <gmp.h>
#include <gmpxx.h>
#endif

#ifdef PICLUSTER_HAVE_MPI
#include <mpi.h>
#endif

#ifdef PICLUSTER_HAVE_GMP

namespace picluster { namespace core {

static const long A_CHUD = 13591409L;
static const long B_CHUD = 545140134L;
static const char* C3_24_STR = "10939058860032000";

struct BSNode {
    mpz_class P, Q, T;
};

static BSNode bs_base_case(std::int64_t a) {
    BSNode n;
    if (a == 0) {
        n.P = 1; n.Q = 1; n.T = mpz_class(A_CHUD);
    } else {
        n.P = -mpz_class(6*a-5) * mpz_class(2*a-1) * mpz_class(6*a-1);
        mpz_class a3(a); a3 = a3*a3*a3;
        n.Q = a3 * mpz_class(C3_24_STR);
        n.T = n.P * (mpz_class(A_CHUD) + mpz_class(B_CHUD) * mpz_class(a));
    }
    return n;
}

#ifdef PICLUSTER_HAVE_CUDA
extern void gpu_multiply_mpz(mpz_t result, const mpz_t a, const mpz_t b);
static const size_t GPU_MUL_THRESH = 50000;
static void big_mul(mpz_class& r, const mpz_class& a, const mpz_class& b) {
    if (mpz_size(a.get_mpz_t())+mpz_size(b.get_mpz_t()) > GPU_MUL_THRESH)
        gpu_multiply_mpz(r.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
    else r = a * b;
}
#else
static void big_mul(mpz_class& r, const mpz_class& a, const mpz_class& b) { r = a * b; }
#endif

static BSNode bs_merge_nodes(const BSNode& L, const BSNode& R) {
    BSNode n;
    big_mul(n.P, L.P, R.P);
    big_mul(n.Q, L.Q, R.Q);
    mpz_class tq, pt;
    big_mul(tq, L.T, R.Q);
    big_mul(pt, L.P, R.T);
    n.T = tq + pt;
    return n;
}

static BSNode bs_recursive(std::int64_t a, std::int64_t b) {
    if (b - a == 1) return bs_base_case(a);
    std::int64_t m = (a + b) / 2;
    return bs_merge_nodes(bs_recursive(a, m), bs_recursive(m, b));
}

// --- GMP serialization for MPI transfer of BSNode ---
static std::vector<char> serialize_mpz(const mpz_class& val) {
    char* raw = mpz_get_str(nullptr, 62, val.get_mpz_t()); // base 62 for compactness
    std::string s(raw);
    free(raw);
    return std::vector<char>(s.begin(), s.end());
}

static mpz_class deserialize_mpz(const std::vector<char>& data) {
    std::string s(data.begin(), data.end());
    mpz_class val;
    val.set_str(s, 62);
    return val;
}

static std::vector<char> serialize_bsnode(const BSNode& node) {
    auto sp = serialize_mpz(node.P);
    auto sq = serialize_mpz(node.Q);
    auto st = serialize_mpz(node.T);
    // Format: len_P|P|len_Q|Q|len_T|T
    std::vector<char> out;
    auto push_field = [&](const std::vector<char>& field) {
        std::int32_t len = (std::int32_t)field.size();
        out.insert(out.end(), (char*)&len, (char*)&len + 4);
        out.insert(out.end(), field.begin(), field.end());
    };
    push_field(sp);
    push_field(sq);
    push_field(st);
    return out;
}

static BSNode deserialize_bsnode(const std::vector<char>& data) {
    BSNode node;
    std::size_t pos = 0;
    auto read_field = [&]() -> std::vector<char> {
        std::int32_t len;
        std::memcpy(&len, data.data() + pos, 4); pos += 4;
        std::vector<char> field(data.data() + pos, data.data() + pos + len);
        pos += len;
        return field;
    };
    node.P = deserialize_mpz(read_field());
    node.Q = deserialize_mpz(read_field());
    node.T = deserialize_mpz(read_field());
    return node;
}

// --- MPI-distributed binary splitting ---
std::string compute_pi_binary_splitting_mpi(std::int64_t digits, int mpi_rank, int mpi_size,
                                             ProgressCallback cb) {
    if (digits <= 0) return "";

    mp_bitcnt_t prec = static_cast<mp_bitcnt_t>((digits + 50) * 3.4);
    mpf_set_default_prec(prec);

    std::int64_t N = digits / 14 + 10;

    // Partition [0, N) across ranks
    std::int64_t chunk = N / mpi_size;
    std::int64_t remainder = N % mpi_size;
    std::int64_t my_start = mpi_rank * chunk + std::min((std::int64_t)mpi_rank, remainder);
    std::int64_t my_end = my_start + chunk + (mpi_rank < remainder ? 1 : 0);

    if (cb && mpi_rank == 0)
        cb(0.05, "MPI BS: computing local subtree", 0);

    // Each rank computes its local subtree
    BSNode local = bs_recursive(my_start, my_end);

    if (cb && mpi_rank == 0)
        cb(0.50, "MPI BS: tree-reduce merge across ranks", my_end * 7);

#ifdef PICLUSTER_HAVE_MPI
    // Pairwise tree-reduce of BSNodes
    // Same pattern as the partial-sum tree-reduce but with BSNode merge rule
    for (int step = 1; step < mpi_size; step *= 2) {
        if (mpi_rank % (2 * step) == 0) {
            int partner = mpi_rank + step;
            if (partner < mpi_size) {
                // Receive partner's BSNode
                int partner_sz = 0;
                MPI_Recv(&partner_sz, 1, MPI_INT, partner, 300+step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<char> rbuf(partner_sz);
                MPI_Recv(rbuf.data(), partner_sz, MPI_CHAR, partner, 400+step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                BSNode partner_node = deserialize_bsnode(rbuf);
                // Merge: local is left, partner is right (partner has higher range)
                local = bs_merge_nodes(local, partner_node);
            }
        } else if (mpi_rank % (2 * step) == step) {
            int partner = mpi_rank - step;
            auto buf = serialize_bsnode(local);
            int sz = (int)buf.size();
            MPI_Send(&sz, 1, MPI_INT, partner, 300+step, MPI_COMM_WORLD);
            MPI_Send(buf.data(), sz, MPI_CHAR, partner, 400+step, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (mpi_rank == 0 && cb)
        cb(0.85, "MPI BS: tree-reduce complete, finalizing pi", N * 10);
#endif

    // Rank 0 has the final merged BSNode → compute pi
    std::string result;
    if (mpi_rank == 0) {
        mpf_class Q_f(local.Q);
        mpf_class T_f(local.T);
        mpf_class denom = T_f + mpf_class(A_CHUD) * Q_f;
        mpf_class sqrt10005 = sqrt(mpf_class(10005));
        mpf_class C_val = mpf_class(426880) * sqrt10005;
        mpf_class pi = Q_f * C_val / denom;

        std::size_t bufsize = static_cast<std::size_t>(digits) + 20;
        std::vector<char> buf(bufsize);
        gmp_snprintf(buf.data(), bufsize, "%.*Ff", static_cast<int>(digits), pi.get_mpf_t());
        result = std::string(buf.data());
        if (cb) cb(1.0, "MPI BS: done", digits);
    }
    return result;
}

// --- Single-node binary splitting (unchanged) ---
std::string compute_pi_binary_splitting(std::int64_t digits, ProgressCallback cb) {
    if (digits <= 0) return "";
    mp_bitcnt_t prec = static_cast<mp_bitcnt_t>((digits + 50) * 3.4);
    mpf_set_default_prec(prec);
    std::int64_t N = digits / 14 + 10;
    if (cb) cb(0.05, "Binary splitting: computing product tree", 0);
    BSNode root = bs_recursive(0, N);
    if (cb) cb(0.80, "Binary splitting: computing sqrt(10005)", N * 7);
    mpf_class Q_f(root.Q); mpf_class T_f(root.T);
    mpf_class denom = T_f + mpf_class(A_CHUD) * Q_f;
    mpf_class pi = mpf_class(426880) * sqrt(mpf_class(10005)) * Q_f / denom;
    if (cb) cb(0.95, "Binary splitting: formatting", digits);
    std::size_t bufsize = static_cast<std::size_t>(digits) + 20;
    std::vector<char> buf(bufsize);
    gmp_snprintf(buf.data(), bufsize, "%.*Ff", static_cast<int>(digits), pi.get_mpf_t());
    if (cb) cb(1.0, "Binary splitting: done", digits);
    return std::string(buf.data());
}

BSState binary_split(std::int64_t a, std::int64_t b) { return {a, b}; }
BSState merge_bs(const BSState& l, const BSState& r) { return {l.a, r.b}; }

bool should_use_binary_splitting(std::int64_t digits) {
    return digits > 50000;
}

}} // namespace

#else
// No-GMP stubs
namespace picluster { namespace core {
std::string compute_pi_binary_splitting(std::int64_t, ProgressCallback) { return ""; }
std::string compute_pi_binary_splitting_mpi(std::int64_t, int, int, ProgressCallback) { return ""; }
BSState binary_split(std::int64_t a, std::int64_t b) { return {a, b}; }
BSState merge_bs(const BSState& l, const BSState& r) { return {l.a, r.b}; }
bool should_use_binary_splitting(std::int64_t) { return false; }
}} // namespace
#endif
