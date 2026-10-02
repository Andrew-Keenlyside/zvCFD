"""CUDA source for the fast finite-volume kernels (double precision).

The same discrete equations as :mod:`zvcfd.fv._cuda` (the specification is
``docs/spec/fv_numerics.md``); ``tests/test_fv_gpu.py`` holds both kernel
sets to the CPU reference. What changes is how the work is laid out:

- **Slots.** Elements of one type are stored in colour order ("slots"),
  so the elements of one launch are consecutive: their connectivity,
  mass flows, block positions and (stored) geometry are read and written
  in contiguous, coalesced runs.
- **Threads per block, not per element.** An element's matrix has
  ``n × n`` 4 × 4 blocks; the assembly gives each ``(row node, column
  node)`` pair its own thread, which accumulates its block over the flux
  points at the row node in registers and writes it once. There is no
  per-thread element matrix (which spilled to local memory) and 16–64
  times the parallelism.
- **Shared memory.** A thread block holds a few elements. Their geometry
  (flux-point areas and centroids, shape-function gradients, sub-volumes)
  is copied to shared memory in one coalesced sweep, or rebuilt there
  cooperatively (one thread per flux point) when it is not stored; the
  node values the kernels need are gathered once per element.
- **Topology in constant memory.** Each type's tables (flux-point edges,
  shape functions at the flux points, the flux points at each node) are
  compiled in.

Colouring (no two elements of a launch share a node) keeps writes free of
atomics, so results are deterministic.

Per type, the source is formatted with ``%(n)d`` nodes, ``%(nip)d`` flux
points, ``%(npts)d`` dual points, ``%(epb)d`` elements per thread block,
``%(stored)d`` (geometry read from ``georec``) and ``%(constg)d`` (one
gradient set per element: tetrahedra); the tables come from
:func:`tables`.
"""

import numpy as np

HEADER = r"""
#define N_NODES %(n)d
#define N_IP %(nip)d
#define N_PTS %(npts)d
#define EPB %(epb)d
#define STORED %(stored)d
#define CONST_G %(constg)d
#define T_ELEM (N_NODES * N_NODES)
#define NG (CONST_G ? 1 : N_IP)
#define OFF_A 0
#define OFF_X (N_IP * 3)
#define OFF_G (2 * N_IP * 3)
#define OFF_S (2 * N_IP * 3 + NG * N_NODES * 3)
#define REC (OFF_S + N_NODES)
#define MAXI 8
%(tables)s

// the tables in shared memory: threads of a warp index them by their own node or flux
// point, and divergent constant-memory reads serialise
struct Tab {
    int ips[N_NODES][MAXI];
    double sgn[N_NODES][MAXI];
    double nip[N_IP][N_NODES];
    int edge[N_IP][2];
#if !STORED
    int tris[N_IP][3];
    double wpt[N_PTS][N_NODES];
    double dn[N_IP][N_NODES][3];
#endif
};

__device__ inline void load_tab(Tab &T) {
    for (int i = threadIdx.x; i < N_NODES * MAXI; i += blockDim.x) {
        (&T.ips[0][0])[i] = (&NODE_IPS[0][0])[i];
        (&T.sgn[0][0])[i] = (&NODE_SGN[0][0])[i];
    }
    for (int i = threadIdx.x; i < N_IP * N_NODES; i += blockDim.x) (&T.nip[0][0])[i] = (&NIP[0][0])[i];
    for (int i = threadIdx.x; i < N_IP * 2; i += blockDim.x) (&T.edge[0][0])[i] = (&EDGE[0][0])[i];
#if !STORED
    for (int i = threadIdx.x; i < N_IP * 3; i += blockDim.x) (&T.tris[0][0])[i] = (&TRIS[0][0])[i];
    for (int i = threadIdx.x; i < N_PTS * N_NODES; i += blockDim.x) (&T.wpt[0][0])[i] = (&WPT[0][0])[i];
    for (int i = threadIdx.x; i < N_IP * N_NODES * 3; i += blockDim.x) (&T.dn[0][0][0])[i] = (&DN[0][0][0])[i];
#endif
}

__device__ inline const double *Gp(const double *rec, int s, int a) {
    return rec + OFF_G + ((CONST_G ? 0 : s) * N_NODES + a) * 3;
}

__device__ inline void inv3f(const double J[3][3], double I[3][3]) {
    double a = J[0][0], b = J[0][1], c = J[0][2];
    double d = J[1][0], e = J[1][1], f = J[1][2];
    double g = J[2][0], h = J[2][1], i = J[2][2];
    double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    double r = 1.0 / (a * A + b * B + c * C);
    I[0][0] = A * r; I[0][1] = -(b * i - c * h) * r; I[0][2] = (b * f - c * e) * r;
    I[1][0] = B * r; I[1][1] = (a * i - c * g) * r;  I[1][2] = -(a * f - c * d) * r;
    I[2][0] = C * r; I[2][1] = -(a * h - b * g) * r; I[2][2] = (a * e - b * d) * r;
}

__device__ inline double visc(const double gu[3][3], int rheo, double mu0, double muinf,
                              double lam, double ca, double cn) {
    if (rheo == 0) return mu0;
    double ss = 0.0;
    for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) { double s = 0.5 * (gu[j][k] + gu[k][j]); ss += s * s; }
    double g = sqrt(2.0 * ss);
    return muinf + (mu0 - muinf) * pow(1.0 + pow(lam * g, ca), (cn - 1.0) / ca);
}

// Geometry of one element into rec[REC], by the element's T_ELEM threads (t = 0..T_ELEM-1):
// copied from the stored record, or rebuilt from the node coordinates xs[N_NODES][3].
// Every thread of the block must call it (it synchronises).
__device__ void element_geometry(const double *georec, long long slot, bool active,
                                 const double *xs, double *rec, int t, const Tab &T) {
#if STORED
    if (active)
        for (int i = t; i < REC; i += T_ELEM) rec[i] = georec[slot * REC + i];
    __syncthreads();
#else
    if (active) {
        for (int s = t; s < N_IP; s += T_ELEM) {
            double p[3][3];
            for (int v = 0; v < 3; ++v) {
                int q = T.tris[s][v];
                for (int k = 0; k < 3; ++k) {
                    double acc = 0.0;
                    for (int a = 0; a < N_NODES; ++a) acc += T.wpt[q][a] * xs[a * 3 + k];
                    p[v][k] = acc;
                }
            }
            double u[3], w[3];
            for (int k = 0; k < 3; ++k) { u[k] = p[1][k] - p[0][k]; w[k] = p[2][k] - p[0][k]; }
            rec[OFF_A + s * 3 + 0] = 0.5 * (u[1] * w[2] - u[2] * w[1]);
            rec[OFF_A + s * 3 + 1] = 0.5 * (u[2] * w[0] - u[0] * w[2]);
            rec[OFF_A + s * 3 + 2] = 0.5 * (u[0] * w[1] - u[1] * w[0]);
            for (int k = 0; k < 3; ++k) rec[OFF_X + s * 3 + k] = (p[0][k] + p[1][k] + p[2][k]) / 3.0;
            if (!CONST_G || s == 0) {
                double J[3][3], I[3][3];
                for (int k = 0; k < 3; ++k)
                    for (int d = 0; d < 3; ++d) {
                        double acc = 0.0;
                        for (int a = 0; a < N_NODES; ++a) acc += xs[a * 3 + k] * T.dn[s][a][d];
                        J[k][d] = acc;
                    }
                inv3f(J, I);
                double *G = rec + OFF_G + (CONST_G ? 0 : s) * N_NODES * 3;
                for (int a = 0; a < N_NODES; ++a)
                    for (int k = 0; k < 3; ++k)
                        G[a * 3 + k] = T.dn[s][a][0] * I[0][k] + T.dn[s][a][1] * I[1][k] + T.dn[s][a][2] * I[2][k];
            }
        }
    }
    __syncthreads();
    if (active && t < N_NODES) {                  // sub-control volumes (divergence theorem)
        double v = 0.0;
        for (int i = 0; i < MAXI; ++i) {
            int s = T.ips[t][i];
            if (s < 0) break;
            double d = 0.0;
            for (int k = 0; k < 3; ++k) d += (rec[OFF_X + s * 3 + k] - xs[t * 3 + k]) * rec[OFF_A + s * 3 + k];
            v += T.sgn[t][i] * d / 3.0;
        }
        rec[OFF_S + t] = v;
    }
    __syncthreads();
#endif
}
"""

# the geometry record of every slot, once (compiled with STORED 0)
GEOBUILD = r"""
extern "C" __global__ void geobuild(const int *conn, const double *X, long long nslots,
                                    double *georec) {
    __shared__ double s_x[EPB][N_NODES * 3];
    __shared__ double s_rec[EPB][REC];
    __shared__ Tab T;
    load_tab(T);
    int le = threadIdx.x / T_ELEM, t = threadIdx.x %% T_ELEM;
    long long slot = (long long)blockIdx.x * EPB + le;
    bool active = le < EPB && slot < nslots;
    if (active && t < N_NODES) {
        long long n = conn[slot * N_NODES + t];
        for (int k = 0; k < 3; ++k) s_x[le][t * 3 + k] = X[n * 3 + k];
    }
    __syncthreads();
    element_geometry(nullptr, slot, active, s_x[le < EPB ? le : 0], s_rec[le < EPB ? le : 0], t, T);
    if (active)
        for (int i = t; i < REC; i += T_ELEM) georec[slot * REC + i] = s_rec[le][i];
}
"""

# node volumes, the static (Newtonian) viscous diagonal, and the nodal gradient operator
STATICS = r"""
extern "C" __global__ void statics(const int *conn, long long base, long long count,
                                   const double *X, const double *georec, const int *pos,
                                   double mu, int transpose, double *vol, double *dvisc,
                                   double *gop) {
    __shared__ double s_x[EPB][N_NODES * 3];
    __shared__ double s_rec[EPB][REC];
    __shared__ long long s_n[EPB][N_NODES];
    __shared__ Tab T;
    load_tab(T);
    int le = threadIdx.x / T_ELEM, t = threadIdx.x %% T_ELEM;
    long long slot = base + (long long)blockIdx.x * EPB + le;
    bool active = le < EPB && slot < base + count;
    int lx = le < EPB ? le : 0;
    if (active && t < N_NODES) {
        long long n = conn[slot * N_NODES + t];
        s_n[le][t] = n;
        for (int k = 0; k < 3; ++k) s_x[le][t * 3 + k] = X[n * 3 + k];
    }
    __syncthreads();
    element_geometry(georec, slot, active, s_x[lx], s_rec[lx], t, T);
    if (!active) return;
    const double *rec = s_rec[le];
    int a = t / N_NODES, c = t %% N_NODES;
    // gradient operator: row a, column c, the SCV-weighted element gradient coefficient
    double gbar[3] = {0.0, 0.0, 0.0};
    for (int s = 0; s < NG; ++s)
        for (int k = 0; k < 3; ++k) gbar[k] += Gp(rec, s, c)[k];
    long long p = pos[slot * T_ELEM + t];
    for (int k = 0; k < 3; ++k) gop[p * 3 + k] += rec[OFF_S + a] * gbar[k] / NG;
    if (c == 0) {
        vol[s_n[le][a]] += rec[OFF_S + a];
        double d = 0.0;
        for (int i = 0; i < MAXI; ++i) {
            int s = T.ips[a][i];
            if (s < 0) break;
            const double *A = rec + OFF_A + s * 3;
            const double *G = Gp(rec, s, a);
            double r = -mu * (G[0] * A[0] + G[1] * A[1] + G[2] * A[2]);
            if (transpose) r -= mu * G[0] * A[0];
            d += T.sgn[a][i] * r;
        }
        dvisc[s_n[le][a]] += d;
    }
}
"""

# momentum diagonal with a generalised-Newtonian viscosity (or any mdot passed in)
DIAG = r"""
extern "C" __global__ void diag(const int *conn, long long base, long long count,
                                const double *X, const double *georec, const double *mdot,
                                const double *U, int rheo, double mu0, double muinf,
                                double lam, double ca, double cn, int transpose, int stokes,
                                double *out) {
    __shared__ double s_x[EPB][N_NODES * 3];
    __shared__ double s_u[EPB][N_NODES * 3];
    __shared__ double s_rec[EPB][REC];
    __shared__ double s_mu[EPB][N_IP];
    __shared__ long long s_n[EPB][N_NODES];
    __shared__ Tab T;
    load_tab(T);
    int le = threadIdx.x / T_ELEM, t = threadIdx.x %% T_ELEM;
    long long slot = base + (long long)blockIdx.x * EPB + le;
    bool active = le < EPB && slot < base + count;
    int lx = le < EPB ? le : 0;
    if (active && t < N_NODES) {
        long long n = conn[slot * N_NODES + t];
        s_n[le][t] = n;
        for (int k = 0; k < 3; ++k) { s_x[le][t * 3 + k] = X[n * 3 + k]; s_u[le][t * 3 + k] = U[n * 3 + k]; }
    }
    __syncthreads();
    element_geometry(georec, slot, active, s_x[lx], s_rec[lx], t, T);
    const double *rec = s_rec[lx];
    if (active && t < N_IP) {
        double mu = mu0;
        if (rheo) {
            double gu[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
            for (int b = 0; b < N_NODES; ++b) {
                const double *G = Gp(rec, t, b);
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < 3; ++k) gu[j][k] += s_u[le][b * 3 + j] * G[k];
            }
            mu = visc(gu, rheo, mu0, muinf, lam, ca, cn);
        }
        s_mu[le][t] = mu;
    }
    __syncthreads();
    if (!active || t >= N_NODES) return;
    int a = t;
    double d = 0.0;
    for (int i = 0; i < MAXI; ++i) {
        int s = T.ips[a][i];
        if (s < 0) break;
        double sg = T.sgn[a][i], mu = s_mu[le][s];
        const double *A = rec + OFF_A + s * 3;
        const double *G = Gp(rec, s, a);
        double r = -mu * (G[0] * A[0] + G[1] * A[1] + G[2] * A[2]);
        if (transpose) r -= mu * G[0] * A[0];
        d += sg * r;
        if (!stokes) {
            double m = mdot[slot * N_IP + s];
            d += sg > 0 ? (m > 0 ? m : 0.0) : (m < 0 ? -m : 0.0);
        }
    }
    out[s_n[le][a]] += d;
}
"""

ASSEMBLE = r"""
// The coupled system's blocks and right-hand side (see _cuda.ASSEMBLE for the terms).
// Thread (a, c) of an element: block (row a, column c), summed over the flux points at a.
extern "C" __global__ void assemble(const int *conn, long long base, long long count,
                                    const double *X, const double *georec,
                                    const double *U, const double *mdot, const double *gradU,
                                    const double *beta, const double *gradP,
                                    const double *dnode, const double *snode, int nlev,
                                    double c1, const double *Uo1, const double *mo1,
                                    double c2, const double *Uo2, const double *mo2,
                                    double rho, int rheo, double mu0, double muinf,
                                    double lam, double ca, double cn, int transpose,
                                    int stokes, const int *own, const int *pos, double *data,
                                    double *b) {
    __shared__ double s_x[EPB][N_NODES * 3];
    __shared__ double s_u[EPB][N_NODES * 3];
    __shared__ double s_gp[EPB][N_NODES * 3];
    __shared__ double s_gu[EPB][N_NODES * 9];
    __shared__ double s_be[EPB][N_NODES * 3];
    __shared__ double s_u1[EPB][N_NODES * 3];
    __shared__ double s_u2[EPB][N_NODES * 3];
    __shared__ double s_dn[EPB][N_NODES];
    __shared__ double s_sn[EPB][N_NODES];
    __shared__ double s_rec[EPB][REC];
    __shared__ double s_ip[EPB][N_IP][3];          // mu, mdot, d
    __shared__ double s_rhs[EPB][N_IP][4];         // each flux point's rhs (momentum, continuity)
    __shared__ long long s_n[EPB][N_NODES];
    __shared__ Tab T;
    load_tab(T);
    int le = threadIdx.x / T_ELEM, t = threadIdx.x %% T_ELEM;
    long long slot = base + (long long)blockIdx.x * EPB + le;
    bool active = le < EPB && slot < base + count;
    int lx = le < EPB ? le : 0;
    // gather the element's node values (one thread per node and field group)
    if (active && t < N_NODES) {
        long long n = conn[slot * N_NODES + t];
        s_n[le][t] = n;
        for (int k = 0; k < 3; ++k) {
            s_x[le][t * 3 + k] = X[n * 3 + k];
            s_u[le][t * 3 + k] = U[n * 3 + k];
            s_gp[le][t * 3 + k] = gradP[n * 3 + k];
        }
        s_dn[le][t] = dnode[n];
        s_sn[le][t] = snode[n];
    } else if (active && t < 2 * N_NODES) {
        int a = t - N_NODES;
        long long n = conn[slot * N_NODES + a];
        if (!stokes) {
            for (int q = 0; q < 9; ++q) s_gu[le][a * 9 + q] = gradU[n * 9 + q];
            for (int k = 0; k < 3; ++k) s_be[le][a * 3 + k] = beta[n * 3 + k];
        }
        if (nlev > 0) for (int k = 0; k < 3; ++k) s_u1[le][a * 3 + k] = Uo1[n * 3 + k];
        if (nlev > 1) for (int k = 0; k < 3; ++k) s_u2[le][a * 3 + k] = Uo2[n * 3 + k];
    }
    __syncthreads();
    element_geometry(georec, slot, active, s_x[lx], s_rec[lx], t, T);
    const double *rec = s_rec[lx];
    // flux-point scalars
    if (active && t < N_IP) {
        int s = t, ia = T.edge[s][0], ib = T.edge[s][1];
        double mu = mu0;
        if (rheo) {
            double gu[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
            for (int q = 0; q < N_NODES; ++q) {
                const double *G = Gp(rec, s, q);
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < 3; ++k) gu[j][k] += s_u[le][q * 3 + j] * G[k];
            }
            mu = visc(gu, rheo, mu0, muinf, lam, ca, cn);
        }
        double m = mdot[slot * N_IP + s];
        double d = 0.5 * (s_dn[le][ia] + s_dn[le][ib]);
        s_ip[le][s][0] = mu;
        s_ip[le][s][1] = m;
        s_ip[le][s][2] = d;
        // right-hand side of this flux point, once (the rows at ia and ib take -+ it):
        // continuity: the lagged Rhie-Chow term and the transient part
        const double *A = rec + OFF_A + s * 3;
        double gA = 0.0, u1 = 0.0, u2 = 0.0;
        for (int q = 0; q < N_NODES; ++q) {
            double Nq = T.nip[s][q];
            for (int j = 0; j < 3; ++j) {
                gA += Nq * s_gp[le][q * 3 + j] * A[j];
                if (nlev > 0) u1 += Nq * s_u1[le][q * 3 + j] * A[j];
                if (nlev > 1) u2 += Nq * s_u2[le][q * 3 + j] * A[j];
            }
        }
        double tr = 0.0;
        if (nlev > 0) {
            double sbar = 0.5 * (s_sn[le][ia] + s_sn[le][ib]);
            tr = c1 * (mo1[slot * N_IP + s] - rho * u1);
            if (nlev > 1) tr += c2 * (mo2[slot * N_IP + s] - rho * u2);
            tr *= 1.0 - d / sbar;
        }
        s_rhs[le][s][3] = rho * d * gA + tr;
        // momentum: the deferred correction at the upwind node
        int up = m >= 0 ? ia : ib;
        for (int k = 0; k < 3; ++k) {
            double corr = 0.0;
            if (!stokes)
                for (int q = 0; q < 3; ++q)
                    corr += s_gu[le][up * 9 + k * 3 + q] * (rec[OFF_X + s * 3 + q] - s_x[le][up * 3 + q]);
            s_rhs[le][s][k] = stokes ? 0.0 : m * s_be[le][up * 3 + k] * corr;
        }
    }
    __syncthreads();
    if (!active) return;
    int a = t / N_NODES, c = t %% N_NODES;
    // block (a, c) = sum over the flux points at a, with sign sg, of the flux point's block:
    // momentum diagonal -mu G_c.A (+ upwind m), transpose -mu G_c[k] A[j], pressure N_c A[k],
    // continuity rho N_c A[k] and -rho d G_c.A. The vector sums are factored out.
    const double *Gc0 = Gp(rec, 0, c);
    double vis = 0.0, pv0 = 0.0, pv1 = 0.0, pv2 = 0.0, rcd = 0.0, adv = 0.0;
    double tr[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < MAXI; ++i) {
        int s = T.ips[a][i];
        if (s < 0) break;
        double sg = T.sgn[a][i];
        double mu = s_ip[le][s][0], m = s_ip[le][s][1], d = s_ip[le][s][2];
        const double *A = rec + OFF_A + s * 3;
        const double *G = CONST_G ? Gc0 : Gp(rec, s, c);
        double GA = G[0] * A[0] + G[1] * A[1] + G[2] * A[2];
        double sN = sg * T.nip[s][c];
        vis -= sg * mu * GA;
        rcd -= sg * d * GA;
        pv0 += sN * A[0]; pv1 += sN * A[1]; pv2 += sN * A[2];
        if (transpose) {
            double smu = sg * mu;
            for (int k = 0; k < 3; ++k)
                for (int j = 0; j < 3; ++j) tr[k * 3 + j] -= smu * G[k] * A[j];
        }
        if (!stokes) {
            int ia = T.edge[s][0], ib = T.edge[s][1];
            if ((c == ia && m > 0) || (c == ib && m < 0)) adv += sg * m;
        }
    }
    double acc[16];
    acc[0] = vis + adv + tr[0]; acc[1] = tr[1];            acc[2] = tr[2];             acc[3] = pv0;
    acc[4] = tr[3];             acc[5] = vis + adv + tr[4]; acc[6] = tr[5];            acc[7] = pv1;
    acc[8] = tr[6];             acc[9] = tr[7];            acc[10] = vis + adv + tr[8]; acc[11] = pv2;
    acc[12] = rho * pv0;        acc[13] = rho * pv1;       acc[14] = rho * pv2;        acc[15] = rho * rcd;
    long long p = pos[slot * T_ELEM + t];
    double2 *blk = reinterpret_cast<double2 *>(data + p * 16);    // 128-byte aligned blocks
    for (int q = 0; q < 8; ++q) {
        double2 v = blk[q];
        v.x += acc[2 * q]; v.y += acc[2 * q + 1];
        blk[q] = v;
    }
    // right-hand side: thread (a, k), k < 4, component k of row a. A node flagged in own
    // (flow entering through a pressure boundary) keeps out of its own row the corrections
    // of the faces it is upwind of: its boundary flux carries that momentum in.
    if (c < 4) {
        int k = c;
        bool mirror = k < 3 && own[s_n[le][a]];
        double r = 0.0;
        for (int i = 0; i < MAXI; ++i) {
            int s = T.ips[a][i];
            if (s < 0) break;
            if (mirror && (s_ip[le][s][1] >= 0 ? T.edge[s][0] : T.edge[s][1]) == a) continue;
            r -= T.sgn[a][i] * s_rhs[le][s][k];
        }
        b[s_n[le][a] * 4 + k] += r;
    }
}
"""

MASSFLOW = r"""
// New Rhie-Chow mass flows, each node's net interior outflow, and the advective part of
// the momentum diagonal for the next assembly (so no diagonal pass is needed).
extern "C" __global__ void massflow(const int *conn, long long base, long long count,
                                    const double *X, const double *georec,
                                    const double *U, const double *P, const double *gradP,
                                    const double *dnode, const double *snode, int nlev,
                                    double c1, const double *Uo1, const double *mo1,
                                    double c2, const double *Uo2, const double *mo2,
                                    double rho, double *mdot, double *imbalance,
                                    double *adv) {
    __shared__ double s_x[EPB][N_NODES * 3];
    __shared__ double s_u[EPB][N_NODES * 3];
    __shared__ double s_gp[EPB][N_NODES * 3];
    __shared__ double s_u1[EPB][N_NODES * 3];
    __shared__ double s_u2[EPB][N_NODES * 3];
    __shared__ double s_p[EPB][N_NODES];
    __shared__ double s_dn[EPB][N_NODES];
    __shared__ double s_sn[EPB][N_NODES];
    __shared__ double s_rec[EPB][REC];
    __shared__ double s_m[EPB][N_IP];
    __shared__ long long s_n[EPB][N_NODES];
    __shared__ Tab T;
    load_tab(T);
    int le = threadIdx.x / T_ELEM, t = threadIdx.x %% T_ELEM;
    long long slot = base + (long long)blockIdx.x * EPB + le;
    bool active = le < EPB && slot < base + count;
    int lx = le < EPB ? le : 0;
    if (active && t < N_NODES) {
        long long n = conn[slot * N_NODES + t];
        s_n[le][t] = n;
        for (int k = 0; k < 3; ++k) {
            s_x[le][t * 3 + k] = X[n * 3 + k];
            s_u[le][t * 3 + k] = U[n * 3 + k];
            s_gp[le][t * 3 + k] = gradP[n * 3 + k];
            if (nlev > 0) s_u1[le][t * 3 + k] = Uo1[n * 3 + k];
            if (nlev > 1) s_u2[le][t * 3 + k] = Uo2[n * 3 + k];
        }
        s_p[le][t] = P[n];
        s_dn[le][t] = dnode[n];
        s_sn[le][t] = snode[n];
    }
    __syncthreads();
    element_geometry(georec, slot, active, s_x[lx], s_rec[lx], t, T);
    const double *rec = s_rec[lx];
    if (active && t < N_IP) {
        int s = t, ia = T.edge[s][0], ib = T.edge[s][1];
        const double *A = rec + OFF_A + s * 3;
        double ua = 0.0, u1 = 0.0, u2 = 0.0, rc = 0.0;
        for (int q = 0; q < N_NODES; ++q) {
            double Nq = T.nip[s][q];
            const double *G = Gp(rec, s, q);
            for (int k = 0; k < 3; ++k) {
                ua += Nq * s_u[le][q * 3 + k] * A[k];
                if (nlev > 0) u1 += Nq * s_u1[le][q * 3 + k] * A[k];
                if (nlev > 1) u2 += Nq * s_u2[le][q * 3 + k] * A[k];
                rc += (Nq * s_gp[le][q * 3 + k] - G[k] * s_p[le][q]) * A[k];
            }
        }
        double d = 0.5 * (s_dn[le][ia] + s_dn[le][ib]);
        double m = rho * ua + rho * d * rc;
        if (nlev > 0) {
            double sbar = 0.5 * (s_sn[le][ia] + s_sn[le][ib]);
            double tr = c1 * (mo1[slot * N_IP + s] - rho * u1);
            if (nlev > 1) tr += c2 * (mo2[slot * N_IP + s] - rho * u2);
            m += (1.0 - d / sbar) * tr;
        }
        mdot[slot * N_IP + s] = m;
        s_m[le][s] = m;
    }
    __syncthreads();
    if (!active || t >= N_NODES) return;
    int a = t;
    double out = 0.0, ad = 0.0;
    for (int i = 0; i < MAXI; ++i) {
        int s = T.ips[a][i];
        if (s < 0) break;
        double sg = T.sgn[a][i], m = s_m[le][s];
        out += sg * m;
        ad += sg > 0 ? (m > 0 ? m : 0.0) : (m < 0 ? -m : 0.0);
    }
    imbalance[s_n[le][a]] += out;
    adv[s_n[le][a]] += ad;
}
"""

LIMITER = r"""
// Barth-Jespersen blend at each node and velocity component: the smallest ratio over the
// flux points of the node's elements that keeps the reconstruction within the extremes
extern "C" __global__ void limiter(const int *conn, long long base, long long count,
                                   const double *X, const double *georec, const double *U,
                                   const double *gradU, const double *lo, const double *hi,
                                   double *beta) {
    __shared__ double s_x[EPB][N_NODES * 3];
    __shared__ double s_rec[EPB][REC];
    __shared__ long long s_n[EPB][N_NODES];
    __shared__ Tab T;
    load_tab(T);
    int le = threadIdx.x / T_ELEM, t = threadIdx.x %% T_ELEM;
    long long slot = base + (long long)blockIdx.x * EPB + le;
    bool active = le < EPB && slot < base + count;
    int lx = le < EPB ? le : 0;
    if (active && t < N_NODES) {
        long long n = conn[slot * N_NODES + t];
        s_n[le][t] = n;
        for (int k = 0; k < 3; ++k) s_x[le][t * 3 + k] = X[n * 3 + k];
    }
    __syncthreads();
    element_geometry(georec, slot, active, s_x[lx], s_rec[lx], t, T);
    // thread (node a, component k)
    if (!active || t >= 3 * N_NODES) return;
    int a = t / 3, k = t %% 3;
    long long n = s_n[le][a];
    const double *rec = s_rec[le];
    double g0 = gradU[(n * 3 + k) * 3], g1 = gradU[(n * 3 + k) * 3 + 1], g2 = gradU[(n * 3 + k) * 3 + 2];
    double u = U[n * 3 + k], h = hi[n * 3 + k], l = lo[n * 3 + k];
    // min over flux points of (h - u)/delta (delta > 0) is (h - u)/max delta, and division
    // is monotone in the denominator when correctly rounded: bit-identical to taking the
    // ratio at every flux point, with two divisions instead of one per flux point
    double dpos = 0.0, dneg = 0.0;
    for (int i = 0; i < MAXI; ++i) {
        int s = T.ips[a][i];
        if (s < 0) break;
        double delta = g0 * (rec[OFF_X + s * 3] - s_x[le][a * 3])
                     + g1 * (rec[OFF_X + s * 3 + 1] - s_x[le][a * 3 + 1])
                     + g2 * (rec[OFF_X + s * 3 + 2] - s_x[le][a * 3 + 2]);
        if (delta > 1e-300) dpos = fmax(dpos, delta);
        else if (delta < -1e-300) dneg = fmin(dneg, delta);
    }
    double bmin = beta[n * 3 + k];
    if (dpos > 0.0) bmin = fmin(bmin, fmin(fmax((h - u) / dpos, 0.0), 1.0));
    if (dneg < 0.0) bmin = fmin(bmin, fmin(fmax((l - u) / dneg, 0.0), 1.0));
    beta[n * 3 + k] = bmin;
}
"""

BLOCKPOS = r"""
// once per mesh and pattern: the block of every (row node, column node) pair of every slot
extern "C" __global__ void blockpos(const int *conn, long long nslots,
                                    const long long *indptr, const int *indices, int *pos) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nslots * T_ELEM) return;
    long long slot = i / T_ELEM;
    int t = (int)(i %% T_ELEM), a = t / N_NODES, c = t %% N_NODES;
    long long row = conn[slot * N_NODES + a], col = conn[slot * N_NODES + c];
    long long lo = indptr[row], hi = indptr[row + 1] - 1, found = -1;
    while (lo <= hi) {
        long long mid = (lo + hi) >> 1;
        long long v = indices[mid];
        if (v == col) { found = mid; break; }
        if (v < col) lo = mid + 1; else hi = mid - 1;
    }
    pos[i] = (int)found;
}
"""

# ------------------------------------------------------------------ node-based kernels
NODE = r"""
// nodal gradients of nc fields from the precomputed operator (one row per 8 lanes)
extern "C" __global__ void csr_gradient(long long n, int nc, const long long *indptr,
                                        const int *indices, const double *gop,
                                        const double *phi, double *out) {
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long row = gid >> 3;
    int lane = threadIdx.x & 7;
    double acc[12];
    for (int q = 0; q < 12; ++q) acc[q] = 0.0;
    if (row < n) {
        for (long long p = indptr[row] + lane; p < indptr[row + 1]; p += 8) {
            long long j = indices[p];
            double g0 = gop[p * 3], g1 = gop[p * 3 + 1], g2 = gop[p * 3 + 2];
            for (int c = 0; c < nc; ++c) {
                double v = phi[j * nc + c];
                acc[c * 3] += g0 * v; acc[c * 3 + 1] += g1 * v; acc[c * 3 + 2] += g2 * v;
            }
        }
    }
    unsigned mask = 0xffffffffu;
    for (int off = 4; off > 0; off >>= 1)
        for (int q = 0; q < 12; ++q) acc[q] += __shfl_down_sync(mask, acc[q], off, 8);
    if (row < n && lane == 0)
        for (int q = 0; q < nc * 3; ++q) out[row * nc * 3 + q] = acc[q];
}

// extremes of each velocity component over the nodes sharing an element with each node
extern "C" __global__ void csr_minmax(long long n, const long long *indptr, const int *indices,
                                      const double *U, double *lo, double *hi) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double l0 = 1e300, l1 = 1e300, l2 = 1e300, h0 = -1e300, h1 = -1e300, h2 = -1e300;
    for (long long p = indptr[i]; p < indptr[i + 1]; ++p) {
        long long j = indices[p];
        double u0 = U[j * 3], u1 = U[j * 3 + 1], u2 = U[j * 3 + 2];
        l0 = fmin(l0, u0); l1 = fmin(l1, u1); l2 = fmin(l2, u2);
        h0 = fmax(h0, u0); h1 = fmax(h1, u1); h2 = fmax(h2, u2);
    }
    lo[i * 3] = l0; lo[i * 3 + 1] = l1; lo[i * 3 + 2] = l2;
    hi[i * 3] = h0; hi[i * 3 + 1] = h1; hi[i * 3 + 2] = h2;
}
"""


def _carray(name, a, ctype, fmt):
    a = np.asarray(a)
    body = ", ".join(fmt(v) for v in a.reshape(-1))
    dims = "".join(f"[{d}]" for d in a.shape)
    return f"__constant__ {ctype} {name}{dims} = {{{body}}};"


def tables(topo) -> str:
    """Constant-memory tables of one element type (edges, shape functions, incidence)."""
    n, nip = topo.n, topo.edges.shape[0]
    ips = -np.ones((n, 8), np.int64)
    sgn = np.zeros((n, 8))
    cnt = np.zeros(n, np.int64)
    for s, (ia, ib) in enumerate(topo.edges):
        for node, sg in ((ia, 1.0), (ib, -1.0)):
            ips[node, cnt[node]] = s
            sgn[node, cnt[node]] = sg
            cnt[node] += 1
    f = repr
    return "\n".join([
        _carray("EDGE", topo.edges.astype(np.int64), "int", lambda v: str(int(v))),
        _carray("NIP", topo.N_ip, "double", lambda v: f(float(v))),
        _carray("DN", topo.dN_ip, "double", lambda v: f(float(v))),
        _carray("TRIS", topo.tris.astype(np.int64), "int", lambda v: str(int(v))),
        _carray("WPT", topo.point_weights, "double", lambda v: f(float(v))),
        _carray("NODE_IPS", ips, "int", lambda v: str(int(v))),
        _carray("NODE_SGN", sgn, "double", lambda v: f(float(v))),
    ]) + f"\n// nip {nip}\n"


def epb(n: int) -> int:
    """Elements per thread block: blocks of about 128 threads, ``n²`` threads per element."""
    return max(1, 128 // (n * n))


__all__ = ["ASSEMBLE", "BLOCKPOS", "DIAG", "GEOBUILD", "HEADER", "LIMITER", "MASSFLOW", "NODE",
           "STATICS", "epb", "tables"]
