"""CUDA sources for the D3Q19 lattice-Boltzmann kernels.

Clean-room implementation of the textbook scheme (Krueger et al., *The
Lattice Boltzmann Method: Principles and Practice*, Springer 2017, ch. 3,
5, 6): BGK collision, Guo forcing, pull streaming between two population
buffers, half-way bounce-back walls, fixed-density reservoir cells. See
``docs/cleanroom.md`` for what may and may not be consulted.

Layout is structure-of-arrays: population ``q`` of cell ``i`` sits at
``q * n + i``, so a warp touches one contiguous run per population.
Indices are 32-bit (a 64-bit div/mod per cell made the first version
integer-bound rather than memory-bound); callers keep ``Q * n < 2**31``
per device buffer.

Populations are stored and computed shifted by their rest weight,
``g_q = f_q - w_q``, so the density is ``1 + sum(g)``. In a slow flow the
populations differ from ``w_q`` only in the fourth or fifth digit, and
fp32 arithmetic on the full ``f_q`` loses those digits. Shifted, fp32
spends its precision on the deviation. ``STORE_HALF=1`` stores ``g`` as
fp16 and computes in fp32. Densities handed to the kernels (patch and
reservoir densities) are deviations ``rho - 1`` for the same reason.
"""

from __future__ import annotations

Q = 19
CX = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0]
CY = [0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1]
CZ = [0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1]
OPP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]
W = [1 / 3] + [1 / 18] * 6 + [1 / 36] * 12

COMMON = r"""
#include <cuda_fp16.h>
#define Q 19
__constant__ int CX[19] = {%(cx)s};
__constant__ int CY[19] = {%(cy)s};
__constant__ int CZ[19] = {%(cz)s};
__constant__ int OPP[19] = {%(opp)s};
__constant__ float W[19] = {%(w)s};

// Buffers hold g = f - w (see the module docstring); ld and st move g.
#if STORE_HALF
  typedef __half store_t;
  __device__ __forceinline__ float ld(const store_t* a, int k, int q) {
      return __half2float(a[k]); }
  __device__ __forceinline__ void st(store_t* a, int k, int q, float v) {
      a[k] = __float2half_rn(v); }
#else
  typedef float store_t;
  __device__ __forceinline__ float ld(const store_t* a, int k, int q) { return a[k]; }
  __device__ __forceinline__ void st(store_t* a, int k, int q, float v) { a[k] = v; }
#endif

// Rheology / collision parameters, passed by value to every step kernel.
struct Params {
    float omega;        // 1/tau (Newtonian, or the zero-shear value for Carreau-Yasuda)
    float magic;        // TRT magic parameter Lambda (3/16 puts half-way walls exactly)
    float nu_inf, nu_0; // Carreau-Yasuda viscosities, lattice units
    float cy_lambda;    // Carreau-Yasuda time constant, lattice units
    float cy_a, cy_n;   // Carreau-Yasuda exponents
};

#ifndef COLLISION_TRT
#define COLLISION_TRT 0
#endif
#ifndef NONNEWTONIAN
#define NONNEWTONIAN 0
#endif

// Shifted equilibrium feq_q - w_q, from the density deviation drho = rho - 1.
__device__ __forceinline__ float geq_q(int q, float drho, float rho, float ux, float uy, float uz,
                                       float usq) {
    float cu = CX[q] * ux + CY[q] * uy + CZ[q] * uz;
    return W[q] * (drho + rho * (3.f * cu + 4.5f * cu * cu - usq));
}

// f holds shifted populations g on entry and exit.
__device__ __forceinline__ void collide(float* f, const Params& P, float Fx, float Fy, float Fz) {
    float drho = 0.f, jx = 0.f, jy = 0.f, jz = 0.f;
    #pragma unroll
    for (int q = 0; q < Q; q++) {
        drho += f[q]; jx += CX[q] * f[q]; jy += CY[q] * f[q]; jz += CZ[q] * f[q];
    }
    float rho = 1.f + drho;
    float ux = (jx + 0.5f * Fx) / rho, uy = (jy + 0.5f * Fy) / rho, uz = (jz + 0.5f * Fz) / rho;
    float usq = 1.5f * (ux * ux + uy * uy + uz * uz);
    float omega = P.omega;
#if NONNEWTONIAN
    // Strain rate from the non-equilibrium second moment (no finite differences):
    // S = -(3 omega / 2 rho) Pi_neq, shear rate = sqrt(2 S:S). omega depends on the
    // shear rate through the viscosity: iterate the fixed point from the zero-shear
    // value until omega settles (1e-6 relative, at most 12 iterations). A fixed three
    // left a ~1e-2 error in omega where thinning is strong (a 6e-3 channel-profile
    // error that did not fall with resolution). A fixed eight was accurate but 20 to 40
    // per cent slower at low shear, where one or two iterations suffice.
    float pxx = 0.f, pyy = 0.f, pzz = 0.f, pxy = 0.f, pxz = 0.f, pyz = 0.f;
    #pragma unroll
    for (int q = 0; q < Q; q++) {
        float fn = f[q] - geq_q(q, drho, rho, ux, uy, uz, usq);
        pxx += CX[q] * CX[q] * fn; pyy += CY[q] * CY[q] * fn; pzz += CZ[q] * CZ[q] * fn;
        pxy += CX[q] * CY[q] * fn; pxz += CX[q] * CZ[q] * fn; pyz += CY[q] * CZ[q] * fn;
    }
    float pn = sqrtf(pxx * pxx + pyy * pyy + pzz * pzz + 2.f * (pxy * pxy + pxz * pxz + pyz * pyz));
    #pragma unroll
    for (int it = 0; it < 12; it++) {
        float gd = 1.41421356f * 1.5f * omega / rho * pn;
        float nu = P.nu_inf + (P.nu_0 - P.nu_inf)
                   * powf(1.f + powf(P.cy_lambda * gd, P.cy_a), (P.cy_n - 1.f) / P.cy_a);
        float om = fminf(fmaxf(1.f / (3.f * nu + 0.5f), 0.05f), 1.95f);
        bool done = fabsf(om - omega) <= 1e-6f * om;
        omega = om;
        if (done) break;
    }
#endif
    float uF = ux * Fx + uy * Fy + uz * Fz;
#if COLLISION_TRT
    // Two relaxation times: omega on the symmetric (even) parts, omega_m on the
    // antisymmetric (odd) parts, tied by Lambda = (1/omega - 1/2)(1/omega_m - 1/2).
    float om_m = 1.f / (P.magic / (1.f / omega - 0.5f) + 0.5f);
    {
        float fe = geq_q(0, drho, rho, ux, uy, uz, usq);
        float Fp = W[0] * (-3.f * uF);
        f[0] += omega * (fe - f[0]) + (1.f - 0.5f * omega) * Fp;
    }
    #pragma unroll
    for (int q = 1; q < Q; q += 2) {               // (q, q+1) are opposite velocities
        float fep = geq_q(q, drho, rho, ux, uy, uz, usq);
        float fem = geq_q(q + 1, drho, rho, ux, uy, uz, usq);
        float sp = 0.5f * (f[q] + f[q + 1]), sm = 0.5f * (f[q] - f[q + 1]);
        float ep = 0.5f * (fep + fem), em = 0.5f * (fep - fem);
        float cu = CX[q] * ux + CY[q] * uy + CZ[q] * uz;
        float cF = CX[q] * Fx + CY[q] * Fy + CZ[q] * Fz;
        float Fp = W[q] * (-3.f * uF + 9.f * cu * cF);   // even part of Guo's term
        float Fm = W[q] * (3.f * cF);                     // odd part
        float dp = -omega * (sp - ep) + (1.f - 0.5f * omega) * Fp;
        float dm = -om_m * (sm - em) + (1.f - 0.5f * om_m) * Fm;
        f[q] += dp + dm;
        f[q + 1] += dp - dm;
    }
#else
    float pref = 1.f - 0.5f * omega;
    #pragma unroll
    for (int q = 0; q < Q; q++) {
        float cu = CX[q] * ux + CY[q] * uy + CZ[q] * uz;
        float feq = W[q] * (drho + rho * (3.f * cu + 4.5f * cu * cu - usq));
        float cF = CX[q] * Fx + CY[q] * Fy + CZ[q] * Fz;
        float fq = W[q] * pref * (3.f * (cF - uF) + 9.f * cu * cF);
        f[q] += omega * (feq - f[q]) + fq;
    }
#endif
}

__device__ __forceinline__ void reservoir(float* f, float drho) {
    #pragma unroll
    for (int q = 0; q < Q; q++) f[q] = W[q] * drho;
}
""" % {
    "cx": ",".join(map(str, CX)),
    "cy": ",".join(map(str, CY)),
    "cz": ",".join(map(str, CZ)),
    "opp": ",".join(map(str, OPP)),
    "w": ",".join(f"{w:.9f}f" for w in W),
}

DENSE = COMMON + r"""
// 3-D launch (x threads, y and z from the grid) and 32-bit indices: 64-bit
// div/mod per cell made the first version integer-bound, not memory-bound.
extern "C" __global__ void step_dense(
    const store_t* __restrict__ fin, store_t* __restrict__ fout,
    const unsigned char* __restrict__ flag, int nx, int ny, int nz,
    float omega, float Fx, float Fy, float Fz, float drho_in, float drho_out)
{
    const int n = nx * ny * nz;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y, z = blockIdx.z;
    if (x >= nx) return;
    int i = x + nx * (y + ny * z);
    unsigned char fl = flag[i];
    if (fl == 1) return;
    float f[Q];
    if (fl == 2) {
        reservoir(f, x < nx / 2 ? drho_in : drho_out);
    } else {
        #pragma unroll
        for (int q = 0; q < Q; q++) {
            int xs = x - CX[q]; xs += (xs < 0) * nx - (xs >= nx) * nx;
            int ys = y - CY[q]; ys += (ys < 0) * ny - (ys >= ny) * ny;
            int zs = z - CZ[q]; zs += (zs < 0) * nz - (zs >= nz) * nz;
            int j = xs + nx * (ys + ny * zs);
            f[q] = (flag[j] == 1) ? ld(fin, OPP[q] * n + i, OPP[q]) : ld(fin, q * n + j, q);
        }
        Params P = {omega, 0.1875f, 0.f, 0.f, 0.f, 2.f, 1.f};
        collide(f, P, Fx, Fy, Fz);
    }
    #pragma unroll
    for (int q = 0; q < Q; q++) st(fout, q * n + i, q, f[q]);
}
"""

# Sparse bricks: only bricks holding fluid are stored.  One CUDA block per
# brick; each thread owns one cell.  The 26 neighbour bricks (27 incl.
# self) come from a table; a missing neighbour reads as solid.
SPARSE = COMMON + r"""
#define B 8
#define BB 512
extern "C" __global__ void step_sparse(
    const store_t* __restrict__ fin, store_t* __restrict__ fout,
    const unsigned char* __restrict__ flag, const int* __restrict__ nbr,
    const int* __restrict__ brick_x, int nb, int nx_cells,
    float omega, float magic, float nu_inf, float nu_0, float cy_lambda, float cy_a,
    float cy_n, float Fx, float Fy, float Fz, float drho_in, float drho_out)
{
    // Launched over the first (owned) bricks only; nb counts owned + ghost bricks,
    // and fixes the stride between population planes.
    const Params P = {omega, magic, nu_inf, nu_0, cy_lambda, cy_a, cy_n};
    __shared__ int s_nbr[27];
    int b = blockIdx.x;
    int l = threadIdx.x;
    if (l < 27) s_nbr[l] = nbr[b * 27 + l];
    __syncthreads();
    const int nq = nb * BB;     // stride between q planes (32-bit: < 2^31 / Q cells)
    int i = b * BB + l;
    unsigned char fl = flag[i];
    if (fl == 1 || fl >= 3) return;     // solid; or a patch cell, written by boundary_neem
    int lx = l & 7, ly = (l >> 3) & 7, lz = l >> 6;
    float f[Q];
    if (fl == 2) {
        int gx = brick_x[b] * B + lx;
        reservoir(f, gx < nx_cells / 2 ? drho_in : drho_out);
    } else {
        #pragma unroll
        for (int q = 0; q < Q; q++) {
            int sx = lx - CX[q], sy = ly - CY[q], sz = lz - CZ[q];
            int dx = (sx < 0) ? -1 : (sx >= B ? 1 : 0);
            int dy = (sy < 0) ? -1 : (sy >= B ? 1 : 0);
            int dz = (sz < 0) ? -1 : (sz >= B ? 1 : 0);
            int nbk = s_nbr[(dz + 1) * 9 + (dy + 1) * 3 + (dx + 1)];
            int j = nbk * BB + ((sx - dx * B) + B * ((sy - dy * B) + B * (sz - dz * B)));
            bool wall = (nbk < 0) || (flag[j] == 1);
            f[q] = wall ? ld(fin, OPP[q] * nq + i, OPP[q]) : ld(fin, q * nq + j, q);
        }
        collide(f, P, Fx, Fy, Fz);
    }
    #pragma unroll
    for (int q = 0; q < Q; q++) st(fout, q * nq + i, q, f[q]);
}
"""

# Inlet/outlet patch cells: Guo's non-equilibrium extrapolation (Guo, Zheng, Shi,
# Chinese Physics 11, 366 (2002)). A patch cell b takes the equilibrium at its
# prescribed state plus the non-equilibrium part of its interior neighbour n:
#   f*(b) = feq(rho_b, u_b) + [f*(n) - feq(rho_n, u_n)]
# Pressure patches (flag 3) prescribe rho and extrapolate u from n; velocity
# patches (flag 4) prescribe u (times a per-cell profile factor) and take rho
# from n. Runs after the step, on the freshly written buffer. Works on shifted
# populations throughout; pdrho holds each patch's density deviation rho - 1.
BOUNDARY = COMMON + r"""
extern "C" __global__ void boundary_neem(
    store_t* __restrict__ f, int nq,
    const unsigned char* __restrict__ flag,
    const int* __restrict__ bidx, const int* __restrict__ bnbr, const int* __restrict__ bpid,
    const float* __restrict__ bscale, int nbnd,
    const float* __restrict__ pdrho, const float* __restrict__ pux,
    const float* __restrict__ puy, const float* __restrict__ puz)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nbnd) return;
    int b = bidx[t], n = bnbr[t], p = bpid[t];
    float fn[Q];
    float drn = 0.f, jx = 0.f, jy = 0.f, jz = 0.f;
    if (n >= 0) {
        #pragma unroll
        for (int q = 0; q < Q; q++) {
            fn[q] = ld(f, q * nq + n, q);
            drn += fn[q]; jx += CX[q] * fn[q]; jy += CY[q] * fn[q]; jz += CZ[q] * fn[q];
        }
    }
    float rn = 1.f + drn;
    float unx = jx / rn, uny = jy / rn, unz = jz / rn;
    float usn = 1.5f * (unx * unx + uny * uny + unz * unz);
    float drb, ubx, uby, ubz;
    if (flag[b] == 3) { drb = pdrho[p]; ubx = unx; uby = uny; ubz = unz; }
    else { float s = bscale[t]; drb = drn; ubx = s * pux[p]; uby = s * puy[p]; ubz = s * puz[p]; }
    float rb = 1.f + drb;
    float usb = 1.5f * (ubx * ubx + uby * uby + ubz * ubz);
    #pragma unroll
    for (int q = 0; q < Q; q++) {
        float v = geq_q(q, drb, rb, ubx, uby, ubz, usb);
        if (n >= 0) v += fn[q] - geq_q(q, drn, rn, unx, uny, unz, usn);
        st(f, q * nq + b, q, v);
    }
}
"""

# Exact lattice mass flux through each patch cell: populations it sends into
# fluid neighbours minus populations fluid neighbours send into it, read from
# the newest post-collision buffer. Positive = into the domain. Summed per
# patch on the host; this, not velocity x area, is what flow splits use.
FLUX = COMMON + r"""
#define B 8
#define BB 512
__device__ __forceinline__ int neighbour_cell(const int* nbr, int cell, int qx, int qy, int qz) {
    int b = cell / BB, l = cell % BB;
    int sx = (l & 7) + qx, sy = ((l >> 3) & 7) + qy, sz = (l >> 6) + qz;
    int dx = (sx < 0) ? -1 : (sx >= B ? 1 : 0);
    int dy = (sy < 0) ? -1 : (sy >= B ? 1 : 0);
    int dz = (sz < 0) ? -1 : (sz >= B ? 1 : 0);
    int nbk = nbr[b * 27 + (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1)];
    if (nbk < 0) return -1;
    return nbk * BB + ((sx - dx * B) + B * ((sy - dy * B) + B * (sz - dz * B)));
}

extern "C" __global__ void patch_flux(
    const store_t* __restrict__ f, int nq, const unsigned char* __restrict__ flag,
    const int* __restrict__ nbr, const int* __restrict__ bidx, int nbnd,
    float* __restrict__ out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nbnd) return;
    int b = bidx[t];
    float net = 0.f;
    #pragma unroll
    for (int q = 1; q < Q; q++) {
        int j = neighbour_cell(nbr, b, CX[q], CY[q], CZ[q]);          // b sends q to j
        if (j >= 0 && flag[j] == 0) net += ld(f, q * nq + b, q);
        int s = neighbour_cell(nbr, b, -CX[q], -CY[q], -CZ[q]);       // s sends q to b
        if (s >= 0 && flag[s] == 0) net -= ld(f, q * nq + s, q);
    }
    out[t] = net;
}
"""

MACROS = COMMON + r"""
extern "C" __global__ void macros(
    const store_t* __restrict__ f, const unsigned char* __restrict__ flag, int n,
    float Fx, float Fy, float Fz,
    float* __restrict__ rho, float* __restrict__ ux, float* __restrict__ uy,
    float* __restrict__ uz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (flag[i] == 1) { rho[i] = 0.f; ux[i] = 0.f; uy[i] = 0.f; uz[i] = 0.f; return; }
    float dr = 0.f, jx = 0.f, jy = 0.f, jz = 0.f;
    #pragma unroll
    for (int q = 0; q < Q; q++) {
        float v = ld(f, q * n + i, q);
        dr += v; jx += CX[q] * v; jy += CY[q] * v; jz += CZ[q] * v;
    }
    float r = 1.f + dr;
    rho[i] = r;
    // The buffer holds post-collision populations, whose momentum already carries the
    // whole step's force: the physical (half-step) velocity is (j - F/2) / rho.
    ux[i] = (jx - 0.5f * Fx) / r; uy[i] = (jy - 0.5f * Fy) / r; uz[i] = (jz - 0.5f * Fz) / r;
}
"""

SOURCES = {"step_dense": DENSE, "step_sparse": SPARSE, "boundary_neem": BOUNDARY,
           "patch_flux": FLUX, "macros": MACROS}
