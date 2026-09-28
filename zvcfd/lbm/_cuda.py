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

``STORE_HALF=1`` stores populations as fp16 offset by their rest weight
(``f - w_q``) and computes in fp32.
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

#if STORE_HALF
  typedef __half store_t;
  // Store f - w (the rest equilibrium) so fp16 spends its bits on the deviation.
  __device__ __forceinline__ float ld(const store_t* a, int k, int q) {
      return __half2float(a[k]) + W[q]; }
  __device__ __forceinline__ void st(store_t* a, int k, int q, float v) {
      a[k] = __float2half_rn(v - W[q]); }
#else
  typedef float store_t;
  __device__ __forceinline__ float ld(const store_t* a, int k, int q) { return a[k]; }
  __device__ __forceinline__ void st(store_t* a, int k, int q, float v) { a[k] = v; }
#endif

__device__ __forceinline__ void collide(float* f, float omega, float Fx, float Fy, float Fz) {
    float rho = 0.f, jx = 0.f, jy = 0.f, jz = 0.f;
    #pragma unroll
    for (int q = 0; q < Q; q++) {
        rho += f[q]; jx += CX[q] * f[q]; jy += CY[q] * f[q]; jz += CZ[q] * f[q];
    }
    float ux = (jx + 0.5f * Fx) / rho, uy = (jy + 0.5f * Fy) / rho, uz = (jz + 0.5f * Fz) / rho;
    float usq = 1.5f * (ux * ux + uy * uy + uz * uz);
    float pref = 1.f - 0.5f * omega;
    #pragma unroll
    for (int q = 0; q < Q; q++) {
        float cu = CX[q] * ux + CY[q] * uy + CZ[q] * uz;
        float feq = W[q] * rho * (1.f + 3.f * cu + 4.5f * cu * cu - usq);
        float cF = CX[q] * Fx + CY[q] * Fy + CZ[q] * Fz;
        float uF = ux * Fx + uy * Fy + uz * Fz;
        float fq = W[q] * pref * (3.f * (cF - uF) + 9.f * cu * cF);
        f[q] += omega * (feq - f[q]) + fq;
    }
}

__device__ __forceinline__ void reservoir(float* f, float rho) {
    #pragma unroll
    for (int q = 0; q < Q; q++) f[q] = W[q] * rho;
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
    float omega, float Fx, float Fy, float Fz, float rho_in, float rho_out)
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
        reservoir(f, x < nx / 2 ? rho_in : rho_out);
    } else {
        #pragma unroll
        for (int q = 0; q < Q; q++) {
            int xs = x - CX[q]; xs += (xs < 0) * nx - (xs >= nx) * nx;
            int ys = y - CY[q]; ys += (ys < 0) * ny - (ys >= ny) * ny;
            int zs = z - CZ[q]; zs += (zs < 0) * nz - (zs >= nz) * nz;
            int j = xs + nx * (ys + ny * zs);
            f[q] = (flag[j] == 1) ? ld(fin, OPP[q] * n + i, OPP[q]) : ld(fin, q * n + j, q);
        }
        collide(f, omega, Fx, Fy, Fz);
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
    float omega, float Fx, float Fy, float Fz, float rho_in, float rho_out)
{
    __shared__ int s_nbr[27];
    int b = blockIdx.x;
    int l = threadIdx.x;
    if (l < 27) s_nbr[l] = nbr[b * 27 + l];
    __syncthreads();
    const int nq = nb * BB;     // stride between q planes (32-bit: < 2^31 / Q cells)
    int i = b * BB + l;
    unsigned char fl = flag[i];
    if (fl == 1) return;
    int lx = l & 7, ly = (l >> 3) & 7, lz = l >> 6;
    float f[Q];
    if (fl == 2) {
        int gx = brick_x[b] * B + lx;
        reservoir(f, gx < nx_cells / 2 ? rho_in : rho_out);
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
        collide(f, omega, Fx, Fy, Fz);
    }
    #pragma unroll
    for (int q = 0; q < Q; q++) st(fout, q * nq + i, q, f[q]);
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
    float r = 0.f, jx = 0.f, jy = 0.f, jz = 0.f;
    #pragma unroll
    for (int q = 0; q < Q; q++) {
        float v = ld(f, q * n + i, q);
        r += v; jx += CX[q] * v; jy += CY[q] * v; jz += CZ[q] * v;
    }
    rho[i] = r;
    ux[i] = (jx + 0.5f * Fx) / r; uy[i] = (jy + 0.5f * Fy) / r; uz[i] = (jz + 0.5f * Fz) / r;
}
"""

SOURCES = {"step_dense": DENSE, "step_sparse": SPARSE, "macros": MACROS}
