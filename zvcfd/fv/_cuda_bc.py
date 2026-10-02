"""CUDA source: the boundary rows, residual statistics and row scaling of the coupled system.

One kernel (``finalise``) replaces some forty small array operations per
outer iteration. Thread ``(i, r)`` owns component ``r`` of block row ``i``
(its row in every block of the row), so threads never write the same entry
and the result is deterministic. In order, for its row:

1. the known terms: time (``tdiag`` on the diagonal, ``trhs``), body force,
   the outlet pressure force, the known inflow (continuity), the outflowing
   momentum at pressure outlets (``acc`` on the diagonal, from the lagged
   boundary mass flow and the backflow weights), and a lumped outlet's
   implicit pressure force ``A_j`` in the pressure column;
2. a copy of the row before boundary conditions replace it, for rows the
   solver keeps (reactions, wall shear, lumped flows);
3. Dirichlet rows: a fixed component's row (``fixed``) becomes the identity
   with the known value; with ``eliminate`` the columns of known values
   (``known``: fixed components except a lumped outlet's pressure, which is
   an unknown) are also moved to the right-hand side (symmetric
   elimination: the same solution, and multigrid no longer mixes identity
   rows into its coarse operators);
4. the residual ``b − A x`` of the current state, as sums and maxima per
   equation for the CFX-style report (normalised by ``a_P`` or
   ``ρ V^{2/3}``; the host divides by the largest ``|u|``);
5. the dimensionless row scaling (``1/(a_P U_ref)``, ``1/(ρ U_ref V^{2/3})``,
   ``1/U_ref`` or ``1/p_ref`` for fixed rows, a lumped zone's own scale),
   with ``U_ref`` and ``p_ref`` read from device memory (no host sync).

Per thread block it writes 8 partial statistics (Σ r², max, for u v w p),
reduced by the caller.
"""

FINALISE = r"""
extern "C" __global__ void finalise(
        long long n, const long long *indptr, const int *indices, const long long *dpos,
        double *data, double *b,
        int has_time, const double *tdiag, const double *trhs, const double *fV,
        const double *force,
        const double *inflow, const double *mb, const double *wcons, const double *wonly,
        const double *wstab, int stokes, const double *snl,
        const unsigned char *fixed, const unsigned char *known, const double *vel,
        const double *pval, int eliminate,
        const long long *rawoff, double *rawdata, double *rawb,
        const double *x, const double *aP, const double *V, double rho,
        const double *scal, const double *lscale, double *rs_out, double *partial) {
    __shared__ double red[8][64];
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long i = gid >> 2;
    int r = (int)(gid & 3);
    double sq = 0.0, mx = 0.0;
    if (i < n) {
        long long p0 = indptr[i], p1 = indptr[i + 1], pd = dpos[i];
        double bi = b[i * 4 + r];
        // 1. known terms
        if (r < 3) {
            bi += (has_time ? trhs[i * 3 + r] : 0.0) + fV[i * 3 + r] - force[i * 3 + r];
            double dd = has_time ? tdiag[i] : 0.0;
            if (!stokes) {
                double m = mb[i];
                dd += wcons[i] * m + wonly[i] * fmax(m, 0.0) + wstab[i] * fmax(-m, 0.0);
            }
            data[pd * 16 + r * 5] += dd;
            data[pd * 16 + r * 4 + 3] += snl[i * 3 + r];
        } else {
            bi -= inflow[i];
        }
        // 2. the row before boundary conditions
        long long ro = rawoff[i];
        if (ro >= 0) {
            for (long long p = p0; p < p1; ++p)
                for (int c = 0; c < 4; ++c)
                    rawdata[(ro + (p - p0)) * 16 + r * 4 + c] = data[p * 16 + r * 4 + c];
            rawb[i * 4 + r] = bi;
        }
        // 3. Dirichlet
        bool fx = fixed[i * 4 + r];
        if (fx) {
            for (long long p = p0; p < p1; ++p)
                for (int c = 0; c < 4; ++c) data[p * 16 + r * 4 + c] = 0.0;
            data[pd * 16 + r * 5] = 1.0;
            bi = r < 3 ? vel[i * 3 + r] : pval[i];
        } else if (eliminate) {
            for (long long p = p0; p < p1; ++p) {
                long long j = indices[p];
                for (int c = 0; c < 4; ++c)
                    if (known[j * 4 + c]) {
                        double v = c < 3 ? vel[j * 3 + c] : pval[j];
                        bi -= data[p * 16 + r * 4 + c] * v;
                        data[p * 16 + r * 4 + c] = 0.0;
                    }
            }
        }
        // 4. residual of the current state
        double ax = 0.0;
        for (long long p = p0; p < p1; ++p) {
            long long j = indices[p];
            const double *row = data + p * 16 + r * 4;
            ax += row[0] * x[j * 4] + row[1] * x[j * 4 + 1] + row[2] * x[j * 4 + 2]
                + row[3] * x[j * 4 + 3];
        }
        if (!fx) {
            // with elimination b holds -A_known x_known and those columns are zero: the
            // residual is the same, since x takes the known values there
            double res = bi - ax;
            double nrm = r < 3 ? aP[i] : rho * pow(V[i], 2.0 / 3.0);
            double v = fabs(res) / nrm;
            sq = v * v;
            mx = v;
        }
        // 5. scaling
        double uref = scal[0], pref = scal[1];
        double s;
        if (lscale[i] > 0.0 && r == 3) s = lscale[i] / uref;
        else if (fx) s = r < 3 ? 1.0 / uref : 1.0 / pref;
        else s = r < 3 ? 1.0 / (aP[i] * uref) : 1.0 / (rho * uref * pow(V[i], 2.0 / 3.0));
        for (long long p = p0; p < p1; ++p)
            for (int c = 0; c < 4; ++c) data[p * 16 + r * 4 + c] *= s;
        b[i * 4 + r] = bi * s;
        rs_out[i * 4 + r] = s;
    }
    // block partials: sum of squares and max per equation (u v w p = r)
    int lane = threadIdx.x;                      // 256 threads = 64 rows x 4 components
    int row = lane >> 2;
    for (int q = 0; q < 8; ++q) red[q][row] = 0.0;
    __syncthreads();
    red[r][row] = sq;
    red[4 + r][row] = mx;
    __syncthreads();
    for (int s2 = 32; s2 > 0; s2 >>= 1) {
        if (row < s2)
            for (int q = 0; q < 8; ++q) {
                if (q < 4) red[q][row] += red[q][row + s2];
                else red[q][row] = fmax(red[q][row], red[q][row + s2]);
            }
        __syncthreads();
    }
    if (lane < 8) partial[blockIdx.x * 8 + lane] = red[lane][0];
}
"""

UPDATE = r"""
// The state from the solution, and the largest changes (block partials: max |dU|, max |U|,
// max |dp|, min p, max p).
extern "C" __global__ void update(long long n, const double *X, double *U, double *P,
                                  double *partial) {
    __shared__ double red[5][256];
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    double du = 0.0, um = 0.0, dp = 0.0, pmin = 1e300, pmax = -1e300;
    if (i < n) {
        for (int k = 0; k < 3; ++k) {
            double v = X[i * 4 + k];
            du = fmax(du, fabs(v - U[i * 3 + k]));
            um = fmax(um, fabs(v));
            U[i * 3 + k] = v;
        }
        double p = X[i * 4 + 3];
        dp = fabs(p - P[i]);
        pmin = p; pmax = p;
        P[i] = p;
    }
    int t = threadIdx.x;
    red[0][t] = du; red[1][t] = um; red[2][t] = dp; red[3][t] = pmin; red[4][t] = pmax;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) {
            red[0][t] = fmax(red[0][t], red[0][t + s]);
            red[1][t] = fmax(red[1][t], red[1][t + s]);
            red[2][t] = fmax(red[2][t], red[2][t + s]);
            red[3][t] = fmin(red[3][t], red[3][t + s]);
            red[4][t] = fmax(red[4][t], red[4][t + s]);
        }
        __syncthreads();
    }
    if (t < 5) partial[blockIdx.x * 5 + t] = red[t][0];
}
"""

__all__ = ["FINALISE", "UPDATE"]
