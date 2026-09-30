"""CUDA source for the finite-volume element kernels (double precision).

Written from ``docs/spec/fv_numerics.md``, not ported from the CPU
reference, so the two act as independent implementations of the same
discrete equations. One thread handles one element of one type. It
recomputes the element's dual points, flux-point areas and shape-function
gradients from node coordinates (nothing geometric is stored). It adds its
contributions to its nodes without atomics: kernels run one element
colour at a time, and no two elements of a colour share a node. Results
are therefore deterministic.

Kernels (templated per element type by ``%(n)d`` nodes, ``%(nip)d`` flux
points, ``%(npts)d`` dual points):

- ``gradient``: SCV-weighted element gradients of a node field with ``%(nc)d``
  components (sums; the host divides by the node volumes);
- ``diagonal``: the momentum-equation diagonal ``a_P`` (x component), which
  sets the Rhie–Chow coefficient;
- ``residual``: momentum and continuity residuals of the ip fluxes, with the
  lagged ``ṁ`` for advection and a given blend ``β`` and ``∇u``, ``∇̄p``
  for the deferred correction and the Rhie–Chow term.
"""

COMMON = r"""
#define N_NODES %(n)d
#define N_IP %(nip)d
#define N_PTS %(npts)d

__device__ inline void inv3(const double J[3][3], double I[3][3], double *det) {
    double a = J[0][0], b = J[0][1], c = J[0][2];
    double d = J[1][0], e = J[1][1], f = J[1][2];
    double g = J[2][0], h = J[2][1], i = J[2][2];
    double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    *det = a * A + b * B + c * C;
    double r = 1.0 / *det;
    I[0][0] = A * r; I[0][1] = -(b * i - c * h) * r; I[0][2] = (b * f - c * e) * r;
    I[1][0] = B * r; I[1][1] = (a * i - c * g) * r;  I[1][2] = -(a * f - c * d) * r;
    I[2][0] = C * r; I[2][1] = -(a * h - b * g) * r; I[2][2] = (a * e - b * d) * r;
}

// element geometry: node coordinates, dual points, flux-point areas and centroids,
// shape-function gradients at each flux point
struct Geo {
    double x[N_NODES][3];
    double A[N_IP][3];
    double xip[N_IP][3];
    double G[N_IP][N_NODES][3];
};

__device__ void geometry(const long long *elem, long long e, const double *X,
                         const double *W, const int *tris, const double *dN, Geo &g,
                         long long *node) {
    for (int a = 0; a < N_NODES; ++a) {
        node[a] = elem[e * N_NODES + a];
        for (int k = 0; k < 3; ++k) g.x[a][k] = X[node[a] * 3 + k];
    }
    double P[N_PTS][3];
    for (int p = 0; p < N_PTS; ++p)
        for (int k = 0; k < 3; ++k) {
            double s = 0.0;
            for (int a = 0; a < N_NODES; ++a) s += W[p * N_NODES + a] * g.x[a][k];
            P[p][k] = s;
        }
    for (int s = 0; s < N_IP; ++s) {
        const double *p0 = P[tris[s * 3]], *p1 = P[tris[s * 3 + 1]], *p2 = P[tris[s * 3 + 2]];
        double u[3], v[3];
        for (int k = 0; k < 3; ++k) { u[k] = p1[k] - p0[k]; v[k] = p2[k] - p0[k]; }
        g.A[s][0] = 0.5 * (u[1] * v[2] - u[2] * v[1]);
        g.A[s][1] = 0.5 * (u[2] * v[0] - u[0] * v[2]);
        g.A[s][2] = 0.5 * (u[0] * v[1] - u[1] * v[0]);
        for (int k = 0; k < 3; ++k) g.xip[s][k] = (p0[k] + p1[k] + p2[k]) / 3.0;
        double J[3][3], I[3][3], det;
        for (int k = 0; k < 3; ++k)
            for (int d = 0; d < 3; ++d) {
                double t = 0.0;
                for (int a = 0; a < N_NODES; ++a) t += g.x[a][k] * dN[(s * N_NODES + a) * 3 + d];
                J[k][d] = t;                     // dx_k / dxi_d
            }
        inv3(J, I, &det);
        for (int a = 0; a < N_NODES; ++a)
            for (int k = 0; k < 3; ++k) {
                double t = 0.0;
                for (int d = 0; d < 3; ++d) t += dN[(s * N_NODES + a) * 3 + d] * I[d][k];
                g.G[s][a][k] = t;                // dN_a / dx_k
            }
    }
}
"""

GRADIENT = r"""
extern "C" __global__ void gradient(const long long *elem, const long long *list, long long count,
                                    const double *X, const double *W, const int *tris,
                                    const double *dN, const int *edges, const double *phi,
                                    double *out) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= count) return;
    long long e = list[t];
    Geo g; long long node[N_NODES];
    geometry(elem, e, X, W, tris, dN, g, node);
    // SCV volumes from the flux-point triangles (divergence theorem about each node)
    double scv[N_NODES];
    for (int a = 0; a < N_NODES; ++a) scv[a] = 0.0;
    for (int s = 0; s < N_IP; ++s) {
        int ia = edges[s * 2], ib = edges[s * 2 + 1];
        double da = 0.0, db = 0.0;
        for (int k = 0; k < 3; ++k) {
            da += (g.xip[s][k] - g.x[ia][k]) * g.A[s][k];
            db += (g.xip[s][k] - g.x[ib][k]) * g.A[s][k];
        }
        scv[ia] += da / 3.0;
        scv[ib] -= db / 3.0;
    }
    // element gradient: mean over flux points of the shape-function gradient
    double grad[%(nc)d][3];
    for (int c = 0; c < %(nc)d; ++c)
        for (int k = 0; k < 3; ++k) {
            double s2 = 0.0;
            for (int s = 0; s < N_IP; ++s)
                for (int a = 0; a < N_NODES; ++a) s2 += g.G[s][a][k] * phi[node[a] * %(nc)d + c];
            grad[c][k] = s2 / N_IP;
        }
    for (int a = 0; a < N_NODES; ++a) {
        for (int c = 0; c < %(nc)d; ++c)
            for (int k = 0; k < 3; ++k) out[(node[a] * %(nc)d + c) * 3 + k] += scv[a] * grad[c][k];
        out[(long long)%(N)d * %(nc)d * 3 + node[a]] += scv[a];       // node volume, last block
    }
}
"""

DIAGONAL = r"""
extern "C" __global__ void diagonal(const long long *elem, const long long *list, long long count,
                                    const double *X, const double *W, const int *tris,
                                    const double *dN, const int *edges, const double *mdot,
                                    double mu, int transpose, int stokes, double *diag) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= count) return;
    long long e = list[t];
    Geo g; long long node[N_NODES];
    geometry(elem, e, X, W, tris, dN, g, node);
    double d[N_NODES];
    for (int a = 0; a < N_NODES; ++a) d[a] = 0.0;
    for (int s = 0; s < N_IP; ++s) {
        int ia = edges[s * 2], ib = edges[s * 2 + 1];
        double GAa = 0.0, GAb = 0.0;
        for (int k = 0; k < 3; ++k) { GAa += g.G[s][ia][k] * g.A[s][k]; GAb += g.G[s][ib][k] * g.A[s][k]; }
        double ra = -mu * GAa, rb = -mu * GAb;             // x-momentum, own column
        if (transpose) { ra -= mu * g.G[s][ia][0] * g.A[s][0]; rb -= mu * g.G[s][ib][0] * g.A[s][0]; }
        if (!stokes) {
            double m = mdot[e * N_IP + s];
            ra += m > 0 ? m : 0.0;
            rb += m < 0 ? m : 0.0;
        }
        d[ia] += ra;                                      // flux out of a: + at a
        d[ib] -= rb;                                      // and into b: - at b
    }
    for (int a = 0; a < N_NODES; ++a) diag[node[a]] += d[a];
}
"""

RESIDUAL = r"""
extern "C" __global__ void residual(const long long *elem, const long long *list, long long count,
                                    const double *X, const double *W, const int *tris,
                                    const double *dN, const int *edges, const double *Nip,
                                    const double *U, const double *P, const double *mdot,
                                    const double *gradU, const double *beta,
                                    const double *gradP, const double *dnode,
                                    double rho, double mu, int transpose, int stokes,
                                    double *R) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= count) return;
    long long e = list[t];
    Geo g; long long node[N_NODES];
    geometry(elem, e, X, W, tris, dN, g, node);
    double r[N_NODES][4];
    for (int a = 0; a < N_NODES; ++a) for (int c = 0; c < 4; ++c) r[a][c] = 0.0;
    for (int s = 0; s < N_IP; ++s) {
        int ia = edges[s * 2], ib = edges[s * 2 + 1];
        const double *A = g.A[s];
        // values and gradients at the flux point
        double pip = 0.0, uip[3] = {0, 0, 0}, gp[3] = {0, 0, 0}, gpbar[3] = {0, 0, 0};
        double gu[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};       // du_j/dx_k
        for (int a = 0; a < N_NODES; ++a) {
            double Na = Nip[s * N_NODES + a];
            long long na = node[a];
            pip += Na * P[na];
            for (int k = 0; k < 3; ++k) {
                uip[k] += Na * U[na * 3 + k];
                gp[k] += g.G[s][a][k] * P[na];
                gpbar[k] += Na * gradP[na * 3 + k];
                for (int j = 0; j < 3; ++j) gu[j][k] += U[na * 3 + j] * g.G[s][a][k];
            }
        }
        // momentum flux leaving a through this face: advection + pressure - viscous
        double m = mdot[e * N_IP + s];
        long long nA = node[ia], nB = node[ib];
        for (int k = 0; k < 3; ++k) {
            double f = pip * A[k];
            for (int j = 0; j < 3; ++j) {
                f -= mu * gu[k][j] * A[j];                          // mu grad u_k . A
                if (transpose) f -= mu * gu[j][k] * A[j];           // mu (du_j/dx_k) A_j
            }
            if (!stokes) {
                long long up = m >= 0 ? nA : nB;
                double uup = U[up * 3 + k];
                double corr = 0.0;
                for (int q = 0; q < 3; ++q) corr += gradU[(up * 3 + k) * 3 + q] * (g.xip[s][q] - X[up * 3 + q]);
                f += m * (uup + beta[up * 3 + k] * corr);
            }
            r[ia][k] += f;
            r[ib][k] -= f;
        }
        // continuity: Rhie-Chow mass flow from the current state
        double d = 0.5 * (dnode[nA] + dnode[nB]);
        double mf = 0.0;
        for (int k = 0; k < 3; ++k) mf += (uip[k] + d * (gpbar[k] - gp[k])) * A[k];
        mf *= rho;
        r[ia][3] += mf;
        r[ib][3] -= mf;
    }
    for (int a = 0; a < N_NODES; ++a)
        for (int c = 0; c < 4; ++c) R[node[a] * 4 + c] += r[a][c];
}
"""

__all__ = ["COMMON", "DIAGONAL", "GRADIENT", "RESIDUAL"]
