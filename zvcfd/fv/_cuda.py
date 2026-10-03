"""CUDA source for the finite-volume element kernels (double precision).

Written from ``docs/spec/fv_numerics.md``, not ported from the CPU
reference, so the two act as independent implementations of the same
discrete equations. One thread handles one element of one type. With
``STORED`` 0 it recomputes the element's dual points, flux-point areas and
shape-function gradients from node coordinates; with ``STORED`` 1 it reads
them from arrays the ``geostore`` kernel filled once (gradients once per
tetrahedron, where they are constant: ``CONST_G``), and the assembly reads
precomputed block positions instead of searching the pattern. It adds its
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
#define STORED %(stored)d
#define CONST_G %(constg)d

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

// viscosity at a flux point from the velocity gradient gu[j][k] = du_j/dx_k:
// rheo 0 Newtonian (mu0), 1 Carreau-Yasuda mu_inf + (mu0 - mu_inf)(1 + (lam g)^a)^((n-1)/a)
__device__ inline double viscosity(const double gu[3][3], int rheo, double mu0, double muinf,
                                   double lam, double ca, double cn) {
    if (rheo == 0) return mu0;
    double ss = 0.0;
    for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) { double s = 0.5 * (gu[j][k] + gu[k][j]); ss += s * s; }
    double g = sqrt(2.0 * ss);
    return muinf + (mu0 - muinf) * pow(1.0 + pow(lam * g, ca), (cn - 1.0) / ca);
}

__device__ inline void ip_gradient(const double *U, const long long *node, const double G[][3],
                                   double gu[3][3]) {
    for (int j = 0; j < 3; ++j) for (int k = 0; k < 3; ++k) gu[j][k] = 0.0;
    for (int a = 0; a < N_NODES; ++a)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) gu[j][k] += U[node[a] * 3 + j] * G[a][k];
}

__device__ inline long long find_block(const long long *indptr, const int *indices,
                                       long long row, long long col) {
    long long lo = indptr[row], hi = indptr[row + 1] - 1;
    while (lo <= hi) {
        long long mid = (lo + hi) >> 1;
        long long c = indices[mid];
        if (c == col) return mid;
        if (c < col) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
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

// geometry access: recomputed (Geo g) or stored per element (Gst, Ast, Xst, Sst)
#define GEO_ARGS const double *Gst, const double *Ast, const double *Xst, const double *Sst
#define GEO_ARGS_OUT double *Go, double *Ao, double *Xo, double *So
#if STORED
#define ELEMENT(e) long long node[N_NODES]; for (int a_ = 0; a_ < N_NODES; ++a_) node[a_] = elem[(e) * N_NODES + a_];
#if CONST_G
#define GPTR(s) ((const double (*)[3]) (Gst + e * N_NODES * 3))
#else
#define GPTR(s) ((const double (*)[3]) (Gst + (e * N_IP + (s)) * N_NODES * 3))
#endif
#define APTR(s) (Ast + (e * N_IP + (s)) * 3)
#define XIP(s, q) Xst[(e * N_IP + (s)) * 3 + (q)]
#else
#define ELEMENT(e) Geo g; long long node[N_NODES]; geometry(elem, e, X, W, tris, dN, g, node);
#define GPTR(s) (g.G[s])
#define APTR(s) (g.A[s])
#define XIP(s, q) (g.xip[s][q])
#endif
"""

GRADIENT = r"""
extern "C" __global__ void gradient(const long long *elem, const long long *list, long long count,
                                    const double *X, const double *W, const int *tris,
                                    const double *dN, const int *edges, GEO_ARGS, const double *phi,
                                    double *out) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= count) return;
    long long e = list[t];
    ELEMENT(e)
    // SCV volumes from the flux-point triangles (divergence theorem about each node)
    double scv[N_NODES];
#if STORED
    for (int a = 0; a < N_NODES; ++a) scv[a] = Sst[e * N_NODES + a];
#else
    for (int a = 0; a < N_NODES; ++a) scv[a] = 0.0;
    for (int s = 0; s < N_IP; ++s) {
        int ia = edges[s * 2], ib = edges[s * 2 + 1];
        double da = 0.0, db = 0.0;
        for (int k = 0; k < 3; ++k) {
            da += (XIP(s, k) - g.x[ia][k]) * APTR(s)[k];
            db += (XIP(s, k) - g.x[ib][k]) * APTR(s)[k];
        }
        scv[ia] += da / 3.0;
        scv[ib] -= db / 3.0;
    }
#endif
    // element gradient: mean over flux points of the shape-function gradient
    double grad[%(nc)d][3];
    for (int c = 0; c < %(nc)d; ++c)
        for (int k = 0; k < 3; ++k) {
            double s2 = 0.0;
            for (int s = 0; s < N_IP; ++s)
                for (int a = 0; a < N_NODES; ++a) s2 += GPTR(s)[a][k] * phi[node[a] * %(nc)d + c];
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
                                    const double *dN, const int *edges, GEO_ARGS, const double *mdot,
                                    const double *U, int rheo, double mu0, double muinf,
                                    double lam, double ca, double cn,
                                    int transpose, int stokes, int has_mut, const double *mut,
                                    const double *Nip, double *diag) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= count) return;
    long long e = list[t];
    ELEMENT(e)
    double d[N_NODES];
    for (int a = 0; a < N_NODES; ++a) d[a] = 0.0;
    for (int s = 0; s < N_IP; ++s) {
        int ia = edges[s * 2], ib = edges[s * 2 + 1];
        double gu[3][3];
        ip_gradient(U, node, GPTR(s), gu);
        double mu = viscosity(gu, rheo, mu0, muinf, lam, ca, cn);
        if (has_mut)                                      // eddy viscosity, interpolated
            for (int a = 0; a < N_NODES; ++a) mu += Nip[s * N_NODES + a] * mut[node[a]];
        double GAa = 0.0, GAb = 0.0;
        for (int k = 0; k < 3; ++k) { GAa += GPTR(s)[ia][k] * APTR(s)[k]; GAb += GPTR(s)[ib][k] * APTR(s)[k]; }
        double ra = -mu * GAa, rb = -mu * GAb;             // x-momentum, own column
        if (transpose) { ra -= mu * GPTR(s)[ia][0] * APTR(s)[0]; rb -= mu * GPTR(s)[ib][0] * APTR(s)[0]; }
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
                                    const double *dN, const int *edges, GEO_ARGS, const double *Nip,
                                    const double *U, const double *P, const double *mdot,
                                    const double *gradU, const double *beta,
                                    const double *gradP, const double *dnode,
                                    double rho, int rheo, double mu0, double muinf,
                                    double lam, double ca, double cn, int transpose, int stokes,
                                    int has_mut, const double *mut, double *R) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= count) return;
    long long e = list[t];
    ELEMENT(e)
    double r[N_NODES][4];
    for (int a = 0; a < N_NODES; ++a) for (int c = 0; c < 4; ++c) r[a][c] = 0.0;
    for (int s = 0; s < N_IP; ++s) {
        int ia = edges[s * 2], ib = edges[s * 2 + 1];
        const double *A = APTR(s);
        // values and gradients at the flux point
        double pip = 0.0, uip[3] = {0, 0, 0}, gp[3] = {0, 0, 0}, gpbar[3] = {0, 0, 0};
        double gu[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};       // du_j/dx_k
        for (int a = 0; a < N_NODES; ++a) {
            double Na = Nip[s * N_NODES + a];
            long long na = node[a];
            pip += Na * P[na];
            for (int k = 0; k < 3; ++k) {
                uip[k] += Na * U[na * 3 + k];
                gp[k] += GPTR(s)[a][k] * P[na];
                gpbar[k] += Na * gradP[na * 3 + k];
                for (int j = 0; j < 3; ++j) gu[j][k] += U[na * 3 + j] * GPTR(s)[a][k];
            }
        }
        double mu = viscosity(gu, rheo, mu0, muinf, lam, ca, cn);
        if (has_mut)                                      // eddy viscosity, interpolated
            for (int a = 0; a < N_NODES; ++a) mu += Nip[s * N_NODES + a] * mut[node[a]];
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
                for (int q = 0; q < 3; ++q) corr += gradU[(up * 3 + k) * 3 + q] * (XIP(s, q) - X[up * 3 + q]);
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

ASSEMBLE = r"""
// Block (4 x 4, row-major) contributions of every flux point into a BSR matrix and its
// right-hand side. Momentum: advection (upwind in the lagged mdot), pressure (shape
// functions), viscous (mu grad u . A, and the transpose); continuity: rho N A on u and
// -rho d G.A on p. Right-hand side: the deferred correction, the lagged Rhie-Chow term
// rho d gradbar(p) . A, and the transient Rhie-Chow part f sum_l c_l (mdot_l - rho ubar_l . A)
// over up to two old levels (a false time step: one level, the current iterate, c = 1).
extern "C" __global__ void assemble(const long long *elem, const long long *list, long long count,
                                    const double *X, const double *W, const int *tris,
                                    const double *dN, const int *edges, GEO_ARGS, const double *Nip,
                                    const double *U, const double *mdot, const double *gradU,
                                    const double *beta, const double *gradP,
                                    const double *dnode, const double *snode, int nlev,
                                    double c1, const double *Uo1, const double *mo1,
                                    double c2, const double *Uo2, const double *mo2,
                                    double rho, int rheo, double mu0, double muinf,
                                    double lam, double ca, double cn, int transpose, int stokes,
                                    int has_mut, const double *mut, const int *own,
                                    const long long *indptr, const int *indices,
                                    const int *pos, double *data, double *b) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= count) return;
    long long e = list[t];
    ELEMENT(e)
    // the element's blocks, accumulated over its flux points and written once per
    // (row node, column node): N_NODES^2 read-modify-writes instead of 2 N_IP N_NODES
    double K[N_NODES][N_NODES][16];
    for (int a = 0; a < N_NODES; ++a)
        for (int c = 0; c < N_NODES; ++c)
            for (int q = 0; q < 16; ++q) K[a][c][q] = 0.0;
    for (int s = 0; s < N_IP; ++s) {
        int ia = edges[s * 2], ib = edges[s * 2 + 1];
        long long nA = node[ia], nB = node[ib];
        const double *A = APTR(s);
        double gu[3][3];
        ip_gradient(U, node, GPTR(s), gu);
        double mu = viscosity(gu, rheo, mu0, muinf, lam, ca, cn);
        if (has_mut)                                      // eddy viscosity, interpolated
            for (int a = 0; a < N_NODES; ++a) mu += Nip[s * N_NODES + a] * mut[node[a]];
        double m = mdot[e * N_IP + s];
        double d = 0.5 * (dnode[nA] + dnode[nB]);
        double sbar = 0.5 * (snode[nA] + snode[nB]);
        for (int c = 0; c < N_NODES; ++c) {
            double R[16];
            for (int q = 0; q < 16; ++q) R[q] = 0.0;
            double GA = 0.0;
            for (int k = 0; k < 3; ++k) GA += GPTR(s)[c][k] * A[k];
            double Nc = Nip[s * N_NODES + c];
            for (int k = 0; k < 3; ++k) {
                R[k * 4 + k] -= mu * GA;
                if (transpose)
                    for (int j = 0; j < 3; ++j) R[k * 4 + j] -= mu * GPTR(s)[c][k] * A[j];
                R[k * 4 + 3] += Nc * A[k];
                R[12 + k] = rho * Nc * A[k];
            }
            R[15] = -rho * d * GA;
            if (!stokes) {
                if (c == ia && m > 0) for (int k = 0; k < 3; ++k) R[k * 4 + k] += m;
                if (c == ib && m < 0) for (int k = 0; k < 3; ++k) R[k * 4 + k] += m;
            }
            for (int q = 0; q < 16; ++q) { K[ia][c][q] += R[q]; K[ib][c][q] -= R[q]; }
        }
        // right-hand side
        double gpbar[3] = {0, 0, 0}, ub[3] = {0, 0, 0};
        for (int a = 0; a < N_NODES; ++a) {
            double Na = Nip[s * N_NODES + a];
            for (int k = 0; k < 3; ++k) {
                gpbar[k] += Na * gradP[node[a] * 3 + k];
                ub[k] += Na * U[node[a] * 3 + k];
            }
        }
        double rc = 0.0, ua = 0.0;
        for (int k = 0; k < 3; ++k) { rc += gpbar[k] * A[k]; ua += ub[k] * A[k]; }
        rc *= rho * d;
        double tr = 0.0;
        if (nlev > 0) {
            double u1 = 0.0, u2 = 0.0;
            for (int a = 0; a < N_NODES; ++a) {
                double Na = Nip[s * N_NODES + a];
                for (int k = 0; k < 3; ++k) {
                    u1 += Na * Uo1[node[a] * 3 + k] * A[k];
                    if (nlev > 1) u2 += Na * Uo2[node[a] * 3 + k] * A[k];
                }
            }
            tr = c1 * (mo1[e * N_IP + s] - rho * u1);
            if (nlev > 1) tr += c2 * (mo2[e * N_IP + s] - rho * u2);
            tr *= 1.0 - d / sbar;
        }
        b[nA * 4 + 3] -= rc + tr;
        b[nB * 4 + 3] += rc + tr;
        if (!stokes) {
            long long up = m >= 0 ? nA : nB;
            // a node flagged in own (inflow through a pressure boundary) leaves this
            // correction out of its own row: its boundary flux carries that momentum in
            bool skip = own[up] != 0;
            for (int k = 0; k < 3; ++k) {
                double corr = 0.0;
                for (int q = 0; q < 3; ++q)
                    corr += gradU[(up * 3 + k) * 3 + q] * (XIP(s, q) - X[up * 3 + q]);
                corr *= m * beta[up * 3 + k];
                if (!(skip && up == nA)) b[nA * 4 + k] -= corr;
                if (!(skip && up == nB)) b[nB * 4 + k] += corr;
            }
        }
    }
    for (int a = 0; a < N_NODES; ++a)
        for (int c = 0; c < N_NODES; ++c) {
#if STORED
            long long p = pos[(e * N_NODES + a) * N_NODES + c];
#else
            long long p = find_block(indptr, indices, node[a], node[c]);
#endif
            for (int q = 0; q < 16; ++q) data[p * 16 + q] += K[a][c][q];
        }
}
"""

MASSFLOW = r"""
// Rhie-Chow mass flows at the flux points from the new state, plus each node's net
// outflow (for the consistent outlet flows), with the transient part over up to two
// old levels as in assemble.
extern "C" __global__ void massflow(const long long *elem, const long long *list, long long count,
                                    const double *X, const double *W, const int *tris,
                                    const double *dN, const int *edges, GEO_ARGS, const double *Nip,
                                    const double *U, const double *P, const double *gradP,
                                    const double *dnode, const double *snode, int nlev,
                                    double c1, const double *Uo1, const double *mo1,
                                    double c2, const double *Uo2, const double *mo2, double rho,
                                    double *mdot, double *imbalance) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= count) return;
    long long e = list[t];
    ELEMENT(e)
    double out[N_NODES];
    for (int a = 0; a < N_NODES; ++a) out[a] = 0.0;
    for (int s = 0; s < N_IP; ++s) {
        int ia = edges[s * 2], ib = edges[s * 2 + 1];
        long long nA = node[ia], nB = node[ib];
        const double *A = APTR(s);
        double ua = 0.0, u1 = 0.0, u2 = 0.0, gp[3] = {0, 0, 0}, gpbar[3] = {0, 0, 0};
        for (int a = 0; a < N_NODES; ++a) {
            double Na = Nip[s * N_NODES + a];
            for (int k = 0; k < 3; ++k) {
                ua += Na * U[node[a] * 3 + k] * A[k];
                if (nlev > 0) u1 += Na * Uo1[node[a] * 3 + k] * A[k];
                if (nlev > 1) u2 += Na * Uo2[node[a] * 3 + k] * A[k];
                gp[k] += GPTR(s)[a][k] * P[node[a]];
                gpbar[k] += Na * gradP[node[a] * 3 + k];
            }
        }
        double d = 0.5 * (dnode[nA] + dnode[nB]);
        double m = rho * ua;
        for (int k = 0; k < 3; ++k) m += rho * d * (gpbar[k] - gp[k]) * A[k];
        if (nlev > 0) {
            double sbar = 0.5 * (snode[nA] + snode[nB]);
            double tr = c1 * (mo1[e * N_IP + s] - rho * u1);
            if (nlev > 1) tr += c2 * (mo2[e * N_IP + s] - rho * u2);
            m += (1.0 - d / sbar) * tr;
        }
        mdot[e * N_IP + s] = m;
        out[ia] += m;
        out[ib] -= m;
    }
    for (int a = 0; a < N_NODES; ++a) imbalance[node[a]] += out[a];
}
"""

MINMAX = r"""
// neighbour extremes of each velocity component (node's elements), for the limiter
extern "C" __global__ void minmax(const long long *elem, const long long *list, long long count,
                                  const double *X, const double *W, const int *tris,
                                  const double *dN, const int *edges, GEO_ARGS, const double *U,
                                  double *lo, double *hi) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= count) return;
    long long e = list[t];
    double mn[3] = {1e300, 1e300, 1e300}, mx[3] = {-1e300, -1e300, -1e300};
    for (int a = 0; a < N_NODES; ++a) {
        long long na = elem[e * N_NODES + a];
        for (int k = 0; k < 3; ++k) { double v = U[na * 3 + k]; mn[k] = fmin(mn[k], v); mx[k] = fmax(mx[k], v); }
    }
    for (int a = 0; a < N_NODES; ++a) {
        long long na = elem[e * N_NODES + a];
        for (int k = 0; k < 3; ++k) { lo[na * 3 + k] = fmin(lo[na * 3 + k], mn[k]); hi[na * 3 + k] = fmax(hi[na * 3 + k], mx[k]); }
    }
}
"""

LIMITER = r"""
// Barth-Jespersen: beta at each node and component, the smallest ratio over the flux
// points of the node's elements that keeps the reconstruction within the extremes
extern "C" __global__ void limiter(const long long *elem, const long long *list, long long count,
                                   const double *X, const double *W, const int *tris,
                                   const double *dN, const int *edges, GEO_ARGS, const double *U,
                                   const double *gradU, const double *lo, const double *hi,
                                   double *beta) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= count) return;
    long long e = list[t];
    ELEMENT(e)
    for (int s = 0; s < N_IP; ++s) {
        for (int side = 0; side < 2; ++side) {
            long long n = node[edges[s * 2 + side]];
            for (int k = 0; k < 3; ++k) {
                double delta = 0.0;
                for (int q = 0; q < 3; ++q) delta += gradU[(n * 3 + k) * 3 + q] * (XIP(s, q) - X[n * 3 + q]);
                double r = 1.0, u = U[n * 3 + k];
                if (delta > 1e-300) r = (hi[n * 3 + k] - u) / delta;
                else if (delta < -1e-300) r = (lo[n * 3 + k] - u) / delta;
                r = fmin(fmax(r, 0.0), 1.0);
                beta[n * 3 + k] = fmin(beta[n * 3 + k], r);
            }
        }
    }
}
"""

GEOSTORE = r"""
// once per mesh: the geometry each thread would otherwise recompute (compiled with STORED 0)
extern "C" __global__ void geostore(const long long *elem, const long long *list, long long count,
                                    const double *X, const double *W, const int *tris,
                                    const double *dN, const int *edges, GEO_ARGS_OUT) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= count) return;
    long long e = list[t];
    Geo g; long long node[N_NODES];
    geometry(elem, e, X, W, tris, dN, g, node);
    double scv[N_NODES];
    for (int a = 0; a < N_NODES; ++a) scv[a] = 0.0;
    for (int s = 0; s < N_IP; ++s) {
        int ia = edges[s * 2], ib = edges[s * 2 + 1];
        double da = 0.0, db = 0.0;
        for (int k = 0; k < 3; ++k) {
            da += (g.xip[s][k] - g.x[ia][k]) * g.A[s][k];
            db += (g.xip[s][k] - g.x[ib][k]) * g.A[s][k];
            Ao[(e * N_IP + s) * 3 + k] = g.A[s][k];
            Xo[(e * N_IP + s) * 3 + k] = g.xip[s][k];
        }
        scv[ia] += da / 3.0;
        scv[ib] -= db / 3.0;
        if (!CONST_G || s == 0)
            for (int a = 0; a < N_NODES; ++a)
                for (int k = 0; k < 3; ++k)
                    Go[((CONST_G ? e : e * N_IP + s) * N_NODES + a) * 3 + k] = g.G[s][a][k];
    }
    for (int a = 0; a < N_NODES; ++a) So[e * N_NODES + a] = scv[a];
}
"""

BLOCKPOS = r"""
// once per mesh and pattern: the block position of every (row node, column node) pair
extern "C" __global__ void blockpos(const long long *elem, const long long *list, long long count,
                                    const double *X, const double *W, const int *tris,
                                    const double *dN, const int *edges, GEO_ARGS,
                                    const long long *indptr, const int *indices, int *pos) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= count) return;
    long long e = list[t];
    for (int a = 0; a < N_NODES; ++a)
        for (int c = 0; c < N_NODES; ++c)
            pos[(e * N_NODES + a) * N_NODES + c] = (int)find_block(
                indptr, indices, elem[e * N_NODES + a], elem[e * N_NODES + c]);
}
"""

__all__ = ["ASSEMBLE", "BLOCKPOS", "GEOSTORE", "COMMON", "DIAGONAL", "GRADIENT", "LIMITER", "MASSFLOW", "MINMAX",
           "RESIDUAL"]
