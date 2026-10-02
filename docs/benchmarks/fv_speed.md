# Finite-volume solver speed

**Question.** Where does the GPU finite-volume solver spend its time?
How much faster did the speed review of 30 September 2026 make it,
without changing its answers? And how does it compare with other methods
on the same problem?

Everything here was run on the workstation of the
[benchmark index](index.md): one RTX A2000 12 GB and a 16-core
Threadripper. The GPU is shared with other users' jobs. The tables below
were measured on 1 October with no other process on it, except where a
row says otherwise. Every GPU timing records how much of the GPU was
available ([A shared GPU](#a-shared-gpu) says why that matters).

## What changed

| Part | Before | After | Where |
|---|---|---|---|
| Element kernels (gradient, limiter, diagonal, assembly, mass flows) | one thread per element, geometry recomputed per flux point, per-element atomics avoided by colouring (136 tetrahedral colours) | elements stored in colour order ("slots"), so each colour is one contiguous launch with coalesced reads; `N²` threads per element, one per 4 × 4 block; node data, geometry and the topology tables staged in shared memory; geometry stored per slot or rebuilt cooperatively in the kernel; one node-based CSR operator for all gradients; the advective diagonal produced by the mass-flow pass; first-fit GPU colouring (38 colours) | `zvcfd/fv/_cuda_fast.py`, `zvcfd/fv/fast.py` |
| Set-up (dual geometry, node graph, colouring, block positions) | on the host (NumPy) | on the GPU | `zvcfd/fv/fast.py` |
| Boundary rows, residual statistics, row scaling | about forty array operations per outer iteration, several host synchronisations | one fused kernel (thread per row and component, deterministic), plus symmetric elimination of known values | `zvcfd/fv/_cuda_bc.py` |
| Host read-backs per outer iteration | one per statistic, per zone flow, per lumped outlet | one | `GPUSolver._iterate_fused` |
| Linear solver | AmgX, ILU(0) or DILU smoothing | AmgX, symmetric multicolour Gauss–Seidel (`gs`), smoothed coarsest level; `linear: auto` times `robust` on its third system and keeps the faster | `zvcfd/fv/linear.py` |
| Memory | full FGMRES basis (2 × 31 vectors); the last matrix alive during re-assembly | the basis grows in steps of 8; one matrix at a time; one AmgX resources handle shared by every solver | `zvcfd/fv/linear.py`, `zvcfd/fv/solver.py` |

The last row fixed a crash too. Destroying AmgX resources also destroys
AmgX's process-wide cuBLAS and cuSPARSE handles, so closing one solver
aborted every other live one ("Cuda failure: invalid argument").

## Correctness

The review changed no answer beyond round-off:

- `tests/test_fv_fast.py`: 48 fuzz cases (element mixes × gradient
  schemes × time steps × stored or rebuilt geometry) hold every fast
  kernel to the classic one, plus colouring and determinism checks.
- The GPU colouring once failed on the 2.07 M-tetrahedron SimVascular mesh
  ("more than 256 element colours"). One 32-bit counter held both the
  round's uncoloured elements and the out-of-colours flag, so more than 2²⁰
  of the former looked like the latter. The two counters are now separate
  and 64-bit, and the colour masks widen when a high-valence node needs
  more than 256 colours. A test colours 2.2 M tetrahedra and a 200-element
  fan on one node.
- The fused boundary pass against the staged one it replaces, on the
  gate B tube with an exact linear solve: **bit-identical** `u` and `p`
  over six steady outer iterations; with symmetric elimination, 2 × 10⁻¹⁵
  (the same solution of an equivalent system); over three BDF2 steps
  1 × 10⁻¹⁴, residual reports to 10⁻¹².
- The whole solver before and after the review, on the benchmark pipe
  (9.5 k nodes, converged to 10⁻¹⁰) with a fixed advection blend: flow rates
  agree to **4 × 10⁻¹²**, velocities to 3 × 10⁻¹⁰ of their maximum and
  pressures to 5 × 10⁻¹¹ of their range. With High Resolution they differ
  by 0.3 %. The limiter is frozen after 10 outer iterations
  (`freeze_limiter`), so the converged answer depends on the state at
  that iteration, which differs between linear solvers.
- The finite-volume test files all pass, 212 tests: `test_fv_fast` (50),
  `test_fv_gpu` (122), `test_fv_solver_gpu` (8), `test_fv_linear` (5),
  `test_fv_run` (4), `test_fv_multi` (7) and `test_fv_boundaries` (16).

## The linear solver

These are the four hard coupled systems of
[numerics](../spec/fv_numerics.md#linear-solvers-on-thin-cells-zvcfdfvlinear),
solved to the outer loop's reduction of 0.1, on an idle GPU. Each entry
is the number of iterations; 300 means not converged. Scripts:
`benchmarks/fv/capture_hard_systems.py` (the systems) and
`benchmarks/fv/amgx_smoothers.py`; results:
`benchmarks/results/fv/amgx_smoothers.json` (keys `synth2-`).

| AmgX configuration | DFG slab | gate B tube | graded layers | thin slab |
|---|---:|---:|---:|---:|
| DILU, 1 pre- / 2 post-sweeps (the usual set-up, the old default) | 300 | 4 | 7 | 300 |
| DILU, 0 / 1 sweeps | 111 | 5 | 4 | 300 |
| **symmetric Gauss–Seidel, 1 / 2 sweeps, pressure-weighted SIZE_4 (`gs`, new default)** | **2** | **3** | **3** | 300 |
| ILU(0), pairwise, pressure-weighted (`robust`) | 1 | 2 | 2 | 12 |

On the DFG slab, one cell deep, DILU stalls: its second post-sweep
amplifies the saddle-point modes, and a symmetric forward-and-back sweep
damps them. Of the two that work everywhere else, `robust` takes as few
or fewer iterations on these small systems and alone solves the thin
slab, while `gs` sets up about twice as fast. Which one is faster overall depends
on the mesh, so `linear: auto` measures it (see below).

The systems were captured again on 1 October, after a fix to the tube
generator. Its `growth` had graded the layers toward the core, not the
wall, so the "graded layers" and gate B tubes had their largest cells at
the wall. With layers that are finest at the wall, the graded-layer
system is much easier than before. The old DILU default took 166
iterations on the old one and takes 7 on the new.

**On the coronary mesh.** These are three systems captured from the
SimVascular coronary run on the corrected crop (398,105 nodes, 2,089,669
tetrahedra), solved to 0.1 on an idle GPU. Each entry is iterations and
total time (set-up and solve), in double precision; mixed precision gives
the same iterations and times within 0.1 s.

| AmgX configuration | system 0 | system 1 | system 2 |
|---|---:|---:|---:|
| **`gs` (new default)** | **1, 0.18 s** | **2, 0.22 s** | **4, 0.31 s** |
| `robust` (ILU(0), SIZE_2) | 1, 0.32 s | 2, 0.37 s | 3, 0.42 s |
| DILU 1 / 2 (the old default) | 13, 0.92 s | 23, 1.56 s | 53, 3.48 s |
| DILU 0 / 1 | 5, 0.30 s | 10, 0.46 s | 17, 0.66 s |

`gs` is 5–11× faster than the old default here and 1.4–1.8× faster than
`robust`. `robust` converges in as few iterations, but its ILU set-up costs
0.22 s against 0.08 s. The results file also keeps two systems captured
earlier, from a crop later found defective (337 isolated pieces with
undetermined pressure: singular systems). On those, DILU appeared to stall
outright. They are kept for the record only.

Reusing the hierarchy across outer iterations (`structure_reuse_levels:
−1` with `AMGX_solver_setup`) keeps the *first* matrix's whole hierarchy:
set-up drops to 4 ms, and every later solve stalls at the iteration
limit (gate B tube, 72 k nodes: 200 iterations from the second outer
iteration on). `AMGX_solver_resetup` is the call meant for this: it keeps
the aggregates and colourings and recomputes the operators. zvCFD now
uses it when `reuse` is set, with a rebuild when the linear iterations
grow by half. On the 22 k-node gate B tube it keeps the iterations of a
full rebuild ([4, 5, 5, 4, 4]) and cuts set-up 3–20×. On the 72 k-node
tube and the benchmark pipe, though, the solve after the first re-setup
fails inside AmgX 2.5 ("Matrix was not initialized"). That happens with
every reuse depth and coarse solver tried, so rebuilding each outer
iteration stays the default.

## Where the time goes

One outer iteration of the gate B tube (169 k nodes, 0.8 M elements,
steady Re = 100, High Resolution), before and after, stage by stage, on
an idle GPU. Script: `benchmarks/fv/profile_iteration.py tube 16`, whose
timers synchronise the device around every stage, so the totals are
upper bounds. Results:
`benchmarks/results/fv/profile_tube16_{before,after}.json`.

| Stage | Before | After |
|---|---:|---:|
| gradients, limiter, momentum diagonal (stage 1) | 144 ms | 21 ms |
| of which the limiter | 89 ms | 18 ms |
| assembly kernel | 192 ms | 36 ms |
| boundary rows, residuals, row scaling | 20 ms (three passes) | 7 ms (fused, in the 64 ms below) |
| assembly with boundary rows, all together | 357 ms | 64 ms |
| mass flows (with gradient of `p`) | 58 ms | 23 ms |
| **everything but the linear solve** | **416 ms** | **87 ms** |
| linear solve (set-up and FGMRES) | 191 ms (ILU(0), 2–3 iterations) | 229 ms (GS, 4–7 iterations) |
| **per outer iteration** | **0.62 s** | **0.32 s** |
| solver set-up (once) | 7.4 s | 1.3 s |

Everything except the linear solve is now **4.8× faster**, and the linear
solve is 73 % of an iteration. On this tube, `robust` solves each system
in fewer iterations than `gs`. So `linear: auto` measures instead of
guessing: on its third system it also solves with `robust`, and keeps
`robust` if that converges in under 80 % of the time `gs` took.

Without the stage timers, gate B with the current code runs at 0.084,
0.142 and 0.254 s per outer iteration at 22 k, 72 k and 169 k nodes
(AmgX `gs`, 24–25 outer iterations to convergence;
`benchmarks/results/fv/gate_b.json`). On 30 September, the old code on
the old meshes took 0.25, 0.49 and 0.81 s.

## The coronary case

The SimVascular coronary model (VMR 0066, cropped to its two coronary
trees: 398,105 nodes, 2,089,669 tetrahedra), BDF2 at 1 ms with five
coefficient loops per step and mixed precision. Measured by
`benchmarks/simvascular/run_fv.py`, with the current defaults (fast kernels,
fused boundary pass, AmgX `gs`), on 1 October:

| Run | Linear rtol | Linear iterations per solve (mean, max) | Per outer iteration | Per time step |
|---|---:|---:|---|---:|
| **full cardiac cycle, 1,000 steps** | 0.01 | 10.5, 22 (0 of 5,000 unconverged) | assembly 0.19 s, linear 0.66 s | **4.85 s (81 min in all)** |
| ten-step probe | 0.01 | 11, 21 (0 of 50 unconverged) | assembly 0.19 s, linear 0.68 s | 4.94 s |
| capture run (system capture on) | 0.1 | 2.3, 4 (all converged) | linear 0.06 s set-up + 0.18 s solve | — |

Solver set-up took 25 s. The mass imbalance stayed within 0.75 % at
every step. No before-and-after comparison exists on this mesh: the
earlier timings (34 s per step, a 7.6 h cycle) were made on a crop later
found defective. The answers are compared with SimVascular's in
[the SimVascular comparison](../validation/simvascular.md).

## Against other methods: one pipe

The pressure-driven pipe of the [OpenFOAM comparison](openfoam.md):
radius 0.5, length 4, a tetrahedral core with three wedge layers at the
wall, fixed pressure at both ends (Δp = 2.56), no-slip walls, Re = 50 on
the developed mean velocity. Every finite-volume method ran on **the same
mesh** (written as a Fluent file and imported by OpenFOAM). The LBM ran
on a voxelised pipe of the same size. Timed is the wall time until the
flow rate stays within 0.1 % of that method's own converged value (1 % in
brackets), from each solver's per-iteration history, after set-up. Script:
`benchmarks/fv/compare_methods.py`; results:
`benchmarks/results/fv/compare_methods*.json`.

| Mesh | Method | Hardware | Flow rate to 0.1 % (1 %) | Converged: iterations, time | Q / Q_Poiseuille |
|---|---|---|---:|---:|---:|
| 1.3 k nodes, 5.3 k cells | OpenFOAM v2506 `simpleFoam` | 1 core | 0.43 s (0.29 s) | 269, 2.0 s | 0.840 |
| | zvCFD GPU FV, now | A2000 | 0.44 s (0.39 s) | 40, 0.91 s | 0.888 |
| | zvCFD CPU reference FV | 1 process (SciPy) | — | 27, 11 s | 0.888 |
| 9.5 k nodes, 38 k cells | OpenFOAM | 1 core | 7.0 s (4.5 s) | 735, 17 s | 0.910 |
| | **zvCFD GPU FV, now** | A2000 | **0.60 s (0.49 s)** | 33, 1.5 s | 0.970 |
| | zvCFD GPU FV, before the review | A2000 | 0.71 s (0.48 s) | 33, 2.5 s | 0.970 |
| | zvCFD CPU reference FV | 1 process | — | 32, 714 s | 0.970 |
| 72 k nodes, 304 k cells | OpenFOAM | 16 cores | 61 s (40 s) | 2,300, 129 s | 0.936 |
| | **zvCFD GPU FV, now** | A2000 | **1.7 s (1.4 s)** | 34, 5.1 s | 0.988 |
| | zvCFD GPU FV, before the review | A2000 | 2.1 s (1.6 s) | 33, 8.3 s | 0.988 |
| voxels, radius 8 (13 k fluid cells) | zvCFD LBM (TRT, fp32) | A2000, contended | 0.35 s (0.24 s) | 15,300 steps, 1.1 s | 1.031 |
| voxels, radius 16 (104 k) | zvCFD LBM | A2000, contended | 3.4 s (2.2 s) | 39,600 steps, 11 s | 0.998 |
| voxels, radius 32 (826 k) | zvCFD LBM | A2000, contended | 43 s (28 s) | 79,200 steps, 141 s | 0.990 |

The finite-volume rows of zvCFD were measured on 2 October, after the fix
to momentum entering through pressure boundaries (below), with no other
process on the GPU (`gpu_contended_fraction` 0). The OpenFOAM rows are from
1 October, on the same meshes. The LBM rows are from 30 September, with
another job on the GPU throughout. On every pipe `linear: auto` kept
`robust` after its probe.

What this shows:

- **At 72 k nodes, the GPU solver gets the flow rate to 0.1 % 36× sooner
  than OpenFOAM on 16 cores (1.7 s against 61 s), and to 1 % 29× sooner.**
  At 9.5 k nodes it is 12× sooner than OpenFOAM on one core. Most of that
  is the algorithm: the coupled solver converges in 33–40 outer
  iterations, where OpenFOAM's segregated SIMPLE needs 269–2,300.
- **Before and after on the pipe:** 1.7× less time per outer iteration
  (0.044 against 0.075 s at 9.5 k nodes, 0.151 against 0.252 s at 72 k),
  and 1.2–1.3× sooner to 0.1 %. That is less than on the gate B tube
  because the linear solve, which the review sped up least, is most of an
  iteration here.
- **The inflow fix cut the outer iterations by 3–11×** (93 → 33 at 9.5 k
  nodes, 362 → 34 at 72 k) and moved the flow rates. Before it, momentum
  entering through the pressure inlet was inconsistent and the inflow
  formed a jet. zvCFD's flow rate was then 3–4 % above the exact one, and
  4–10 % above OpenFOAM's
  ([developed pipe](../validation/fv_solver.md#developed-pipe-flow-through-pressure-boundaries)).
- **The methods still differ on the flow rate, but zvCFD is now the
  closer.** The exact answer for this pipe is developed Poiseuille flow on
  the polygonal section, 0.977 and 0.994 of the circle's at 9.5 k and 72 k
  nodes. zvCFD is 0.70 % and 0.66 % below it, and OpenFOAM 6.8 % and
  5.8 %. What remains in zvCFD is the Stokes solution's own error and a
  0.5 % floor of this mesh's structured tetrahedra. Before and after the
  review agree to 10⁻¹² with a fixed blend. The LBM, on a
  voxelised pipe with a different pressure boundary, sits within 1–3 % of
  the circle's Poiseuille value.

## A shared GPU

Another user's image-registration job ran on the A2000 throughout: a
batch of ten subjects, one after another, at 60–99 % of the SMs. It is
not ours to stop, and it does not share memory bandwidth politely.
Measured with the probe `compare_methods.py` records before and after
every GPU run:

| | Idle A2000 | During these runs |
|---|---:|---:|
| Device copy bandwidth | 253 GB/s | 83–112 GB/s |
| One tiny kernel launch plus a synchronisation (median) | 13 µs | 2.4–4.6 ms |

The second row matters more. With another process's kernels resident,
the GPU time-slices between the two contexts, and every host
synchronisation waits for this process's next slice. Work that
synchronises often slows down by orders of magnitude. AmgX's hierarchy
set-up is hundreds of small steps (aggregation passes, colouring rounds,
coarse products), each ending in a synchronisation. On the 72 k-node
gate B tube it took **1.06 s per outer iteration, for every smoother**,
where the same library on the idle card took 0.03 s (gate B, earlier the
same day). Assembly, whose kernels are launched asynchronously, was not
slowed that much: 58 ms.

Timings taken under contention are therefore pessimistic, by an amount
that varies within a run as the other job's phases come and go. So
`compare_methods.py` samples `nvidia-smi pmon` once a second during each
run, records the fraction of samples with another process on the SMs, and
can repeat a run until one is clean (`--clean-tries`). The tables above
were re-measured on an idle GPU.
