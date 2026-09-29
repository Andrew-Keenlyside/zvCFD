# Kernels

**Question.** How close do the D3Q19 kernels run to the GPU's memory
roofline, and what does the sparse brick layout cost?

**Method.** `benchmarks/bench_lbm.py`. It measures a device-to-device copy
(the achievable bandwidth), then times 50 steps after 5 warm-up steps for
each case. It reports fluid-cell updates per second (MLUPS) and converts
them to effective bandwidth, at 153 bytes per update in fp32 and 77 in
fp16 (every population read and written once, plus a flag byte).

```{figure} ../_static/figures/kernel_efficiency.png
:width: 100%
:figclass: zv-figure

**Kernel efficiency on the RTX A2000.** Effective bandwidth as a
percentage of device copy bandwidth (253 GB/s); higher is better. Bars are
fp32 (blue) and fp16 (orange) population storage.
```

## Results (RTX A2000, copy bandwidth 253 GB/s)

BGK collision, the series these pages have tracked since the feasibility
study:

| Case | Storage | MLUPS | Effective GB/s | % of copy | ms/step |
|---|---|---:|---:|---:|---:|
| dense, open 256³ | fp32 | 1,617 | 247 | **98 %** | 10.4 |
| sparse, open 256³ | fp32 | 1,287 | 197 | **78 %** | 13.0 |
| dense, open 256³ | fp16 | 2,148 | 165 | 65 % | 7.8 |
| sparse, open 256³ | fp16 | 972 | 75 | 30 % | 17.3 |
| dense, porous φ = 0.45, 256³ | fp32 | 891 | 136 | 54 % | 8.5 |
| sparse, porous φ = 0.45, 256³ | fp32 | 686 | 105 | 41 % | 11.0 |
| dense, porous | fp16 | 993 | 76 | 30 % | 7.6 |
| sparse, porous | fp16 | 432 | 33 | 13 % | 17.5 |
| sparse, vessels 512³ (3.6 % fluid, 10.4 % bricks) | fp32 | 624 | 96 | **38 %** | 7.8 |
| sparse, vessels 512³ | fp16 | 413 | 32 | 13 % | 11.8 |

MLUPS count **fluid** cells only. The vessel case would need 20.5 GB as a
dense fp32 grid and does not fit this 12 GB GPU. Sparse, it needs 2.1 GB.

## Reading the results

**The dense fp32 kernel is at the roofline.** At 98 % of copy bandwidth
there is nothing left to gain from the kernel itself. This took one fix:
the first version computed 64-bit division and modulo per cell to find its
coordinates. It was integer-bound at 85 %, and fp16 ran no faster than
fp32. A 3-D launch with 32-bit indices removed that.

**Sparse bricks cost 20 % on open domains.** The neighbour-table
lookup and less regular access take the sparse kernel from 98 % to 78 %.
That cost buys memory proportional to the fluid. The vessel case uses a
tenth of the dense memory.

**Complex geometry costs more than sparsity.** Per *fluid* cell, porous
and vessel cases run at 38–54 %. Two things cause it: warps whose threads
are partly solid voxels (they idle), and bounce-back reads at walls. For
vessels only 35 % of stored voxels are fluid (the brick fill). The
published sparse LBM on a coronary geometry (waLBerla, A100, FP64) runs
at a similar fraction: about 30 % of copy bandwidth, kernel-only.

**The fp16 kernel is latency-bound, not bandwidth-bound.** It moves half
the bytes but runs only 1.3× faster dense, and slower than fp32 sparse. It
issues 2-byte loads with the same thread structure as fp32, so not enough
bytes are in flight. Vectorising (two cells per thread with `half2`, or
wider loads) is the known fix: FluidX3D reaches 68 % of roofline in fp16 on
an H100. Until that is done, fp16 is a **memory** option (2× capacity), not
a speed option. The [performance model](../how_to/plan_a_run.md)'s
planning basis reflects this.

## Production kernels: TRT and Carreau–Yasuda

Runs use TRT collision (and Carreau–Yasuda viscosity for blood), not BGK.
They were measured in the same session, sparse layout:

| Case | Collision | Storage | MLUPS | % of copy | vs BGK |
|---|---|---|---:|---:|---:|
| open 256³ | TRT | fp32 | 1,259 | 76 % | −2 % |
| open 256³ | TRT | fp16 | 959 | 29 % | −1 % |
| open 256³ | TRT + Carreau–Yasuda | fp32 | 1,099 | 66 % | −15 % |
| porous φ = 0.45, 256³ | TRT | fp32 | 668 | 40 % | −3 % |
| porous φ = 0.45, 256³ | TRT + Carreau–Yasuda | fp32 | 530 | 32 % | −23 % |
| vessels 512³ | TRT | fp32 | 622 | 38 % | 0 % |
| vessels 512³ | TRT + Carreau–Yasuda | fp32 | 491 | 30 % | −21 % |

**TRT costs almost nothing** (0–3 %): the kernel is memory-bound, and TRT
adds arithmetic, not traffic. **Carreau–Yasuda costs 13–21 % over TRT.**
Each cell solves for its own relaxation rate by fixed-point iteration
(`powf` twice per iteration), which makes the kernel partly compute-bound.

## What the MVP changes cost

Several accuracy fixes went into the kernels during validation
([Validation](../validation/index.md)). Each was timed against the code
before it, back to back on the idle GPU (`benchmarks/results/bench_lbm_a2000_before_mvp.json`,
`kernel_ab_a2000.json`):

| Change | Why | Throughput effect |
|---|---|---|
| Shifted populations (`f − w` stored and computed) | fp32 lost the digits that carry a slow flow | fp32: −2.8 % to +0.9 % (within noise except dense porous); fp16: dense +7–8 % (no weight added and subtracted on every access), sparse −2.6 % to +1 % |
| Carreau–Yasuda: iterate until ω settles (10⁻⁶, at most 12) instead of a fixed 3 | a fixed 3 left a 0.6 % error that did not fall with resolution | +11 % (open), +5 % (porous) against the fixed 3, because cells at low shear settle in one or two iterations |
| … instead of a fixed 8 (the first fix) | as accurate, fewer iterations | +27–38 % at low shear; −3 % to +6 % in a strongly shear-thinning flow (λγ̇ up to 10–100) |
| Per-patch flux summed on the host in a fixed order | reruns must be bit-identical under flow control | faster: 0.07 ms per check against 0.18 ms for GPU atomics (HiP-CT coronary, 8,143 patch voxels); a check interval of 2,000 steps takes 13.7 s |

The planning basis of the [performance model](../how_to/plan_a_run.md)
was left unchanged: the fp32 efficiencies moved by at most a point, and
the fp16 gains (dense 61 → 65 %, sparse vessels 12 → 13 %) only make it
more conservative.

## Projection to the H100

An H100 SXM copies at ~3.12 TB/s (STREAM), 12.4× this GPU. The planning
basis takes the lower of the A2000 efficiency and the published efficiency
of tuned kernels:

| Kernel | A2000 measured | Published tuned | Planning | H100 G fluid-updates/s |
|---|---:|---:|---:|---:|
| fp32 dense, open | 98 % | 80 % (FluidX3D) | 80 % | 16.3 |
| fp32 sparse, open | 77 % | 70 % (assumed) | 70 % | 14.3 |
| fp32 sparse, vessels | 38 % | 55 % (assumed) | 38 % | 7.7 |
| fp16 dense, open | 61 % | 68 % (FluidX3D FP16S) | 61 % | 24.7 |
| fp16 sparse, vessels | 12 % | 45 % (assumed) | 12 % | 4.9 |

For reference, FluidX3D reports 17.6 (fp32) and 29.6 (fp16) G updates/s
dense on one H100 SXM. XLB's JAX backend reports 1.43 per A100. Python
kernels reach the roofline only as hand-written CUDA (CuPy `RawKernel`,
Warp, Numba-CUDA); array-level JAX or NumPy code loses 5–10×.
