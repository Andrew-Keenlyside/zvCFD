## Per-GPU throughput (H100 SXM, fluid-cell updates per second)

| Kernel | Efficiency basis | GLUPS |
|---|---|---:|
| zvCFD lbm-fp32 dense open | measured | 19.5 |
| zvCFD lbm-fp32 dense open | target | 16.3 |
| zvCFD lbm-fp32 dense open | planning | 16.3 |
| zvCFD lbm-fp32 sparse open | measured | 15.9 |
| zvCFD lbm-fp32 sparse open | target | 14.3 |
| zvCFD lbm-fp32 sparse open | planning | 14.3 |
| zvCFD lbm-fp32 sparse vessel | measured | 7.1 |
| zvCFD lbm-fp32 sparse vessel | target | 11.2 |
| zvCFD lbm-fp32 sparse vessel | planning | 7.1 |
| zvCFD lbm-fp16 dense open | measured | 24.7 |
| zvCFD lbm-fp16 dense open | target | 27.5 |
| zvCFD lbm-fp16 dense open | planning | 24.7 |
| zvCFD lbm-fp16 sparse open | measured | 12.1 |
| zvCFD lbm-fp16 sparse open | target | 24.3 |
| zvCFD lbm-fp16 sparse open | planning | 12.1 |
| zvCFD lbm-fp16 sparse vessel | measured | 4.5 |
| zvCFD lbm-fp16 sparse vessel | target | 18.2 |
| zvCFD lbm-fp16 sparse vessel | planning | 4.5 |
| FluidX3D FP32 / FP16S dense | published, H100 SXM | 17.6 / 29.6 |
| waLBerla sparse coronary (FP64, A100) | published | 1.37 kernel-only |
| XLB (JAX), A100 | published | 1.43 |

## HiP-CT coronary tree: one cardiac cycle (1 s), pulsatile, u_peak = 0.5 m/s

| Solver | Resolution | Cells | Memory | Hardware | Time per cycle |
|---|---|---:|---:|---|---:|
| Fluent-class FV, CPU | Simpleware mesh (~35 um equiv.) | 1.48e+07 | 37 GB RAM | 128 cores | 5.3 h |
| Fluent-class FV, CPU | same | 1.48e+07 | | 1024 cores | 40.1 min |
| Fluent-class FV, GPU | same | 1.48e+07 | 27 GB | 1 x H100 | 24.7 min |
| zvCFD lbm-fp32 | 20 um voxels (tau 0.605) | 7.86e+07 fluid | 22 GB/GPU | 1 x H100 | 45.9 min (2.5e+05 steps) |
| zvCFD lbm-fp32 | 20 um voxels (tau 0.605) | 7.86e+07 fluid | 3 GB/GPU | 8 x H100 | 6.8 min (2.5e+05 steps) |
| zvCFD lbm-fp16 | 20 um voxels (tau 0.605) | 7.86e+07 fluid | 12 GB/GPU | 1 x H100 | 1.2 h (2.5e+05 steps) |
| zvCFD lbm-fp16 | 20 um voxels (tau 0.605) | 7.86e+07 fluid | 1 GB/GPU | 8 x H100 | 10.8 min (2.5e+05 steps) |
| zvCFD lbm-fp32 | 10 um voxels (tau 0.710) | 6.29e+08 fluid | 141 GB/GPU | 1 x H100 | does not fit (5e+05 steps) |
| zvCFD lbm-fp32 | 10 um voxels (tau 0.710) | 6.29e+08 fluid | 18 GB/GPU | 8 x H100 | 1.8 h (5e+05 steps) |
| zvCFD lbm-fp16 | 10 um voxels (tau 0.710) | 6.29e+08 fluid | 73 GB/GPU | 1 x H100 | does not fit (5e+05 steps) |
| zvCFD lbm-fp16 | 10 um voxels (tau 0.710) | 6.29e+08 fluid | 9 GB/GPU | 8 x H100 | 2.9 h (5e+05 steps) |
| Fluent-class FV at the same cell count | 10 um-equivalent mesh | 6.29e+08 | 1.1 TB GPU | 1024 cores | 1.2 d |
| zvCFD lbm-fp32 | 5 um voxels (tau 0.920) | 5.03e+09 fluid | 963 GB/GPU | 1 x H100 | does not fit (1e+06 steps) |
| zvCFD lbm-fp32 | 5 um voxels (tau 0.920) | 5.03e+09 fluid | 120 GB/GPU | 8 x H100 | does not fit (1e+06 steps) |
| zvCFD lbm-fp16 | 5 um voxels (tau 0.920) | 5.03e+09 fluid | 497 GB/GPU | 1 x H100 | does not fit (1e+06 steps) |
| zvCFD lbm-fp16 | 5 um voxels (tau 0.920) | 5.03e+09 fluid | 62 GB/GPU | 8 x H100 | 1.9 d (1e+06 steps) |

## Digital rock, steady permeability (porosity 0.2)

| Image | Solver | Memory | Hardware | Time to steady state |
|---|---|---:|---|---:|
| 1024^3 | zvCFD lbm-fp32, dense | 21 GB/GPU | 8 x H100 | 1.5 min |
| 1024^3 | zvCFD lbm-fp16, dense | 11 GB/GPU | 8 x H100 | 1.4 min |
| 1024^3 | FV (snappy/Fluent mesh of the pore space) | 0.54 TB RAM | 1024 cores | 58.3 min |
| 1536^3 | zvCFD lbm-fp32, dense | 71 GB/GPU | 8 x H100 | 4.9 min |
| 1536^3 | zvCFD lbm-fp16, dense | 37 GB/GPU | 8 x H100 | 4.9 min |
| 1536^3 | FV (snappy/Fluent mesh of the pore space) | 1.81 TB RAM | 1024 cores | 3.3 h |
| 2048^3 | zvCFD lbm-fp32, dense | 169 GB/GPU | 8 x H100 | does not fit |
| 2048^3 | zvCFD lbm-fp16, dense | 87 GB/GPU | 8 x H100 | does not fit |
| 2048^3 | FV (snappy/Fluent mesh of the pore space) | 4.29 TB RAM | 1024 cores | 7.8 h |

Reference points: GeoChemFoam (OpenFOAM DBS) solved 8e9 voxels in 1230 s on 81,920 ARCHER2 cores; Palabos (A100) solved Berea 400^3 in under 10 min.

(10 um, tau 0.6: dt = 9.52e-07 s, 1 m/s is 0.095 lattice)

