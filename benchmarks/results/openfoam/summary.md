| Case | Fluid cells | zvCFD to 0.1 % | OpenFOAM to 0.1 % | Speed-up | Flux zvCFD / OpenFOAM | vs analytic (zvCFD, OpenFOAM) |
|---|---:|---:|---:|---:|---:|---|
| duct16 | 32,768 | 0.113 s | 0.39 s | 3× | 0.9889 | 1.004, 1.015 |
| duct32 | 262,144 | 1.04 s | 14.1 s | 14× | 0.9972 | 1.001, 1.004 |
| pipe6 | 10,752 | 0.0168 s | 0.14 s | 8× | 0.9739 | 0.958, 0.984 |
| pipe12 | 86,016 | 0.21 s | 1.52 s | 7× | 0.9932 | 0.981, 0.988 |
| pipe24 | 692,736 | 4.66 s | 102 s | 22× | 0.9979 | 0.990, 0.992 |
| porous | 1,042,190 | 2.45 s | 13.5 s | 6× | 1.0100 | — |
| vessels | 1,315,097 | 12.4 s | 41.9 s | 3× | 0.9904 | — |

| Solver | Cells | Hardware | Inlet pressure | Time to 1 pp | Time to 0.1 pp | Splits vs OpenFOAM (mean / max abs, pp) |
|---|---:|---|---:|---:|---:|---|
| OpenFOAM simpleFoam | 14,790,642 | 16 cores | 65.7 Pa | 9.3 min | 27.6 min | — |
| zvCFD 50 µm | 5,028,753 | RTX A2000 | 62.7 Pa | 0.4 min | 0.7 min | 0.01 / 0.11 |
| zvCFD 35 µm | 14,661,804 | RTX A2000 | 63.1 Pa | 1.7 min | 3.5 min | 0.01 / 0.07 |
