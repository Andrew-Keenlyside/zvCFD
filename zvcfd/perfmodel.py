"""Roofline performance and memory model behind ``zvcfd plan`` and ``docs/performance.md``.

LBM is memory-bound: one fluid-cell update reads and writes every
population once, so

    updates/s per GPU = achievable bandwidth x kernel efficiency / bytes per update

*Achievable bandwidth* is a device-to-device copy (STREAM-like), not the
datasheet peak. *Kernel efficiency* is the fraction of that a kernel
reaches per fluid cell. It was measured for the zvCFD kernels on an RTX
A2000 (``benchmarks/bench_lbm.py``) and is assumed to carry over to an
H100. This is the main uncertainty. ``efficiency="target"`` uses published
numbers for tuned kernels instead (FluidX3D, waLBerla); ``"planning"`` (the
default) takes the lower of the two, so a plan never counts on a kernel
that does not exist yet nor on the A2000's unusually high copy efficiency.

Memory: two population buffers plus flags per *stored* cell. Stored cells
are the fluid cells divided by the brick *fill* (the fraction of an
active brick's voxels that are fluid).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

Q = 19


@dataclass(frozen=True)
class GPU:
    name: str
    memory_gb: float
    peak_tbs: float          # datasheet HBM/GDDR bandwidth
    copy_fraction: float     # achievable copy bandwidth / peak
    peer_gbs: float          # GPU-GPU per direction (NVLink via NVSwitch; 0 = through host)
    host_gbs: float          # PCIe host link per direction, achievable

    @property
    def copy_gbs(self) -> float:
        return self.peak_tbs * 1e3 * self.copy_fraction


GPUS = {
    # STREAM 3.12 TB/s measured on H100 SXM (Kempner Institute benchmarks)
    "H100-SXM": GPU("H100-SXM", 80, 3.35, 0.93, 450, 55),
    "H100-PCIe": GPU("H100-PCIe", 80, 2.0, 0.90, 0, 55),
    "A100-SXM-80": GPU("A100-SXM-80", 80, 2.04, 0.90, 300, 25),
    # measured here: 249 GB/s copy
    "RTX-A2000": GPU("RTX-A2000", 12, 0.288, 0.865, 0, 25),
}


@dataclass(frozen=True)
class Method:
    name: str
    bytes_per_update: float     # per fluid cell per step (or per unknown per iteration)
    bytes_per_cell: float       # device memory per stored cell
    efficiency: dict            # {"measured"|"target": {geometry: fraction of copy bandwidth}}


_GEOMETRIES = ("open", "porous", "vessel")

METHODS = {
    "lbm-fp32": Method(
        "lbm-fp32", 2 * Q * 4 + 1, 2 * Q * 4 + 1 + 4,
        {"measured": {"dense-open": 0.98, "dense-porous": 0.55,
                      "sparse-open": 0.77, "sparse-porous": 0.42, "sparse-vessel": 0.38},
         # FluidX3D 80% dense (H100); sparse assumed 70% open, 55% complex
         "target": {"dense-open": 0.80, "dense-porous": 0.60,
                    "sparse-open": 0.70, "sparse-porous": 0.55, "sparse-vessel": 0.55}}),
    "lbm-fp16": Method(
        "lbm-fp16", 2 * Q * 2 + 1, 2 * Q * 2 + 1 + 4,
        # the spike's fp16 path is latency-bound (not yet vectorised)
        {"measured": {"dense-open": 0.61, "dense-porous": 0.27,
                      "sparse-open": 0.29, "sparse-porous": 0.13, "sparse-vessel": 0.12},
         # FluidX3D FP16S reaches 68% on H100
         "target": {"dense-open": 0.68, "dense-porous": 0.50,
                    "sparse-open": 0.60, "sparse-porous": 0.45, "sparse-vessel": 0.45}}),
    # Matrix-free 7-point operator, fp32 values, CG + smoothed-aggregation-like
    # V-cycle (operator complexity ~1.5): ~400 B per unknown per preconditioned iteration.
    "lubrication-amg": Method(
        "lubrication-amg", 400.0, 120.0,
        {"measured": {g: 0.5 for g in ("dense-open", "dense-porous", "sparse-open",
                                       "sparse-porous", "sparse-vessel")},
         "target": {g: 0.6 for g in ("dense-open", "dense-porous", "sparse-open",
                                     "sparse-porous", "sparse-vessel")}}),
}


@dataclass
class Estimate:
    gpu: str
    gpus: int
    method: str
    layout: str
    geometry: str
    fluid_cells: float
    stored_cells: float
    memory_per_gpu_gb: float
    fits: bool
    updates_per_s: float        # aggregate, after parallel efficiency
    seconds_per_step: float
    steps: int | None
    seconds: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def estimate(
    fluid_cells: float,
    *,
    fill: float = 0.6,
    box_cells: float | None = None,
    gpus: int = 8,
    gpu: str = "H100-SXM",
    method: str = "lbm-fp32",
    layout: str = "sparse",
    geometry: str = "vessel",
    steps: int | None = None,
    efficiency: str = "planning",
    parallel_efficiency: float | None = None,
) -> Estimate:
    """Memory and time for ``steps`` updates of ``fluid_cells`` on ``gpus`` x ``gpu``.

    Args:
        fluid_cells: Non-solid voxels.
        fill: Brick fill (fluid / stored cells), sparse layout only.
        box_cells: Bounding-box voxels; required for ``layout="dense"``.
        geometry: ``open`` | ``porous`` | ``vessel`` (kernel-efficiency class).
        parallel_efficiency: Multi-GPU efficiency; default 1.0 for one GPU,
            0.85 for peer-connected GPUs (waLBerla measures >= 82% weak
            scaling), 0.6 when halos go through the host.
    """
    g = GPUS[gpu]
    m = METHODS[method]
    if geometry not in _GEOMETRIES:
        raise ValueError(f"geometry must be one of {_GEOMETRIES}")
    if layout == "dense":
        if box_cells is None:
            raise ValueError("dense layout needs box_cells")
        stored = float(box_cells)
        key = f"dense-{'open' if geometry == 'open' else 'porous'}"
    else:
        stored = fluid_cells / max(fill, 1e-6)
        key = f"sparse-{geometry}"
    if efficiency == "planning":
        eff = min(m.efficiency["measured"][key], m.efficiency["target"][key])
    else:
        eff = m.efficiency[efficiency][key]
    if parallel_efficiency is None:
        parallel_efficiency = 1.0 if gpus == 1 else (0.85 if g.peer_gbs > 0 else 0.6)
    per_gpu = g.copy_gbs * 1e9 * eff / m.bytes_per_update
    rate = per_gpu * gpus * parallel_efficiency
    mem = stored * m.bytes_per_cell / gpus / 1e9
    sps = fluid_cells / rate
    return Estimate(gpu, gpus, method, layout, geometry, fluid_cells, stored, mem,
                    mem <= 0.9 * g.memory_gb, rate, sps, steps,
                    None if steps is None else steps * sps)


def cpu_fv_estimate(cells: float, iterations: int, *, cores: int = 128,
                    cell_iters_per_core_s: float = 1.2e5) -> float:
    """Wall seconds for an unstructured FV (SIMPLE-type) solve on CPU cores.

    Default throughput ~0.1-0.13 M cell-iterations/s/core, the level both
    Fluent (CADFEM, DrivAer) and OpenFOAM (OHC-1 benchmark) report.
    """
    return cells * iterations / (cores * cell_iters_per_core_s)


def gpu_fv_estimate(cells: float, iterations: int, *, gpus: int = 1,
                    cell_iters_per_gpu_s: float = 2.0e8) -> float:
    """Wall seconds for a GPU-native unstructured FV solve (Fluent GPU class).

    Default ~200 M cell-iterations/s per H100, between the vendor-reported
    ~150 (A100) and ~300 (H200) figures; independent OpenFOAM GPU ports are
    ~15-60 M/s per A100.
    """
    return cells * iterations / (gpus * cell_iters_per_gpu_s)


def fv_memory_gb(cells: float, gb_per_million: float = 1.8) -> float:
    """Fluent GPU memory: 1.0-5.6 GB per million cells (SP segregated tet .. DP coupled poly)."""
    return cells / 1e6 * gb_per_million
