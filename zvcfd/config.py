"""Run configuration: one YAML or JSON file per run, validated strictly.

Unknown keys are errors (a typo must not silently fall back to a default).
The resolved configuration is written into the run collection
(``config`` node), and its hash names the run, as in BRIDGE-Simulation.

Example (the HiP-CT coronary tree from its Fluent mesh)::

    name: coronary-50um
    source: {kind: mesh, path: "mesh 1.msh", unit: mm, voxel_size: 50.0}
    physics: {nu: 3.5e-6, rho: 1060.0, u_ref: 0.2}
    boundaries:
      patches:
        - {match: inlet, kind: velocity, flow_rate: 1.9e-7, profile: parabolic}
        - {match: outlet, kind: pressure, pressure: 0.0}
    solver: {collision: trt, mach: 0.05, steps: 200000, check_every: 2000, tolerance: 1.0e-4}
    output: {path: runs, every: 0}
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Source:
    kind: str = "phantom"            # phantom | omezarr | npy | mesh
    name: str | None = None          # phantom name
    path: str | None = None          # omezarr / npy / mesh path
    level: int = 0                   # pyramid level to solve on
    array: str | None = None         # array path inside the multiscale (default: datasets[level])
    threshold: float | None = None   # fluid = value > threshold (or == label)
    label: int | None = None
    voxel_size: float | None = None  # micrometre; taken from OME metadata when omitted
    unit: str = "mm"                 # mesh coordinate unit (mesh sources)
    region: list[list[int]] | None = None  # [[z0, z1], [y0, y1], [x0, x1]] voxel crop


@dataclass
class Physics:
    nu: float = 3.5e-6               # m^2/s (blood ~3.5e-6, water 1.0e-6)
    rho: float = 1060.0              # kg/m^3
    u_ref: float | None = None       # m/s, reference (peak) velocity: sets dt with solver.mach
    pressure_drop: float | None = None  # Pa between the x faces (shorthand for two face patches)
    body_force: list[float] | None = None  # lattice units (z, y, x)
    rheology: dict | None = None     # {model: carreau-yasuda, mu_0, mu_inf, lam, a, n} (SI)


@dataclass
class Boundaries:
    faces: dict = field(default_factory=dict)      # {xmin: {kind, pressure|velocity|...}, ...}
    patches: list = field(default_factory=list)    # [{match: substring, kind, ...}, ...]


@dataclass
class Solver:
    method: str = "lbm"              # lbm | lubrication
    collision: str = "trt"           # trt | bgk
    precision: str = "fp32"          # fp32 | fp16 (population storage)
    tau: float | None = None         # relaxation time; or derived from mach and u_ref
    mach: float = 0.05               # lattice velocity at u_ref (keeps compressibility error small)
    steps: int = 10_000
    check_every: int = 500
    tolerance: float = 1e-5          # relative change of every patch flux between checks
    mass_tolerance: float = 1e-3     # and |inflow - outflow| / inflow: mean pressure settled
    flow_control: bool = True        # correct velocity patches to hit their flow_rate


@dataclass
class Domain:
    brick: int = 8
    chunk_bricks: int = 32
    periodic: bool = False


@dataclass
class Parallel:
    gpus: int = 1                    # devices to use
    partitions: int | None = None    # default: one per GPU (more than GPUs: shared round-robin)


@dataclass
class Output:
    path: str = "runs"
    every: int = 0                   # 0: final state only
    fields: list[str] = field(default_factory=lambda: ["rho", "ux", "uy", "uz"])
    dtype: str = "float32"
    compressor: str | None = "zstd"
    shard_shape: int | None = None


@dataclass
class RunConfig:
    name: str = "run"
    source: Source = field(default_factory=Source)
    physics: Physics = field(default_factory=Physics)
    boundaries: Boundaries = field(default_factory=Boundaries)
    solver: Solver = field(default_factory=Solver)
    domain: Domain = field(default_factory=Domain)
    parallel: Parallel = field(default_factory=Parallel)
    output: Output = field(default_factory=Output)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def hash(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


_SECTIONS = {"source": Source, "physics": Physics, "boundaries": Boundaries, "solver": Solver,
             "domain": Domain, "parallel": Parallel, "output": Output}
_PATCH_KEYS = {"match", "kind", "pressure", "velocity", "flow_rate", "profile", "rcr", "waveform",
               "coronary", "pim"}


def from_dict(d: dict[str, Any]) -> RunConfig:
    unknown = set(d) - {"name", *_SECTIONS}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    kw: dict[str, Any] = {"name": d.get("name", "run")}
    for key, cls in _SECTIONS.items():
        sec = d.get(key) or {}
        names = {f.name for f in dataclasses.fields(cls)}
        bad = set(sec) - names
        if bad:
            raise ValueError(f"unknown keys in {key}: {sorted(bad)}")
        kw[key] = cls(**sec)
    cfg = RunConfig(**kw)
    _validate(cfg)
    return cfg


def load(path: str | Path) -> RunConfig:
    text = Path(path).read_text()
    if str(path).endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML configs need pyyaml: pip install 'zvcfd[yaml]'") from exc
        return from_dict(yaml.safe_load(text) or {})
    return from_dict(json.loads(text))


def _validate(cfg: RunConfig) -> None:
    if cfg.source.kind not in ("phantom", "omezarr", "npy", "mesh"):
        raise ValueError(f"source.kind {cfg.source.kind!r}")
    if cfg.source.kind == "mesh" and not cfg.source.voxel_size:
        raise ValueError("mesh sources need source.voxel_size (micrometre)")
    if cfg.solver.method not in ("lbm", "lubrication"):
        raise ValueError(f"solver.method {cfg.solver.method!r}")
    if cfg.solver.collision not in ("trt", "bgk"):
        raise ValueError(f"solver.collision {cfg.solver.collision!r}")
    if cfg.solver.precision not in ("fp32", "fp16"):
        raise ValueError(f"solver.precision {cfg.solver.precision!r}")
    if cfg.solver.tau is not None and cfg.solver.tau <= 0.5:
        raise ValueError("solver.tau must exceed 0.5")
    if cfg.domain.brick != 8:
        raise ValueError("domain.brick: only 8 is compiled")
    for spec in list(cfg.boundaries.faces.values()) + list(cfg.boundaries.patches):
        bad = set(spec) - _PATCH_KEYS
        if bad:
            raise ValueError(f"unknown patch keys: {sorted(bad)}")
        if spec.get("kind") not in ("pressure", "velocity", "rcr", "coronary"):
            raise ValueError(f"patch kind {spec.get('kind')!r}")
    for face in cfg.boundaries.faces:
        if face not in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
            raise ValueError(f"face {face!r}")
