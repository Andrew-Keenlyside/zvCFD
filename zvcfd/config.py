"""Run configuration: one YAML or JSON file per run, validated strictly.

Unknown keys are errors (a typo must not silently fall back to a default).
The resolved configuration is written into the run collection
(``config`` node), and its hash names the run, as in BRIDGE-Simulation.

``solver.method`` defaults to ``auto``: the finite-volume solver (the main
one, CFX-style, on the mesh itself) for mesh sources, and the
lattice-Boltzmann solver for voxel sources and for meshes given a
``source.voxel_size`` (:func:`resolve_method`). The resolved method is what
is stored and hashed.

Example (the HiP-CT coronary tree on its Ansys Fluent mesh, finite volume;
settings in ``fv``)::

    name: coronary-fv
    source: {kind: mesh, path: "mesh 1.msh", unit: mm}
    physics: {nu: 3.5e-6, rho: 1060.0}
    boundaries:
      patches:
        - {match: inlet, kind: velocity, flow_rate: 1.9e-7, profile: parabolic}
        - {match: outlet, kind: rcr, rcr: [1.0e9, 1.0e-10, 1.0e10]}
    fv: {linear: amgx, iterations: 300, tolerance: 1.0e-6}

The same mesh voxelised for the lattice-Boltzmann solver (a voxel size
selects it; ``solver.method: lbm`` says so explicitly)::

    name: coronary-50um
    source: {kind: mesh, path: "mesh 1.msh", unit: mm, voxel_size: 50.0}
    physics: {nu: 3.5e-6, rho: 1060.0, u_ref: 0.2}
    boundaries:
      patches:
        - {match: inlet, kind: velocity, flow_rate: 1.9e-7, profile: parabolic}
        - {match: outlet, kind: pressure, pressure: 0.0}
    solver: {method: lbm, collision: trt, mach: 0.05, steps: 200000, check_every: 2000,
             tolerance: 1.0e-4}
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
    path: str | None = None          # omezarr / npy / mesh: a .zvmesh collection, or a raw mesh
                                     # (.msh, .vtu, mesh-complete dir, meshio) imported to one
    level: int = 0                   # pyramid level to solve on
    array: str | None = None         # array path inside the multiscale (default: datasets[level])
    threshold: float | None = None   # fluid = value > threshold (or == label)
    label: int | None = None
    voxel_size: float | None = None  # micrometre; taken from OME metadata when omitted
    unit: str = "mm"                 # raw mesh coordinate unit (a .zvmesh is in metres)
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
    method: str = "auto"             # auto | fv | lbm | lubrication (auto: resolve_method)
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
class Fv:
    """Finite-volume settings (``solver.method: fv``)."""

    # auto (AmgX ladder, zvcfd.fv.linear.Auto) | amgx | simple | acm | host-direct
    linear: str = "auto"
    linear_rtol: float = 0.1         # residual reduction per outer iteration
    precision: str = "double"        # double | mixed (AmgX: single-precision hierarchy)
    advection: Any = "high-resolution"   # high-resolution | upwind | blend factor 0..1
    false_dt: float | None = None    # steady runs: pseudo time step (s); None = none
    iterations: int = 300            # steady: maximum outer iterations
    tolerance: float = 1e-6          # steady: relative change of u and p
    residual_target: float | None = None  # or: largest RMS normalised residual (steady, loops)
    backflow_stabilisation: float = 0.2  # default for pressure and lumped outlets
    # transient (dt set): after a steady start, march with BDF2
    dt: float | None = None          # s; None = steady only
    steps: int | None = None         # time steps; or periods x period / dt
    periods: float | None = None     # cardiac cycles (with period)
    period: float | None = None      # s; default: the longest waveform period
    loops: int = 5                   # coefficient loops per step
    loop_tolerance: float = 1e-5
    scheme: str = "bdf2"
    steady_start: bool = True        # solve the steady state at t = 0 first
    wss: bool = True                 # wall shear stress in snapshots and the boundary store
    tawss_from: float | None = None  # s; default: the last period
    chunk: float | None = None       # mesh store chunk edge (m); default: bounding box / 4
    # RANS turbulence: None (laminar) or {model: k-kl, k_inf, kl_inf | speed, mach; relax}
    turbulence: dict | None = None


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
    fv: Fv = field(default_factory=Fv)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def hash(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


_SECTIONS = {"source": Source, "physics": Physics, "boundaries": Boundaries, "solver": Solver,
             "domain": Domain, "parallel": Parallel, "output": Output, "fv": Fv}
_PATCH_KEYS = {"match", "kind", "pressure", "velocity", "flow_rate", "profile", "rcr", "waveform",
               "coronary", "pim", "backflow_stabilisation", "pressure_profile", "opening"}
_FV_ONLY_KEYS = {"backflow_stabilisation", "pressure_profile", "opening"}


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
        return from_dict(yaml.load(text, Loader=_yaml_loader(yaml)) or {})
    return from_dict(json.loads(text))


def _yaml_loader(yaml):
    """PyYAML's safe loader, reading exponent forms such as ``1e8`` and ``2.0e-7`` as floats.

    PyYAML follows YAML 1.1, whose floats need a decimal point and a signed
    exponent, so ``rcr: [1.0e8, ...]`` would arrive as a string; YAML 1.2 (and
    every config author) reads it as a number.
    """
    import re

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(r"^[-+]?(?:\d[\d_]*(?:\.[\d_]*)?|\.\d[\d_]*)[eE][-+]?\d+$"),
        list("-+0123456789."))
    return Loader


def resolve_method(cfg: RunConfig) -> str:
    """The solver ``solver.method: auto`` stands for.

    The finite-volume solver is the main one: it takes every mesh source. The
    lattice-Boltzmann solver takes voxel sources (OME-Zarr images, arrays,
    phantoms), and meshes given a ``source.voxel_size``, which asks for a
    voxelised run.
    """
    if cfg.solver.method != "auto":
        return cfg.solver.method
    return "fv" if cfg.source.kind == "mesh" and not cfg.source.voxel_size else "lbm"


def _validate(cfg: RunConfig) -> None:
    if cfg.source.kind not in ("phantom", "omezarr", "npy", "mesh"):
        raise ValueError(f"source.kind {cfg.source.kind!r}")
    cfg.solver.method = resolve_method(cfg)        # stored resolved: run hashes do not change
    fv = cfg.solver.method == "fv"
    if cfg.solver.method not in ("lbm", "lubrication", "fv"):
        raise ValueError(f"solver.method {cfg.solver.method!r}")
    if fv and cfg.source.kind != "mesh":
        raise ValueError("solver.method fv needs a mesh source")
    if cfg.source.kind == "mesh" and not fv and not cfg.source.voxel_size:
        raise ValueError("mesh sources need source.voxel_size (micrometre)")
    if cfg.fv.linear not in ("auto", "amgx", "simple", "acm", "host-direct"):
        raise ValueError(f"fv.linear {cfg.fv.linear!r}")
    if cfg.fv.precision not in ("double", "mixed"):
        raise ValueError(f"fv.precision {cfg.fv.precision!r}")
    if cfg.fv.scheme not in ("bdf1", "bdf2"):
        raise ValueError(f"fv.scheme {cfg.fv.scheme!r}")
    if cfg.fv.turbulence is not None:
        t = cfg.fv.turbulence
        bad = set(t) - {"model", "k_inf", "kl_inf", "speed", "mach", "relax"}
        if bad:
            raise ValueError(f"unknown fv.turbulence keys: {sorted(bad)}")
        if t.get("model") != "k-kl":
            raise ValueError(f"fv.turbulence.model {t.get('model')!r}: k-kl (k-kL-MEAH2015m)")
        if not fv:
            raise ValueError("fv.turbulence needs solver.method fv")
        given = ("k_inf" in t and "kl_inf" in t, "speed" in t and "mach" in t)
        if given.count(True) != 1:
            raise ValueError("fv.turbulence: give k_inf and kl_inf, or speed and mach "
                             "(the NASA TMR freestream values)")
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
        kinds = ("pressure", "velocity", "rcr", "coronary") + (("wall", "symmetry") if fv else ())
        if spec.get("kind") not in kinds:
            raise ValueError(f"patch kind {spec.get('kind')!r}")
        if not fv and set(spec) & _FV_ONLY_KEYS:
            keys = sorted(set(spec) & _FV_ONLY_KEYS)
            raise ValueError(f"patch keys {keys} need solver.method fv")
    for face in cfg.boundaries.faces:
        if face not in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
            raise ValueError(f"face {face!r}")
