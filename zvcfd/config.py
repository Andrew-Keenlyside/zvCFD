"""Run configuration: one YAML or JSON file per run, validated strictly.

Unknown keys are errors (a typo must not silently fall back to a default).
The resolved configuration is written into the run collection
(``config`` node), and its hash names the run, as in BRIDGE-Simulation.

Example::

    name: vessels-demo
    source: {kind: phantom, name: network-128x128x512, voxel_size: 5.0}
    physics: {nu: 3.5e-6, rho: 1060.0, pressure_drop: 50.0}
    solver: {method: lbm, precision: fp32, tau: 1.0, steps: 20000, check_every: 500}
    domain: {brick: 8, chunk_bricks: 32}
    output: {path: runs/, every: 5000, fields: [rho, ux, uy, uz]}
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
    kind: str = "phantom"            # phantom | omezarr | npy
    name: str | None = None          # phantom name
    path: str | None = None          # omezarr / npy path
    level: int = 0                   # pyramid level to solve on
    array: str | None = None         # array path inside the multiscale (default: datasets[level])
    threshold: float | None = None   # fluid = value > threshold (or == label)
    label: int | None = None
    voxel_size: float | None = None  # micrometre; taken from OME metadata when omitted
    region: list[list[int]] | None = None  # [[z0, z1], [y0, y1], [x0, x1]] voxel crop


@dataclass
class Physics:
    nu: float = 3.5e-6               # m^2/s (blood ~3.5e-6, water 1.0e-6)
    rho: float = 1060.0              # kg/m^3
    pressure_drop: float | None = None  # Pa, between the x faces (reservoir BC)
    body_force: list[float] | None = None  # lattice units (z, y, x)


@dataclass
class Solver:
    method: str = "lbm"              # lbm | lubrication
    precision: str = "fp32"          # fp32 | fp16 (population storage)
    tau: float = 1.0
    steps: int = 10_000
    check_every: int = 500
    tolerance: float = 1e-5          # relative flux change between checks, to stop early


@dataclass
class Domain:
    brick: int = 8
    chunk_bricks: int = 32
    periodic: bool = False


@dataclass
class Parallel:
    gpus: int = 1


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


_SECTIONS = {"source": Source, "physics": Physics, "solver": Solver, "domain": Domain,
             "parallel": Parallel, "output": Output}


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
    if cfg.source.kind not in ("phantom", "omezarr", "npy"):
        raise ValueError(f"source.kind {cfg.source.kind!r}")
    if cfg.solver.method not in ("lbm", "lubrication"):
        raise ValueError(f"solver.method {cfg.solver.method!r}")
    if cfg.solver.precision not in ("fp32", "fp16"):
        raise ValueError(f"solver.precision {cfg.solver.precision!r}")
    if cfg.solver.tau <= 0.5:
        raise ValueError("solver.tau must exceed 0.5")
    if cfg.domain.brick != 8:
        raise ValueError("domain.brick: only 8 is compiled")
