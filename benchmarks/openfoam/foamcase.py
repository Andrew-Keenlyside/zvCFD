"""OpenFOAM (ESI v2506, official image) cases for the zvCFD comparison benchmarks.

Two kinds of case, both steady laminar ``simpleFoam`` (SIMPLEC, GAMG pressure
solver, second-order schemes), run in parallel with MPI:

- **voxel cases**: the *same* voxel geometry the lattice-Boltzmann solver
  sees. ``blockMesh`` makes the bounding box, a ``cellSet`` lists the fluid
  voxels (blockMesh numbers cells x fastest, exactly like a C-ordered
  (z, y, x) array), and ``subsetMesh`` keeps them; exposed faces become the
  ``walls`` patch. Staircase walls on both sides, so any difference is the
  solver, not the geometry.
- **Fluent cases**: an ASCII Fluent mesh imported with ``fluent3DMeshToFoam``
  (body-fitted, with the zones as patches).

Per-patch volume flow is written every iteration by a ``surfaceFieldValue``
function object; :func:`read_flows` parses it, :func:`read_log` the
timing, so time-to-converged-flux can be compared with zvCFD's monitors.

Environment: ``FOAM_SIF`` names an Apptainer image of an official release
(``apptainer pull esi2506.sif docker://opencfd/openfoam-default:2506``);
``FOAM_ENV`` alternatively names a shell file that puts a native install on
PATH. Do **not** use conda-forge ``openfoam=2412``: its laminar steady and
PIMPLE solvers under-predict duct and pipe flow by ~19 % (see
docs/benchmarks/openfoam.md), while the official v2506 image is exact.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def _w(path: Path, cls: str, obj: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER.format(cls=cls, obj=obj) + body)


def foam(case: Path, cmd: str, log: str, *, env_file: str | None = None, check=True) -> float:
    """Run an OpenFOAM command in ``case``; returns wall seconds.

    ``FOAM_SIF`` (an Apptainer image of an official OpenFOAM release, e.g.
    ``docker://opencfd/openfoam-default:2506``) takes precedence over
    ``FOAM_ENV`` (a shell file that puts a native install on PATH).
    """
    sif = os.environ.get("FOAM_SIF")
    t = time.time()
    if sif:
        bashrc = os.environ.get("FOAM_BASHRC", "/usr/lib/openfoam/openfoam2506/etc/bashrc")
        argv = ["apptainer", "exec", "--bind", str(case.resolve().parent), sif, "bash", "-c",
                f"source {bashrc} && cd {case} && {cmd}"]
    else:
        env_file = env_file or os.environ.get("FOAM_ENV")
        pre = f". {env_file} && " if env_file else ""
        argv = ["bash", "-c", f"{pre}cd {case} && {cmd}"]
    with open(case / log, "w") as fh:
        r = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd} failed in {case}; see {case / log}")
    return time.time() - t


# ---------------------------------------------------------------- common dictionaries

def write_system(case: Path, *, n_procs: int, end: int, flow_patches: list[str],
                 nonorth: int = 0, write_interval: int | None = None,
                 pressure_patches: list[str] = (), modifiable: bool = False) -> None:
    fos = "\n".join(f"""    flow_{i}
    {{
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        regionType      patch;
        name            {p};
        operation       sum;
        fields          (phi);
        writeFields     false;
        log             false;
        writeControl    timeStep;
        writeInterval   1;
    }}""" for i, p in enumerate(flow_patches))
    fos += "\n" + "\n".join(f"""    pressure_{i}
    {{
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        regionType      patch;
        name            {p};
        operation       areaAverage;
        fields          (p);
        writeFields     false;
        log             false;
        writeControl    timeStep;
        writeInterval   1;
    }}""" for i, p in enumerate(pressure_patches))
    _w(case / "system/controlDict", "dictionary", "controlDict", f"""
application     simpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {end};
deltaT          1;
writeControl    timeStep;
writeInterval   {write_interval or end};
purgeWrite      1;
writeFormat     binary;
writePrecision  8;
timeFormat      general;
runTimeModifiable {"true" if modifiable else "false"};

functions
{{
{fos}
}}
""")
    lap = "corrected" if nonorth == 0 else "limited corrected 0.5"
    _w(case / "system/fvSchemes", "dictionary", "fvSchemes", f"""
ddtSchemes      {{ default steadyState; }}
gradSchemes     {{ default Gauss linear; }}
divSchemes
{{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default Gauss linear {lap}; }}
interpolationSchemes {{ default linear; }}
snGradSchemes   {{ default {lap}; }}
""")
    _w(case / "system/fvSolution", "dictionary", "fvSolution", f"""
solvers
{{
    p
    {{
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.05;
    }}
    U
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-9;
        relTol          0.1;
    }}
}}
SIMPLE
{{
    consistent      yes;
    nNonOrthogonalCorrectors {nonorth};
    residualControl {{ p 1e-7; U 1e-8; }}
}}
relaxationFactors
{{
    equations {{ U 0.9; ".*" 0.9; }}
}}
""")
    _w(case / "system/decomposeParDict", "dictionary", "decomposeParDict", f"""
numberOfSubdomains {n_procs};
method          scotch;
""")


def make_transient_ico(case: Path, *, dt: float, steps: int, n_procs: int) -> None:
    """Switch a written case to icoFoam run towards steady state (Euler, PISO).

    icoFoam's momentum equation has only the Laplacian viscous term, which
    for incompressible Newtonian flow is the same PDE as simpleFoam's; see
    docs/benchmarks/openfoam.md for why both are run.
    """
    cd = (case / "system/controlDict").read_text()
    cd = re.sub(r"application\s+\w+;", "application     icoFoam;", cd)
    cd = re.sub(r"endTime\s+[\d.eE+-]+;", f"endTime         {dt * steps:.9g};", cd)
    cd = re.sub(r"deltaT\s+[\d.eE+-]+;", f"deltaT          {dt:.9g};", cd)
    cd = re.sub(r"writeInterval\s+\d+;", f"writeInterval   {steps};", cd)
    (case / "system/controlDict").write_text(cd)
    _w(case / "system/fvSchemes", "dictionary", "fvSchemes", """
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes      { default none; div(phi,U) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
""")
    _w(case / "system/fvSolution", "dictionary", "fvSolution", """
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-10; relTol 0.01; }
    pFinal { $p; relTol 0; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-12; relTol 0; }
}
PISO { nCorrectors 2; nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0; }
""")
    tp = case / "constant/transportProperties"
    tp.write_text(tp.read_text().replace("transportModel Newtonian;\n", ""))


def write_physics(case: Path, nu: float) -> None:
    _w(case / "constant/transportProperties", "dictionary", "transportProperties",
       f"\ntransportModel Newtonian;\nnu {nu:.9g};\n")
    _w(case / "constant/turbulenceProperties", "dictionary", "turbulenceProperties",
       "\nsimulationType laminar;\n")


def write_fields(case: Path, bcs: dict[str, dict]) -> None:
    """``bcs``: patch name or regex -> {"p": ..., "U": ...} OpenFOAM entries (strings)."""
    def block(field):
        return "\n".join(f"    \"{k}\"\n    {{\n        {v[field]}\n    }}" for k, v in bcs.items())
    _w(case / "0/U", "volVectorField", "U", f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{{
{block("U")}
}}
""")
    _w(case / "0/p", "volScalarField", "p", f"""
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{{
{block("p")}
}}
""")


WALL = {"U": "type noSlip;", "p": "type zeroGradient;"}


def pressure(p: float) -> dict:
    return {"U": "type zeroGradient;", "p": f"type fixedValue; value uniform {p:.9g};"}


def flow_rate(q: float) -> dict:
    return {"U": f"type flowRateInletVelocity; volumetricFlowRate constant {q:.9g}; "
                 "value uniform (0 0 0);",
            "p": "type zeroGradient;"}


# ---------------------------------------------------------------- voxel cases

def write_voxel_case(case: Path, fluid: np.ndarray, dx: float, nu: float, *,
                     faces: dict[str, dict], n_procs: int, end: int) -> int:
    """Write a case whose mesh is exactly the fluid voxels of ``fluid`` (z, y, x).

    ``faces`` maps box faces (``xmin`` ...) to BC dicts (:func:`pressure`,
    :func:`flow_rate`); other exposed faces are no-slip walls. Returns the
    number of cells.
    """
    if case.exists():
        shutil.rmtree(case)
    nz, ny, nx = fluid.shape
    X, Y, Z = nx * dx, ny * dx, nz * dx
    names = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
    quads = {"xmin": "(0 4 7 3)", "xmax": "(1 2 6 5)", "ymin": "(0 1 5 4)",
             "ymax": "(3 7 6 2)", "zmin": "(0 3 2 1)", "zmax": "(4 5 6 7)"}
    _w(case / "system/blockMeshDict", "dictionary", "blockMeshDict", f"""
scale 1;
vertices
(
    (0 0 0) ({X:.9g} 0 0) ({X:.9g} {Y:.9g} 0) (0 {Y:.9g} 0)
    (0 0 {Z:.9g}) ({X:.9g} 0 {Z:.9g}) ({X:.9g} {Y:.9g} {Z:.9g}) (0 {Y:.9g} {Z:.9g})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
boundary
(
{chr(10).join(f"    {n} {{ type patch; faces ( {quads[n]} ); }}" for n in names)}
    walls {{ type wall; faces (); }}
);
""")
    write_system(case, n_procs=n_procs, end=end, flow_patches=list(faces))
    write_physics(case, nu)
    bcs = {"walls": WALL}
    for n in names:
        bcs[n] = faces.get(n, WALL)
    # fields are written after subsetMesh (mesh_voxel_case): subsetMesh -overwrite rewrites
    # 0/ and gives the new "walls" patch a calculated (frictionless) condition
    import json

    (case / "bcs.json").write_text(json.dumps(bcs))
    ids = np.flatnonzero(fluid.reshape(-1))
    body = f"\n{len(ids)}\n(\n" + "\n".join(map(str, ids)) + "\n)\n"
    # staged outside polyMesh: blockMesh clears constant/polyMesh
    (case / "fluid.cellSet").write_text(HEADER.format(cls="cellSet", obj="fluid") + body)
    return int(len(ids))


def mesh_voxel_case(case: Path) -> dict:
    """blockMesh the box, then subset it to the fluid voxels."""
    t = {"blockMesh_s": foam(case, "blockMesh", "log.blockMesh")}
    sets = case / "constant/polyMesh/sets"
    sets.mkdir(parents=True, exist_ok=True)
    shutil.copy(case / "fluid.cellSet", sets / "fluid")
    t["subsetMesh_s"] = foam(case, "subsetMesh fluid -patch walls -overwrite", "log.subsetMesh")
    import json

    shutil.rmtree(case / "0", ignore_errors=True)
    write_fields(case, json.loads((case / "bcs.json").read_text()))
    return t


# ---------------------------------------------------------------- Fluent cases

def import_fluent(case: Path, msh: str, *, scale: float = 1e-3) -> dict:
    """Import an ASCII Fluent mesh (``fluent3DMeshToFoam``) and scale it to metres."""
    case.mkdir(parents=True, exist_ok=True)
    write_system(case, n_procs=1, end=1, flow_patches=[])
    link = case / "source.msh"                    # OpenFOAM's argument parser splits on spaces
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(msh).resolve())
    t = {"fluent3DMeshToFoam_s": foam(case, "fluent3DMeshToFoam source.msh", "log.import")}
    t["transformPoints_s"] = foam(case, f"transformPoints -scale '({scale} {scale} {scale})'",
                                  "log.transform")
    return t


def patch_names(case: Path) -> list[str]:
    txt = (case / "constant/polyMesh/boundary").read_text()
    return re.findall(r"^\s{4}(\S+)\s*\n\s{4}\{", txt, flags=re.M)


# ---------------------------------------------------------------- running and reading

def run_parallel(case: Path, n_procs: int, solver: str = "simpleFoam") -> dict:
    t = {"decomposePar_s": foam(case, "decomposePar -force", "log.decomposePar")}
    cmd = f"mpirun -np {n_procs} {solver} -parallel" if n_procs > 1 else solver
    t["solve_s"] = foam(case, cmd, "log.simpleFoam")      # same log name for read_log
    return t


def read_log(case: Path) -> dict:
    """Iterations, per-iteration ExecutionTime, final residuals."""
    txt = (case / "log.simpleFoam").read_text()
    its = [int(x) for x in re.findall(r"^Time = (\d+)", txt, flags=re.M)]
    exe = [float(x) for x in re.findall(r"ExecutionTime = ([\d.eE+-]+) s", txt)]
    conv = "SIMPLE solution converged" in txt
    cells = re.search(r"nCells:\s+(\d+)", (case / "log.checkMesh").read_text()) \
        if (case / "log.checkMesh").exists() else None
    return {"iterations": its[-1] if its else 0, "exec_s": exe, "converged": conv,
            "cells": int(cells.group(1)) if cells else None}


def read_flows(case: Path, n: int) -> np.ndarray:
    """``(iterations, n)`` summed phi per monitored patch (outward positive in OpenFOAM)."""
    cols = []
    for i in range(n):
        f = sorted((case / "postProcessing" / f"flow_{i}").glob("*/surfaceFieldValue*.dat"))
        data = np.loadtxt(f[-1], comments="#")
        cols.append(data[:, 1])
    m = min(len(c) for c in cols)
    return np.stack([c[:m] for c in cols], 1)
