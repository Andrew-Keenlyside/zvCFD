"""Lattice <-> physical unit conversion, and diffusive scaling between pyramid levels.

A lattice is fixed by the voxel size ``dx`` (m), the physical kinematic
viscosity ``nu`` (m^2/s) and the relaxation time ``tau``; the time step
follows from ``nu = (tau - 1/2)/3 * dx^2/dt``.

Coarsening by 2 at fixed ``tau`` (diffusive scaling) gives ``dt -> 4 dt``:
lattice velocity doubles, lattice pressure (density) differences
quadruple, and a lattice body force grows eightfold. These are the factors
used to restrict boundary conditions and prolong solutions between
OME-Zarr pyramid levels (``docs/multiresolution.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

CS2 = 1.0 / 3.0


@dataclass(frozen=True)
class Lattice:
    dx: float          # m per voxel
    nu: float          # m^2/s
    tau: float = 1.0
    rho: float = 1060.0  # kg/m^3 (blood); 1000 for water

    @property
    def nu_lattice(self) -> float:
        return (self.tau - 0.5) * CS2

    @property
    def dt(self) -> float:
        return self.nu_lattice * self.dx ** 2 / self.nu

    def velocity(self, u_lattice: float) -> float:
        return u_lattice * self.dx / self.dt

    def u_lattice(self, u: float) -> float:
        return u * self.dt / self.dx

    def pressure(self, drho_lattice: float) -> float:
        """Physical pressure difference (Pa) of a lattice density difference."""
        return drho_lattice * CS2 * self.rho * (self.dx / self.dt) ** 2

    def drho_lattice(self, dp: float) -> float:
        return dp / (CS2 * self.rho * (self.dx / self.dt) ** 2)

    def steps(self, seconds: float) -> int:
        return int(round(seconds / self.dt))

    def mach(self, u: float) -> float:
        return self.u_lattice(u) / CS2 ** 0.5

    def coarsened(self, factor: int = 2) -> Lattice:
        return Lattice(self.dx * factor, self.nu, self.tau, self.rho)


def level_factors(k: int) -> dict[str, float]:
    """Lattice-unit multipliers from level 0 to level ``k`` (2^k coarser) at fixed tau."""
    return {"velocity": 2.0 ** k, "drho": 4.0 ** k, "force": 8.0 ** k, "dt": 4.0 ** k}
