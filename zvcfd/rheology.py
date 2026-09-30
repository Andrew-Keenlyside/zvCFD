"""Generalised-Newtonian viscosity models, in SI, shared by the solvers.

Each model is a callable ``mu(gamma_dot)`` (Pa·s) of the shear rate
``γ̇ = sqrt(2 S:S)`` (1/s), with optional clips of the shear rate to
``[gamma_min, gamma_max]`` as CFX offers (CFX-Solver Modeling Guide §1.2.13).
Without clips the models are written so that complex arguments pass
through, which the manufactured solutions use (complex-step derivatives).

Models:

- :class:`CarreauYasuda`: ``μ∞ + (μ0 − μ∞) [1 + (λ γ̇)^a]^((n − 1)/a)``;
  :meth:`CarreauYasuda.blood` gives Cho & Kensey's (1991) parameters.
  ``a = 2`` is the Bird–Carreau model.
- :class:`Cross`: ``μ∞ + (μ0 − μ∞) / (1 + (λ γ̇)^m)``.
- :class:`PowerLaw`: ``K γ̇^(n − 1)``, clipped to ``[mu_min, mu_max]``.
- :class:`Casson`: ``(sqrt(τ_y / γ̇) + sqrt(μ_p))²``, regularised below
  ``gamma_min`` (the model is singular at zero shear).

References: Y. I. Cho, K. R. Kensey, Biorheology 28, 241 (1991); R. B.
Bird, R. C. Armstrong, O. Hassager, *Dynamics of Polymeric Liquids*,
vol. 1 (1987).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clip(g, lo, hi):
    if lo is None and hi is None:
        return g
    return np.clip(g, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)


@dataclass(frozen=True)
class CarreauYasuda:
    mu_0: float = 0.056
    mu_inf: float = 0.00345
    lam: float = 3.313
    a: float = 2.0
    n: float = 0.3568
    gamma_min: float | None = None
    gamma_max: float | None = None

    @classmethod
    def blood(cls, **kw) -> CarreauYasuda:
        """Cho & Kensey (1991): μ0 0.056, μ∞ 0.00345 Pa·s, λ 3.313 s, n 0.3568, a 2."""
        return cls(**kw)

    def __call__(self, gamma):
        g = _clip(gamma, self.gamma_min, self.gamma_max)
        return self.mu_inf + (self.mu_0 - self.mu_inf) * (1 + (self.lam * g) ** self.a) ** (
            (self.n - 1) / self.a)

    def of_gamma_squared(self, g2):
        """The same for ``a = 2`` as a function of ``γ̇²``: smooth at zero shear."""
        if self.a != 2:
            raise ValueError("of_gamma_squared needs a = 2")
        return self.mu_inf + (self.mu_0 - self.mu_inf) * (1 + self.lam ** 2 * g2) ** (
            (self.n - 1) / 2)


@dataclass(frozen=True)
class Cross:
    mu_0: float
    mu_inf: float
    lam: float
    m: float
    gamma_min: float | None = None
    gamma_max: float | None = None

    def __call__(self, gamma):
        g = _clip(gamma, self.gamma_min, self.gamma_max)
        return self.mu_inf + (self.mu_0 - self.mu_inf) / (1 + (self.lam * g) ** self.m)


@dataclass(frozen=True)
class PowerLaw:
    k: float
    n: float
    mu_min: float = 0.0
    mu_max: float = np.inf
    gamma_min: float | None = 1e-3

    def __call__(self, gamma):
        g = _clip(gamma, self.gamma_min, None)
        return np.clip(self.k * g ** (self.n - 1), self.mu_min, self.mu_max)


@dataclass(frozen=True)
class Casson:
    tau_y: float
    mu_p: float
    gamma_min: float = 1e-2

    def __call__(self, gamma):
        g = np.maximum(gamma, self.gamma_min)
        return (np.sqrt(self.tau_y / g) + np.sqrt(self.mu_p)) ** 2


MODELS = {"carreau-yasuda": CarreauYasuda, "cross": Cross, "power-law": PowerLaw,
          "casson": Casson}


def from_config(spec: dict):
    """A model from a config table ``{model: carreau-yasuda, mu_0: ..., ...}`` (SI)."""
    spec = dict(spec)
    name = spec.pop("model", "carreau-yasuda")
    if name not in MODELS:
        raise ValueError(f"rheology model {name!r}; expected one of {sorted(MODELS)}")
    return MODELS[name](**spec)


__all__ = ["Casson", "CarreauYasuda", "Cross", "MODELS", "PowerLaw", "from_config"]
