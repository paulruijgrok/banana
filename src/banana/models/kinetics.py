"""Kinetic binding models: 1:1, piecewise exponential, etc."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class KineticModel(ABC):
    """Base class for kinetic binding models."""

    @abstractmethod
    def __call__(self, t: np.ndarray, *params) -> np.ndarray:
        """Evaluate model at times t with given parameters."""
        pass

    @abstractmethod
    def n_params(self) -> int:
        """Number of free parameters."""
        pass

    @abstractmethod
    def param_names(self) -> list:
        """Names of parameters for reporting."""
        pass


class OneToOneBinding(KineticModel):
    """
    1:1 binding kinetics model.

    Association: R(t) = R_eq * (1 - exp(-k_obs * t))
    where k_obs = k_on * [C] + k_off, R_eq = R_max * [C] / (K_D + [C])

    Dissociation: R(t) = R_0 * exp(-k_off * t)

    For global fitting of association phase across concentrations:
    Shared params: k_on, k_off
    Local params per trace: R_max (optional)
    """

    def __init__(self, phase: str = "association"):
        """
        Parameters
        ----------
        phase : 'association' or 'dissociation'
        """
        self.phase = phase.lower()
        if self.phase not in ("association", "dissociation"):
            raise ValueError("phase must be 'association' or 'dissociation'")

    def association(self, t: np.ndarray, k_on: float, k_off: float, R_max: float, conc: float) -> np.ndarray:
        """Association phase: R(t) = R_eq * (1 - exp(-k_obs * t))."""
        k_obs = k_on * (conc * 1e-9) + k_off  # conc in nM -> M
        K_D = k_off / k_on if k_on > 0 else np.inf
        conc_M = conc * 1e-9
        R_eq = R_max * conc_M / (K_D + conc_M)
        return R_eq * (1 - np.exp(-k_obs * t))

    def dissociation(self, t: np.ndarray, k_off: float, R_0: float) -> np.ndarray:
        """Dissociation phase: R(t) = R_0 * exp(-k_off * t)."""
        return R_0 * np.exp(-k_off * t)

    def __call__(self, t: np.ndarray, *params) -> np.ndarray:
        if self.phase == "association":
            k_on, k_off, R_max, conc = params[:4]
            return self.association(t, k_on, k_off, R_max, conc)
        else:
            k_off, R_0 = params[:2]
            return self.dissociation(t, k_off, R_0)

    def n_params(self) -> int:
        return 4 if self.phase == "association" else 2

    def param_names(self) -> list:
        if self.phase == "association":
            return ["k_on", "k_off", "R_max", "conc"]
        return ["k_off", "R_0"]


class PiecewiseExponential(KineticModel):
    """
    Piecewise exponential: association (rise) + dissociation (decay).

    Association (t1 <= t < t2): R(t) = A1 + (A0 - A1) * exp(-(t-t1)/tau1)
    Dissociation (t >= t2):     R(t) = A2 + (A1_eff - A2) * exp(-(t-t2)/tau2)

    where A1_eff = A1 + (A0 - A1) * exp(-(t2-t1)/tau1)

    Parameters: A1, tau1, A2, tau2 (A0, t1, t2 are fixed from data)
    k_obs = 1/tau1, k_off = 1/tau2
    k_on = (k_obs - k_off) / [conc]
    """

    def __init__(self, A0: float, t1: float, t2: float):
        """
        Parameters
        ----------
        A0 : float
            Baseline value before association.
        t1 : float
            Start of association phase.
        t2 : float
            Start of dissociation phase.
        """
        self.A0 = A0
        self.t1 = t1
        self.t2 = t2

    def __call__(self, t: np.ndarray, A1: float, tau1: float, A2: float, tau2: float) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        tau1 = np.abs(tau1)  # Ensure positive
        tau2 = np.abs(tau2)

        # Association: R = A1 + (A0 - A1) * exp(-(t-t1)/tau1)
        assoc = A1 + (self.A0 - A1) * np.exp(-(t - self.t1) / tau1)

        # Value at end of association
        A1_eff = A1 + (self.A0 - A1) * np.exp(-(self.t2 - self.t1) / tau1)

        # Dissociation: R = A2 + (A1_eff - A2) * exp(-(t-t2)/tau2)
        dissoc = A2 + (A1_eff - A2) * np.exp(-(t - self.t2) / tau2)

        # Do not use np.piecewise with full-length assoc/dissoc arrays; NumPy expects
        # scalars or callables there. Combine phase masks with np.where instead.
        before = t < self.t1
        during = (t >= self.t1) & (t < self.t2)
        return np.where(before, self.A0, np.where(during, assoc, dissoc))

    def n_params(self) -> int:
        return 4

    def param_names(self) -> list:
        return ["A1", "tau1", "A2", "tau2"]

    def rates_from_params(
        self, A1: float, tau1: float, A2: float, tau2: float, concentration_nM: float
    ) -> Tuple[float, float, float, float]:
        """Compute k_obs, k_off, k_on, K_d from fitted parameters."""
        k_obs = 1.0 / np.abs(tau1)
        k_off = 1.0 / np.abs(tau2)
        conc_M = concentration_nM * 1e-9
        k_on = (k_obs - k_off) / conc_M if conc_M > 0 else np.nan
        K_d = k_off / k_on if k_on > 0 else np.nan
        return k_obs, k_off, k_on, K_d
