"""Standard LPILE nonlinear p-y soil resistance models & layered soil profile builder.

Internal calculation units:
    - Depth z, pile diameter b, deflection y: inches (in)
    - Effective vertical stress sigma_v_eff, cohesion c_u: ksi (kips / in^2)
    - Soil reaction p: kips / in (force per unit length of pile)
    - Secant soil modulus Es = p / y: ksi (kips / in^2)
User-facing constructor inputs use standard US geotechnical units (ft, pcf, psf, pci, degrees).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

GAMMA_WATER_PCF = 62.4
PSF_TO_KSI = 1.0 / 144000.0
PCI_TO_KSI_PER_IN = 1.0 / 1000.0


def api_sand_k_pci(phi_deg: float, submerged: bool = True) -> float:
    """Interpolate API RP 2A initial subgrade modulus k (pci) from friction angle phi (deg)."""
    phi_grid = np.array([25.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 42.0])
    if submerged:
        k_grid = np.array([15.0, 20.0, 40.0, 55.0, 75.0, 95.0, 125.0, 160.0, 200.0])
    else:
        k_grid = np.array([20.0, 25.0, 60.0, 85.0, 115.0, 150.0, 200.0, 250.0, 300.0])
    return float(np.interp(phi_deg, phi_grid, k_grid))


def reese_sand_k_pci(phi_deg: float, submerged: bool = True) -> float:
    """Interpolate Reese, Cox & Koop (1974) initial subgrade modulus k (pci) from phi (deg).

    Matches LPILE Table values: Loose (20/25 pci), Medium (60/90 pci), Dense (125/225 pci)
    for submerged / above-water-table sand respectively.
    """
    phi_grid = np.array([28.0, 33.0, 38.0, 42.0])
    if submerged:
        k_grid = np.array([20.0, 60.0, 125.0, 160.0])
    else:
        k_grid = np.array([25.0, 90.0, 225.0, 275.0])
    return float(np.interp(phi_deg, phi_grid, k_grid))


def reese_sand_ab_factors(z_over_b: float, loading: str = "static") -> Tuple[float, float]:
    """Interpolate Reese, Cox & Koop (1974) empirical depth factors (A, B) vs. z / b."""
    zb_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
    if loading.lower() == "cyclic":
        a_grid = np.array([0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88])
        b_grid = np.array([0.70, 0.67, 0.64, 0.61, 0.58, 0.54, 0.51, 0.49, 0.47, 0.46, 0.45, 0.45])
    else:
        a_grid = np.array([2.83, 2.66, 2.48, 2.28, 2.08, 1.72, 1.44, 1.22, 1.05, 0.96, 0.90, 0.88])
        b_grid = np.array([2.15, 1.98, 1.82, 1.67, 1.53, 1.28, 1.08, 0.90, 0.75, 0.63, 0.55, 0.50])
    zb = float(np.clip(z_over_b, 0.0, 5.0))
    return float(np.interp(zb, zb_grid, a_grid)), float(np.interp(zb, zb_grid, b_grid))


def api_sand_c_coeffs(phi_deg: float) -> Tuple[float, float, float]:
    """Compute API RP 2A (O'Neill & Murchison 1983) C1, C2, C3 coefficients from phi (degrees)."""
    phi = np.radians(np.clip(phi_deg, 20.0, 45.0))
    alpha = 0.5 * phi
    beta = np.pi / 4.0 + 0.5 * phi
    k0 = 0.4
    ka = np.tan(np.pi / 4.0 - 0.5 * phi) ** 2

    tan_b = np.tan(beta)
    tan_a = np.tan(alpha)
    tan_p = np.tan(phi)
    cos_a = np.cos(alpha)
    sin_b = np.sin(beta)

    c1 = (
        (tan_b**2 * tan_a / np.tan(beta - phi))
        + k0 * (tan_p * sin_b / (np.tan(beta - phi) * cos_a) + tan_b * (tan_p * sin_b - tan_a))
    )
    c2 = (tan_b / np.tan(beta - phi)) - ka
    c3 = (
        k0 * tan_p * tan_b**4
        + ka * (tan_b**8 - 1.0)
    )
    return float(c1), float(c2), float(c3)


class SoilModel(ABC):
    """Abstract base class for a p-y soil constitutive model."""

    model_type: str = "Base"
    color: str = "#DCE1E6"

    @abstractmethod
    def p_reaction(
        self,
        y_in: np.ndarray | float,
        z_in: float,
        b_in: float,
        sigma_v_eff_ksi: float,
        submerged: bool = False,
        loading: str = "static",
    ) -> np.ndarray:
        """Return soil resistance p (kips/in) for deflection(s) y_in (in), preserving sign of y."""

    def secant_modulus(
        self,
        y_in: float,
        z_in: float,
        b_in: float,
        sigma_v_eff_ksi: float,
        submerged: bool = False,
        loading: str = "static",
        y_eps: float = 1e-4,
    ) -> float:
        """Return secant subgrade modulus Es = p(|y|) / |y| in ksi (kips/in^2)."""
        y_abs = abs(float(y_in))
        y_eval = max(y_abs, y_eps)
        p_val = float(
            self.p_reaction(
                y_eval,
                z_in=z_in,
                b_in=b_in,
                sigma_v_eff_ksi=sigma_v_eff_ksi,
                submerged=submerged,
                loading=loading,
            )
        )
        return max(p_val / y_eval, 0.0)


@dataclass
class MatlockSoftClay(SoilModel):
    """Soft Clay p-y model (Matlock 1970 / API RP 2A)."""

    cu_psf: float               # Undrained shear strength c_u (psf)
    eps50: float = 0.020        # Strain at 50% max stress difference (typical 0.01 - 0.02)
    J: float = 0.50             # Empirical constant (0.5 soft clay, 0.25 medium clay)
    model_type: str = "Soft Clay (Matlock)"
    color: str = "#D4E6F1"

    def p_reaction(
        self,
        y_in: np.ndarray | float,
        z_in: float,
        b_in: float,
        sigma_v_eff_ksi: float,
        submerged: bool = False,
        loading: str = "static",
    ) -> np.ndarray:
        y_arr = np.asarray(y_in, dtype=float)
        sgn = np.sign(y_arr)
        y = np.abs(y_arr)

        cu_ksi = self.cu_psf * PSF_TO_KSI
        z_eff = max(z_in, 1e-3)
        pu_shallow = (3.0 + sigma_v_eff_ksi / max(cu_ksi, 1e-9) + self.J * (z_eff / b_in)) * cu_ksi * b_in
        pu_deep = 9.0 * cu_ksi * b_in
        pu = min(pu_shallow, pu_deep)

        y50 = max(2.5 * self.eps50 * b_in, 1e-6)
        # Linearize tiny deflections y < y_lin to prevent singular infinite tangent at y = 0
        y_lin = 1e-3 * y50
        p_lin = 0.5 * pu * (y_lin / y50) ** (1.0 / 3.0)

        p = np.where(
            y <= y_lin,
            (p_lin / y_lin) * y,
            np.where(y <= 8.0 * y50, 0.5 * pu * (y / y50) ** (1.0 / 3.0), pu),
        )

        if loading.lower() == "cyclic":
            # Critical depth z_r where shallow wedge transitions to deep flow-around
            gamma_eff_pci = sigma_v_eff_ksi / max(z_eff, 1.0)
            zr = 6.0 * b_in / max((gamma_eff_pci * b_in / max(cu_ksi, 1e-9)) + self.J, 1e-6)
            p_cyc_ult = 0.72 * pu * min(z_eff / max(zr, 1e-3), 1.0)
            # For y > 3*y50: ramp down to p_cyc_ult at 15*y50 (if z < zr), else hold at 0.72*pu
            p_at_3 = 0.5 * pu * (3.0 ** (1.0 / 3.0))  # ~0.721 * pu
            p_cyc = np.where(
                y <= 3.0 * y50,
                p,
                np.where(
                    y <= 15.0 * y50,
                    p_at_3 + (p_cyc_ult - p_at_3) * ((y - 3.0 * y50) / (12.0 * y50)),
                    p_cyc_ult,
                ),
            )
            p = p_cyc

        return sgn * p


@dataclass
class ReeseStiffClay(SoilModel):
    """Stiff Clay without Free Water p-y model (Welch & Reese 1972 / Reese & Welch 1975)."""

    cu_psf: float               # Undrained shear strength c_u (psf)
    eps50: float = 0.007        # Strain at 50% max stress (typical 0.004 - 0.007)
    J: float = 0.50
    model_type: str = "Stiff Clay (Reese-Welch)"
    color: str = "#E8DAEF"

    def p_reaction(
        self,
        y_in: np.ndarray | float,
        z_in: float,
        b_in: float,
        sigma_v_eff_ksi: float,
        submerged: bool = False,
        loading: str = "static",
    ) -> np.ndarray:
        y_arr = np.asarray(y_in, dtype=float)
        sgn = np.sign(y_arr)
        y = np.abs(y_arr)

        cu_ksi = self.cu_psf * PSF_TO_KSI
        z_eff = max(z_in, 1e-3)
        pu = min(
            (3.0 + sigma_v_eff_ksi / max(cu_ksi, 1e-9) + self.J * (z_eff / b_in)) * cu_ksi * b_in,
            9.0 * cu_ksi * b_in,
        )
        y50 = max(2.5 * self.eps50 * b_in, 1e-6)
        y_lin = 1e-3 * y50
        p_lin = 0.5 * pu * (y_lin / y50) ** 0.25

        p = np.where(
            y <= y_lin,
            (p_lin / y_lin) * y,
            np.where(y <= 16.0 * y50, 0.5 * pu * (y / y50) ** 0.25, pu),
        )
        if loading.lower() == "cyclic":
            # Mild cyclic softening factor for stiff clay (100 cycles approx)
            p = np.where(y > 2.0 * y50, 0.80 * p, p)
        return sgn * p


@dataclass
class APISand(SoilModel):
    """Cohesionless Sand p-y model (API RP 2A / O'Neill & Murchison 1983)."""

    phi_deg: float                      # Internal friction angle (degrees)
    k_pci: Optional[float] = None       # Initial subgrade modulus k (pci); auto-computed if None
    model_type: str = "Sand (API RP 2A)"
    color: str = "#FCF3CF"

    def p_reaction(
        self,
        y_in: np.ndarray | float,
        z_in: float,
        b_in: float,
        sigma_v_eff_ksi: float,
        submerged: bool = False,
        loading: str = "static",
    ) -> np.ndarray:
        y_arr = np.asarray(y_in, dtype=float)
        sgn = np.sign(y_arr)
        y = np.abs(y_arr)

        if z_in <= 0.0 or sigma_v_eff_ksi <= 0.0:
            return np.zeros_like(y_arr)

        c1, c2, c3 = api_sand_c_coeffs(self.phi_deg)
        p_us = (c1 * z_in + c2 * b_in) * sigma_v_eff_ksi
        p_ud = c3 * b_in * sigma_v_eff_ksi
        pu = max(min(p_us, p_ud), 1e-9)

        if loading.lower() == "cyclic":
            A = 0.9
        else:
            A = max(3.0 - 0.8 * (z_in / b_in), 0.9)

        k_val_pci = self.k_pci if self.k_pci is not None else api_sand_k_pci(self.phi_deg, submerged=submerged)
        k_ksi_per_in = k_val_pci * PCI_TO_KSI_PER_IN

        arg = (k_ksi_per_in * z_in / (A * pu)) * y
        p = A * pu * np.tanh(np.clip(arg, 0.0, 50.0))
        return sgn * p


@dataclass
class ReeseSand(SoilModel):
    """Cohesionless Sand p-y model (Reese, Cox & Koop 1974 — LPILE 'Sand (Reese)').

    Constructs the classic 4-segment piecewise curve:
        1. Initial linear branch:         p = (k * z) * y                  for 0 <= y <= y_k
        2. Parabolic transition branch:   p = C * y^(1/n)                  for y_k < y <= y_m
        3. Linear hardening branch:       p = p_m + m * (y - y_m)          for y_m < y <= y_u
        4. Ultimate plastic plateau:      p = p_u = A * p_c                for y > y_u
    where y_m = b / 60 and y_u = 3*b / 80.
    """

    phi_deg: float                      # Internal friction angle (degrees)
    k_pci: Optional[float] = None       # Initial subgrade modulus k (pci); auto-computed if None
    model_type: str = "Sand (Reese 1974)"
    color: str = "#FDEBD0"

    def p_reaction(
        self,
        y_in: np.ndarray | float,
        z_in: float,
        b_in: float,
        sigma_v_eff_ksi: float,
        submerged: bool = False,
        loading: str = "static",
    ) -> np.ndarray:
        y_arr = np.asarray(y_in, dtype=float)
        sgn = np.sign(y_arr)
        y = np.abs(y_arr)

        if z_in <= 0.0 or sigma_v_eff_ksi <= 0.0:
            return np.zeros_like(y_arr)

        c1, c2, c3 = api_sand_c_coeffs(self.phi_deg)
        p_st = (c1 * z_in + c2 * b_in) * sigma_v_eff_ksi
        p_sd = c3 * b_in * sigma_v_eff_ksi
        p_c = max(min(p_st, p_sd), 1e-9)

        A, B = reese_sand_ab_factors(z_in / b_in, loading=loading)
        y_m = b_in / 60.0
        y_u = 3.0 * b_in / 80.0
        p_m = B * p_c
        p_u = A * p_c
        m = max((p_u - p_m) / (y_u - y_m), 1e-12)

        k_val_pci = self.k_pci if self.k_pci is not None else reese_sand_k_pci(self.phi_deg, submerged=submerged)
        k_z = max(k_val_pci * PCI_TO_KSI_PER_IN * z_in, 1e-12)

        n = p_m / (m * y_m)

        # Standard 4-branch Reese (1974) regime: initial line intersects parabola before y_m
        if n > 1.001 and (k_z * y_m) > p_m:
            C = p_m / (y_m ** (1.0 / n))
            y_k = (C / k_z) ** (n / (n - 1.0))
            p = np.where(
                y <= y_k,
                k_z * y,
                np.where(
                    y <= y_m,
                    C * np.maximum(y, 1e-15) ** (1.0 / n),
                    np.where(y <= y_u, p_m + m * (y - y_m), p_u),
                ),
            )
        else:
            # Shallow / low-stiffness fallback: linear initial branch directly into segment/plateau
            p_hard = np.where(y <= y_u, p_m + m * (y - y_m), p_u)
            p = np.where(y <= y_m, np.minimum(k_z * y, p_u), np.minimum(k_z * y, p_hard))
            p = np.minimum(p, p_u)

        return sgn * p


@dataclass
class ReeseWeakRock(SoilModel):
    """Weak Rock p-y model (Reese 1997)."""

    qu_psi: float               # Uniaxial compressive strength q_ur (psi)
    E_ir_ksi: float             # Initial rock mass modulus E_ir (ksi)
    rqd_pct: float = 50.0       # Rock Quality Designation (%)
    k_rm: float = 0.0005        # Dimensionless strain constant (0.00005 - 0.0005)
    model_type: str = "Weak Rock (Reese 1997)"
    color: str = "#D5D8DC"

    def p_reaction(
        self,
        y_in: np.ndarray | float,
        z_in: float,
        b_in: float,
        sigma_v_eff_ksi: float,
        submerged: bool = False,
        loading: str = "static",
    ) -> np.ndarray:
        y_arr = np.asarray(y_in, dtype=float)
        sgn = np.sign(y_arr)
        y = np.abs(y_arr)

        qu_ksi = self.qu_psi / 1000.0
        alpha_r = max(1.0 - (2.0 / 3.0) * (self.rqd_pct / 100.0), 1.0 / 3.0)

        # Ultimate rock resistance p_ur (Reese 1997)
        if z_in <= 3.0 * b_in:
            pur = alpha_r * qu_ksi * b_in * (1.0 + 1.4 * (z_in / b_in))
        else:
            pur = 5.2 * alpha_r * qu_ksi * b_in

        # Initial stiffness K_ir
        if z_in <= 3.0 * b_in:
            k_ir = (100.0 + 400.0 * (z_in / (3.0 * b_in))) * self.E_ir_ksi
        else:
            k_ir = 500.0 * self.E_ir_ksi

        y_rm = self.k_rm * b_in
        y_a = min((pur / (2.0 * (y_rm**0.25) * k_ir)) ** (4.0 / 3.0), 16.0 * y_rm)

        p = np.where(
            y <= y_a,
            k_ir * y,
            np.where(
                y <= 16.0 * y_rm,
                0.5 * pur * (y / y_rm) ** 0.25,
                pur,
            ),
        )
        return sgn * np.minimum(p, pur)


@dataclass
class LinearWinklerSoil(SoilModel):
    """Constant linear Winkler spring p = k_h * y (useful for analytical Hetényi verification)."""

    kh_ksi: float               # Constant subgrade modulus Es = p / y (kips / in^2)
    model_type: str = "Linear Winkler"
    color: str = "#EAFAF1"

    def p_reaction(
        self,
        y_in: np.ndarray | float,
        z_in: float,
        b_in: float,
        sigma_v_eff_ksi: float,
        submerged: bool = False,
        loading: str = "static",
    ) -> np.ndarray:
        return np.asarray(y_in, dtype=float) * self.kh_ksi


@dataclass
class SoilLayer:
    """One horizontal soil stratum in a layered geotechnical profile."""

    name: str
    top_ft: float               # Depth to top of layer from ground surface (ft)
    bottom_ft: float            # Depth to bottom of layer (ft)
    gamma_pcf: float            # Total (saturated below GWL) unit weight (pcf)
    model: SoilModel
    p_multiplier: float = 1.0   # Optional group-effect / p-multiplier (default 1.0)

    def __post_init__(self) -> None:
        if self.bottom_ft <= self.top_ft:
            raise ValueError(f"Layer '{self.name}' bottom_ft must exceed top_ft.")


class SoilProfile:
    """Multi-layer soil stratigraphy with groundwater table & exact effective vertical stress."""

    def __init__(self, layers: Sequence[SoilLayer], gwl_ft: float = 1e6) -> None:
        if not layers:
            raise ValueError("SoilProfile requires at least one SoilLayer.")
        self.layers: List[SoilLayer] = sorted(layers, key=lambda l: l.top_ft)
        self.gwl_ft = float(gwl_ft)

    def layer_at(self, z_ft: float) -> Optional[SoilLayer]:
        """Return the soil layer containing depth z_ft (ft), or None if above ground/free-standing."""
        if z_ft < self.layers[0].top_ft:
            return None
        for lyr in self.layers:
            if lyr.top_ft <= z_ft <= lyr.bottom_ft:
                return lyr
        # Extend bottom layer downward if pile extends slightly past defined bottom
        return self.layers[-1]

    def effective_stress_psf(self, z_ft: float) -> float:
        """Integrate effective vertical overburden stress sigma'_v(z) in psf down to depth z_ft."""
        if z_ft <= self.layers[0].top_ft:
            return 0.0
        stress = 0.0
        for lyr in self.layers:
            if z_ft <= lyr.top_ft:
                break
            z0 = lyr.top_ft
            z1 = min(z_ft, lyr.bottom_ft)
            if z1 <= z0:
                continue
            # Split across groundwater table if GWL lies inside [z0, z1]
            if self.gwl_ft <= z0:
                stress += (z1 - z0) * max(lyr.gamma_pcf - GAMMA_WATER_PCF, 0.0)
            elif self.gwl_ft >= z1:
                stress += (z1 - z0) * lyr.gamma_pcf
            else:
                stress += (self.gwl_ft - z0) * lyr.gamma_pcf
                stress += (z1 - self.gwl_ft) * max(lyr.gamma_pcf - GAMMA_WATER_PCF, 0.0)
        # If depth extends past last layer bottom
        last = self.layers[-1]
        if z_ft > last.bottom_ft:
            z0, z1 = last.bottom_ft, z_ft
            eff_g = max(last.gamma_pcf - GAMMA_WATER_PCF, 0.0) if self.gwl_ft <= z0 else last.gamma_pcf
            stress += (z1 - z0) * eff_g
        return float(stress)

    def evaluate_py(
        self,
        z_ft: float,
        y_in: np.ndarray | float,
        b_in: float,
        loading: str = "static",
    ) -> np.ndarray:
        """Evaluate soil reaction p (kips/in) at depth z_ft (ft) for deflection(s) y_in (in)."""
        lyr = self.layer_at(z_ft)
        if lyr is None:
            return np.zeros_like(np.asarray(y_in, dtype=float))
        sigma_v_eff_ksi = self.effective_stress_psf(z_ft) * PSF_TO_KSI
        submerged = z_ft >= self.gwl_ft
        # Depth below ground surface (top of first layer) in inches
        z_below_gs_in = max(z_ft - self.layers[0].top_ft, 0.0) * 12.0
        p = lyr.model.p_reaction(
            y_in=y_in,
            z_in=z_below_gs_in,
            b_in=b_in,
            sigma_v_eff_ksi=sigma_v_eff_ksi,
            submerged=submerged,
            loading=loading,
        )
        return lyr.p_multiplier * p

    def secant_modulus(
        self,
        z_ft: float,
        y_in: float,
        b_in: float,
        loading: str = "static",
    ) -> float:
        """Evaluate secant subgrade modulus Es (ksi) at depth z_ft (ft) and deflection y_in (in)."""
        lyr = self.layer_at(z_ft)
        if lyr is None:
            return 0.0
        sigma_v_eff_ksi = self.effective_stress_psf(z_ft) * PSF_TO_KSI
        submerged = z_ft >= self.gwl_ft
        z_below_gs_in = max(z_ft - self.layers[0].top_ft, 0.0) * 12.0
        es = lyr.model.secant_modulus(
            y_in=y_in,
            z_in=z_below_gs_in,
            b_in=b_in,
            sigma_v_eff_ksi=sigma_v_eff_ksi,
            submerged=submerged,
            loading=loading,
        )
        return lyr.p_multiplier * es
