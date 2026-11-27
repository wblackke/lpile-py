"""Pile structural cross-section definitions (US Customary units: inches, kips, ksi)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PileSection:
    """Structural properties of a pile cross-section along a length segment."""

    name: str
    diameter_in: float       # Width/diameter b perpendicular to loading (in)
    area_in2: float          # Cross-sectional area (in^2)
    moment_of_inertia_in4: float  # Flexural moment of inertia I (in^4)
    elastic_modulus_ksi: float    # Young's modulus E (ksi)
    yield_moment_kip_ft: float | None = None  # Optional nominal moment capacity (kip-ft)

    @property
    def ei_kip_in2(self) -> float:
        """Flexural rigidity EI in kip*in^2."""
        return self.elastic_modulus_ksi * self.moment_of_inertia_in4

    @property
    def section_modulus_in3(self) -> float:
        """Extreme fiber elastic section modulus S = I / (b/2) in in^3."""
        return self.moment_of_inertia_in4 / (0.5 * self.diameter_in)


def SteelPipePile(
    outer_diameter_in: float,
    wall_thickness_in: float,
    E_ksi: float = 29000.0,
    fy_ksi: float = 50.0,
    name: str | None = None,
) -> PileSection:
    """Create a circular hollow steel pipe pile section."""
    if wall_thickness_in <= 0 or 2.0 * wall_thickness_in >= outer_diameter_in:
        raise ValueError("Invalid wall thickness relative to outer diameter.")
    d_out = float(outer_diameter_in)
    d_in = d_out - 2.0 * float(wall_thickness_in)
    area = 0.25 * np.pi * (d_out**2 - d_in**2)
    inertia = (np.pi / 64.0) * (d_out**4 - d_in**4)
    # Elastic yield moment My = Fy * S (converted from kip-in to kip-ft)
    my_kip_ft = (fy_ksi * (inertia / (0.5 * d_out))) / 12.0
    label = name or f"Steel Pipe {d_out:.1f} in x {wall_thickness_in:.3f} in"
    return PileSection(
        name=label,
        diameter_in=d_out,
        area_in2=float(area),
        moment_of_inertia_in4=float(inertia),
        elastic_modulus_ksi=float(E_ksi),
        yield_moment_kip_ft=float(my_kip_ft),
    )


def DrilledShaft(
    diameter_in: float,
    fc_ksi: float = 4.0,
    cracked_factor: float = 0.50,
    name: str | None = None,
) -> PileSection:
    """Create a reinforced concrete drilled shaft section (ACI 318 Ec = 57*sqrt(fc_psi) ksi)."""
    d = float(diameter_in)
    area = 0.25 * np.pi * d**2
    i_gross = (np.pi / 64.0) * d**4
    i_eff = cracked_factor * i_gross
    ec_ksi = 57.0 * np.sqrt(fc_ksi * 1000.0)  # ACI 318: 57,000*sqrt(fc_psi) psi = 57*sqrt(fc_psi) ksi
    label = name or f"Drilled Shaft {d:.0f} in (f'c={fc_ksi:.1f} ksi, Ieff={cracked_factor:.2f}Ig)"
    return PileSection(
        name=label,
        diameter_in=d,
        area_in2=float(area),
        moment_of_inertia_in4=float(i_eff),
        elastic_modulus_ksi=float(ec_ksi),
    )


def HPile(
    depth_in: float,
    flange_width_in: float,
    area_in2: float,
    moment_of_inertia_in4: float,
    E_ksi: float = 29000.0,
    fy_ksi: float = 50.0,
    strong_axis: bool = True,
    name: str | None = None,
) -> PileSection:
    """Create a steel H-pile section (strong-axis or weak-axis bending)."""
    b = float(flange_width_in) if strong_axis else float(depth_in)
    c_fiber = 0.5 * (float(depth_in) if strong_axis else float(flange_width_in))
    my_kip_ft = (fy_ksi * (moment_of_inertia_in4 / c_fiber)) / 12.0
    label = name or f"H-Pile ({'Strong' if strong_axis else 'Weak'} Axis)"
    return PileSection(
        name=label,
        diameter_in=b,
        area_in2=float(area_in2),
        moment_of_inertia_in4=float(moment_of_inertia_in4),
        elastic_modulus_ksi=float(E_ksi),
        yield_moment_kip_ft=float(my_kip_ft),
    )


def CustomPileSection(
    diameter_in: float,
    ei_kip_in2: float,
    E_ksi: float = 29000.0,
    name: str = "Custom Pile",
) -> PileSection:
    """Create a custom pile section from diameter and flexural rigidity EI (kip*in^2)."""
    return PileSection(
        name=name,
        diameter_in=float(diameter_in),
        area_in2=0.25 * np.pi * float(diameter_in) ** 2,
        moment_of_inertia_in4=float(ei_kip_in2) / float(E_ksi),
        elastic_modulus_ksi=float(E_ksi),
    )
