"""User-friendly dictionary/table builders & formatted echo tables for pile sections and soil profiles."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from .py_curves import (
    APISand,
    LinearWinklerSoil,
    MatlockSoftClay,
    ReeseSand,
    ReeseStiffClay,
    ReeseWeakRock,
    SoilLayer,
    SoilProfile,
    api_sand_k_pci,
    reese_sand_k_pci,
)
from .sections import CustomPileSection, DrilledShaft, HPile, PileSection, SteelPipePile

SOIL_TYPE_ALIASES: Dict[str, str] = {
    "soft_clay": "soft_clay",
    "matlock": "soft_clay",
    "matlock_soft_clay": "soft_clay",
    "stiff_clay": "stiff_clay",
    "reese_stiff_clay": "stiff_clay",
    "api_sand": "api_sand",
    "sand_api": "api_sand",
    "sand": "reese_sand",
    "reese_sand": "reese_sand",
    "sand_reese": "reese_sand",
    "weak_rock": "weak_rock",
    "rock": "weak_rock",
    "reese_weak_rock": "weak_rock",
    "linear": "linear",
    "winkler": "linear",
}


def build_pile_section(config: Mapping[str, Any]) -> PileSection:
    """Build a PileSection from a plain dictionary with self-documenting keys.

    Supported ``type`` values:
        - ``"steel_pipe"``: ``diameter_in``, ``wall_thickness_in``, optional ``E_ksi`` (29000), ``fy_ksi`` (50)
        - ``"drilled_shaft"``: ``diameter_in``, optional ``fc_ksi`` (4.0), ``cracked_factor`` (0.50)
        - ``"h_pile"``: ``depth_in``, ``flange_width_in``, ``area_in2``, ``inertia_in4``,
          optional ``strong_axis`` (True), ``E_ksi`` (29000), ``fy_ksi`` (50)
        - ``"custom"``: ``diameter_in``, ``ei_kip_in2``, optional ``E_ksi`` (29000)
    """
    ptype = str(config.get("type", "")).strip().lower()
    name = config.get("name")

    if ptype == "steel_pipe":
        return SteelPipePile(
            outer_diameter_in=float(config["diameter_in"]),
            wall_thickness_in=float(config["wall_thickness_in"]),
            E_ksi=float(config.get("E_ksi", 29000.0)),
            fy_ksi=float(config.get("fy_ksi", 50.0)),
            name=name,
        )
    if ptype == "drilled_shaft":
        return DrilledShaft(
            diameter_in=float(config["diameter_in"]),
            fc_ksi=float(config.get("fc_ksi", 4.0)),
            cracked_factor=float(config.get("cracked_factor", 0.50)),
            name=name,
        )
    if ptype == "h_pile":
        return HPile(
            depth_in=float(config["depth_in"]),
            flange_width_in=float(config["flange_width_in"]),
            area_in2=float(config["area_in2"]),
            moment_of_inertia_in4=float(config["inertia_in4"]),
            E_ksi=float(config.get("E_ksi", 29000.0)),
            fy_ksi=float(config.get("fy_ksi", 50.0)),
            strong_axis=bool(config.get("strong_axis", True)),
            name=name,
        )
    if ptype == "custom":
        return CustomPileSection(
            diameter_in=float(config["diameter_in"]),
            ei_kip_in2=float(config["ei_kip_in2"]),
            E_ksi=float(config.get("E_ksi", 29000.0)),
            name=name or "Custom Pile",
        )

    raise ValueError(
        f"Unknown pile type '{ptype}'. Expected one of: 'steel_pipe', 'drilled_shaft', 'h_pile', 'custom'."
    )


def build_soil_profile(
    layers_config: Sequence[Mapping[str, Any]],
    gwl_ft: float = 1e6,
) -> SoilProfile:
    """Build a SoilProfile from a clean list of layer dictionaries.

    Each layer dictionary accepts:
        - ``name`` (str): Display name for tables and plots
        - ``type`` (str): ``"soft_clay"``, ``"stiff_clay"``, ``"reese_sand"``, ``"api_sand"``, ``"weak_rock"``, or ``"linear"``
        - ``top_ft`` / ``bottom_ft`` (float): Layer depth bounds below pile head (ft)
          *(Optional: if ``top_ft`` is omitted, it automatically continues from the previous layer's ``bottom_ft``, starting at 0.0!)*
        - ``gamma_pcf`` (float): Total / saturated unit weight (pcf)
        - ``p_mult`` (float, optional): Group p-multiplier (default 1.0)
        - Model-specific parameters:
            * ``soft_clay``: ``cu_psf``, optional ``eps50`` (0.020), ``J`` (0.50)
            * ``stiff_clay``: ``cu_psf``, optional ``eps50`` (0.007), ``J`` (0.50)
            * ``reese_sand`` / ``api_sand``: ``phi_deg``, optional ``k_pci`` (auto-interpolated if omitted/None)
            * ``weak_rock``: ``qu_psi``, ``E_ir_ksi``, optional ``rqd_pct`` (50.0), ``k_rm`` (0.0005)
            * ``linear``: ``kh_ksi``
    """
    layers: List[SoilLayer] = []
    cursor_ft = 0.0

    for i, cfg in enumerate(layers_config):
        raw_type = str(cfg.get("type", "")).strip().lower()
        if raw_type not in SOIL_TYPE_ALIASES:
            valid = sorted(set(SOIL_TYPE_ALIASES.values()))
            raise ValueError(f"Layer #{i+1} has unknown soil type '{raw_type}'. Valid types: {valid}")
        stype = SOIL_TYPE_ALIASES[raw_type]

        top_ft = float(cfg["top_ft"]) if "top_ft" in cfg and cfg["top_ft"] is not None else cursor_ft
        if "bottom_ft" in cfg and cfg["bottom_ft"] is not None:
            bottom_ft = float(cfg["bottom_ft"])
        elif "thickness_ft" in cfg and cfg["thickness_ft"] is not None:
            bottom_ft = top_ft + float(cfg["thickness_ft"])
        else:
            raise ValueError(f"Layer #{i+1} must specify either 'bottom_ft' or 'thickness_ft'.")
        cursor_ft = bottom_ft

        gamma_pcf = float(cfg["gamma_pcf"])
        p_mult = float(cfg.get("p_mult", 1.0))
        name = str(cfg.get("name", f"Layer {i+1} ({stype})"))

        if stype == "soft_clay":
            if "cu_psf" not in cfg:
                raise ValueError(f"Layer '{name}' (soft_clay) requires 'cu_psf'.")
            model = MatlockSoftClay(
                cu_psf=float(cfg["cu_psf"]),
                eps50=float(cfg.get("eps50", 0.020)),
                J=float(cfg.get("J", 0.50)),
            )
        elif stype == "stiff_clay":
            if "cu_psf" not in cfg:
                raise ValueError(f"Layer '{name}' (stiff_clay) requires 'cu_psf'.")
            model = ReeseStiffClay(
                cu_psf=float(cfg["cu_psf"]),
                eps50=float(cfg.get("eps50", 0.007)),
                J=float(cfg.get("J", 0.50)),
            )
        elif stype == "reese_sand":
            if "phi_deg" not in cfg:
                raise ValueError(f"Layer '{name}' (reese_sand) requires 'phi_deg'.")
            k_val = None if cfg.get("k_pci") is None else float(cfg["k_pci"])
            model = ReeseSand(phi_deg=float(cfg["phi_deg"]), k_pci=k_val)
        elif stype == "api_sand":
            if "phi_deg" not in cfg:
                raise ValueError(f"Layer '{name}' (api_sand) requires 'phi_deg'.")
            k_val = None if cfg.get("k_pci") is None else float(cfg["k_pci"])
            model = APISand(phi_deg=float(cfg["phi_deg"]), k_pci=k_val)
        elif stype == "weak_rock":
            if "qu_psi" not in cfg or "E_ir_ksi" not in cfg:
                raise ValueError(f"Layer '{name}' (weak_rock) requires 'qu_psi' and 'E_ir_ksi'.")
            model = ReeseWeakRock(
                qu_psi=float(cfg["qu_psi"]),
                E_ir_ksi=float(cfg["E_ir_ksi"]),
                rqd_pct=float(cfg.get("rqd_pct", 50.0)),
                k_rm=float(cfg.get("k_rm", 0.0005)),
            )
        else:  # linear
            if "kh_ksi" not in cfg:
                raise ValueError(f"Layer '{name}' (linear) requires 'kh_ksi'.")
            model = LinearWinklerSoil(kh_ksi=float(cfg["kh_ksi"]))

        layers.append(
            SoilLayer(
                name=name,
                top_ft=top_ft,
                bottom_ft=bottom_ft,
                gamma_pcf=gamma_pcf,
                model=model,
                p_multiplier=p_mult,
            )
        )

    return SoilProfile(layers=layers, gwl_ft=gwl_ft)


def _describe_soil_params(lyr: SoilLayer, gwl_ft: float) -> str:
    """Return concise human-readable key parameters string for echo tables and schematics."""
    m = lyr.model
    z_mid = 0.5 * (lyr.top_ft + lyr.bottom_ft)
    sub = z_mid >= gwl_ft
    if isinstance(m, (MatlockSoftClay, ReeseStiffClay)):
        return f"cu={m.cu_psf:.0f} psf, ε50={m.eps50:.3f}, J={m.J:.2f}"
    if isinstance(m, ReeseSand):
        k_used = m.k_pci if m.k_pci is not None else reese_sand_k_pci(m.phi_deg, submerged=sub)
        tag = "" if m.k_pci is not None else " (auto)"
        return f"φ={m.phi_deg:.1f}°, k={k_used:.0f} pci{tag}"
    if isinstance(m, APISand):
        k_used = m.k_pci if m.k_pci is not None else api_sand_k_pci(m.phi_deg, submerged=sub)
        tag = "" if m.k_pci is not None else " (auto)"
        return f"φ={m.phi_deg:.1f}°, k={k_used:.0f} pci{tag}"
    if isinstance(m, ReeseWeakRock):
        return f"qu={m.qu_psi:.0f} psi, Eir={m.E_ir_ksi:.0f} ksi, RQD={m.rqd_pct:.0f}%"
    if isinstance(m, LinearWinklerSoil):
        return f"kh={m.kh_ksi:.3f} ksi"
    return ""


def soil_profile_table(soil: SoilProfile) -> pd.DataFrame:
    """Return a clean engineering summary DataFrame echoing all soil layers and effective stresses."""
    rows = []
    for idx, lyr in enumerate(soil.layers, start=1):
        rows.append(
            {
                "Layer #": idx,
                "Name": lyr.name,
                "Depth (ft)": f"{lyr.top_ft:.1f} – {lyr.bottom_ft:.1f}",
                "Thickness (ft)": lyr.bottom_ft - lyr.top_ft,
                "p-y Model": lyr.model.model_type,
                "γ (pcf)": lyr.gamma_pcf,
                "Key Parameters": _describe_soil_params(lyr, soil.gwl_ft),
                "p-Mult": lyr.p_multiplier,
                "σ'v at Bottom (psf)": soil.effective_stress_psf(lyr.bottom_ft),
            }
        )
    return pd.DataFrame(rows).set_index("Layer #")


def pile_summary_table(pile: PileSection, length_ft: float) -> pd.DataFrame:
    """Return a clean engineering echo table for the pile cross-section."""
    rows = [
        ("Section Name", pile.name),
        ("Embedded Length, L", f"{length_ft:.1f} ft ({length_ft * 12:.0f} in)"),
        ("Diameter / Width, b", f"{pile.diameter_in:.2f} in ({pile.diameter_in / 12:.2f} ft)"),
        ("Cross-Sectional Area, A", f"{pile.area_in2:.2f} in²"),
        ("Moment of Inertia, I", f"{pile.moment_of_inertia_in4:,.1f} in⁴"),
        ("Elastic Modulus, E", f"{pile.elastic_modulus_ksi:,.1f} ksi"),
        ("Flexural Stiffness, EI", f"{pile.ei_kip_in2:,.0f} kip·in² ({pile.ei_kip_in2 / 144:,.0f} kip·ft²)"),
        (
            "Elastic Yield Moment, My",
            f"{pile.yield_moment_kip_ft:,.1f} kip·ft" if pile.yield_moment_kip_ft is not None else "Not specified",
        ),
    ]
    return pd.DataFrame(rows, columns=["Property", "Value"]).set_index("Property")
