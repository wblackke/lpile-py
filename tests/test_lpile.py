"""Verification & unit tests for the LPILE-style nonlinear p-y pile solver."""

from __future__ import annotations

import numpy as np
import pytest

from src.group import (
    CapLoadCase,
    GroupPileSpec,
    PileCapSpec,
    aashto_p_multipliers,
    solve_pile_group,
)
from src.inputs import (
    build_pile_section,
    build_soil_profile,
    pile_summary_table,
    soil_profile_table,
)
from src.py_curves import (
    APISand,
    LinearWinklerSoil,
    MatlockSoftClay,
    ReeseSand,
    ReeseStiffClay,
    ReeseWeakRock,
    SoilLayer,
    SoilProfile,
)
from src.sections import CustomPileSection, DrilledShaft, SteelPipePile
from src.solver import BoundaryCondition, run_pushover, solve_lateral_pile


def test_exact_cantilever_beam_shear_and_moment() -> None:
    """Verify FEM solver reproduces exact Euler-Bernoulli cantilever beam formulas."""
    ei = 1.0e8  # kip*in^2
    L_ft = 20.0
    L_in = L_ft * 12.0
    V_t = 50.0  # kips
    pile = CustomPileSection(diameter_in=24.0, ei_kip_in2=ei)
    zero_soil = SoilProfile([SoilLayer("Air", 0.0, 30.0, 0.0, LinearWinklerSoil(0.0))])

    # Pure lateral shear at free tip z=0, clamped at z=L
    res_v = solve_lateral_pile(
        pile, L_ft, zero_soil,
        BoundaryCondition(head="free", lateral_load_kips=V_t, tip="fixed"),
        n_elements=80,
    )
    assert res_v.converged
    expected_y0 = V_t * L_in**3 / (3.0 * ei)
    expected_theta0 = -V_t * L_in**2 / (2.0 * ei)
    assert np.isclose(res_v.head_deflection_in, expected_y0, rtol=1e-4)
    assert np.isclose(res_v.head_slope_rad, expected_theta0, rtol=1e-4)
    assert np.isclose(res_v.profile["moment_kip_ft"].iloc[-1], V_t * L_ft, rtol=1e-3)

    # Pure applied head moment M_t (positive M_t bends top in +y direction)
    M_t_kft = 200.0
    M_t_kin = M_t_kft * 12.0
    res_m = solve_lateral_pile(
        pile, L_ft, zero_soil,
        BoundaryCondition(head="free", lateral_load_kips=0.0, moment_kip_ft=M_t_kft, tip="fixed"),
        n_elements=80,
    )
    expected_ym = M_t_kin * L_in**2 / (2.0 * ei)
    assert np.isclose(res_m.head_deflection_in, expected_ym, rtol=1e-4)
    assert np.isclose(res_m.head_moment_kip_ft, M_t_kft, rtol=1e-3)


def test_exact_hetenyi_winkler_free_and_fixed_head() -> None:
    """Verify against Hetényi (1946) exact closed-form solutions for a long beam on a linear Winkler foundation."""
    ei = 2.0e8        # kip*in^2
    k0 = 0.5          # ksi (kips / in^2)
    V_t = 100.0       # kips
    beta = (k0 / (4.0 * ei)) ** 0.25  # ~0.00889 1/in => beta*L >> 4 for L = 80 ft (960 in)
    L_ft = 80.0

    pile = CustomPileSection(diameter_in=24.0, ei_kip_in2=ei)
    soil = SoilProfile([SoilLayer("Linear Winkler", 0.0, 100.0, 120.0, LinearWinklerSoil(k0))])

    # 1. Free-Head Hetényi exact solution
    res_free = solve_lateral_pile(
        pile, L_ft, soil,
        BoundaryCondition(head="free", lateral_load_kips=V_t, tip="free"),
        n_elements=200,
    )
    assert res_free.converged
    y0_exact_free = V_t / (2.0 * ei * beta**3)
    th0_exact_free = -V_t / (2.0 * ei * beta**2)
    mmax_exact_free_kft = (V_t / beta) * np.exp(-np.pi / 4.0) * np.sin(np.pi / 4.0) / 12.0

    assert np.isclose(res_free.head_deflection_in, y0_exact_free, rtol=5e-3)
    assert np.isclose(res_free.head_slope_rad, th0_exact_free, rtol=5e-3)
    assert np.isclose(res_free.max_moment_kip_ft, mmax_exact_free_kft, rtol=1e-2)

    # 2. Fixed-Head Hetényi exact solution (theta_0 = 0)
    res_fixed = solve_lateral_pile(
        pile, L_ft, soil,
        BoundaryCondition(head="fixed", lateral_load_kips=V_t, tip="free"),
        n_elements=200,
    )
    assert res_fixed.converged
    y0_exact_fixed = V_t / (4.0 * ei * beta**3)
    m0_exact_fixed_kft = -(V_t / (2.0 * beta)) / 12.0

    assert np.isclose(res_fixed.head_deflection_in, y0_exact_fixed, rtol=5e-3)
    assert abs(res_fixed.head_slope_rad) < 1e-10
    assert np.isclose(res_fixed.head_moment_kip_ft, m0_exact_fixed_kft, rtol=1e-2)


def test_layered_nonlinear_soil_equilibrium_and_p_delta() -> None:
    """Verify multi-layer nonlinear p-y convergence, horizontal force equilibrium, and P-Delta softening."""
    pile = SteelPipePile(outer_diameter_in=24.0, wall_thickness_in=0.5)
    soil = SoilProfile(
        layers=[
            SoilLayer("Soft Clay", 0.0, 15.0, 110.0, MatlockSoftClay(cu_psf=600.0, eps50=0.02)),
            SoilLayer("Medium Sand", 15.0, 35.0, 122.0, APISand(phi_deg=34.0)),
            SoilLayer("Stiff Clay", 35.0, 50.0, 125.0, ReeseStiffClay(cu_psf=2500.0, eps50=0.007)),
            SoilLayer("Weak Rock", 50.0, 70.0, 140.0, ReeseWeakRock(qu_psi=500.0, E_ir_ksi=50.0, rqd_pct=60.0)),
        ],
        gwl_ft=6.0,
    )

    V_t = 60.0
    res_no_p = solve_lateral_pile(
        pile, 60.0, soil,
        BoundaryCondition(head="free", lateral_load_kips=V_t, axial_load_kips=0.0),
        n_elements=120,
    )
    assert res_no_p.converged
    # Global horizontal equilibrium: integral of soil reaction p(z) dz must equal applied shear V_t
    p_integral = float(np.trapezoid(res_no_p.profile["p_kips_per_ft"], res_no_p.profile["z_ft"]))
    assert np.isclose(p_integral, V_t, rtol=0.02)

    # Adding compressive axial load P_x > 0 must increase lateral deflection & bending moment (P-Delta)
    res_with_p = solve_lateral_pile(
        pile, 60.0, soil,
        BoundaryCondition(head="free", lateral_load_kips=V_t, axial_load_kips=400.0),
        n_elements=120,
    )
    assert res_with_p.converged
    assert res_with_p.head_deflection_in > res_no_p.head_deflection_in
    assert abs(res_with_p.max_moment_kip_ft) > abs(res_no_p.max_moment_kip_ft)

    # Cyclic loading degrades Matlock soft clay above y = 3*y50 (= 3.6 in for b=24, eps50=0.02)
    res_high_static = solve_lateral_pile(
        pile, 60.0, soil,
        BoundaryCondition(head="free", lateral_load_kips=100.0),
        loading="static",
        n_elements=120,
    )
    res_high_cyclic = solve_lateral_pile(
        pile, 60.0, soil,
        BoundaryCondition(head="free", lateral_load_kips=100.0),
        loading="cyclic",
        n_elements=120,
    )
    assert res_high_static.converged and res_high_cyclic.converged
    assert res_high_cyclic.head_deflection_in > res_high_static.head_deflection_in


def test_pushover_and_drilled_shaft_monotonicity() -> None:
    shaft = DrilledShaft(diameter_in=36.0, fc_ksi=4.0, cracked_factor=0.5)
    assert np.isclose(shaft.elastic_modulus_ksi, 3605.0, rtol=1e-2)  # ACI 318 Ec for 4 ksi concrete
    assert shaft.ei_kip_in2 > 1.0e8
    soil = SoilProfile(
        layers=[SoilLayer("Dense Sand", 0.0, 50.0, 125.0, APISand(phi_deg=36.0))],
        gwl_ft=10.0,
    )
    po = run_pushover(shaft, 45.0, soil, loads_kips=[20, 40, 60, 80, 100], head="free")
    assert po.table["converged"].all()
    # Head deflection and max moment must increase strictly with applied load (and nonlinearly!)
    defs = po.table["head_deflection_in"].to_numpy()
    assert np.all(np.diff(defs) > 0)
    # Secant stiffness softens as load increases -> deflection grows faster than linearly
    assert (defs[-1] / 100.0) > (defs[0] / 20.0)


def test_reese_sand_four_branch_continuity_cyclic_and_solver() -> None:
    """Verify Reese, Cox & Koop (1974) sand p-y continuity, plateau, cyclic softening, and FEM convergence."""
    reese = ReeseSand(phi_deg=34.0)
    b = 24.0
    z_in = 120.0  # 10 ft (z/b = 5.0)
    sig_v = 10.0 * (122.0 - 62.4) / 144000.0  # ksi

    y_grid = np.linspace(0.0, 2.0, 500)
    p_stat = reese.p_reaction(y_grid, z_in=z_in, b_in=b, sigma_v_eff_ksi=sig_v, submerged=True, loading="static")
    p_cyc = reese.p_reaction(y_grid, z_in=z_in, b_in=b, sigma_v_eff_ksi=sig_v, submerged=True, loading="cyclic")

    # Non-decreasing, zero at origin, continuous, plateau for y >= 3*b/80 (= 0.90 in)
    assert np.isclose(p_stat[0], 0.0)
    assert np.all(np.diff(p_stat) >= -1e-12)
    assert np.all(np.diff(p_cyc) >= -1e-12)
    y_u = 3.0 * b / 80.0
    plateau_vals = p_stat[y_grid >= y_u + 0.01]
    assert np.allclose(plateau_vals, plateau_vals[0], rtol=1e-10)
    # Cyclic resistance must be <= static at shallow depth (z/b = 1.0)
    p_stat_sh = reese.p_reaction(y_grid, z_in=24.0, b_in=b, sigma_v_eff_ksi=sig_v * 0.2, submerged=True, loading="static")
    p_cyc_sh = reese.p_reaction(y_grid, z_in=24.0, b_in=b, sigma_v_eff_ksi=sig_v * 0.2, submerged=True, loading="cyclic")
    assert p_cyc_sh[-1] < p_stat_sh[-1]

    # Full FEM solve in ReeseSand vs APISand
    pile = SteelPipePile(outer_diameter_in=24.0, wall_thickness_in=0.5)
    soil_reese = SoilProfile([SoilLayer("Sand (Reese)", 0.0, 50.0, 122.0, ReeseSand(phi_deg=34.0))], gwl_ft=6.0)
    res_r = solve_lateral_pile(pile, 45.0, soil_reese, BoundaryCondition(head="free", lateral_load_kips=60.0), n_elements=120)
    assert res_r.converged
    p_int = float(np.trapezoid(res_r.profile["p_kips_per_ft"], res_r.profile["z_ft"]))
    assert np.isclose(p_int, 60.0, rtol=0.02)


def test_pile_group_equilibrium_shadowing_and_cap_fixity() -> None:
    """Verify 2D coupled pile-cap FEM equilibrium (H, V, M), AASHTO p-multipliers, and cap stiffness."""
    assert np.allclose(aashto_p_multipliers(3.0, 3), [0.80, 0.40, 0.30])
    assert np.allclose(aashto_p_multipliers(5.0, 3), [1.00, 0.85, 0.70])

    pile = SteelPipePile(outer_diameter_in=24.0, wall_thickness_in=0.5)
    soil = SoilProfile(
        [
            SoilLayer("Soft Clay", 0.0, 15.0, 110.0, MatlockSoftClay(cu_psf=600.0)),
            SoilLayer("Sand", 15.0, 60.0, 122.0, ReeseSand(phi_deg=34.0)),
        ],
        gwl_ft=6.0,
    )
    # 3x3 bridge pier group at 3B = 6 ft spacing
    piles_fixed = [
        GroupPileSpec(x_ft=-6.0, pile=pile, length_ft=50.0, n_piles_in_row=3, connection="fixed"),
        GroupPileSpec(x_ft=0.0, pile=pile, length_ft=50.0, n_piles_in_row=3, connection="fixed"),
        GroupPileSpec(x_ft=6.0, pile=pile, length_ft=50.0, n_piles_in_row=3, connection="fixed"),
    ]
    cap_rigid = PileCapSpec(thickness_ft=5.0, width_ft=16.0, overhang_ft=2.0)
    loads = CapLoadCase(lateral_load_kips=360.0, vertical_load_kips=1800.0, moment_kip_ft=2400.0, x_load_ft=0.0)

    res_fixed = solve_pile_group(piles_fixed, cap_rigid, soil, loads, n_pile_elements=60)
    assert res_fixed.converged

    # Exact global 2D static equilibrium (H, V, and Moment about x_load)
    eq = res_fixed.equilibrium_check
    assert np.isclose(eq["sum_pile_H_kips"], eq["applied_H_kips"], rtol=5e-3)
    assert np.isclose(eq["sum_pile_V_kips"], eq["applied_V_kips"], rtol=5e-3)
    assert np.isclose(eq["sum_resisting_M_kip_ft"], eq["applied_M_kip_ft"], rtol=1e-2)

    # Leading row (x = +6 ft, fm=0.80) takes more shear than trailing row (x = -6 ft, fm=0.30)
    summ = res_fixed.pile_summary.set_index("x (ft)")
    assert summ.loc[6.0, "Head Shear V (kips/pile)"] > summ.loc[0.0, "Head Shear V (kips/pile)"]
    assert summ.loc[0.0, "Head Shear V (kips/pile)"] > summ.loc[-6.0, "Head Shear V (kips/pile)"]
    # Clockwise overturning moment puts rightmost row (+6 ft) into higher compression than leftmost (-6 ft)
    assert summ.loc[6.0, "Head Axial P (kips/pile)"] > summ.loc[-6.0, "Head Axial P (kips/pile)"]

    # Pinned pile-to-cap connections => zero head moment and larger lateral deflection
    piles_pinned = [
        GroupPileSpec(x_ft=x, pile=pile, length_ft=50.0, n_piles_in_row=3, connection="pinned")
        for x in (-6.0, 0.0, 6.0)
    ]
    res_pinned = solve_pile_group(piles_pinned, cap_rigid, soil, loads, n_pile_elements=60)
    assert res_pinned.converged
    assert np.allclose(res_pinned.pile_summary["Head Moment M₀ (kip-ft)"], 0.0, atol=1e-3)
    assert res_pinned.cap_lateral_defl_in > res_fixed.cap_lateral_defl_in

    # Flexible thin cap (1.25 ft) deflects and rotates more than rigid thick cap (5.0 ft)
    cap_thin = PileCapSpec(thickness_ft=1.25, width_ft=16.0, overhang_ft=2.0)
    res_thin = solve_pile_group(piles_fixed, cap_thin, soil, loads, n_pile_elements=60)
    assert res_thin.converged
    assert res_thin.cap_lateral_defl_in > res_fixed.cap_lateral_defl_in


def test_clean_dictionary_input_builders_and_tables() -> None:
    """Verify dictionary-based pile & soil builders, auto-depth chaining, and echo tables."""
    pile = build_pile_section(
        {"type": "steel_pipe", "diameter_in": 24.0, "wall_thickness_in": 0.5, "fy_ksi": 50.0}
    )
    shaft = build_pile_section(
        {"type": "drilled_shaft", "diameter_in": 36.0, "fc_ksi": 4.0, "cracked_factor": 0.5}
    )
    assert np.isclose(pile.diameter_in, 24.0)
    assert np.isclose(shaft.elastic_modulus_ksi, 3605.0, rtol=1e-2)

    soil = build_soil_profile(
        [
            {"name": "Soft Clay", "type": "soft_clay", "thickness_ft": 15.0, "gamma_pcf": 110.0, "cu_psf": 600.0},
            {"name": "Sand", "type": "reese_sand", "thickness_ft": 20.0, "gamma_pcf": 122.0, "phi_deg": 34.0},
            {"name": "Rock", "type": "weak_rock", "bottom_ft": 65.0, "gamma_pcf": 140.0, "qu_psi": 500.0, "E_ir_ksi": 50.0},
        ],
        gwl_ft=6.0,
    )
    assert soil.layers[0].top_ft == 0.0 and soil.layers[0].bottom_ft == 15.0
    assert soil.layers[1].top_ft == 15.0 and soil.layers[1].bottom_ft == 35.0
    assert soil.layers[2].top_ft == 35.0 and soil.layers[2].bottom_ft == 65.0

    stbl = soil_profile_table(soil)
    ptbl = pile_summary_table(pile, length_ft=60.0)
    assert len(stbl) == 3
    assert "Key Parameters" in stbl.columns
    assert "Flexural Stiffness, EI" in ptbl.index



