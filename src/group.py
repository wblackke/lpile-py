"""Coupled 2D Frame Pile-Cap & Multi-Pile Group Nonlinear FEM Solver (Ensoft GROUP-style).

Coordinate & Sign Convention:
    - x: horizontal coordinate along pile cap (positive rightward, in ft externally / in internally)
    - z: vertical coordinate downward from pile-cap centerline (positive downward)
    - Global nodal DOFs at every cap node: [u_x (right), u_z (down), phi (clockwise rotation, rad)]
    - Rigid-body compatibility at a fixed pile-to-cap joint:
        * Horizontal cap beam: du_z/dx = +phi
        * Vertical pile shaft: du_x/dz = -phi
    - Applied external loads at cap load point x_load:
        * H_kips > 0 pushes cap rightward (+x)
        * V_kips > 0 pushes cap downward (+z, compression)
        * M_kip_ft > 0 applies clockwise overturning moment (pushes right side down, left side up)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .py_curves import SoilLayer, SoilProfile
from .sections import PileSection
from .solver import BoundaryCondition, LateralPileResult, _beam_element_matrices

ConnectionType = Literal["fixed", "pinned", "spring"]


def aashto_p_multipliers(spacing_over_b: float, n_rows: int) -> List[float]:
    """Return AASHTO LRFD Table 10.7.2.4-1 group p-multipliers from leading row (Row 1) to trailing rows.

    Assumes loading acts toward +x, so the rightmost row (largest x) is Row 1 (leading row)
    and rows behind it are Row 2, Row 3+. Returned list is ordered [Row 1, Row 2, ..., Row N].
    """
    if n_rows <= 1:
        return [1.0]
    sb_grid = np.array([3.0, 5.0, 8.0])
    r1 = float(np.interp(spacing_over_b, sb_grid, [0.80, 1.00, 1.00]))
    r2 = float(np.interp(spacing_over_b, sb_grid, [0.40, 0.85, 1.00]))
    r3 = float(np.interp(spacing_over_b, sb_grid, [0.30, 0.70, 1.00]))
    mults = [r1]
    if n_rows >= 2:
        mults.append(r2)
    for _ in range(2, n_rows):
        mults.append(r3)
    return mults


@dataclass
class PileCapSpec:
    """Rectangular reinforced concrete pile cap discretized into 2D Euler-Bernoulli frame elements."""

    thickness_ft: float         # Cap depth / thickness h_c (ft)
    width_ft: float             # Out-of-plane cap width B_c (ft)
    overhang_ft: float = 2.0    # Edge overhang beyond outermost pile centerlines (ft)
    fc_ksi: float = 4.0         # Concrete compressive strength f'c (ksi)
    cracked_factor: float = 0.70  # Effective flexural stiffness multiplier I_eff / I_gross
    elements_per_bay: int = 8   # Number of 2D frame elements between adjacent pile rows

    @property
    def ec_ksi(self) -> float:
        return float(57.0 * np.sqrt(self.fc_ksi * 1000.0))

    @property
    def area_in2(self) -> float:
        return (self.width_ft * 12.0) * (self.thickness_ft * 12.0)

    @property
    def inertia_in4(self) -> float:
        b_in = self.width_ft * 12.0
        h_in = self.thickness_ft * 12.0
        return self.cracked_factor * (b_in * h_in**3) / 12.0

    @property
    def ei_kip_in2(self) -> float:
        return self.ec_ksi * self.inertia_in4

    @property
    def ea_kips(self) -> float:
        return self.ec_ksi * self.area_in2


@dataclass
class GroupPileSpec:
    """One pile (or out-of-plane row of identical piles) connected to the 2D pile cap."""

    x_ft: float                                 # Horizontal position along pile cap (ft)
    pile: PileSection                           # Structural pile cross-section
    length_ft: float                            # Embedded pile length below cap soffit (ft)
    n_piles_in_row: int = 1                     # Out-of-plane multiplicity (e.g. 3 for a 3x3 group)
    p_multiplier: Optional[float] = None        # Group shadowing p-multiplier (auto AASHTO if None)
    connection: ConnectionType = "fixed"        # Pile-to-cap joint: 'fixed', 'pinned', or 'spring'
    rotational_stiffness_kip_in_per_rad: float = 0.0  # Used if connection == 'spring' (per pile)
    axial_stiffness_kips_per_in: Optional[float] = None  # Per-pile axial stiffness (auto ~0.75*EA/L if None)
    tip: Literal["free", "pinned", "fixed"] = "free"
    label: Optional[str] = None

    def effective_axial_stiffness_per_pile(self) -> float:
        if self.axial_stiffness_kips_per_in is not None:
            return float(self.axial_stiffness_kips_per_in)
        # Standard elastic-soil coupled axial head stiffness (~0.75 * E*A / L)
        L_in = self.length_ft * 12.0
        return 0.75 * (self.pile.elastic_modulus_ksi * self.pile.area_in2) / L_in


@dataclass(frozen=True)
class CapLoadCase:
    """External loads applied to the pile cap at horizontal coordinate x_load_ft."""

    lateral_load_kips: float = 0.0      # H > 0 rightward (+x)
    vertical_load_kips: float = 0.0     # V > 0 downward (+z, compression)
    moment_kip_ft: float = 0.0          # M > 0 clockwise overturning moment
    x_load_ft: float = 0.0              # Point of application on cap (ft)


@dataclass
class PileGroupResult:
    """Coupled results for the pile cap and every individual pile in the group."""

    cap_profile: pd.DataFrame           # x_ft, ux_in, uz_in, rot_rad, moment_kip_ft, shear_kips, axial_kips
    pile_summary: pd.DataFrame          # Per-pile/row summary: head shear, moment, axial, defl, M_max, p_mult
    pile_results: Dict[str, LateralPileResult]  # Full depth profiles per pile row (per single pile!)
    cap_lateral_defl_in: float
    cap_vertical_disp_in: float
    cap_rotation_rad: float
    cap_max_moment_kip_ft: float
    cap_max_shear_kips: float
    converged: bool
    iterations: int
    rel_error: float
    equilibrium_check: Dict[str, float]


def _frame_element_horizontal(ea: float, ei: float, L: float) -> np.ndarray:
    """6x6 2D Euler-Bernoulli horizontal frame element in global DOFs [ux1, uz1, phi1, ux2, uz2, phi2]."""
    k = np.zeros((6, 6), dtype=float)
    # Axial (ux1, ux2)
    ka = ea / L
    k[0, 0] = ka
    k[0, 3] = -ka
    k[3, 0] = -ka
    k[3, 3] = ka
    # Bending (uz1, phi1, uz2, phi2) with duz/dx = +phi
    kb = (ei / L**3) * np.array(
        [
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, 4.0 * L**2, -6.0 * L, 2.0 * L**2],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, 2.0 * L**2, -6.0 * L, 4.0 * L**2],
        ],
        dtype=float,
    )
    b_idx = [1, 2, 4, 5]
    k[np.ix_(b_idx, b_idx)] += kb
    return k


def _scale_soil_profile(soil: SoilProfile, p_mult: float) -> SoilProfile:
    """Return a copy of SoilProfile with every layer's p_multiplier multiplied by p_mult."""
    scaled_layers = [
        SoilLayer(
            name=lyr.name,
            top_ft=lyr.top_ft,
            bottom_ft=lyr.bottom_ft,
            gamma_pcf=lyr.gamma_pcf,
            model=lyr.model,
            p_multiplier=lyr.p_multiplier * p_mult,
        )
        for lyr in soil.layers
    ]
    return SoilProfile(scaled_layers, gwl_ft=soil.gwl_ft)


def solve_pile_group(
    piles: Sequence[GroupPileSpec],
    cap: PileCapSpec,
    soil: SoilProfile,
    loads: CapLoadCase,
    loading: str = "static",
    use_aashto_p_multipliers: bool = True,
    n_pile_elements: int = 80,
    max_iter: int = 200,
    tol: float = 1e-6,
    damping: float = 0.65,
) -> PileGroupResult:
    """Solve coupled 2D Frame Pile-Cap + Multi-Pile Group with nonlinear p-y curves and P-Delta.

    All pile rows and the concrete pile cap are assembled into a single global stiffness system
    and solved iteratively via damped secant stiffness until full cap-soil-pile equilibrium.
    """
    if not piles:
        raise ValueError("At least one GroupPileSpec is required.")

    # Sort piles by x ascending (left -> right); under H >= 0, rightmost row is Row 1 (leading)
    sorted_piles = sorted(piles, key=lambda p: p.x_ft)
    n_rows = len(sorted_piles)

    # Determine spacing S/b and assign AASHTO p-multipliers if not explicitly provided
    if n_rows > 1:
        xs = np.array([p.x_ft for p in sorted_piles])
        avg_spacing_in = float(np.mean(np.diff(xs)) * 12.0)
        avg_b_in = float(np.mean([p.pile.diameter_in for p in sorted_piles]))
        sb_ratio = avg_spacing_in / max(avg_b_in, 1e-6)
        auto_mults_lead_first = aashto_p_multipliers(sb_ratio, n_rows)
    else:
        auto_mults_lead_first = [1.0]

    # Map leading-first multipliers to sorted_piles (if H >= 0, rightmost is leading; if H < 0, leftmost is leading)
    p_mults: List[float] = []
    row_labels: List[str] = []
    for i, pspec in enumerate(sorted_piles):
        rank_from_lead = (n_rows - 1 - i) if loads.lateral_load_kips >= 0 else i
        if pspec.p_multiplier is not None:
            pm = float(pspec.p_multiplier)
        elif use_aashto_p_multipliers:
            pm = auto_mults_lead_first[rank_from_lead]
        else:
            pm = 1.0
        p_mults.append(pm)
        default_lbl = f"Row {rank_from_lead + 1} (x={pspec.x_ft:+.1f} ft)"
        row_labels.append(pspec.label or default_lbl)

    # Build 1D horizontal mesh for the pile cap including overhangs, every pile x, and x_load_ft
    x_min = sorted_piles[0].x_ft - cap.overhang_ft
    x_max = sorted_piles[-1].x_ft + cap.overhang_ft
    key_x = sorted(set([x_min, x_max, loads.x_load_ft] + [p.x_ft for p in sorted_piles]))
    cap_x_list: List[float] = []
    for k in range(len(key_x) - 1):
        seg = np.linspace(key_x[k], key_x[k + 1], cap.elements_per_bay + 1)
        cap_x_list.extend(seg[:-1].tolist())
    cap_x_list.append(key_x[-1])
    cap_x_ft = np.array(cap_x_list, dtype=float)
    cap_x_in = cap_x_ft * 12.0
    n_cap_nodes = len(cap_x_ft)
    n_cap_dofs = 3 * n_cap_nodes

    def _find_cap_node(x_target: float) -> int:
        idx = int(np.argmin(np.abs(cap_x_ft - x_target)))
        assert abs(cap_x_ft[idx] - x_target) < 1e-6
        return idx

    load_node = _find_cap_node(loads.x_load_ft)
    pile_cap_nodes = [_find_cap_node(p.x_ft) for p in sorted_piles]

    # Allocate global DOFs:
    #   [0 .. 3*n_cap_nodes - 1] : pile cap nodes (ux, uz, phi)
    #   Then for each pile row k:
    #     - If connection == 'fixed': head (ux, phi) shares cap node's (ux, phi)!
    #     - If connection in ('pinned', 'spring'): head ux shares cap node's ux, while head phi gets its own DOF!
    #     - Interior & tip nodes (i = 1 .. n_pile_elements): each gets 2 DOFs (ux, phi)
    dof_cursor = n_cap_dofs
    pile_dof_maps: List[np.ndarray] = []  # shape (n_pile_elements + 1, 2) -> global (ux_dof, phi_dof)
    pile_z_ft_list: List[np.ndarray] = []
    pile_trib_in_list: List[np.ndarray] = []
    pile_h_in_list: List[float] = []
    pile_soils: List[SoilProfile] = []
    fixed_dofs: List[int] = []

    for k, pspec in enumerate(sorted_piles):
        cnode = pile_cap_nodes[k]
        cap_ux_dof = 3 * cnode
        cap_phi_dof = 3 * cnode + 2

        n_pnodes = n_pile_elements + 1
        dmap = np.zeros((n_pnodes, 2), dtype=int)
        dmap[0, 0] = cap_ux_dof
        if pspec.connection == "fixed":
            dmap[0, 1] = cap_phi_dof
        else:
            dmap[0, 1] = dof_cursor
            dof_cursor += 1

        for i in range(1, n_pnodes):
            dmap[i, 0] = dof_cursor
            dmap[i, 1] = dof_cursor + 1
            dof_cursor += 2

        if pspec.tip == "pinned":
            fixed_dofs.append(int(dmap[-1, 0]))
        elif pspec.tip == "fixed":
            fixed_dofs.extend([int(dmap[-1, 0]), int(dmap[-1, 1])])

        L_in = pspec.length_ft * 12.0
        h_in = L_in / n_pile_elements
        z_ft = np.linspace(0.0, pspec.length_ft, n_pnodes)
        trib = np.full(n_pnodes, h_in, dtype=float)
        trib[0] = 0.5 * h_in
        trib[-1] = 0.5 * h_in

        pile_dof_maps.append(dmap)
        pile_z_ft_list.append(z_ft)
        pile_trib_in_list.append(trib)
        pile_h_in_list.append(h_in)
        pile_soils.append(_scale_soil_profile(soil, p_mults[k]))

    n_total_dof = dof_cursor
    free_dofs = np.array([d for d in range(n_total_dof) if d not in set(fixed_dofs)], dtype=int)

    # Assemble constant pile-cap frame stiffness + axial pile springs + head rotational springs
    K_base = np.zeros((n_total_dof, n_total_dof), dtype=float)
    cap_elem_mats: List[np.ndarray] = []
    for e in range(n_cap_nodes - 1):
        Lc = float(cap_x_in[e + 1] - cap_x_in[e])
        ke_cap = _frame_element_horizontal(cap.ea_kips, cap.ei_kip_in2, Lc)
        cap_elem_mats.append(ke_cap)
        dofs_e = [3 * e, 3 * e + 1, 3 * e + 2, 3 * (e + 1), 3 * (e + 1) + 1, 3 * (e + 1) + 2]
        K_base[np.ix_(dofs_e, dofs_e)] += ke_cap

    # Axial vertical springs & semi-rigid head rotational springs for each pile row
    row_k_axial: List[float] = []
    for k, pspec in enumerate(sorted_piles):
        cnode = pile_cap_nodes[k]
        k_ax_row = pspec.n_piles_in_row * pspec.effective_axial_stiffness_per_pile()
        row_k_axial.append(k_ax_row)
        K_base[3 * cnode + 1, 3 * cnode + 1] += k_ax_row

        if pspec.connection == "spring" and pspec.rotational_stiffness_kip_in_per_rad > 0:
            k_rot = pspec.n_piles_in_row * pspec.rotational_stiffness_kip_in_per_rad
            d_cap_phi = 3 * cnode + 2
            d_pile_phi = int(pile_dof_maps[k][0, 1])
            K_base[d_cap_phi, d_cap_phi] += k_rot
            K_base[d_pile_phi, d_pile_phi] += k_rot
            K_base[d_cap_phi, d_pile_phi] -= k_rot
            K_base[d_pile_phi, d_cap_phi] -= k_rot

    # Global external load vector applied at load_node
    F = np.zeros(n_total_dof, dtype=float)
    F[3 * load_node] = loads.lateral_load_kips
    F[3 * load_node + 1] = loads.vertical_load_kips
    F[3 * load_node + 2] = loads.moment_kip_ft * 12.0

    # Similarity transform T for vertical pile element: theta = du_x/dz = -phi => diag(1, -1, 1, -1)
    T_sign = np.diag([1.0, -1.0, 1.0, -1.0])

    # Initialize pile deflections & per-pile axial loads
    y_curr_list = [np.full(n_pile_elements + 1, 0.01, dtype=float) for _ in range(n_rows)]
    total_piles_count = sum(p.n_piles_in_row for p in sorted_piles)
    axial_per_pile = [loads.vertical_load_kips / max(total_piles_count, 1) for _ in range(n_rows)]

    u = np.zeros(n_total_dof, dtype=float)
    converged = False
    rel_err = np.inf
    it = 0

    for it in range(1, max_iter + 1):
        K_total = K_base.copy()

        # Assemble every pile row's beam-column stiffness (with current axial P-Delta) & nonlinear soil springs
        for k, pspec in enumerate(sorted_piles):
            dmap = pile_dof_maps[k]
            h_in = pile_h_in_list[k]
            z_ft = pile_z_ft_list[k]
            trib = pile_trib_in_list[k]
            s_prof = pile_soils[k]
            mult_n = float(pspec.n_piles_in_row)

            # Only compressive axial load (>0) or tension (<0) enters geometric stiffness
            ke_local = _beam_element_matrices(pspec.pile.ei_kip_in2, h_in, axial_per_pile[k])
            ke_global = mult_n * (T_sign @ ke_local @ T_sign)

            for e in range(n_pile_elements):
                edofs = [int(dmap[e, 0]), int(dmap[e, 1]), int(dmap[e + 1, 0]), int(dmap[e + 1, 1])]
                K_total[np.ix_(edofs, edofs)] += ke_global

            # Nonlinear soil secant springs on horizontal DOFs
            for i in range(n_pile_elements + 1):
                es = s_prof.secant_modulus(z_ft[i], y_curr_list[k][i], pspec.pile.diameter_in, loading=loading)
                ux_dof = int(dmap[i, 0])
                K_total[ux_dof, ux_dof] += mult_n * es * trib[i]

        u_new = np.zeros(n_total_dof, dtype=float)
        u_new[free_dofs] = np.linalg.solve(K_total[np.ix_(free_dofs, free_dofs)], F[free_dofs])

        # Compute convergence error across all pile lateral deflections
        y_new_list = [u_new[pile_dof_maps[k][:, 0]] for k in range(n_rows)]
        diff_vec = np.concatenate([yn - yc for yn, yc in zip(y_new_list, y_curr_list)])
        norm_new = max(float(np.linalg.norm(np.concatenate(y_new_list))), 1e-8)
        rel_err = float(np.linalg.norm(diff_vec) / norm_new)

        # Update per-pile axial forces from cap vertical displacements for P-Delta coupling
        new_axial = [
            float(u_new[3 * pile_cap_nodes[k] + 1] * sorted_piles[k].effective_axial_stiffness_per_pile())
            for k in range(n_rows)
        ]

        if rel_err < tol:
            u = u_new
            y_curr_list = y_new_list
            axial_per_pile = new_axial
            converged = True
            break

        y_curr_list = [(1.0 - damping) * yc + damping * yn for yc, yn in zip(y_curr_list, y_new_list)]
        axial_per_pile = [(1.0 - damping) * ao + damping * an for ao, an in zip(axial_per_pile, new_axial)]
        u = u_new

    # --- Recover Pile Cap Internal Forces ---
    cap_ux = u[0:n_cap_dofs:3]
    cap_uz = u[1:n_cap_dofs:3]
    cap_phi = u[2:n_cap_dofs:3]

    cap_m_sum = np.zeros(n_cap_nodes)
    cap_v_sum = np.zeros(n_cap_nodes)
    cap_n_sum = np.zeros(n_cap_nodes)
    cap_cnt = np.zeros(n_cap_nodes)

    for e in range(n_cap_nodes - 1):
        dofs_e = [3 * e, 3 * e + 1, 3 * e + 2, 3 * (e + 1), 3 * (e + 1) + 1, 3 * (e + 1) + 2]
        fe = cap_elem_mats[e] @ u[dofs_e]
        # Internal forces: N (tension +) = -fe[0] / +fe[3], V (downward right face +) = fe[1] / -fe[4],
        # M (sagging/bottom-tension + with z down & duz/dx=+phi) = -fe[2] at left, +fe[5] at right
        cap_n_sum[e] += -fe[0]
        cap_v_sum[e] += fe[1]
        cap_m_sum[e] += -fe[2]
        cap_cnt[e] += 1.0
        cap_n_sum[e + 1] += fe[3]
        cap_v_sum[e + 1] += -fe[4]
        cap_m_sum[e + 1] += fe[5]
        cap_cnt[e + 1] += 1.0

    cap_profile_df = pd.DataFrame(
        {
            "x_ft": cap_x_ft,
            "ux_in": cap_ux,
            "uz_in": cap_uz,
            "rot_rad": cap_phi,
            "moment_kip_ft": (cap_m_sum / cap_cnt) / 12.0,
            "shear_kips": cap_v_sum / cap_cnt,
            "axial_kips": cap_n_sum / cap_cnt,
        }
    )

    # --- Recover Individual Pile Profiles (per single pile!) ---
    pile_results: Dict[str, LateralPileResult] = {}
    summary_rows = []
    sum_head_shear_total = 0.0
    sum_head_axial_total = 0.0
    sum_moment_about_load = 0.0

    for k, pspec in enumerate(sorted_piles):
        lbl = row_labels[k]
        dmap = pile_dof_maps[k]
        h_in = pile_h_in_list[k]
        z_ft = pile_z_ft_list[k]
        s_prof = pile_soils[k]

        y_p = u[dmap[:, 0]]
        phi_p = u[dmap[:, 1]]
        theta_p = -phi_p  # Pile slope dy/dz = -phi

        p_kin = np.array([float(s_prof.evaluate_py(z_ft[i], y_p[i], pspec.pile.diameter_in, loading=loading)) for i in range(len(z_ft))])
        es_p = np.array([s_prof.secant_modulus(z_ft[i], y_p[i], pspec.pile.diameter_in, loading=loading) for i in range(len(z_ft))])

        # Single-pile internal M(z), V(z) in standard pile local sign convention (y, theta = dy/dz)
        ke_local = _beam_element_matrices(pspec.pile.ei_kip_in2, h_in, axial_per_pile[k])
        n_pnodes = len(z_ft)
        m_sum = np.zeros(n_pnodes)
        v_sum = np.zeros(n_pnodes)
        cnt = np.zeros(n_pnodes)

        for e in range(n_pile_elements):
            u_loc = np.array([y_p[e], theta_p[e], y_p[e + 1], theta_p[e + 1]])
            f_loc = ke_local @ u_loc
            v_sum[e] += f_loc[0]
            m_sum[e] += -f_loc[1]
            cnt[e] += 1.0
            v_sum[e + 1] += -f_loc[2]
            m_sum[e + 1] += f_loc[3]
            cnt[e + 1] += 1.0

        m_kft = (m_sum / cnt) / 12.0
        v_kips = v_sum / cnt
        p_kft = p_kin * 12.0

        # Exact pile head shear & moment transferred from cap to a single pile
        # Head node has tributary soil spring es_p[0] * (h_in/2) in parallel with beam element 0
        head_shear_pile = float(v_kips[0] + p_kin[0] * (0.5 * h_in))
        head_moment_pile = float(m_kft[0])  # Standard LPILE head moment M(0)
        head_axial_pile = float(axial_per_pile[k])

        sum_head_shear_total += pspec.n_piles_in_row * head_shear_pile
        sum_head_axial_total += pspec.n_piles_in_row * head_axial_pile
        # Global clockwise moment equilibrium about x_load:
        # In global DOFs [ux, phi], T_sign = diag(1, -1, 1, -1) gives f_global_pile[1] = -f_loc[1] = +M(0).
        # Hence K_total * u = F implies: sum( P_ax * (x_k - x_load) + M(0) ) == M_applied.
        sum_moment_about_load += pspec.n_piles_in_row * (
            head_axial_pile * (pspec.x_ft - loads.x_load_ft) + head_moment_pile
        )

        idx_mm = int(np.argmax(np.abs(m_kft)))
        prof_df = pd.DataFrame(
            {
                "z_ft": z_ft,
                "y_in": y_p,
                "slope_rad": theta_p,
                "moment_kip_ft": m_kft,
                "shear_kips": v_kips,
                "p_kips_per_ft": p_kft,
                "Es_ksi": es_p,
            }
        )
        pile_results[lbl] = LateralPileResult(
            profile=prof_df,
            head_deflection_in=float(y_p[0]),
            head_slope_rad=float(theta_p[0]),
            head_moment_kip_ft=head_moment_pile,
            max_moment_kip_ft=float(m_kft[idx_mm]),
            depth_max_moment_ft=float(z_ft[idx_mm]),
            max_shear_kips=float(v_kips[np.argmax(np.abs(v_kips))]),
            max_soil_reaction_kips_per_ft=float(p_kft[np.argmax(np.abs(p_kft))]),
            converged=converged,
            iterations=it,
            rel_error=rel_err,
            boundary=BoundaryCondition(
                head=pspec.connection,
                lateral_load_kips=head_shear_pile,
                moment_kip_ft=head_moment_pile,
                axial_load_kips=head_axial_pile,
                tip=pspec.tip,
            ),
            loading=loading,
        )

        summary_rows.append(
            {
                "Row": lbl,
                "x (ft)": pspec.x_ft,
                "Piles in Row": pspec.n_piles_in_row,
                "p-Multiplier": p_mults[k],
                "Connection": pspec.connection,
                "Head Defl y₀ (in)": float(y_p[0]),
                "Head Axial P (kips/pile)": head_axial_pile,
                "Head Shear V (kips/pile)": head_shear_pile,
                "Row Shear Share (%)": (pspec.n_piles_in_row * head_shear_pile / max(abs(loads.lateral_load_kips), 1e-6)) * 100.0,
                "Head Moment M₀ (kip-ft)": head_moment_pile,
                "Max |M| (kip-ft/pile)": float(abs(m_kft[idx_mm])),
                "Depth M_max (ft)": float(z_ft[idx_mm]),
            }
        )

    pile_summary_df = pd.DataFrame(summary_rows)
    equil = {
        "applied_H_kips": loads.lateral_load_kips,
        "sum_pile_H_kips": sum_head_shear_total,
        "applied_V_kips": loads.vertical_load_kips,
        "sum_pile_V_kips": sum_head_axial_total,
        "applied_M_kip_ft": loads.moment_kip_ft,
        "sum_resisting_M_kip_ft": sum_moment_about_load,
    }

    return PileGroupResult(
        cap_profile=cap_profile_df,
        pile_summary=pile_summary_df,
        pile_results=pile_results,
        cap_lateral_defl_in=float(cap_ux[load_node]),
        cap_vertical_disp_in=float(cap_uz[load_node]),
        cap_rotation_rad=float(cap_phi[load_node]),
        cap_max_moment_kip_ft=float(cap_profile_df["moment_kip_ft"].iloc[np.argmax(np.abs(cap_profile_df["moment_kip_ft"]))]),
        cap_max_shear_kips=float(cap_profile_df["shear_kips"].iloc[np.argmax(np.abs(cap_profile_df["shear_kips"]))]),
        converged=converged,
        iterations=it,
        rel_error=rel_err,
        equilibrium_check=equil,
    )
