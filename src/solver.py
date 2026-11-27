"""Nonlinear 1D Hermite Cubic Beam-Column FEM Solver for Laterally Loaded Piles (LPILE-style)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from .py_curves import SoilProfile
from .sections import PileSection

HeadCondition = Literal["free", "fixed", "spring"]
TipCondition = Literal["free", "pinned", "fixed"]


@dataclass(frozen=True)
class BoundaryCondition:
    """Pile head and tip boundary conditions & applied loads.

    Sign convention (z downward from pile head z=0 to tip z=L, y positive rightward):
        - lateral_load_kips (V_t): Positive pushes pile head to the right (+y).
        - moment_kip_ft (M_t): Positive rotates pile head in the direction of +y (clockwise
          for z-down/y-right: increases deflection at z=0 in the same direction as +V_t).
          Specifically, a positive applied head moment corresponds to M(0) = +M_t.
        - axial_load_kips (P_x): Positive compression (triggers second-order P-Delta softening).
    """

    head: HeadCondition = "free"
    lateral_load_kips: float = 0.0
    moment_kip_ft: float = 0.0
    axial_load_kips: float = 0.0
    head_rotational_stiffness_kip_in_per_rad: float = 0.0  # Used when head == "spring"
    tip: TipCondition = "free"


@dataclass
class LateralPileResult:
    """Complete depth-profile results from a single laterally loaded pile analysis."""

    profile: pd.DataFrame          # Depth profiles: z_ft, y_in, slope_rad, moment_kip_ft, shear_kips, p_kips_per_ft, Es_ksi
    head_deflection_in: float
    head_slope_rad: float
    head_moment_kip_ft: float
    max_moment_kip_ft: float
    depth_max_moment_ft: float
    max_shear_kips: float
    max_soil_reaction_kips_per_ft: float
    converged: bool
    iterations: int
    rel_error: float
    boundary: BoundaryCondition
    loading: str


@dataclass
class PushoverResult:
    """Results from incremental lateral pushover sweep across applied head shear loads."""

    table: pd.DataFrame            # Load vs. head_deflection_in, max_moment_kip_ft, depth_max_moment_ft, converged
    runs: list[LateralPileResult]


def _beam_element_matrices(ei: float, h: float, axial_p: float) -> np.ndarray:
    """Return 4x4 Hermite cubic beam-column element stiffness matrix K_b - K_G (in kip, in units).

    Element DOFs: [y_i, theta_i, y_j, theta_j] where theta = dy/dz.
    """
    kb = (ei / h**3) * np.array(
        [
            [12.0, 6.0 * h, -12.0, 6.0 * h],
            [6.0 * h, 4.0 * h**2, -6.0 * h, 2.0 * h**2],
            [-12.0, -6.0 * h, 12.0, -6.0 * h],
            [6.0 * h, 2.0 * h**2, -6.0 * h, 4.0 * h**2],
        ],
        dtype=float,
    )
    if axial_p == 0.0:
        return kb
    kg = (axial_p / (30.0 * h)) * np.array(
        [
            [36.0, 3.0 * h, -36.0, 3.0 * h],
            [3.0 * h, 4.0 * h**2, -3.0 * h, -1.0 * h**2],
            [-36.0, -3.0 * h, 36.0, -3.0 * h],
            [3.0 * h, -1.0 * h**2, -3.0 * h, 4.0 * h**2],
        ],
        dtype=float,
    )
    return kb - kg


def solve_lateral_pile(
    pile: PileSection,
    length_ft: float,
    soil: SoilProfile,
    bc: BoundaryCondition,
    loading: str = "static",
    n_elements: int = 120,
    max_iter: int = 200,
    tol: float = 1e-6,
    damping: float = 0.65,
    initial_y_in: Optional[np.ndarray] = None,
) -> LateralPileResult:
    """Solve nonlinear laterally loaded pile via damped secant-stiffness Hermite beam-column FEM.

    Parameters
    ----------
    pile : PileSection
        Structural section of the pile (diameter b and flexural rigidity EI).
    length_ft : float
        Total pile length L (ft), measured from pile head z=0 downward to tip z=L.
    soil : SoilProfile
        Layered soil stratigraphy with p-y models and groundwater table.
    bc : BoundaryCondition
        Head/tip fixity and applied lateral shear, head moment, and axial compression.
    loading : {'static', 'cyclic'}
        Static or cyclic p-y curve branch.
    n_elements : int
        Number of finite elements along pile length (default 120 => ~4-6 in element spacing).
    max_iter : int
        Maximum nonlinear secant iterations.
    tol : float
        Relative convergence tolerance on deflection vector norm.
    damping : float
        Under-relaxation factor (0 < damping <= 1) for rock-solid convergence on steep p-y curves.
    """
    L_in = float(length_ft) * 12.0
    h = L_in / n_elements
    n_nodes = n_elements + 1
    n_dof = 2 * n_nodes

    z_in = np.linspace(0.0, L_in, n_nodes)
    z_ft = z_in / 12.0
    trib_in = np.full(n_nodes, h, dtype=float)
    trib_in[0] = 0.5 * h
    trib_in[-1] = 0.5 * h

    # Assemble constant structural beam-column stiffness matrix K_struct
    K_struct = np.zeros((n_dof, n_dof), dtype=float)
    ke = _beam_element_matrices(pile.ei_kip_in2, h, bc.axial_load_kips)
    for e in range(n_elements):
        idx = slice(2 * e, 2 * e + 4)
        K_struct[idx, idx] += ke

    if bc.head == "spring" and bc.head_rotational_stiffness_kip_in_per_rad > 0:
        K_struct[1, 1] += bc.head_rotational_stiffness_kip_in_per_rad

    # Load vector F
    # Sign convention check:
    #   +V_t at z=0 acts along +y_0 (DOF 0 => +V_t).
    #   Applied moment M_t that bends the top in the +y direction (negative slope theta_0 < 0,
    #   positive internal bending moment M(0) = EI*y''(0) = +M_t) enters DOF 1 as -M_t_kip_in!
    F = np.zeros(n_dof, dtype=float)
    F[0] = bc.lateral_load_kips
    F[1] = -bc.moment_kip_ft * 12.0

    # Essential (Dirichlet) boundary condition DOFs fixed to zero
    fixed_dofs = []
    if bc.head == "fixed":
        fixed_dofs.append(1)  # theta_0 = 0
    if bc.tip == "pinned":
        fixed_dofs.append(2 * (n_nodes - 1))  # y_L = 0
    elif bc.tip == "fixed":
        fixed_dofs.extend([2 * (n_nodes - 1), 2 * (n_nodes - 1) + 1])  # y_L = 0, theta_L = 0

    free_dofs = np.array([d for d in range(n_dof) if d not in set(fixed_dofs)], dtype=int)

    # Initial deflection guess (small nonzero deflection to initialize secant soil stiffness)
    if initial_y_in is not None and len(initial_y_in) == n_nodes:
        y_curr = np.asarray(initial_y_in, dtype=float).copy()
    else:
        y_curr = np.full(n_nodes, 0.01, dtype=float)

    converged = False
    rel_err = np.inf
    u = np.zeros(n_dof, dtype=float)
    es_vals = np.zeros(n_nodes, dtype=float)
    it = 0

    for it in range(1, max_iter + 1):
        for i in range(n_nodes):
            es_vals[i] = soil.secant_modulus(z_ft[i], y_curr[i], pile.diameter_in, loading=loading)

        K_total = K_struct.copy()
        for i in range(n_nodes):
            K_total[2 * i, 2 * i] += es_vals[i] * trib_in[i]

        u_new = np.zeros(n_dof, dtype=float)
        u_new[free_dofs] = np.linalg.solve(K_total[np.ix_(free_dofs, free_dofs)], F[free_dofs])
        y_new = u_new[0::2]

        denom = max(float(np.linalg.norm(y_new)), 1e-8)
        rel_err = float(np.linalg.norm(y_new - y_curr) / denom)
        if rel_err < tol:
            u = u_new
            y_curr = y_new
            converged = True
            break

        # Damped update for guaranteed monotonic convergence
        y_curr = (1.0 - damping) * y_curr + damping * y_new
        u = u_new

    # Final exact soil reaction & internal forces (M, V) recovered from Hermite shape functions
    y_final = u[0::2]
    theta_final = u[1::2]
    p_kips_per_in = np.array(
        [
            float(soil.evaluate_py(z_ft[i], y_final[i], pile.diameter_in, loading=loading))
            for i in range(n_nodes)
        ]
    )
    es_final = np.array(
        [
            soil.secant_modulus(z_ft[i], y_final[i], pile.diameter_in, loading=loading)
            for i in range(n_nodes)
        ]
    )

    # Exact nodal bending moments M(z) = EI * y''(z) and shears V(z) from element end forces
    # Element local end forces f_e = ke @ u_e = [V_left, -M_left, -V_right, +M_right]
    # Verify: for cantilever with +V_t at z=0 and clamp at z=L, f_e = [+V_t, -M(0), -V(h), +M(h)].
    m_sum = np.zeros(n_nodes, dtype=float)
    v_sum = np.zeros(n_nodes, dtype=float)
    cnt = np.zeros(n_nodes, dtype=float)
    for e in range(n_elements):
        u_e = u[2 * e : 2 * e + 4]
        f_e = ke @ u_e
        v_left = f_e[0]
        m_left = -f_e[1]
        v_right = -f_e[2]
        m_right = f_e[3]
        v_sum[e] += v_left
        m_sum[e] += m_left
        cnt[e] += 1.0
        v_sum[e + 1] += v_right
        m_sum[e + 1] += m_right
        cnt[e + 1] += 1.0

    moment_kip_in = m_sum / cnt
    shear_kips = v_sum / cnt
    moment_kip_ft = moment_kip_in / 12.0
    p_kips_per_ft = p_kips_per_in * 12.0

    profile_df = pd.DataFrame(
        {
            "z_ft": z_ft,
            "y_in": y_final,
            "slope_rad": theta_final,
            "moment_kip_ft": moment_kip_ft,
            "shear_kips": shear_kips,
            "p_kips_per_ft": p_kips_per_ft,
            "Es_ksi": es_final,
        }
    )

    idx_mmax = int(np.argmax(np.abs(moment_kip_ft)))
    max_moment = float(moment_kip_ft[idx_mmax])
    depth_mmax = float(z_ft[idx_mmax])

    return LateralPileResult(
        profile=profile_df,
        head_deflection_in=float(y_final[0]),
        head_slope_rad=float(theta_final[0]),
        head_moment_kip_ft=float(moment_kip_ft[0]),
        max_moment_kip_ft=max_moment,
        depth_max_moment_ft=depth_mmax,
        max_shear_kips=float(shear_kips[np.argmax(np.abs(shear_kips))]),
        max_soil_reaction_kips_per_ft=float(p_kips_per_ft[np.argmax(np.abs(p_kips_per_ft))]),
        converged=converged,
        iterations=it,
        rel_error=rel_err,
        boundary=bc,
        loading=loading,
    )


def run_pushover(
    pile: PileSection,
    length_ft: float,
    soil: SoilProfile,
    loads_kips: Sequence[float],
    head: HeadCondition = "free",
    moment_kip_ft: float = 0.0,
    axial_load_kips: float = 0.0,
    head_rotational_stiffness_kip_in_per_rad: float = 0.0,
    tip: TipCondition = "free",
    loading: str = "static",
    n_elements: int = 120,
) -> PushoverResult:
    """Run incremental lateral pushover sweep with warm-starting across applied head loads."""
    runs: list[LateralPileResult] = []
    rows = []
    prev_y: Optional[np.ndarray] = None

    for v_load in loads_kips:
        bc = BoundaryCondition(
            head=head,
            lateral_load_kips=float(v_load),
            moment_kip_ft=moment_kip_ft,
            axial_load_kips=axial_load_kips,
            head_rotational_stiffness_kip_in_per_rad=head_rotational_stiffness_kip_in_per_rad,
            tip=tip,
        )
        res = solve_lateral_pile(
            pile=pile,
            length_ft=length_ft,
            soil=soil,
            bc=bc,
            loading=loading,
            n_elements=n_elements,
            initial_y_in=prev_y,
        )
        prev_y = res.profile["y_in"].to_numpy()
        runs.append(res)
        rows.append(
            {
                "lateral_load_kips": float(v_load),
                "head_deflection_in": res.head_deflection_in,
                "head_slope_rad": res.head_slope_rad,
                "head_moment_kip_ft": res.head_moment_kip_ft,
                "max_abs_moment_kip_ft": abs(res.max_moment_kip_ft),
                "max_moment_kip_ft": res.max_moment_kip_ft,
                "depth_max_moment_ft": res.depth_max_moment_ft,
                "converged": res.converged,
                "iterations": res.iterations,
            }
        )

    return PushoverResult(table=pd.DataFrame(rows), runs=runs)
