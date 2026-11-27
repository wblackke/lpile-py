# LPILE-Style Laterally Loaded Pile Analysis in Python

A Python and Jupyter Notebook toolkit for nonlinear **$p$-$y$ analysis of laterally loaded piles** (similar to Ensoft LPILE), using a 1D Hermite cubic beam-column finite element solver with damped secant-stiffness iteration.

## Capabilities
- **Pile Cross-Sections** (`src/sections.py`): `SteelPipePile`, `DrilledShaft` (cracked ACI 318 $E_c I_{\text{eff}}$), `HPile`, and `CustomPileSection`.
- **Standard LPILE $p$-$y$ Soil Models** (`src/py_curves.py`):
  - **Soft Clay** (Matlock, 1970 — static & cyclic)
  - **Stiff Clay without Free Water** (Welch & Reese, 1972 / Reese & Welch, 1975)
  - **Sand (API RP 2A / O'Neill & Murchison, 1983)** — smooth hyperbolic tangent, static & cyclic
  - **Sand (Reese, Cox & Koop, 1974)** — classic LPILE 4-branch piecewise curve (linear → parabola → linear hardening → plateau), static & cyclic
  - **Weak Rock** (Reese, 1997)
  - **Linear Winkler** (for analytical Hetényi verification)
  - Supports arbitrary multi-layer soil profiles, groundwater table effective vertical stress integration, and group **$p$-multipliers**.
- **Beam-Column Solver** (`src/solver.py`): Free-head, Fixed-head, and Rotational-Spring head BCs; Free/Pinned/Fixed tip BCs; second-order **axial $P$-$\Delta$ effects**; incremental **lateral pushover** curves.
- **Multi-Pile Group & 2D Finite-Element Pile Cap** (`src/group.py`): Coupled 2D Euler–Bernoulli frame pile cap (`PileCapSpec`) connected to arbitrary pile rows (`GroupPileSpec`, Fixed/Pinned/Spring joints), automatic **AASHTO LRFD group $p$-multipliers**, axial push-pull overturning forces coupled to per-pile $P$-$\Delta$, and full pile-cap moment/shear/settlement diagrams.
- **LPILE-Style Engineering Plots** (`src/plotting.py`): 4-panel depth profiles ($y, M, V, p$ with soil strata & GWL shading), $p$-$y$ curve families, and pushover load-deflection / peak-moment curves.

## Verification
Unit tests in `tests/test_lpile.py` verify the solver against:
1. Exact Euler–Bernoulli cantilever beam formulas (shear & moment)
2. Exact **Hetényi (1946)** analytical solutions for both **free-head** and **fixed-head** beams on a linear Winkler foundation
3. Global horizontal force equilibrium ($\int p(z)\,dz = V_t$), $P$-$\Delta$ second-order amplification, and cyclic $p$-$y$ softening.

## Quick Start
```bash
pytest
jupyter lab lpile_analysis.ipynb
```
