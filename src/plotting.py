"""Classic LPILE-style engineering Matplotlib plots for laterally loaded pile analysis."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .group import PileGroupResult
from .inputs import _describe_soil_params
from .py_curves import SoilProfile
from .sections import PileSection
from .solver import BoundaryCondition, LateralPileResult, PushoverResult

COLORS = {
    "navy": "#1B365D",
    "red": "#C0392B",
    "teal": "#2A9D8F",
    "amber": "#D68910",
    "purple": "#6C3483",
    "gwl": "#2E86C1",
}


def apply_engineering_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#B0B8C1",
            "axes.grid": True,
            "grid.color": "#E5E8EC",
            "grid.linewidth": 0.7,
            "font.size": 9.8,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#DCDCDC",
            "legend.fontsize": 8.5,
        }
    )


def _shade_soil_layers(ax: plt.Axes, soil: SoilProfile, max_depth_ft: float, show_labels: bool = False) -> None:
    """Shade soil strata and groundwater table on depth-inverted LPILE plots."""
    for lyr in soil.layers:
        z0 = max(lyr.top_ft, 0.0)
        z1 = min(lyr.bottom_ft, max_depth_ft)
        if z1 <= z0:
            continue
        ax.axhspan(z0, z1, color=lyr.model.color, alpha=0.42, zorder=0)
        ax.axhline(z0, color="#99A3A4", linestyle=":", linewidth=0.7, zorder=1)
        if show_labels:
            ax.text(
                0.97, 0.5 * (z0 + z1),
                f"{lyr.name}",
                transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=7.6, color="#444444", alpha=0.88,
            )

    if 0.0 <= soil.gwl_ft <= max_depth_ft:
        ax.axhline(soil.gwl_ft, color=COLORS["gwl"], linestyle="--", linewidth=1.3, zorder=2)
        if show_labels:
            ax.text(
                0.03, soil.gwl_ft - 0.6,
                f"GWL ({soil.gwl_ft:.0f} ft)",
                transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=7.8, color=COLORS["gwl"], fontweight="bold",
            )


def plot_lpile_4panel(
    results: Mapping[str, LateralPileResult],
    soil: SoilProfile,
    pile: Optional[PileSection] = None,
    title: str = "Laterally Loaded Pile Response vs. Depth (LPILE-Style Profiles)",
) -> plt.Figure:
    """Classic 4-panel LPILE output: Deflection y(z), Bending Moment M(z), Shear V(z), Soil Reaction p(z)."""
    apply_engineering_style()
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 6.4), sharey=True)
    line_colors = [COLORS["navy"], COLORS["red"], COLORS["teal"], COLORS["amber"], COLORS["purple"]]

    first_res = next(iter(results.values()))
    max_depth = float(first_res.profile["z_ft"].max())

    panels = [
        ("y_in", "Lateral Deflection (in)", "Deflection, y(z)"),
        ("moment_kip_ft", "Bending Moment (kip-ft)", "Bending Moment, M(z)"),
        ("shear_kips", "Shear Force (kips)", "Shear Force, V(z)"),
        ("p_kips_per_ft", "Soil Reaction (kips/ft)", "Soil Reaction, p(z)"),
    ]

    for ax_i, (ax, (col, xlabel, subtitle)) in enumerate(zip(axes, panels)):
        _shade_soil_layers(ax, soil, max_depth, show_labels=(ax_i == 3))
        ax.axvline(0.0, color="#333333", linewidth=0.9, zorder=2)

        for r_i, (label, res) in enumerate(results.items()):
            df = res.profile
            ax.plot(
                df[col], df["z_ft"],
                color=line_colors[r_i % len(line_colors)],
                linewidth=2.1,
                label=label,
                zorder=4,
            )

        if col == "moment_kip_ft" and pile is not None and pile.yield_moment_kip_ft is not None:
            my = pile.yield_moment_kip_ft
            ax.axvline(my, color="#C0392B", linestyle="--", linewidth=1.1, label=f"My = ±{my:.0f} k-ft")
            ax.axvline(-my, color="#C0392B", linestyle="--", linewidth=1.1)

        ax.set_xlabel(xlabel)
        ax.set_title(subtitle)
        ax.invert_yaxis()
        ax.set_ylim(max_depth, 0.0)
        if ax_i == 0:
            ax.set_ylabel("Depth Below Pile Head, z (ft)")
            ax.legend(loc="lower right")
        elif col == "moment_kip_ft" and pile is not None and pile.yield_moment_kip_ft is not None:
            ax.legend(loc="lower right")

    fig.suptitle(title, fontsize=13.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_py_curves(
    soil: SoilProfile,
    pile: PileSection,
    depths_ft: Sequence[float],
    y_max_in: float = 3.0,
    loading: str = "static",
    title: str = "Generated Nonlinear p-y Soil Resistance Curves by Depth",
) -> plt.Figure:
    """Plot nonlinear p-y curves p(z, y) in kips/ft across selected depths."""
    apply_engineering_style()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    y_grid = np.linspace(0.0, y_max_in, 250)
    cmap = plt.cm.viridis(np.linspace(0.10, 0.90, len(depths_ft)))

    for z_ft, col in zip(depths_ft, cmap):
        lyr = soil.layer_at(z_ft)
        lyr_lbl = lyr.name if lyr else "Free-Standing"
        p_kips_per_ft = soil.evaluate_py(z_ft, y_grid, pile.diameter_in, loading=loading) * 12.0
        ax.plot(y_grid, p_kips_per_ft, color=col, linewidth=2.0, label=f"z = {z_ft:.1f} ft ({lyr_lbl})")

    ax.set_xlabel("Lateral Deflection, y (in)")
    ax.set_ylabel("Soil Resistance, p (kips/ft)")
    ax.set_title(f"{title} ({loading.capitalize()} Loading)", loc="left")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    return fig


def plot_pushover_curves(
    pushovers: Mapping[str, PushoverResult],
    pile: Optional[PileSection] = None,
    deflection_limit_in: Optional[float] = 1.0,
    title: str = "Lateral Pushover Analysis: Load vs. Deflection & Maximum Bending Moment",
) -> plt.Figure:
    """Two-panel pushover plot: Head Load vs. Head Deflection, and Head Load vs. Max Bending Moment."""
    apply_engineering_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    line_colors = [COLORS["navy"], COLORS["red"], COLORS["teal"], COLORS["amber"]]

    for i, (label, po) in enumerate(pushovers.items()):
        c = line_colors[i % len(line_colors)]
        df = po.table
        axes[0].plot(df["head_deflection_in"], df["lateral_load_kips"], marker="o", markersize=4.5, lw=2.1, color=c, label=label)
        axes[1].plot(df["max_abs_moment_kip_ft"], df["lateral_load_kips"], marker="s", markersize=4.5, lw=2.1, color=c, label=label)

    if deflection_limit_in is not None:
        axes[0].axvline(deflection_limit_in, color="#C0392B", linestyle="--", lw=1.2, label=f"Service Limit ({deflection_limit_in:.1f} in)")
    if pile is not None and pile.yield_moment_kip_ft is not None:
        axes[1].axvline(pile.yield_moment_kip_ft, color="#C0392B", linestyle="--", lw=1.2, label=f"Yield Moment My ({pile.yield_moment_kip_ft:.0f} k-ft)")

    axes[0].set_xlabel("Pile Head Deflection, y(0) (in)")
    axes[0].set_ylabel("Applied Lateral Load, Vt (kips)")
    axes[0].set_title("Load-Deflection Curve (Pushover)", loc="left")
    axes[0].legend(loc="lower right")

    axes[1].set_xlabel("Maximum Absolute Bending Moment, |M_max| (kip-ft)")
    axes[1].set_ylabel("Applied Lateral Load, Vt (kips)")
    axes[1].set_title("Load vs. Peak Bending Moment", loc="left")
    axes[1].legend(loc="lower right")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_pile_cap_response(
    group_result: PileGroupResult,
    title: str = "Pile Cap 2D Finite Element Response (Settlement, Bending Moment & Shear)",
) -> plt.Figure:
    """Three-panel structural diagram along the concrete pile cap span with pile row markers."""
    apply_engineering_style()
    cap = group_result.cap_profile
    pile_xs = group_result.pile_summary["x (ft)"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(11.2, 7.6), sharex=True)

    # Panel 1: Vertical settlement u_z(x) (downward positive -> invert y-axis for visual intuition)
    ax = axes[0]
    ax.plot(cap["x_ft"], cap["uz_in"], color=COLORS["navy"], lw=2.2)
    ax.axhline(0.0, color="#555555", ls="--", lw=0.9)
    for x_p in pile_xs:
        ax.axvline(x_p, color="#999999", ls=":", lw=1.0)
    ax.invert_yaxis()
    ax.set_ylabel("Vertical Disp. uz (in)\n(Downward ↓)")
    ax.set_title("Pile Cap Vertical Settlement / Rocking Profile", loc="left")

    # Panel 2: Cap Bending Moment M(x)
    ax = axes[1]
    ax.fill_between(cap["x_ft"], 0.0, cap["moment_kip_ft"], color=COLORS["teal"], alpha=0.22)
    ax.plot(cap["x_ft"], cap["moment_kip_ft"], color=COLORS["teal"], lw=2.2)
    ax.axhline(0.0, color="#555555", ls="--", lw=0.9)
    for x_p in pile_xs:
        ax.axvline(x_p, color="#999999", ls=":", lw=1.0)
    ax.set_ylabel("Cap Moment (kip-ft)")
    ax.set_title("Pile Cap Bending Moment Diagram, M_cap(x)", loc="left")

    # Panel 3: Cap Shear Force V(x)
    ax = axes[2]
    ax.fill_between(cap["x_ft"], 0.0, cap["shear_kips"], color=COLORS["red"], alpha=0.18)
    ax.plot(cap["x_ft"], cap["shear_kips"], color=COLORS["red"], lw=2.2)
    ax.axhline(0.0, color="#555555", ls="--", lw=0.9)
    for x_p in pile_xs:
        ax.axvline(x_p, color="#999999", ls=":", lw=1.0)
    ax.set_ylabel("Cap Shear (kips)")
    ax.set_xlabel("Horizontal Position Along Pile Cap, x (ft)")
    ax.set_title("Pile Cap Shear Force Diagram, V_cap(x)", loc="left")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_group_load_sharing(
    group_result: PileGroupResult,
    title: str = "Pile Group Load Distribution Across Rows (AASHTO p-Multiplier Shadowing & Rocking)",
) -> plt.Figure:
    """Two-panel bar chart showing per-pile shear, bending moment, and axial push-pull forces by row."""
    apply_engineering_style()
    df = group_result.pile_summary
    x = np.arange(len(df))
    labels = [f"{r['Row']}\n(fm={r['p-Multiplier']:.2f})" for _, r in df.iterrows()]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))

    # Left: Lateral Shear & Peak Moment per pile
    ax = axes[0]
    bars = ax.bar(x, df["Head Shear V (kips/pile)"], color=COLORS["navy"], width=0.52, alpha=0.9)
    for b, (_, r) in zip(bars, df.iterrows()):
        ax.text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
            f"{r['Head Shear V (kips/pile)']:.1f} k\n({r['Row Shear Share (%)']:.1f}%)",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, df["Head Shear V (kips/pile)"].max() * 1.28)
    ax.set_ylabel("Lateral Head Shear (kips / pile)")
    ax.set_title("Lateral Load Sharing (Leading vs. Trailing Rows)", loc="left")

    # Right: Axial Force (Push-Pull from Overturning)
    ax2 = axes[1]
    ax_vals = df["Head Axial P (kips/pile)"].to_numpy()
    colors = [COLORS["red"] if v < 0 else COLORS["teal"] for v in ax_vals]
    bars2 = ax2.bar(x, ax_vals, color=colors, width=0.52, alpha=0.88)
    ax2.axhline(0.0, color="#333333", lw=0.9)
    for b, v in zip(bars2, ax_vals):
        va = "bottom" if v >= 0 else "top"
        off = 4.0 if v >= 0 else -4.0
        ax2.text(b.get_x() + b.get_width() / 2, v + off, f"{v:+.1f} k", ha="center", va=va, fontsize=8.8, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    pad = max(abs(ax_vals).max() * 0.22, 20.0)
    ax2.set_ylim(min(0.0, ax_vals.min()) - pad, max(0.0, ax_vals.max()) + pad)
    ax2.set_ylabel("Pile Axial Force (+Compression / -Tension, kips)")
    ax2.set_title("Axial Load Distribution (Dead Load + Cap Overturning)", loc="left")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_input_schematic(
    pile: PileSection,
    length_ft: float,
    soil: SoilProfile,
    bc: Optional[BoundaryCondition] = None,
    title: str = "Subsurface Soil Profile & Pile Geometry Schematic",
) -> plt.Figure:
    """Draw a geotechnical boring-log & pile elevation schematic echoing user inputs."""
    import matplotlib.patches as patches

    apply_engineering_style()
    max_z = max(length_ft + 6.0, soil.layers[-1].bottom_ft)
    fig, ax = plt.subplots(figsize=(9.6, 6.0))

    # Shade soil layers with bold readable annotations
    for lyr in soil.layers:
        z0 = lyr.top_ft
        z1 = min(lyr.bottom_ft, max_z)
        ax.axhspan(z0, z1, color=lyr.model.color, alpha=0.65, zorder=0)
        ax.axhline(z0, color="#666666", linestyle="-", linewidth=0.9, zorder=1)
        z_mid = 0.5 * (z0 + z1)
        param_str = _describe_soil_params(lyr, soil.gwl_ft)
        ax.text(
            0.38, z_mid,
            f"{lyr.name}  ({lyr.top_ft:.0f}–{lyr.bottom_ft:.0f} ft)\n"
            f"{lyr.model.model_type} | γ={lyr.gamma_pcf:.0f} pcf | {param_str}",
            transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=8.6, color="#222222",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.72, edgecolor="none"),
            zorder=4,
        )

    # Groundwater table line + inverted triangle symbol
    if 0.0 <= soil.gwl_ft <= max_z:
        ax.axhline(soil.gwl_ft, color=COLORS["gwl"], linestyle="--", linewidth=1.6, zorder=3)
        ax.plot([1.7], [soil.gwl_ft - 0.6], marker="v", markersize=8, color=COLORS["gwl"], zorder=5)
        ax.text(
            0.03, soil.gwl_ft - 0.8,
            f"GWL = {soil.gwl_ft:.1f} ft",
            transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=8.8, color=COLORS["gwl"], fontweight="bold", zorder=5,
        )

    # Draw pile shaft centered at x = 0 (exaggerated width visually for clarity)
    pile_w_vis = max(pile.diameter_in / 12.0, 1.4)
    pile_rect = patches.Rectangle(
        (-0.5 * pile_w_vis, 0.0),
        pile_w_vis,
        length_ft,
        facecolor="#5D6D7E",
        edgecolor=COLORS["navy"],
        linewidth=2.0,
        zorder=4,
    )
    ax.add_patch(pile_rect)
    ax.text(
        0.0, 0.5 * length_ft,
        f"L = {length_ft:.0f} ft\nb = {pile.diameter_in:.0f} in",
        ha="center", va="center", color="white", fontsize=8.4, fontweight="bold", zorder=5,
    )

    # Draw applied load arrows at pile head if BoundaryCondition supplied
    if bc is not None:
        if bc.lateral_load_kips != 0:
            ax.annotate(
                "", xy=(-0.5 * pile_w_vis, 0.0), xytext=(-0.5 * pile_w_vis - 2.6, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["red"], lw=2.4),
                zorder=6,
            )
            ax.text(
                -0.5 * pile_w_vis - 2.8, -1.2,
                f"Vt = {bc.lateral_load_kips:.0f} kips",
                ha="right", va="bottom", color=COLORS["red"], fontweight="bold", fontsize=9.2,
            )
        if bc.axial_load_kips != 0:
            ax.annotate(
                "", xy=(0.0, 0.0), xytext=(0.0, -5.5),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["navy"], lw=2.4),
                zorder=6,
            )
            ax.text(
                0.3, -4.5,
                f"Px = {bc.axial_load_kips:.0f} kips",
                ha="left", va="center", color=COLORS["navy"], fontweight="bold", fontsize=9.2,
            )
        if bc.moment_kip_ft != 0:
            ax.text(
                0.5 * pile_w_vis + 0.5, -1.5,
                f"Mt = {bc.moment_kip_ft:.0f} kip-ft",
                ha="left", va="bottom", color=COLORS["purple"], fontweight="bold", fontsize=9.0,
            )

    ax.axhline(0.0, color="#222222", linewidth=1.6, zorder=3)
    ax.set_xlim(-6.5, 14.0)
    ax.set_ylim(max_z, -7.5)
    ax.set_xticks([])
    ax.set_ylabel("Depth Below Pile Head, z (ft)")
    ax.set_title(f"{title} — {pile.name}", loc="left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


