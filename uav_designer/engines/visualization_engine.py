"""
visualization_engine.py
=======================
All matplotlib figures for the UAV Designer.
Each function returns a matplotlib Figure ready for st.pyplot().
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from models.data_models import MassItem, GeometryDerived
from typing import List, Optional

BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"
ACCENT  = "#58a6ff"
GREEN   = "#3fb950"
RED     = "#f78166"
PURPLE  = "#d2a8ff"
ORANGE  = "#ffa657"

CAT_COLORS = {
    "structure":  "#8b949e",
    "propulsion": "#f78166",
    "energy":     "#ffa657",
    "avionics":   "#58a6ff",
    "payload":    "#3fb950",
    "misc":       "#d2a8ff",
}

STYLE = {
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT,
    "xtick.color":      SUBTEXT,
    "ytick.color":      SUBTEXT,
    "text.color":       TEXT,
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "axes.grid":        True,
    "legend.facecolor": "#1c2230",
    "legend.edgecolor": BORDER,
}
plt.rcParams.update(STYLE)


def _setup_ax(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=SUBTEXT, labelsize=8)


def plot_drag_polar(perf: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        _setup_ax(ax)
    CL = perf["CL_plot"]
    CD = perf["CD_plot"]
    LD = perf["LD_plot"]
    ax1.plot(CD, CL, color=ACCENT, lw=2.5)
    idx = np.argmax(LD)
    ax1.scatter(CD[idx], CL[idx], color=ORANGE, s=80, zorder=6,
                label=f"Best L/D = {perf['LD_max']:.1f}")
    ax1.set_xlabel("CD", fontsize=9)
    ax1.set_ylabel("CL", fontsize=9)
    ax1.set_title("Drag Polar", color=TEXT, fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax2.plot(CL, LD, color=PURPLE, lw=2.5)
    ax2.axvline(CL[idx], color=ORANGE, lw=1.2, linestyle="--",
                label=f"CL = {CL[idx]:.2f}")
    ax2.set_xlabel("CL", fontsize=9)
    ax2.set_ylabel("L/D", fontsize=9)
    ax2.set_title("Lift-to-Drag Ratio", color=TEXT, fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    fig.tight_layout(pad=1.5)
    return fig


def plot_thrust_curves(perf: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(BG)
    _setup_ax(ax)
    V = perf["V_arr"]
    ax.plot(V, perf["D_arr"],   color=RED,   lw=2.5, label="Thrust Required")
    ax.plot(V, perf["T_avail"], color=GREEN, lw=2.5, linestyle="--", label="Thrust Available")
    ax.axvline(perf["V_stall"], color=ACCENT, lw=1.2, linestyle=":",
               label=f"V_stall = {perf['V_stall']:.1f} m/s")
    ax.axvline(perf["V_max"],   color=ORANGE, lw=1.2, linestyle=":",
               label=f"V_max = {perf['V_max']:.1f} m/s")
    ax.set_xlabel("Airspeed  [m/s]", fontsize=9)
    ax.set_ylabel("Force  [N]", fontsize=9)
    ax.set_title("Thrust Required vs. Available", color=TEXT, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_roc(perf: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(BG)
    _setup_ax(ax)
    V   = perf["V_arr"]
    ROC = perf["ROC_arr"]
    ax.plot(V, ROC, color=GREEN, lw=2.5)
    ax.axhline(0, color=BORDER, lw=0.8)
    ax.scatter(perf["V_ROCmax"], perf["ROC_max"], color=ORANGE, s=80, zorder=6,
               label=f"ROC_max = {perf['ROC_max']:.2f} m/s")
    ax.fill_between(V, 0, np.clip(ROC, 0, None), alpha=0.15, color=GREEN)
    ax.fill_between(V, np.clip(ROC, None, 0), 0, alpha=0.15, color=RED)
    ax.set_xlabel("Airspeed  [m/s]", fontsize=9)
    ax.set_ylabel("Rate of Climb  [m/s]", fontsize=9)
    ax.set_title("Rate of Climb", color=TEXT, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_endurance(perf: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(BG)
    _setup_ax(ax)
    V = perf["V_end"]
    T = perf["T_end_min"]
    ax.plot(V, T, color=ACCENT, lw=2.5)
    ax.scatter(perf["V_end_max"], perf["end_max_min"], color=ORANGE, s=80, zorder=6,
               label=f"Max = {perf['end_max_min']:.1f} min  @ {perf['V_end_max']:.1f} m/s")
    ax.set_xlabel("Airspeed  [m/s]", fontsize=9)
    ax.set_ylabel("Endurance  [min]", fontsize=9)
    ax.set_title("Predicted Flight Endurance", color=TEXT, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_wing_planform(gd, sweep_deg, span=1.4, root_chord=0.20, taper=0.70) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    _setup_ax(ax)
    ax.set_aspect("equal")
    ax.grid(True, color="#21262d", lw=0.5)

    half_span    = span / 2
    tip_chord    = root_chord * taper
    sweep_offset = half_span * math.tan(math.radians(sweep_deg))

    for sign in (-1, 1):
        wx = [0, sweep_offset, sweep_offset + tip_chord, root_chord, 0]
        wy = [0, sign * half_span, sign * half_span, 0, 0]
        ax.fill(wx, wy, color=ACCENT, alpha=0.25)
        ax.plot(wx, wy, color=ACCENT, lw=2)

    mac      = gd.MAC
    mac_le_x = sweep_offset * (1 - taper) / (1 + taper) if span > 0 else 0
    ax.plot([mac_le_x, mac_le_x + mac], [0, 0],
            color=ORANGE, lw=2.5, linestyle="--", label=f"MAC = {mac:.3f} m")
    ax.axvline(mac_le_x + 0.25 * mac, color=GREEN, lw=1.2, linestyle=":", label="25% MAC")
    ax.axvline(mac_le_x + 0.35 * mac, color=RED,   lw=1.2, linestyle=":", label="35% MAC")

    ax.set_xlabel("Chord direction [m]", fontsize=9)
    ax.set_ylabel("Span [m]", fontsize=9)
    ax.set_title(
        f"Wing Planform  —  AR={gd.AR:.1f}  |  S={gd.S:.3f} m2  |  "
        f"Taper={taper:.2f}  |  Sweep={sweep_deg:.1f} deg",
        color=TEXT, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_mass_layout(items, gd, configuration, fuselage_length, cg_x, cg_lo, cg_hi, span=1.4, root_chord=0.20, taper=0.70, sweep_deg=0) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(BG)
    _setup_ax(ax)
    ax.set_aspect("equal")
    ax.grid(True, color="#21262d", lw=0.4, alpha=0.6)
    ax.set_axisbelow(True)

    L         = fuselage_length
    half_span = span / 2
    tip_chord = root_chord * taper
    sweep_off = half_span * math.tan(math.radians(sweep_deg))
    fuselage_w = max(L * 0.09, 0.06)

    fuse = mpatches.FancyBboxPatch(
        (0, -fuselage_w / 2), L, fuselage_w,
        boxstyle="round,pad=0.01",
        linewidth=1.5, edgecolor=SUBTEXT,
        facecolor="#1c2230", zorder=2)
    ax.add_patch(fuse)

    root_le_x = L * 0.30
    for sign in (-1, 1):
        tip_le_x = root_le_x + sweep_off
        wx = [root_le_x, tip_le_x, tip_le_x + tip_chord,
              root_le_x + root_chord, root_le_x]
        wy = [0, sign * half_span, sign * half_span, 0, 0]
        ax.fill(wx, wy, color="#21262d", alpha=0.7, zorder=1)
        ax.plot(wx, wy, color=ACCENT, lw=1.5, zorder=3)

    if configuration not in ("Flying Wing",):
        tail_x   = L * 0.88
        h_tail_h = max(span * 0.18, 0.10)
        h_chord  = max(L * 0.14, 0.06)
        v_tail_h = max(span * 0.08, 0.05)
        for sign in (-1, 1):
            htx = [tail_x, tail_x, tail_x + h_chord, tail_x + h_chord, tail_x]
            hty = [0, sign * h_tail_h, sign * h_tail_h, 0, 0]
            ax.fill(htx, hty, color="#21262d", alpha=0.7, zorder=1)
            ax.plot(htx, hty, color=ACCENT, lw=1.2, zorder=3)
        vtx = [tail_x, tail_x + h_chord, tail_x + h_chord, tail_x, tail_x]
        vty = [-v_tail_h/4, -v_tail_h/4, v_tail_h/4, v_tail_h/4, -v_tail_h/4]
        ax.fill(vtx, vty, color=SUBTEXT, alpha=0.25, zorder=2)

    if configuration == "Twin-Boom Pusher":
        boom_y = fuselage_w * 1.8
        for sign in (-1, 1):
            ax.plot([L * 0.28, L], [sign * boom_y, sign * boom_y],
                    color=SUBTEXT, lw=2, zorder=2)

    ax.axvspan(cg_lo, cg_hi, alpha=0.15, color=GREEN, zorder=1, label="Target CG band")
    mac_le = root_le_x + sweep_off * 0.3
    ax.plot([mac_le, mac_le + gd.MAC], [-fuselage_w * 0.5, -fuselage_w * 0.5],
            color=ORANGE, lw=3, zorder=4, label=f"MAC ({gd.MAC:.3f} m)")

    cg_color = GREEN if cg_lo <= cg_x <= cg_hi else RED
    ax.axvline(cg_x, color=cg_color, lw=2.5, zorder=6,
               label=f"CG = {cg_x*100:.1f} cm from nose")
    ax.text(cg_x, -half_span * 0.55, "CG", color=cg_color,
            fontsize=9, fontweight="bold", ha="center", zorder=7)

    total_mass = sum(i.mass_g for i in items)
    placed_y = {}
    for item in items:
        if item.mass_g <= 0:
            continue
        color  = CAT_COLORS.get(item.category, SUBTEXT)
        bw     = item.width if item.width else fuselage_w * 0.6
        bh     = item.length if item.length else 0.05
        bx     = item.x - bh / 2
        by_raw = item.y - bw / 2
        stack_key = round(item.x, 2)
        by = placed_y.get(stack_key, by_raw)
        placed_y[stack_key] = by + bw + 0.005
        rect = FancyBboxPatch(
            (bx, by), bh, bw,
            boxstyle="round,pad=0.003",
            linewidth=1.2, edgecolor=color,
            facecolor=color + "40", zorder=5)
        ax.add_patch(rect)
        pct = item.mass_g / total_mass * 100 if total_mass else 0
        ax.text(item.x, by + bw / 2,
                f"{item.name}\n{item.mass_g:.0f}g ({pct:.0f}%)",
                color=color, fontsize=6.5, ha="center", va="center",
                fontweight="bold", zorder=6)

    legend_patches = [mpatches.Patch(color=c, label=cat.capitalize())
                      for cat, c in CAT_COLORS.items()]
    ax.legend(handles=legend_patches, fontsize=7, loc="lower right", ncol=3, framealpha=0.9)

    margin = span * 0.12
    ax.set_xlim(-0.05, L + 0.08)
    ax.set_ylim(-half_span - margin, half_span + margin)
    ax.set_xlabel("Longitudinal position from nose  [m]", fontsize=9)
    ax.set_ylabel("Lateral position  [m]", fontsize=9)
    ax.set_title(
        f"{configuration}  -  Top-View Mass Layout   "
        f"(Total: {total_mass:.0f} g  |  CG: {cg_x*100:.1f} cm from nose)",
        color=TEXT, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_mass_pie(items: List[MassItem]) -> plt.Figure:
    by_cat: dict = {}
    for item in items:
        by_cat[item.category] = by_cat.get(item.category, 0) + item.mass_g
    labels = list(by_cat.keys())
    sizes  = list(by_cat.values())
    colors = [CAT_COLORS.get(l, SUBTEXT) for l in labels]
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=[l.capitalize() for l in labels],
        colors=colors, autopct="%1.0f%%",
        pctdistance=0.82, startangle=140,
        wedgeprops=dict(linewidth=1.5, edgecolor=BG))
    for t in texts:
        t.set_color(TEXT); t.set_fontsize(9)
    for t in autotexts:
        t.set_color(BG); t.set_fontweight("bold"); t.set_fontsize(8)
    ax.set_title("Mass Breakdown by Category", color=TEXT, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_moment_diagram(items: List[MassItem], cg_x: float,
                         cg_lo: float, cg_hi: float) -> plt.Figure:
    if not items:
        fig, ax = plt.subplots(figsize=(9, 3))
        fig.patch.set_facecolor(BG)
        _setup_ax(ax)
        ax.text(0.5, 0.5, "No mass items defined", transform=ax.transAxes,
                ha="center", va="center", color=SUBTEXT)
        return fig
    items_s = sorted(items, key=lambda i: i.x)
    names   = [i.name for i in items_s]
    masses  = [i.mass_g for i in items_s]
    xs      = [i.x for i in items_s]
    cats    = [i.category for i in items_s]
    colors  = [CAT_COLORS.get(c, SUBTEXT) for c in cats]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        _setup_ax(ax)
    ax1.bar(xs, masses, width=0.03, color=colors, alpha=0.85, zorder=3)
    for x, m, n in zip(xs, masses, names):
        ax1.text(x, m + max(masses)*0.02, n, rotation=45,
                 ha="left", va="bottom", fontsize=7, color=TEXT)
    ax1.axvline(cg_x, color=RED if not (cg_lo<=cg_x<=cg_hi) else GREEN,
                lw=2, linestyle="--", label=f"CG = {cg_x*100:.1f} cm")
    ax1.axvspan(cg_lo, cg_hi, alpha=0.15, color=GREEN, label="Target band")
    ax1.set_ylabel("Mass  [g]", fontsize=9)
    ax1.set_title("Longitudinal Mass Distribution", color=TEXT, fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)
    cum_mass   = 0
    cum_moment = 0
    cg_running = []
    x_running  = []
    for it in items_s:
        cum_mass   += it.mass_g
        cum_moment += it.mass_g * it.x
        cg_running.append(cum_moment / cum_mass)
        x_running.append(it.x)
    ax2.plot(x_running, [c*100 for c in cg_running],
             color=ACCENT, lw=2, marker="o", markersize=4, zorder=4)
    ax2.axhline(cg_lo * 100, color=GREEN, lw=1, linestyle=":")
    ax2.axhline(cg_hi * 100, color=GREEN, lw=1, linestyle=":",
                label=f"Target band  [{cg_lo*100:.1f}-{cg_hi*100:.1f} cm]")
    ax2.set_xlabel("Component position from nose  [m]", fontsize=9)
    ax2.set_ylabel("Running CG  [cm]", fontsize=9)
    ax2.set_title("Running CG as Components Are Loaded", color=TEXT, fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    fig.tight_layout(pad=1.2)
    return fig


def plot_config_scores(scores) -> plt.Figure:
    names  = [s.name for s in scores]
    vals   = [s.score for s in scores]
    colors = [GREEN if i == 0 else ACCENT if i < 3 else SUBTEXT
              for i in range(len(scores))]
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(BG)
    _setup_ax(ax)
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Mission Suitability Score", fontsize=9)
    ax.set_title("Configuration Ranking for This Mission", color=TEXT, fontsize=11, fontweight="bold")
    ax.axvline(50, color=SUBTEXT, lw=0.8, linestyle="--")
    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}", va="center", fontsize=8, color=TEXT)
    fig.tight_layout()
    return fig
