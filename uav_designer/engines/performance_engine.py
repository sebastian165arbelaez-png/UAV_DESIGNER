"""
performance_engine.py
=====================
First-order flight performance estimates.
All speeds in m/s, forces in N, power in W, mass in kg.
"""

import math
import numpy as np
from models.data_models import GeometryDerived, PropulsionDerived

G       = 9.81
RHO_SSL = 1.225

AIRFOIL_DB = {
    "NACA 2412":  {"Cl_a": 6.28, "al0": -2.1, "CL_max": 1.50},
    "NACA 4412":  {"Cl_a": 6.28, "al0": -4.0, "CL_max": 1.65},
    "NACA 0012":  {"Cl_a": 6.28, "al0":  0.0, "CL_max": 1.35},
    "Clark-Y":    {"Cl_a": 6.00, "al0": -3.0, "CL_max": 1.45},
    "S1223":      {"Cl_a": 6.20, "al0": -4.5, "CL_max": 2.20},
    "E214":       {"Cl_a": 6.10, "al0": -3.8, "CL_max": 1.80},
    "Flat plate": {"Cl_a": 6.28, "al0":  0.0, "CL_max": 0.90},
    "NACA 4-digit (custom)": {"Cl_a": 6.28, "al0": -4.0, "CL_max": 1.60},
}


def get_airfoil(name: str) -> dict:
    return AIRFOIL_DB.get(name, AIRFOIL_DB["NACA 4412"])


def cl_alpha_3d(Cl_a2D: float, AR: float, sweep_r: float, e: float) -> float:
    return Cl_a2D / (1 + Cl_a2D / (math.pi * e * AR))


def estimate_CD0(configuration: str = "Conventional Tractor") -> float:
    # These values include realistic parasite drag from fuselage, landing gear,
    # wiring, interference effects, and surface roughness for small UAVs.
    # Small UAVs operate at low Re where skin friction is higher than theoretical.
    base = {
        "Conventional Tractor": 0.055,
        "Conventional Pusher":  0.050,
        "Twin-Boom Pusher":     0.065,
        "Flying Wing":          0.030,
        "Canard":               0.058,
        "Tandem Wing":          0.060,
    }
    return base.get(configuration, 0.055)


def drag_polar(CL: np.ndarray, AR: float, CD0: float, e: float) -> np.ndarray:
    return CD0 + CL**2 / (math.pi * e * AR)


def compute_performance(total_mass_kg: float,
                         gd: GeometryDerived,
                         pd: PropulsionDerived,
                         airfoil_name: str,
                         configuration: str,
                         extra_load_w: float = 13.0,
                         rho: float = RHO_SSL) -> dict:
    af  = get_airfoil(airfoil_name)
    e   = gd.oswald_e if hasattr(gd, 'oswald_e') else 0.82
    CD0 = estimate_CD0(configuration)
    # Real-world correction: small UAVs have additional drag from
    # surface roughness, gaps, hinges, antenna, wires. Add 30% to CD0.
    CD0 = CD0 * 1.30

    Cl_a3D  = cl_alpha_3d(af["Cl_a"], gd.AR, gd.sweep_r, e)
    CL_max  = af["CL_max"] * 0.90

    W = total_mass_kg * G
    S = gd.S

    CL_md   = math.sqrt(math.pi * e * gd.AR * CD0)
    CL_be   = math.sqrt(3 * math.pi * e * gd.AR * CD0)
    CD_md   = drag_polar(np.array([CL_md]), gd.AR, CD0, e)[0]
    LD_max  = CL_md / CD_md

    def speed_from_CL(CL):
        return math.sqrt(2 * W / (rho * S * CL)) if (CL > 0 and S > 0) else 0

    V_stall = speed_from_CL(CL_max)
    V_md    = speed_from_CL(CL_md)
    V_be    = speed_from_CL(CL_be)
    V_br    = V_md

    V_arr   = np.linspace(max(V_stall * 0.90, 1.0), max(V_stall * 5, 55), 500)
    q_arr   = 0.5 * rho * V_arr**2
    CL_arr  = W / (q_arr * S)
    CD_arr  = drag_polar(CL_arr, gd.AR, CD0, e)
    D_arr   = q_arr * S * CD_arr

    T_max   = pd.static_thrust_n
    T_avail = T_max * (1 - 0.45 * (V_arr / V_arr[-1]))
    T_avail = np.clip(T_avail, 0, T_max)

    ROC_arr = (T_avail - D_arr) * V_arr / W

    diff    = T_avail - D_arr
    idx_eq  = np.where(np.diff(np.sign(diff)))[0]
    V_max   = float(V_arr[idx_eq[-1]]) if len(idx_eq) else float(V_arr[-1])
    ROC_max = float(np.max(ROC_arr))
    V_ROCmax= float(V_arr[np.argmax(ROC_arr)])

    V_cruise_use = min(V_md * 1.1, V_max * 0.90)
    q_cr   = 0.5 * rho * V_cruise_use**2
    CL_cr  = W / (q_cr * S)
    CD_cr  = CD0 + CL_cr**2 / (math.pi * e * gd.AR)
    D_cr   = q_cr * S * CD_cr

    # Total efficiency chain
    # Note: prop efficiency at cruise is significantly lower than static (0.75→0.60)
    # Additional losses: battery internal resistance (~0.92), wiring (~0.98)
    eta_prop  = 0.60   # cruise prop efficiency (realistic for small fixed-pitch props)
    eta_motor = pd.eta_motor if hasattr(pd, 'eta_motor') else 0.85
    eta_esc   = pd.eta_esc   if hasattr(pd, 'eta_esc')   else 0.95
    eta_bat   = 0.92   # battery internal resistance losses
    eta_wire  = 0.98   # wiring losses
    eta_total = eta_prop * eta_motor * eta_esc * eta_bat * eta_wire
    eta_total = max(eta_total, 0.35)

    P_shaft_cr = D_cr * V_cruise_use
    P_elec_cr  = P_shaft_cr / eta_total + extra_load_w

    # Endurance sweep — apply 80% usable battery factor
    usable_energy_wh = pd.energy_wh * 0.80
    V_end  = np.linspace(V_stall * 1.05, V_max * 0.95, 400)
    q_end  = 0.5 * rho * V_end**2
    CL_end = W / (q_end * S)
    CD_end = CD0 + CL_end**2 / (math.pi * e * gd.AR)
    D_end  = q_end * S * CD_end
    P_end  = D_end * V_end / eta_total + extra_load_w * 1.10  # 10% margin on aux loads
    T_end  = (usable_energy_wh / P_end) * 60   # minutes

    end_max    = float(np.max(T_end))
    V_end_max  = float(V_end[np.argmax(T_end)])
    end_cruise = (usable_energy_wh / P_elec_cr) * 60 if P_elec_cr > 0 else 0

    CL_plot = np.linspace(0, CL_max, 300)
    CD_plot = drag_polar(CL_plot, gd.AR, CD0, e)
    LD_plot = np.where(CD_plot > 0, CL_plot / CD_plot, 0)

    return {
        "CD0":          CD0,
        "e":            e,
        "CL_max":       CL_max,
        "CL_alpha_3D":  Cl_a3D,
        "LD_max":       LD_max,
        "V_stall":      V_stall,
        "V_md":         V_md,
        "V_be":         V_be,
        "V_br":         V_br,
        "V_cruise":     V_cruise_use,
        "V_max":        V_max,
        "ROC_max":      ROC_max,
        "V_ROCmax":     V_ROCmax,
        "end_max_min":  end_max,
        "V_end_max":    V_end_max,
        "end_cruise_min": end_cruise,
        "P_cruise_w":   P_elec_cr,
        "glide_ratio":  LD_max,
        "V_arr":    V_arr,
        "D_arr":    D_arr,
        "T_avail":  T_avail,
        "ROC_arr":  ROC_arr,
        "V_end":    V_end,
        "T_end_min":T_end,
        "CL_plot":  CL_plot,
        "CD_plot":  CD_plot,
        "LD_plot":  LD_plot,
    }
