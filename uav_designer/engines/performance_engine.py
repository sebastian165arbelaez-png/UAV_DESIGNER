"""
performance_engine.py
=====================
First-order flight performance estimates.
All speeds in m/s, forces in N, power in W, mass in kg.
"""

import math
import numpy as np
from typing import Optional
from models.data_models import GeometryDerived, PropulsionDerived

G       = 9.81
RHO_SSL = 1.225

AIRFOIL_DB = {
    "NACA 2412":  {"Cl_a": 6.28, "al0": -2.1, "CL_max": 1.50,
                   "desc": "Classic general-purpose airfoil. Moderate camber, good all-round performance.",
                   "best_for": ["general", "trainer", "surveillance"],
                   "Re_range": "500k-2M", "LD_typical": 12},
    "NACA 4412":  {"Cl_a": 6.28, "al0": -4.0, "CL_max": 1.65,
                   "desc": "Higher camber variant. Better lift at low speeds, slightly more drag.",
                   "best_for": ["endurance", "cargo", "surveillance"],
                   "Re_range": "500k-2M", "LD_typical": 13},
    "NACA 0012":  {"Cl_a": 6.28, "al0":  0.0, "CL_max": 1.35,
                   "desc": "Symmetric airfoil. Excellent for aerobatics, no pitching moment.",
                   "best_for": ["aerobatics", "racing", "symmetric"],
                   "Re_range": "500k-3M", "LD_typical": 10},
    "NACA 2415":  {"Cl_a": 6.28, "al0": -2.1, "CL_max": 1.55,
                   "desc": "Thicker variant of 2412. Better structural depth, slightly more drag.",
                   "best_for": ["general", "cargo", "trainer"],
                   "Re_range": "500k-2M", "LD_typical": 11},
    "NACA 4415":  {"Cl_a": 6.28, "al0": -4.0, "CL_max": 1.70,
                   "desc": "Thicker high-camber airfoil. Good lift, robust structure.",
                   "best_for": ["endurance", "cargo", "surveillance"],
                   "Re_range": "300k-1.5M", "LD_typical": 12},
    "Clark-Y":    {"Cl_a": 6.00, "al0": -3.0, "CL_max": 1.45,
                   "desc": "Historic flat-bottomed airfoil. Easy to build, predictable stall.",
                   "best_for": ["trainer", "general", "easy-build"],
                   "Re_range": "300k-1M", "LD_typical": 11},
    "S1223":      {"Cl_a": 6.20, "al0": -4.5, "CL_max": 2.20,
                   "desc": "High-lift Selig airfoil. Outstanding CL_max at low Re. Ideal for slow UAVs.",
                   "best_for": ["endurance", "slow-flight", "hand-launch", "surveillance"],
                   "Re_range": "100k-500k", "LD_typical": 15},
    "E214":       {"Cl_a": 6.10, "al0": -3.8, "CL_max": 1.80,
                   "desc": "Eppler high-performance section. Excellent L/D at low Re.",
                   "best_for": ["endurance", "mapping", "efficiency"],
                   "Re_range": "200k-800k", "LD_typical": 16},
    "MH45":       {"Cl_a": 6.15, "al0": -2.5, "CL_max": 1.60,
                   "desc": "Martin Hepperle reflexed airfoil. Designed for flying wings.",
                   "best_for": ["flying-wing", "tailless"],
                   "Re_range": "100k-500k", "LD_typical": 13},
    "NACA 23012": {"Cl_a": 6.28, "al0": -2.7, "CL_max": 1.60,
                   "desc": "5-digit series. High CL_max with low pitching moment.",
                   "best_for": ["cargo", "endurance", "general"],
                   "Re_range": "500k-2M", "LD_typical": 14},
    "Flat plate":  {"Cl_a": 6.28, "al0":  0.0, "CL_max": 0.90,
                   "desc": "Theoretical flat plate. Use only for baseline comparison.",
                   "best_for": [],
                   "Re_range": "all", "LD_typical": 6},
}



def parse_naca4(code: str) -> dict | None:
    """Parse NACA 4-digit airfoil: MPXX"""
    import math
    s = code.replace(" ", "").upper().replace("NACA", "")
    if len(s) != 4 or not s.isdigit():
        return None
    M   = int(s[0]) / 100
    P   = int(s[1]) / 10
    T   = int(s[2:]) / 100
    Cl_a = 2 * math.pi
    if M == 0:
        al0 = 0.0
    else:
        al0 = -math.degrees(2 * M * (1 - P + math.log(max(1 - P, 0.01)) * P))
    CL_max = max(0.8, min(1.0 + 2.0*M + 0.5*(T - 0.12), 2.2))
    return {"Cl_a": round(Cl_a, 3), "al0": round(al0, 2),
            "CL_max": round(CL_max, 3), "M_pct": M*100, "T_pct": T*100,
            "type": "4-digit"}


def parse_naca5(code: str) -> dict | None:
    """Parse NACA 5-digit airfoil: LPSXX"""
    import math
    s = code.replace(" ", "").upper().replace("NACA", "")
    if len(s) != 5 or not s.isdigit():
        return None
    L  = int(s[0])
    T  = int(s[3:]) / 100
    Cl_a      = 2 * math.pi
    CL_design = L * 3 / 20
    al0       = -math.degrees(CL_design / Cl_a)
    CL_max    = max(0.8, min(1.0 + CL_design + 0.3*(T - 0.12), 2.2))
    return {"Cl_a": round(Cl_a, 3), "al0": round(al0, 2),
            "CL_max": round(CL_max, 3), "CL_design": round(CL_design, 3),
            "T_pct": T*100, "type": "5-digit"}


def parse_naca(code: str) -> dict | None:
    """Try to parse a NACA 4 or 5-digit code. Returns None if unrecognized."""
    s = code.replace(" ", "").upper().replace("NACA", "")
    if len(s) == 4 and s.isdigit():
        return parse_naca4(code)
    if len(s) == 5 and s.isdigit():
        return parse_naca5(code)
    return None


def naca4_coords(M_pct: float, P_pct: float, T_pct: float, n: int = 120):
    """
    Generate NACA 4-digit airfoil surface coordinates.
    Returns (x_upper, y_upper, x_lower, y_lower, x_camber, y_camber)
    """
    M = M_pct / 100
    P = P_pct / 10
    T = T_pct / 100
    x = np.linspace(0, 1, n)
    yt = (T/0.2)*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                  + 0.2843*x**3 - 0.1015*x**4)
    if P > 0:
        yc  = np.where(x < P,
                       (M/P**2)*(2*P*x - x**2),
                       (M/(1-P)**2)*((1-2*P) + 2*P*x - x**2))
        dyc = np.where(x < P,
                       (2*M/P**2)*(P - x),
                       (2*M/(1-P)**2)*(P - x))
    else:
        yc  = np.zeros_like(x)
        dyc = np.zeros_like(x)
    theta = np.arctan(dyc)
    return (x - yt*np.sin(theta), yc + yt*np.cos(theta),
            x + yt*np.sin(theta), yc - yt*np.cos(theta),
            x, yc)


def naca5_coords(L: int, P: int, T_pct: float, n: int = 120):
    """
    Generate approximate NACA 5-digit airfoil coordinates.
    Uses modified mean camber line.
    """
    T = T_pct / 100
    x = np.linspace(0, 1, n)
    yt = (T/0.2)*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                  + 0.2843*x**3 - 0.1015*x**4)
    # Approximate 5-digit camber using design CL
    CL_d = L * 3 / 20
    p    = P / 20
    r    = 3.33333*(0.6075*p**3 - 0.1021*p)  # approx r coefficient
    yc   = np.where(x < p,
                    r/6*(x**3 - 3*p*x**2 + p**2*(3-p)*x),
                    r*p**3/6*(1 - x))
    dyc  = np.gradient(yc, x)
    theta = np.arctan(dyc)
    return (x - yt*np.sin(theta), yc + yt*np.cos(theta),
            x + yt*np.sin(theta), yc - yt*np.cos(theta),
            x, yc)


def get_airfoil_coords(name: str):
    """
    Get airfoil coordinates for plotting.
    Returns (xu, yu, xl, yl, xc, yc) or None if not computable.
    """
    s = name.replace(" ","").upper().replace("NACA","")
    if len(s) == 4 and s.isdigit():
        M, P, T = int(s[0]), int(s[1]), int(s[2:])
        return naca4_coords(M, P, T)
    if len(s) == 5 and s.isdigit():
        L, P, S_d = int(s[0]), int(s[1]), int(s[2])
        T = int(s[3:])
        return naca5_coords(L, P, T)
    # Handle preset names that are NACA codes
    for code in ["2412","4412","0012","2415","4415","23012"]:
        if code in name.replace(" ",""):
            if len(code) == 4:
                M,P,T = int(code[0]),int(code[1]),int(code[2:])
                return naca4_coords(M, P, T)
            else:
                L,P_d,S_d = int(code[0]),int(code[1]),int(code[2])
                T = int(code[3:])
                return naca5_coords(L, P_d, T)
    return None


def suggest_airfoils(mission_type: str, endurance_min: float,
                      cruise_speed: float, launch_method: str,
                      payload_mass: float) -> list:
    """
    Suggest ranked airfoils based on mission parameters.
    Returns list of (name, score, reason) tuples.
    """
    suggestions = []

    for name, data in AIRFOIL_DB.items():
        if name == "Flat plate":
            continue
        score  = 50.0
        reason = []

        best = data.get("best_for", [])

        # Endurance priority
        if endurance_min >= 45:
            if any(b in best for b in ["endurance", "efficiency", "slow-flight"]):
                score += 20
                reason.append("high L/D suits endurance mission")
            if data["LD_typical"] >= 14:
                score += 10
                reason.append(f"excellent L/D ~{data['LD_typical']}")
            elif data["LD_typical"] >= 12:
                score += 5

        # Hand launch — need high CL_max for low stall speed
        if launch_method == "Hand launch":
            cl_max = data["CL_max"]
            if cl_max >= 1.8:
                score += 15
                reason.append(f"high CL_max {cl_max:.2f} reduces stall speed")
            elif cl_max >= 1.5:
                score += 8

        # Low cruise speed → high lift needed
        if cruise_speed < 12:
            if any(b in best for b in ["slow-flight","endurance","hand-launch"]):
                score += 12
                reason.append("suited to low-speed cruise")

        # High speed → low drag, symmetric or thin preferred
        if cruise_speed > 20:
            if "racing" in best or "aerobatics" in best:
                score += 10
                reason.append("low-drag profile suits high speed")
            if data["CL_max"] > 1.8:
                score -= 8
                reason.append("high-lift profile adds drag at speed")

        # Mission type matching
        m_lower = mission_type.lower()
        if "surveillance" in m_lower or "isr" in m_lower:
            if any(b in best for b in ["surveillance","endurance","efficiency"]):
                score += 10
                reason.append("good match for surveillance mission")
        if "cargo" in m_lower:
            if "cargo" in best:
                score += 10
                reason.append("handles higher wing loading well")
        if "racing" in m_lower or "fpv" in m_lower:
            if any(b in best for b in ["racing","aerobatics","symmetric"]):
                score += 15
                reason.append("suited to high-speed dynamic flight")
        if "mapping" in m_lower:
            if any(b in best for b in ["mapping","efficiency","endurance"]):
                score += 10

        # Flying wing penalty for most airfoils
        if any(b in best for b in ["flying-wing","tailless"]):
            score += 5  # bonus for reflexed profiles

        suggestions.append((name, round(score, 1),
                             "; ".join(reason) if reason else "general-purpose choice"))

    suggestions.sort(key=lambda x: x[1], reverse=True)
    return suggestions[:5]


def get_airfoil(name: str) -> dict:
    # First try the preset database
    if name in AIRFOIL_DB:
        return AIRFOIL_DB[name]
    # Then try to parse as a custom NACA code
    parsed = parse_naca(name)
    if parsed:
        return {"Cl_a": parsed["Cl_a"], "al0": parsed["al0"], "CL_max": parsed["CL_max"]}
    # Fallback
    return AIRFOIL_DB["NACA 4412"]


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
