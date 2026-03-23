"""
geometry_engine.py
==================
Derives full wing, tail, and fuselage geometry from sparse user inputs.
Validates impossible combinations and returns GeometryDerived.
"""

import math
from models.data_models import GeometryInputs, GeometryDerived


def build_geometry(g: GeometryInputs, total_mass_kg: float,
                   rho: float = 1.225) -> tuple[GeometryDerived, list[str]]:
    errors: list[str] = []
    d = GeometryDerived()

    if g.taper <= 0 or g.taper > 1.0:
        errors.append("Taper ratio must be between 0 (pointed tip) and 1.0 (rectangular).")
        g.taper = max(0.1, min(g.taper, 1.0))

    d.tip_chord = g.root_chord * g.taper
    d.MAC       = (2/3) * g.root_chord * (1 + g.taper + g.taper**2) / (1 + g.taper)
    d.S         = g.span * (g.root_chord + d.tip_chord) / 2.0
    d.AR        = g.span**2 / d.S if d.S > 0 else 0
    d.sweep_r   = math.radians(g.sweep_deg)

    if d.AR < 4:
        errors.append(f"Aspect ratio {d.AR:.1f} is very low — expect poor L/D and high induced drag.")
    if d.AR > 20:
        errors.append(f"Aspect ratio {d.AR:.1f} is very high — structural loads on the wing will be severe.")
    if g.root_chord < d.tip_chord:
        errors.append("Root chord is smaller than tip chord — taper ratio exceeds 1.0.")

    W = total_mass_kg * 9.81
    d.wing_loading = W / d.S if d.S > 0 else 0

    if d.S > 0 and d.MAC > 0 and g.span > 0:
        d.h_tail_volume = (g.h_tail_area * g.tail_arm) / (d.S * d.MAC)
        d.v_tail_volume = (g.v_tail_area * g.tail_arm) / (d.S * g.span)
    else:
        d.h_tail_volume = 0
        d.v_tail_volume = 0

    if d.h_tail_volume < 0.30:
        errors.append(
            f"Horizontal tail volume coefficient {d.h_tail_volume:.2f} is low (typical: 0.35-0.50). "
            "Aircraft may be longitudinally unstable."
        )
    if d.v_tail_volume < 0.02:
        errors.append(
            f"Vertical tail volume coefficient {d.v_tail_volume:.3f} is low (typical: 0.025-0.05). "
            "Aircraft may lack yaw authority."
        )

    return d, errors


def back_solve_geometry(mtow_g: float, endurance_min: float,
                         cruise_speed: float, payload_g: float,
                         rho: float = 1.225) -> dict:
    G    = 9.81
    mtow = mtow_g / 1000
    W    = mtow * G

    CL_max         = 1.4
    V_stall_target = cruise_speed / 1.4
    S_needed       = 2 * W / (rho * V_stall_target**2 * CL_max)

    if endurance_min >= 60:
        AR_rec = (9, 14)
    elif endurance_min >= 30:
        AR_rec = (7, 10)
    else:
        AR_rec = (5, 8)

    span_lo = math.sqrt(AR_rec[0] * S_needed)
    span_hi = math.sqrt(AR_rec[1] * S_needed)

    CD0    = 0.025
    e      = 0.82
    AR_mid = (AR_rec[0] + AR_rec[1]) / 2
    CL_cr  = W / (0.5 * rho * cruise_speed**2 * S_needed)
    CD_cr  = CD0 + CL_cr**2 / (math.pi * e * AR_mid)
    D      = 0.5 * rho * cruise_speed**2 * S_needed * CD_cr
    P_elec = D * cruise_speed / 0.65

    # 80% usable battery factor applied
    E_needed_wh = P_elec * (endurance_min / 60) / 0.80

    # LiPo energy density: 150-200 Wh/kg = 0.150-0.200 Wh/g
    bat_mass_g_lo = E_needed_wh / 0.200
    bat_mass_g_hi = E_needed_wh / 0.150

    return {
        "wing_area_m2":     round(S_needed, 4),
        "AR_range":         AR_rec,
        "span_range_m":     (round(span_lo, 2), round(span_hi, 2)),
        "battery_mass_g":   (round(bat_mass_g_lo), round(bat_mass_g_hi)),
        "V_stall_target":   round(V_stall_target, 1),
        "est_power_w":      round(P_elec, 1),
        "est_energy_wh":    round(E_needed_wh, 1),
        "wing_loading_nm2": round(W / S_needed, 1),
    }
