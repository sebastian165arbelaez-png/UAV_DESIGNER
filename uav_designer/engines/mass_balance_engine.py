"""
mass_balance_engine.py
======================
Tracks component masses, computes CG, longitudinal moments,
and evaluates packaging feasibility.
"""

import math
from models.data_models import MassItem, GeometryDerived
from typing import List, Tuple


G = 9.81


def compute_cg(items: List[MassItem]) -> Tuple[float, float, float]:
    """
    Returns (total_mass_g, cg_x, cg_y).
    cg_x and cg_y are in metres from nose.
    """
    total = sum(i.mass_g for i in items)
    if total == 0:
        return 0, 0, 0
    cg_x = sum(i.mass_g * i.x for i in items) / total
    cg_y = sum(i.mass_g * i.y for i in items) / total
    return total, cg_x, cg_y


def target_cg_range(gd: GeometryDerived,
                     fuselage_length: float,
                     mac_le_x: float) -> Tuple[float, float]:
    """
    Returns (cg_lo, cg_hi) in metres from nose.
    Target: 25%–35% of MAC from MAC leading edge.
    mac_le_x = x position of MAC leading edge from nose.
    """
    cg_lo = mac_le_x + 0.20 * gd.MAC
    cg_hi = mac_le_x + 0.38 * gd.MAC
    return cg_lo, cg_hi


def auto_place_battery(items: List[MassItem],
                        cg_lo: float, cg_hi: float,
                        configuration: str) -> Tuple[float, str]:
    """
    Given the current (non-battery) mass distribution,
    suggest where to place the battery to land CG in target range.
    Returns (suggested_x, message).
    """
    non_bat = [i for i in items if i.category != "energy"]
    if not non_bat:
        return (cg_lo + cg_hi) / 2, "No other masses defined yet."

    total_nb  = sum(i.mass_g for i in non_bat)
    moment_nb = sum(i.mass_g * i.x for i in non_bat)
    bat_items = [i for i in items if i.category == "energy"]
    bat_mass  = sum(i.mass_g for i in bat_items)

    if bat_mass == 0:
        return (cg_lo + cg_hi) / 2, "No battery mass defined — cannot auto-place."

    cg_target  = (cg_lo + cg_hi) / 2
    total_all  = total_nb + bat_mass
    needed_moment = cg_target * total_all - moment_nb
    x_bat = needed_moment / bat_mass

    # Configuration-based constraints
    if configuration == "Conventional Pusher":
        x_bat = min(x_bat, 0.65)   # battery can't go past 65% fuse length
    if configuration == "Flying Wing":
        x_bat_min, x_bat_max = 0.35, 0.55   # tight CG range
        if x_bat < x_bat_min:
            return x_bat_min, (f"Battery must be at least {x_bat_min:.2f} m from nose "
                               f"(flying wing packaging constraint).")
        if x_bat > x_bat_max:
            return x_bat_max, (f"Battery cannot be placed beyond {x_bat_max:.2f} m "
                               f"in a flying wing — CG would be too aft.")

    return round(x_bat, 3), f"Place battery ~{x_bat:.2f} m from nose to balance CG."


def packaging_score(items: List[MassItem],
                     cg_x: float, cg_lo: float, cg_hi: float,
                     fuselage_length: float) -> Tuple[float, List[str]]:
    """
    Returns a packaging score 0–100 and a list of packaging issues.
    100 = perfect, 0 = unflightworthy layout.
    """
    score  = 100.0
    issues: List[str] = []

    # CG in band?
    if cg_lo <= cg_x <= cg_hi:
        pass
    elif cg_x < cg_lo:
        margin = cg_lo - cg_x
        score -= min(40, margin * 200)
        issues.append(
            f"CG is {margin*100:.1f} cm ahead of target band — aircraft will be nose-heavy."
        )
    else:
        margin = cg_x - cg_hi
        score -= min(50, margin * 250)
        issues.append(
            f"CG is {margin*100:.1f} cm behind target band — aircraft will be tail-heavy "
            "(potentially divergent pitch instability)."
        )

    # Any item placed outside fuselage?
    for item in items:
        if item.x < 0:
            score -= 10
            issues.append(f"'{item.name}' is placed ahead of the nose (x={item.x:.2f} m).")
        if item.x > fuselage_length:
            score -= 8
            issues.append(
                f"'{item.name}' is placed beyond fuselage end "
                f"(x={item.x:.2f} m, fuse={fuselage_length:.2f} m)."
            )

    return max(score, 0), issues


def default_mass_items(configuration: str,
                        airframe_g: float,
                        motor_g: float,
                        battery_g: float,
                        avionics_g: float,
                        payload_g: float,
                        esc_g: float,
                        servo_g: float,
                        fuselage_length: float) -> List[MassItem]:
    """
    Creates a default component layout based on configuration.
    Positions are normalised fractions of fuselage length.
    """
    L = fuselage_length

    configs = {
        "Conventional Tractor": {
            "Motor":      (0.05 * L, "propulsion", motor_g,   True),
            "Battery":    (0.30 * L, "energy",     battery_g, False),
            "Avionics":   (0.20 * L, "avionics",   avionics_g,False),
            "ESC":        (0.15 * L, "propulsion", esc_g,     False),
            "Payload":    (0.25 * L, "payload",    payload_g, False),
            "Servos":     (0.55 * L, "avionics",   servo_g,   False),
            "Airframe":   (0.45 * L, "structure",  airframe_g,True),
        },
        "Conventional Pusher": {
            "Motor":      (0.92 * L, "propulsion", motor_g,   True),
            "Battery":    (0.35 * L, "energy",     battery_g, False),
            "Avionics":   (0.18 * L, "avionics",   avionics_g,False),
            "ESC":        (0.80 * L, "propulsion", esc_g,     False),
            "Payload":    (0.12 * L, "payload",    payload_g, False),
            "Servos":     (0.55 * L, "avionics",   servo_g,   False),
            "Airframe":   (0.45 * L, "structure",  airframe_g,True),
        },
        "Twin-Boom Pusher": {
            "Motor":      (0.90 * L, "propulsion", motor_g,   True),
            "Battery":    (0.30 * L, "energy",     battery_g, False),
            "Avionics":   (0.15 * L, "avionics",   avionics_g,False),
            "ESC":        (0.75 * L, "propulsion", esc_g,     False),
            "Payload":    (0.10 * L, "payload",    payload_g, False),
            "Servos":     (0.50 * L, "avionics",   servo_g,   False),
            "Airframe":   (0.42 * L, "structure",  airframe_g,True),
        },
        "Flying Wing": {
            "Motor":      (0.50 * L, "propulsion", motor_g,   True),
            "Battery":    (0.45 * L, "energy",     battery_g, False),
            "Avionics":   (0.40 * L, "avionics",   avionics_g,False),
            "ESC":        (0.50 * L, "propulsion", esc_g,     False),
            "Payload":    (0.38 * L, "payload",    payload_g, False),
            "Servos":     (0.52 * L, "avionics",   servo_g,   False),
            "Airframe":   (0.45 * L, "structure",  airframe_g,True),
        },
    }

    layout = configs.get(configuration, configs["Conventional Tractor"])
    items = []
    for name, (x, cat, mass, fixed) in layout.items():
        items.append(MassItem(
            name=name, category=cat, mass_g=mass,
            x=round(x, 3), y=0.0, fixed=fixed,
        ))
    return items
