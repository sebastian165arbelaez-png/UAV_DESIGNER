"""
mission_engine.py
=================
Translates mission requirements into design drivers.
Detects contradictory requirements and ranks design pressures.
"""

from models.data_models import MissionRequirements
from typing import List, Tuple, Dict


# ── Design pressure labels ────────────────────────────────────────────────────
PRESSURES = [
    "endurance",
    "speed",
    "payload_mass",
    "packaging",
    "low_stall",
    "portability",
    "wind_robustness",
    "simplicity",
]


def interpret_mission(m: MissionRequirements) -> Dict:
    """
    Returns a dict with:
      - design_pressures: ranked list of (label, score 0-1)
      - conflicts:        list of conflict description strings
      - drivers:          short human-readable summary strings
    """
    pressures: Dict[str, float] = {p: 0.0 for p in PRESSURES}
    conflicts: List[str] = []
    drivers: List[str] = []

    # ── Endurance ─────────────────────────────────────────────────────────
    if m.endurance_min >= 60:
        pressures["endurance"] = 1.0
        drivers.append(f"Long endurance ({m.endurance_min:.0f} min) → high wing efficiency priority.")
    elif m.endurance_min >= 30:
        pressures["endurance"] = 0.6
        drivers.append(f"Moderate endurance ({m.endurance_min:.0f} min) → balanced design.")
    else:
        pressures["endurance"] = 0.2

    # ── Speed ─────────────────────────────────────────────────────────────
    if m.cruise_speed >= 25:
        pressures["speed"] = 0.9
        drivers.append(f"High cruise speed ({m.cruise_speed:.1f} m/s) → wing sweep & low drag priority.")
    elif m.cruise_speed >= 15:
        pressures["speed"] = 0.5
    else:
        pressures["speed"] = 0.1
        drivers.append(f"Low cruise speed ({m.cruise_speed:.1f} m/s) → high-lift airfoil preferred.")

    # ── Payload ───────────────────────────────────────────────────────────
    if m.payload_mass >= 500:
        pressures["payload_mass"] = 1.0
        drivers.append(f"Heavy payload ({m.payload_mass:.0f} g) → structure and CG management critical.")
    elif m.payload_mass >= 200:
        pressures["payload_mass"] = 0.6
    else:
        pressures["payload_mass"] = 0.2

    if m.payload_needs_fwd_view:
        pressures["packaging"] += 0.4
        drivers.append("Payload requires forward visibility → tractor or pusher configurations preferred.")

    # ── Launch / recovery ─────────────────────────────────────────────────
    if m.launch_method == "Hand launch":
        pressures["low_stall"] += 0.5
        pressures["portability"] += 0.4
        drivers.append("Hand launch → stall speed < 10 m/s strongly preferred.")
    if m.launch_method == "Runway":
        pressures["low_stall"] += 0.2

    if m.recovery_method == "Belly land":
        pressures["packaging"] += 0.2
        drivers.append("Belly landing → protect sensitive belly-mounted components.")
    if m.recovery_method == "Parachute":
        pressures["packaging"] += 0.3

    # ── Wingspan cap ──────────────────────────────────────────────────────
    if 0 < m.wingspan_cap < 1.2:
        pressures["portability"] = max(pressures["portability"], 0.8)
        drivers.append(f"Tight wingspan cap ({m.wingspan_cap:.1f} m) → low AR or folding wing needed.")

    # ── Wind tolerance ────────────────────────────────────────────────────
    if m.wind_tolerance == "High":
        pressures["wind_robustness"] = 0.8
        drivers.append("High wind tolerance required → higher wing loading acceptable.")
    elif m.wind_tolerance == "Calm":
        pressures["wind_robustness"] = 0.1

    # ── Conflict detection ────────────────────────────────────────────────
    if pressures["endurance"] >= 0.8 and pressures["speed"] >= 0.8:
        conflicts.append(
            "High endurance AND high speed are conflicting pressures. "
            "A high-speed airframe is aerodynamically inefficient for endurance. "
            "Consider separate cruise and dash speed targets."
        )

    if pressures["endurance"] >= 0.8 and 0 < m.wingspan_cap < 1.5:
        conflicts.append(
            f"Long endurance missions favour high AR wings, "
            f"but wingspan is capped at {m.wingspan_cap:.1f} m. "
            "Consider a multi-boom or blended wing layout to maximise effective AR."
        )

    if m.launch_method == "Hand launch" and m.payload_mass > 400:
        conflicts.append(
            f"Hand launch with {m.payload_mass:.0f} g payload leads to high GTOW. "
            "Stall speed may exceed safe hand-launch threshold (~10 m/s). "
            "Consider bungee or runway launch."
        )

    if m.payload_needs_fwd_view and pressures["endurance"] >= 0.7:
        conflicts.append(
            "Forward-facing payload plus endurance priority makes flying wings difficult. "
            "A conventional pusher or twin-boom pusher avoids this conflict well."
        )

    if 0 < m.gtow_cap < 500 and m.payload_mass > m.gtow_cap * 0.35:
        conflicts.append(
            f"Payload ({m.payload_mass:.0f} g) is more than 35% of GTOW cap ({m.gtow_cap:.0f} g). "
            "Little mass budget remains for structure, battery, and avionics."
        )

    # ── Sort pressures ────────────────────────────────────────────────────
    ranked = sorted(pressures.items(), key=lambda x: x[1], reverse=True)

    return {
        "design_pressures": ranked,
        "conflicts": conflicts,
        "drivers": drivers,
        "raw_pressures": pressures,
    }
