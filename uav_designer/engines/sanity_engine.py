"""
sanity_engine.py
================
Generates engineering warnings and corrective suggestions.
Behaves like a senior designer reviewing a sketch, not just a field validator.
Each warning has a severity: INFO, CAUTION, WARNING, CRITICAL.
"""

from models.data_models import (
    MissionRequirements, GeometryDerived, PropulsionDerived, AircraftState
)
from typing import List, Tuple

SEVERITY = ["INFO", "CAUTION", "WARNING", "CRITICAL"]


def _w(sev: str, msg: str, fix: str) -> dict:
    return {"severity": sev, "message": msg, "fix": fix}


def check_geometry(gd: GeometryDerived,
                    configuration: str,
                    mission: MissionRequirements) -> List[dict]:
    warnings = []

    # Aspect ratio
    if gd.AR < 5 and mission.endurance_min > 30:
        warnings.append(_w("WARNING",
            f"Aspect ratio {gd.AR:.1f} is low for an endurance mission (target ≥ 7).",
            "Increase span or reduce root chord to raise AR."))
    elif gd.AR < 4:
        warnings.append(_w("CAUTION",
            f"Aspect ratio {gd.AR:.1f} is low — induced drag will be high.",
            "Increase span or reduce mean chord."))
    if gd.AR > 18:
        warnings.append(_w("CAUTION",
            f"Aspect ratio {gd.AR:.1f} is very high — structural loads may be excessive.",
            "Consider adding a tapered or swept tip, or use carbon spar."))

    # Wing loading
    if mission.launch_method == "Hand launch" and gd.wing_loading > 65:
        warnings.append(_w("WARNING",
            f"Wing loading {gd.wing_loading:.1f} N/m² is high for hand launch (recommended < 60 N/m²).",
            "Increase wing area or reduce GTOW. Consider bungee launch."))

    # Tail volumes
    if gd.h_tail_volume < 0.30:
        warnings.append(_w("WARNING",
            f"Horizontal tail volume {gd.h_tail_volume:.2f} is too small (min ~0.35).",
            "Increase tail area or tail arm. Check longitudinal stability."))
    if gd.v_tail_volume < 0.020:
        warnings.append(_w("CAUTION",
            f"Vertical tail volume {gd.v_tail_volume:.3f} is small (min ~0.025).",
            "Increase fin area or tail arm for adequate directional stability."))

    # Configuration-specific
    if configuration == "Flying Wing" and gd.AR < 6:
        warnings.append(_w("CAUTION",
            "Flying wing with AR < 6 will have limited pitch authority.",
            "Increase span or use reflexed airfoil for better trim range."))
    if configuration == "Flying Wing" and gd.sweep_r < 0.17:
        warnings.append(_w("INFO",
            "Flying wing with sweep < 10° may have poor washout and tip-stall tendency.",
            "Consider 15–25° sweep or add twist (washout) to the tip."))

    return warnings


def check_performance(perf: dict,
                       mission: MissionRequirements,
                       configuration: str) -> List[dict]:
    warnings = []

    # Stall speed vs. launch
    if mission.launch_method == "Hand launch" and perf["V_stall"] > 10.5:
        warnings.append(_w("WARNING",
            f"Stall speed {perf['V_stall']:.1f} m/s is high for hand launch (target < 10 m/s).",
            "Reduce GTOW, increase wing area, or use a higher-lift airfoil (S1223, Clark-Y)."))

    # Endurance vs. mission
    if perf["end_max_min"] < mission.endurance_min * 0.80:
        short = mission.endurance_min - perf["end_max_min"]
        warnings.append(_w("WARNING",
            f"Predicted max endurance {perf['end_max_min']:.0f} min is {short:.0f} min "
            f"short of mission target {mission.endurance_min:.0f} min.",
            "Increase battery capacity, reduce GTOW, or improve aerodynamic efficiency."))
    elif perf["end_max_min"] < mission.endurance_min:
        warnings.append(_w("CAUTION",
            f"Predicted endurance {perf['end_max_min']:.0f} min is slightly below "
            f"mission target {mission.endurance_min:.0f} min.",
            "Consider a small increase in battery capacity or reduction in avionics load."))

    # L/D
    if perf["LD_max"] < 8 and mission.endurance_min >= 30:
        warnings.append(_w("CAUTION",
            f"L/D max {perf['LD_max']:.1f} is modest — efficiency could be improved.",
            "Review airfoil selection and CD0 sources. Minimise fuselage frontal area."))

    # Speed margin
    if perf["V_max"] < mission.cruise_speed * 1.15:
        warnings.append(_w("WARNING",
            f"Max speed {perf['V_max']:.1f} m/s has less than 15% margin over "
            f"cruise target {mission.cruise_speed:.1f} m/s.",
            "Increase static thrust, reduce drag, or revise cruise speed target."))

    return warnings


def check_propulsion(pd: PropulsionDerived,
                      total_mass_kg: float,
                      mission: MissionRequirements) -> List[dict]:
    warnings = []
    W = total_mass_kg * 9.81

    if pd.thrust_to_weight < 0.4:
        warnings.append(_w("WARNING",
            f"T/W = {pd.thrust_to_weight:.2f} is marginal (<0.4). "
            "Aircraft may not sustain level flight in headwind.",
            "Increase motor power, reduce GTOW, or use a larger prop."))

    if pd.thrust_to_weight < 0.25:
        warnings.append(_w("CRITICAL",
            f"T/W = {pd.thrust_to_weight:.2f} — this aircraft is critically underpowered.",
            "Completely revise motor and prop selection."))

    if pd.energy_wh < 10:
        warnings.append(_w("CAUTION",
            f"Battery energy {pd.energy_wh:.1f} Wh is very low.",
            "Check battery capacity and cell count inputs."))

    return warnings


def check_mass_balance(cg_x: float, cg_lo: float, cg_hi: float,
                        pkg_score: float, pkg_issues: List[str]) -> List[dict]:
    warnings = []
    if cg_x < cg_lo:
        warnings.append(_w("WARNING",
            f"CG {cg_x*100:.1f} cm is {(cg_lo-cg_x)*100:.1f} cm ahead of target band.",
            "Move battery aft, shift payload aft, or add nose ballast as last resort."))
    elif cg_x > cg_hi:
        warnings.append(_w("CRITICAL",
            f"CG {cg_x*100:.1f} cm is {(cg_x-cg_hi)*100:.1f} cm behind target band.",
            "Move battery forward, add nose weight, or extend fuselage nose."))

    if pkg_score < 60:
        warnings.append(_w("WARNING",
            f"Packaging score {pkg_score:.0f}/100 — layout has significant issues.",
            "Review component placement in the mass & balance panel."))

    for issue in pkg_issues:
        warnings.append(_w("CAUTION", issue, "Review component placement."))

    return warnings


def check_configuration_fit(configuration: str,
                              mission: MissionRequirements,
                              gd: GeometryDerived) -> List[dict]:
    warnings = []

    if configuration == "Flying Wing":
        if mission.payload_needs_fwd_view:
            warnings.append(_w("WARNING",
                "Flying wing is poorly suited to forward-facing payload missions.",
                "Consider twin-boom pusher or conventional pusher for unobstructed FOV."))
        if mission.payload_mass > 350:
            warnings.append(_w("CAUTION",
                "Flying wing CG range is tight — heavy payload complicates trimming.",
                "Verify payload is centred at 40–45% of MAC. Pre-balance before first flight."))

    if configuration == "Canard" and mission.launch_method == "Hand launch":
        warnings.append(_w("INFO",
            "Canard configurations require careful trim — hand launch needs a clean release.",
            "Verify trim speed is well above stall before attempting hand launch."))

    if configuration == "Conventional Pusher" and mission.recovery_method == "Belly land":
        warnings.append(_w("WARNING",
            "Pusher propeller is vulnerable in belly landings.",
            "Add a landing skid or nose bumper to prevent prop strike."))

    return warnings


def full_sanity_check(configuration: str,
                       mission: MissionRequirements,
                       gd: GeometryDerived,
                       pd: PropulsionDerived,
                       perf: dict,
                       total_mass_kg: float,
                       cg_x: float, cg_lo: float, cg_hi: float,
                       pkg_score: float, pkg_issues: List[str]) -> List[dict]:
    """Run all checks and return combined, severity-sorted warning list."""
    all_warnings = (
        check_geometry(gd, configuration, mission)
        + check_performance(perf, mission, configuration)
        + check_propulsion(pd, total_mass_kg, mission)
        + check_mass_balance(cg_x, cg_lo, cg_hi, pkg_score, pkg_issues)
        + check_configuration_fit(configuration, mission, gd)
    )
    # Sort by severity descending
    order = {"CRITICAL": 0, "WARNING": 1, "CAUTION": 2, "INFO": 3}
    all_warnings.sort(key=lambda w: order.get(w["severity"], 9))
    return all_warnings
