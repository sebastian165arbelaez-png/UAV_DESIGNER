"""
data_models.py
==============
All structured data models for the UAV Designer.
Uses Python dataclasses for speed and simplicity.
"""

from dataclasses import dataclass, field
from typing import Optional, List


# ─────────────────────────────────────────────────────────────────────────────
# MISSION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MissionRequirements:
    mission_type: str = "Surveillance / ISR"   # Surveillance, FPV Racing, Cargo, Mapping, etc.
    endurance_min: float = 30.0                # minutes
    cruise_speed: float = 15.0                 # m/s
    dash_speed: float = 0.0                    # m/s (0 = not required)
    payload_mass: float = 200.0                # grams
    payload_needs_fwd_view: bool = False
    launch_method: str = "Hand launch"         # Hand launch, Runway, Bungee, VTOL
    recovery_method: str = "Hand catch"        # Hand catch, Belly land, Parachute, Net
    wingspan_cap: float = 0.0                  # m (0 = no cap)
    gtow_cap: float = 0.0                      # grams (0 = no cap)
    altitude_m: float = 0.0                    # operating altitude above sea level
    wind_tolerance: str = "Moderate"           # Calm, Moderate, High


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConfigurationScore:
    name: str
    score: float
    penalties: List[str] = field(default_factory=list)
    bonuses: List[str] = field(default_factory=list)
    difficulty: str = "Medium"
    description: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeometryInputs:
    span: float = 1.4              # m
    root_chord: float = 0.20       # m
    taper: float = 0.70
    sweep_deg: float = 0.0
    dihedral_deg: float = 3.0
    # Tail
    tail_arm: float = 0.60         # m  (CG to tail MAC)
    h_tail_area: float = 0.04      # m²
    v_tail_area: float = 0.02      # m²
    # Fuselage
    fuselage_length: float = 0.80  # m


@dataclass
class GeometryDerived:
    tip_chord: float = 0.0
    MAC: float = 0.0
    S: float = 0.0
    AR: float = 0.0
    sweep_r: float = 0.0
    wing_loading: float = 0.0      # N/m²
    h_tail_volume: float = 0.0
    v_tail_volume: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PROPULSION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PropulsionInputs:
    motor_kv: float = 1000.0
    motor_mass: float = 80.0       # grams
    prop_diameter_in: float = 9.0  # inches
    prop_pitch_in: float = 5.0     # inches
    cell_count: int = 4            # LiPo cells (S count)
    battery_capacity_mah: float = 5000.0
    esc_limit_a: float = 40.0
    avionics_load_w: float = 8.0
    payload_load_w: float = 5.0
    eta_motor: float = 0.85
    eta_esc: float = 0.95


@dataclass
class PropulsionDerived:
    battery_voltage: float = 0.0
    energy_wh: float = 0.0
    static_thrust_n: float = 0.0
    max_current_a: float = 0.0
    shaft_power_w: float = 0.0
    eta_prop: float = 0.80
    loaded_rpm: float = 0.0
    power_loading: float = 0.0     # N/W
    thrust_to_weight: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MASS ITEMS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MassItem:
    name: str
    category: str          # structure, propulsion, energy, avionics, payload, misc
    mass_g: float
    x: float = 0.0         # longitudinal position from nose [m]
    y: float = 0.0         # lateral offset [m]
    length: float = 0.05   # approx footprint length [m]
    width: float = 0.04    # approx footprint width [m]
    fixed: bool = False    # if True, cannot be moved by the auto-placer
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# AIRCRAFT STATE  (the big derived result)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AircraftState:
    # Identity
    concept_name: str = "My UAV"
    configuration: str = "Conventional Tractor"

    # Totals
    total_mass_g: float = 0.0
    cg_x: float = 0.0             # m from nose
    cg_y: float = 0.0             # m lateral

    # Aero
    CL_max: float = 0.0
    CL_alpha: float = 0.0
    CD0: float = 0.0
    LD_max: float = 0.0
    oswald_e: float = 0.82

    # Performance
    V_stall: float = 0.0
    V_cruise: float = 0.0
    V_best_endurance: float = 0.0
    V_best_range: float = 0.0
    V_max: float = 0.0
    ROC_max: float = 0.0
    endurance_min: float = 0.0
    glide_ratio: float = 0.0

    # Sizing indices
    wing_loading_nm2: float = 0.0
    power_loading_nw: float = 0.0
    thrust_to_weight: float = 0.0

    # Packaging
    cg_in_target: bool = False
    target_cg_lo: float = 0.0
    target_cg_hi: float = 0.0

    # Warnings
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
