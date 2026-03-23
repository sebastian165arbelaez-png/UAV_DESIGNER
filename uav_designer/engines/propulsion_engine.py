"""
propulsion_engine.py
"""
import math
from models.data_models import PropulsionInputs, PropulsionDerived

RHO_SSL = 1.225
G       = 9.81

def build_propulsion(p: PropulsionInputs,
                     total_mass_kg: float,
                     override_static_thrust_n: float = 0.0):
    warnings = []
    d = PropulsionDerived()

    d.battery_voltage = p.cell_count * 3.7
    d.energy_wh       = (p.battery_capacity_mah / 1000.0) * d.battery_voltage

    no_load_rpm  = p.motor_kv * d.battery_voltage
    d.loaded_rpm = no_load_rpm * 0.85   # 15% droop under load

    prop_dia_m = p.prop_diameter_in * 0.0254
    tip_speed  = math.pi * prop_dia_m * d.loaded_rpm / 60.0
    if tip_speed > 150:
        warnings.append(
            f"Prop tip speed {tip_speed:.0f} m/s is high (>150 m/s). "
            "Consider a lower KV motor or smaller prop diameter."
        )

    if override_static_thrust_n > 0:
        d.static_thrust_n = override_static_thrust_n
        d.eta_prop        = 0.75
    else:
        # Reliable static thrust model: T = Ct * rho * n^2 * D^4
        # Ct = 0.10 is typical for UAV props at static conditions
        n_rps = d.loaded_rpm / 60.0
        D_m   = prop_dia_m
        Ct    = 0.10
        d.static_thrust_n = Ct * RHO_SSL * (n_rps**2) * (D_m**4)
        # Propeller efficiency drops at higher speeds — use 0.75 as realistic average
        d.eta_prop = 0.75

    # Store efficiencies for performance engine
    d.eta_motor = p.eta_motor
    d.eta_esc   = p.eta_esc

    # Shaft power from actuator disk theory
    A = math.pi * (prop_dia_m / 2)**2
    if A > 0 and d.static_thrust_n > 0:
        P_ideal         = (d.static_thrust_n**1.5) / math.sqrt(2 * RHO_SSL * A)
        d.shaft_power_w = P_ideal / 0.55    # figure of merit 0.55 (realistic for small props)
    else:
        d.shaft_power_w = 0

    d.max_current_a = (d.shaft_power_w / (d.battery_voltage * p.eta_motor * p.eta_esc)
                       if d.battery_voltage > 0 else 0)

    if d.max_current_a > p.esc_limit_a * 0.90:
        warnings.append(
            f"Predicted max current {d.max_current_a:.1f} A is within 10% of "
            f"ESC limit {p.esc_limit_a:.0f} A. Increase ESC rating or reduce prop load."
        )
    if d.max_current_a > p.esc_limit_a:
        warnings.append(
            f"CRITICAL: Predicted current {d.max_current_a:.1f} A EXCEEDS "
            f"ESC limit {p.esc_limit_a:.0f} A. This will destroy the ESC."
        )

    W = total_mass_kg * G
    d.power_loading    = (d.static_thrust_n / (d.shaft_power_w / 1000)
                          if d.shaft_power_w > 0 else 0)
    d.thrust_to_weight = d.static_thrust_n / W if W > 0 else 0

    if d.thrust_to_weight < 0.5:
        warnings.append(
            f"Thrust-to-weight ratio {d.thrust_to_weight:.2f} is low (<0.5). "
            "Aircraft may struggle to climb or maintain level flight in wind."
        )
    if d.thrust_to_weight < 0.3:
        warnings.append(
            "CRITICAL: T/W < 0.3. This aircraft is very likely underpowered."
        )

    return d, warnings


def cruise_power(total_mass_kg, S, CD0, e, AR, V_cruise,
                 eta_prop, eta_motor, eta_esc,
                 extra_load_w=0, rho=RHO_SSL):
    W       = total_mass_kg * G
    q       = 0.5 * rho * V_cruise**2
    CL      = W / (q * S)
    CD      = CD0 + CL**2 / (math.pi * e * AR)
    D       = q * S * CD
    P_shaft = D * V_cruise
    P_elec  = P_shaft / (eta_prop * eta_motor * eta_esc) + extra_load_w
    return P_elec, P_shaft


def endurance_from_energy(energy_wh, power_w):
    if power_w <= 0:
        return 0
    return (energy_wh / power_w) * 60
