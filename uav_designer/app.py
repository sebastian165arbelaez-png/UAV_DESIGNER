"""
app.py
======
UAV Preliminary Design Tool — V1
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import math

from models.data_models import (
    MissionRequirements, GeometryInputs, PropulsionInputs, MassItem
)
from engines.mission_engine       import interpret_mission
from engines.config_engine        import score_configurations
from engines.geometry_engine      import build_geometry, back_solve_geometry
from engines.propulsion_engine    import build_propulsion, cruise_power, endurance_from_energy
from engines.performance_engine   import compute_performance, get_airfoil, AIRFOIL_DB, parse_naca, suggest_airfoils, get_airfoil_coords
from engines.mass_balance_engine  import (
    compute_cg, target_cg_range, packaging_score,
    default_mass_items, auto_place_battery
)
from engines.sanity_engine        import full_sanity_check
from engines import visualization_engine as viz

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="UAV Designer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .stSidebar { background-color: #161b22; }
    div[data-testid="stMetricValue"] { color: #58a6ff; font-size: 1.4rem; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #8b949e; font-size: 0.8rem; }
    .warning-critical { background: #3d1a1a; border-left: 4px solid #f78166; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
    .warning-warning  { background: #2d2110; border-left: 4px solid #ffa657; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
    .warning-caution  { background: #1a2a1a; border-left: 4px solid #3fb950; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
    .warning-info     { background: #161b22; border-left: 4px solid #58a6ff; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
    .apply-banner     { background: #1a2a3a; border-left: 4px solid #58a6ff; padding: 12px 16px; border-radius: 6px; margin: 8px 0; }
    h1, h2, h3 { color: #58a6ff !important; }
    .block-container { padding-top: 1rem; }
    button[data-baseweb="tab"] { color: #e6edf3 !important; background-color: transparent !important; font-size: 15px !important; font-weight: 600 !important; padding: 8px 16px !important; }
    button[data-baseweb="tab"]:hover { color: #58a6ff !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #58a6ff !important; border-bottom: 3px solid #58a6ff !important; }
    div[data-baseweb="tab-list"] { background-color: #161b22 !important; border-bottom: 1px solid #30363d !important; gap: 4px !important; }
    div[data-baseweb="tab-panel"] { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

def ss_get(key, default):
    return st.session_state.get(key, default)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ✈️ UAV Designer  V1")
    st.caption("Preliminary Design Environment")
    st.divider()

    concept_name  = st.text_input("Concept name", value="Alpha-1")
    configuration = st.selectbox("Configuration", [
        "Conventional Tractor", "Conventional Pusher",
        "Twin-Boom Pusher", "Flying Wing", "Canard", "Tandem Wing"
    ])

    st.divider()
    workflow = st.radio("Workflow", [
        "🔍  Analyze a concept",
        "🎯  Mission → Concept",
        "🔧  Back-solve requirements",
    ])

    # Show applied values summary if any
    if "mission_applied" in st.session_state:
        st.divider()
        st.markdown("**✅ Mission values applied**")
        ap = st.session_state["mission_applied"]
        st.caption(f"Span: {ap['span']:.2f} m")
        st.caption(f"Chord: {ap['chord']:.3f} m")
        st.caption(f"Battery: {ap['bat_g']:.0f} g est.")
        if st.button("🔄 Clear applied values"):
            for k in ["mission_applied", "apply_span", "apply_chord",
                      "apply_bat_g", "apply_kv", "apply_cells"]:
                st.session_state.pop(k, None)
            st.rerun()

    st.divider()
    st.caption("Aerodynamics Course Tool")
    st.caption("V1 — All Tiers Active")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
tabs = st.tabs([
    "📋 Mission",
    "📐 Geometry",
    "⚡ Propulsion",
    "⚖️ Mass & Balance",
    "🚀 Performance",
    "🔔 Warnings",
    "📊 Compare",
])

tab_mission, tab_geom, tab_prop, tab_mass, tab_perf, tab_warn, tab_compare = tabs


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — MISSION
# ─────────────────────────────────────────────────────────────────────────────

with tab_mission:
    st.header("Mission Requirements")

    col1, col2 = st.columns(2)
    with col1:
        mission_type  = st.selectbox("Mission type", [
            "Surveillance / ISR", "FPV / Racing", "Cargo delivery",
            "Aerial mapping", "BVLOS patrol", "Research platform"])
        endurance_min = st.slider("Required endurance (min)", 5, 180, 45)
        cruise_speed  = st.slider("Cruise speed (m/s)", 5.0, 40.0, 15.0, 0.5)
        dash_speed    = st.slider("Dash speed (m/s, 0 = not required)", 0.0, 60.0, 0.0, 0.5)
        payload_mass  = st.slider("Payload mass (g)", 0, 2000, 250, 25)

    with col2:
        payload_fwd_view = st.checkbox("Payload needs forward visibility", value=False)
        launch_method    = st.selectbox("Launch method", [
            "Hand launch", "Runway", "Bungee", "VTOL"])
        recovery_method  = st.selectbox("Recovery method", [
            "Hand catch", "Belly land", "Parachute", "Net"])
        wingspan_cap     = st.slider("Wingspan cap (m, 0 = no cap)", 0.0, 4.0, 0.0, 0.05)
        gtow_cap         = st.slider("GTOW cap (g, 0 = no cap)", 0, 5000, 0, 50)
        wind_tolerance   = st.selectbox("Wind tolerance", ["Calm", "Moderate", "High"])

    mission = MissionRequirements(
        mission_type=mission_type, endurance_min=endurance_min,
        cruise_speed=cruise_speed, dash_speed=dash_speed,
        payload_mass=payload_mass, payload_needs_fwd_view=payload_fwd_view,
        launch_method=launch_method, recovery_method=recovery_method,
        wingspan_cap=wingspan_cap, gtow_cap=gtow_cap, wind_tolerance=wind_tolerance,
    )
    st.session_state["mission"] = mission

    st.divider()
    result = interpret_mission(mission)
    st.session_state["mission_result"] = result

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🎯 Design Pressures")
        for label, score in result["design_pressures"][:5]:
            bar_pct   = int(score * 100)
            col_color = "#58a6ff" if score > 0.6 else "#3fb950" if score > 0.3 else "#8b949e"
            st.markdown(
                f"**{label.replace('_', ' ').title()}**  "
                f"<span style='color:{col_color}'>"
                f"{'█' * (bar_pct // 10)}{'░' * (10 - bar_pct // 10)} "
                f"{bar_pct}%</span>",
                unsafe_allow_html=True)

    with c2:
        st.subheader("💬 Design Drivers")
        for d in result["drivers"]:
            st.markdown(f"• {d}")

    if result["conflicts"]:
        st.subheader("⚠️ Requirement Conflicts Detected")
        for conflict in result["conflicts"]:
            st.markdown(
                f'<div class="warning-warning">⚠️ {conflict}</div>',
                unsafe_allow_html=True)

    # Configuration recommendation
    st.divider()
    st.subheader("🏆 Recommended Configurations")
    scores = score_configurations(mission, result["raw_pressures"])
    st.session_state["config_scores"] = scores
    st.pyplot(viz.plot_config_scores(scores), use_container_width=True)

    with st.expander("Show detailed scoring"):
        for s in scores:
            color = "#3fb950" if s == scores[0] else "#58a6ff" if scores.index(s) < 3 else "#8b949e"
            st.markdown(f"**<span style='color:{color}'>{s.name}</span>**  "
                        f"— Score: {s.score:.0f}  |  Difficulty: {s.difficulty}",
                        unsafe_allow_html=True)
            st.caption(s.description)
            if s.bonuses:
                for b in s.bonuses:
                    st.markdown(f"  ✅ {b}")
            if s.penalties:
                for p in s.penalties:
                    st.markdown(f"  ❌ {p}")
            st.divider()

    # ── MISSION → GEOMETRY CONNECTION ────────────────────────────────────────
    st.divider()
    st.subheader("🔗 Apply Mission to Design")
    st.caption("Let the mission requirements auto-populate geometry, propulsion, and mass inputs.")

    # Estimate GTOW
    est_gtow = gtow_cap if gtow_cap > 0 else 2000
    bs = back_solve_geometry(est_gtow, endurance_min, cruise_speed, payload_mass)

    span_mid  = (bs["span_range_m"][0] + bs["span_range_m"][1]) / 2
    ar_mid    = (bs["AR_range"][0] + bs["AR_range"][1]) / 2
    chord_rec = round(span_mid / ar_mid, 3)
    chord_rec = max(0.05, min(chord_rec, 0.8))
    bat_g_mid = (bs["battery_mass_g"][0] + bs["battery_mass_g"][1]) / 2

    # Suggest KV and cell count based on cruise speed and wingspan
    if cruise_speed > 20:
        rec_kv, rec_cells = 1400, 4
    elif cruise_speed > 13:
        rec_kv, rec_cells = 1000, 4
    else:
        rec_kv, rec_cells = 800, 3

    # Show what will be applied
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Wingspan", f"{span_mid:.2f} m")
    m2.metric("Root chord", f"{chord_rec:.3f} m")
    m3.metric("Battery est.", f"{bat_g_mid:.0f} g")
    m4.metric("Motor KV", f"{rec_kv}")
    m5.metric("Cell count", f"{rec_cells}S")

    if st.button("🚀  Apply mission values to all tabs", type="primary"):
        st.session_state["apply_span"]   = round(span_mid, 2)
        st.session_state["apply_chord"]  = chord_rec
        st.session_state["apply_bat_g"]  = int(bat_g_mid)
        st.session_state["apply_kv"]     = rec_kv
        st.session_state["apply_cells"]  = rec_cells
        st.session_state["mission_applied"] = {
            "span":   round(span_mid, 2),
            "chord":  chord_rec,
            "bat_g":  bat_g_mid,
            "kv":     rec_kv,
            "cells":  rec_cells,
        }
        st.success(
            f"✅ Applied to design:  "
            f"Span {span_mid:.2f} m  |  "
            f"Chord {chord_rec:.3f} m  |  "
            f"Battery ~{bat_g_mid:.0f} g  |  "
            f"Motor {rec_kv} KV  |  "
            f"{rec_cells}S battery  —  "
            f"Now go to the Geometry, Propulsion, and Mass tabs to review."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

with tab_geom:
    st.header("Wing & Airframe Geometry")

    # Show banner if mission values were applied
    if "mission_applied" in st.session_state:
        ap = st.session_state["mission_applied"]
        st.markdown(
            f'<div class="apply-banner">🔗 <strong>Mission values applied</strong> — '
            f'Wingspan set to {ap["span"]:.2f} m, Root chord to {ap["chord"]:.3f} m. '
            f'Adjust below as needed.</div>',
            unsafe_allow_html=True)

    if "🔧" in workflow:
        st.subheader("📐 Requirement Back-Solver")
        st.caption("Enter design constraints and get recommended geometry ranges.")
        bc1, bc2 = st.columns(2)
        with bc1:
            bs_mtow   = st.number_input("GTOW estimate (g)", 500, 10000, 2000, 100, key="bs_mtow")
            bs_end    = st.slider("Target endurance (min)", 10, 180, 45, key="bs_end")
        with bc2:
            bs_vcruis = st.slider("Cruise speed (m/s)", 5.0, 40.0, 15.0, 0.5, key="bs_vcruis")
            bs_payload= st.slider("Payload (g)", 0, 2000, 250, key="bs_payload")

        if st.button("🔧 Back-Solve"):
            bsr = back_solve_geometry(bs_mtow, bs_end, bs_vcruis, bs_payload)
            st.session_state["backsolve"] = bsr

        if "backsolve" in st.session_state:
            bsr = st.session_state["backsolve"]
            st.markdown("### Recommended Geometry")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Wing Area",    f"{bsr['wing_area_m2']:.3f} m2")
            m2.metric("Aspect Ratio", f"{bsr['AR_range'][0]}-{bsr['AR_range'][1]}")
            m3.metric("Wingspan",     f"{bsr['span_range_m'][0]}-{bsr['span_range_m'][1]} m")
            m4.metric("Battery Mass", f"{bsr['battery_mass_g'][0]:.0f}-{bsr['battery_mass_g'][1]:.0f} g")
            st.info(
                f"Power at cruise: **{bsr['est_power_w']:.0f} W**  |  "
                f"Energy needed: **{bsr['est_energy_wh']:.1f} Wh**  |  "
                f"Wing loading: **{bsr['wing_loading_nm2']:.1f} N/m2**  |  "
                f"Target V_stall: **{bsr['V_stall_target']:.1f} m/s**"
            )
            if st.button("Apply back-solver values to geometry sliders"):
                sp_mid = (bsr["span_range_m"][0] + bsr["span_range_m"][1]) / 2
                ar_m   = (bsr["AR_range"][0] + bsr["AR_range"][1]) / 2
                ch     = round(sp_mid / ar_m, 3)
                st.session_state["apply_span"]  = round(sp_mid, 2)
                st.session_state["apply_chord"] = max(0.05, min(ch, 0.8))
                st.success(f"Applied: Wingspan {sp_mid:.2f} m, Root chord {ch:.3f} m.")
        st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Wing")
        span       = st.slider("Wingspan (m)", 0.3, 4.0,
                               float(ss_get("apply_span", 1.4)), 0.01)
        root_chord = st.slider("Root chord (m)", 0.05, 0.8,
                               float(ss_get("apply_chord", 0.20)), 0.005)
        taper      = st.slider("Taper ratio", 0.2, 1.0, 0.70, 0.01)
        sweep_deg  = st.slider("LE Sweep (deg)", 0.0, 45.0, 0.0, 0.5)
        dihedral   = st.slider("Dihedral (deg)", -5.0, 15.0, 3.0, 0.5)
        oswald_e   = st.slider("Oswald factor e", 0.50, 1.0, 0.82, 0.01)

    with col2:
        st.subheader("Tail & Fuselage")
        fuselage_length = st.slider("Fuselage length (m)", 0.30, 2.0, 0.80, 0.01)
        tail_arm        = st.slider("Tail arm (m)", 0.10, 1.5, 0.60, 0.01)
        h_tail_area     = st.slider("H-tail area (m2)", 0.005, 0.20, 0.04, 0.005)
        v_tail_area     = st.slider("V-tail area (m2)", 0.003, 0.10, 0.02, 0.002)
        st.subheader("Airfoil")
        airfoil_mode = st.radio("Airfoil input mode",
            ["Preset library", "Custom NACA code"],
            horizontal=True, key="airfoil_mode")

        if airfoil_mode == "Preset library":
            airfoil_name = st.selectbox("Select airfoil", list(AIRFOIL_DB.keys()))
        else:
            custom_code = st.text_input(
                "Enter NACA code (4 or 5 digit)",
                value="2412",
                placeholder="e.g. 2412, 4415, 23012",
                key="custom_naca"
            )
            airfoil_name = f"NACA {custom_code.strip().upper().replace('NACA','').strip()}"
            parsed = parse_naca(airfoil_name)
            if parsed:
                st.success(f"✅ Recognized: {airfoil_name}")
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("Cl alpha (/rad)", f"{parsed['Cl_a']:.3f}")
                pc2.metric("Alpha L0 (deg)",  f"{parsed['al0']:.2f}")
                pc3.metric("CL max est.",     f"{parsed['CL_max']:.3f}")
                if parsed.get("type") == "4-digit":
                    st.caption(
                        f"Max camber: {parsed['M_pct']:.1f}%  |  "
                        f"Max thickness: {parsed['T_pct']:.1f}%  |  "
                        f"Parameters computed via thin airfoil theory."
                    )
                else:
                    st.caption(
                        f"Design CL: {parsed.get('CL_design', 'N/A')}  |  "
                        f"Max thickness: {parsed['T_pct']:.1f}%  |  "
                        f"5-digit series (reflexed camber line)."
                    )
            else:
                st.error(
                    f"Could not parse '{custom_code}' as a NACA 4 or 5-digit code. "
                    "Examples: 0012, 2412, 4415, 23012, 63412"
                )
                airfoil_name = "NACA 4412"  # fallback

    geom_in = GeometryInputs(
        span=span, root_chord=root_chord, taper=taper,
        sweep_deg=sweep_deg, dihedral_deg=dihedral,
        tail_arm=tail_arm, h_tail_area=h_tail_area,
        v_tail_area=v_tail_area, fuselage_length=fuselage_length,
    )

    est_mass_kg = ss_get("total_mass_kg", 1.9)
    gd, geom_errors = build_geometry(geom_in, est_mass_kg)
    gd.oswald_e = oswald_e
    st.session_state["geom_in"]         = geom_in
    st.session_state["gd"]              = gd
    st.session_state["airfoil"]         = airfoil_name
    st.session_state["fuselage_length"] = fuselage_length
    st.session_state["span"]            = span
    st.session_state["root_chord"]      = root_chord
    st.session_state["taper"]           = taper
    st.session_state["sweep_deg"]       = sweep_deg

    if geom_errors:
        for e in geom_errors:
            st.warning(e)

    st.divider()
    st.subheader("Derived Geometry")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Wing Area S",  f"{gd.S:.4f} m2")
    m2.metric("Aspect Ratio", f"{gd.AR:.2f}")
    m3.metric("MAC",          f"{gd.MAC:.3f} m")
    m4.metric("Tip Chord",    f"{gd.tip_chord:.3f} m")
    m5.metric("H-tail Vol",   f"{gd.h_tail_volume:.3f}")
    m6.metric("Wing Loading", f"{gd.wing_loading:.1f} N/m2")

    st.divider()
    st.pyplot(viz.plot_wing_planform(gd, sweep_deg, span, root_chord, taper),
              use_container_width=True)

    # ── AIRFOIL VISUALIZER ───────────────────────────────────────────────────
    st.divider()
    st.subheader("✈️ Airfoil Visualizer")

    coords = get_airfoil_coords(airfoil_name)
    if coords:
        st.pyplot(viz.plot_airfoil_shape(airfoil_name, coords),
                  use_container_width=True)
    else:
        st.info(f"Shape preview not available for {airfoil_name}. "
                "Enter a NACA 4 or 5-digit code in Custom mode to see the profile.")

    # ── AIRFOIL SUGGESTER ────────────────────────────────────────────────────
    st.divider()
    st.subheader("🎯 Airfoil Suggestions for Your Mission")

    mission_now = st.session_state.get("mission")
    if mission_now:
        suggestions = suggest_airfoils(
            mission_now.mission_type,
            mission_now.endurance_min,
            mission_now.cruise_speed,
            mission_now.launch_method,
            mission_now.payload_mass,
        )

        st.pyplot(viz.plot_airfoil_suggestions(suggestions),
                  use_container_width=True)

        st.markdown("### Top Recommendations")
        for i, (name, score, reason) in enumerate(suggestions):
            icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            af   = get_airfoil(name)
            desc = AIRFOIL_DB.get(name, {}).get("desc", "")
            with st.expander(f"{icon} {name}  —  Score: {score:.0f}"):
                st.markdown(f"**{desc}**")
                st.caption(f"Why: {reason}")
                c1, c2, c3 = st.columns(3)
                c1.metric("CL max",        f"{af['CL_max']:.2f}")
                c2.metric("Cl alpha /rad", f"{af['Cl_a']:.3f}")
                c3.metric("Alpha L0",      f"{af['al0']:.1f} deg")
                re_range = AIRFOIL_DB.get(name, {}).get("Re_range", "N/A")
                ld_typ   = AIRFOIL_DB.get(name, {}).get("LD_typical", "N/A")
                st.caption(f"Best Reynolds number range: {re_range}  |  Typical L/D: {ld_typ}")

                # Show shape for this suggestion
                sug_coords = get_airfoil_coords(name)
                if sug_coords:
                    st.pyplot(viz.plot_airfoil_shape(name, sug_coords),
                              use_container_width=True)

                if st.button(f"Use {name} as my airfoil", key=f"use_{name}"):
                    st.session_state["apply_airfoil"] = name
                    st.success(f"Set airfoil to {name}. Go to Airfoil selector above to confirm.")
    else:
        st.info("Define your mission in the Mission tab first to get airfoil suggestions.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — PROPULSION
# ─────────────────────────────────────────────────────────────────────────────

with tab_prop:
    st.header("Propulsion & Power")

    if "mission_applied" in st.session_state:
        ap = st.session_state["mission_applied"]
        st.markdown(
            f'<div class="apply-banner">🔗 <strong>Mission values applied</strong> — '
            f'Motor KV set to {ap["kv"]}, Cell count to {ap["cells"]}S. '
            f'Adjust below as needed.</div>',
            unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Motor & Prop")
        motor_kv     = st.slider("Motor KV", 100, 3000,
                                  int(ss_get("apply_kv", 1000)), 10)
        motor_mass_g = st.slider("Motor mass (g)", 20, 500, 80, 5)
        prop_dia     = st.slider("Prop diameter (in)", 4.0, 20.0, 9.0, 0.5)
        prop_pitch   = st.slider("Prop pitch (in)", 2.0, 12.0, 5.0, 0.5)
        override_thr = st.number_input(
            "Override static thrust (N, 0 = auto-estimate)", 0.0, 200.0, 0.0, 0.5)

    with col2:
        st.subheader("Battery & ESC")
        cell_count  = st.select_slider("Cell count (S)", options=[2,3,4,5,6],
                                        value=int(ss_get("apply_cells", 4)))
        bat_cap_mah = st.slider("Battery capacity (mAh)", 500, 25000, 5000, 100)
        esc_limit   = st.slider("ESC current limit (A)", 10, 120, 40, 5)
        eta_motor   = st.slider("eta motor", 0.70, 0.95, 0.85, 0.01)
        eta_esc_val = st.slider("eta ESC",   0.85, 0.99, 0.95, 0.01)
        st.subheader("Electrical Loads")
        avionics_w  = st.slider("Avionics load (W)",  2.0, 30.0, 8.0, 0.5)
        payload_w   = st.slider("Payload power (W)",  0.0, 50.0, 5.0, 0.5)

    prop_in = PropulsionInputs(
        motor_kv=motor_kv, motor_mass=motor_mass_g,
        prop_diameter_in=prop_dia, prop_pitch_in=prop_pitch,
        cell_count=cell_count, battery_capacity_mah=bat_cap_mah,
        esc_limit_a=esc_limit,
        avionics_load_w=avionics_w, payload_load_w=payload_w,
        eta_motor=eta_motor, eta_esc=eta_esc_val,
    )
    total_mass_kg = ss_get("total_mass_kg", 1.9)
    pd_result, prop_warns = build_propulsion(prop_in, total_mass_kg, override_thr)
    st.session_state["prop_in"]      = prop_in
    st.session_state["pd"]           = pd_result
    st.session_state["extra_load_w"] = avionics_w + payload_w

    if prop_warns:
        for w in prop_warns:
            st.warning(w)

    st.divider()
    st.subheader("Propulsion Summary")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Battery Voltage", f"{pd_result.battery_voltage:.1f} V")
    m2.metric("Energy",          f"{pd_result.energy_wh:.1f} Wh")
    m3.metric("Static Thrust",   f"{pd_result.static_thrust_n:.1f} N")
    m4.metric("Max Current",     f"{pd_result.max_current_a:.1f} A")
    m5.metric("T/W Ratio",       f"{pd_result.thrust_to_weight:.2f}")
    m6.metric("Loaded RPM",      f"{pd_result.loaded_rpm:.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — MASS & BALANCE
# ─────────────────────────────────────────────────────────────────────────────

with tab_mass:
    st.header("Mass & Balance")

    if "mission_applied" in st.session_state:
        ap = st.session_state["mission_applied"]
        st.markdown(
            f'<div class="apply-banner">🔗 <strong>Mission values applied</strong> — '
            f'Battery pack set to estimated {ap["bat_g"]:.0f} g. '
            f'Adjust below as needed.</div>',
            unsafe_allow_html=True)

    st.subheader("Component Masses")
    col1, col2 = st.columns(2)
    with col1:
        airframe_g     = st.slider("Airframe / structure (g)", 100, 3000, 800,  10)
        battery_g      = st.slider("Battery pack (g)", 50, 5000,
                                    int(ss_get("apply_bat_g", 520)), 10)
        motor_g_mass   = st.slider("Motor(s) (g)", 20,  500,  80,   5)
        esc_g          = st.slider("ESC (g)",       10,  200,  45,   5)
    with col2:
        avionics_g     = st.slider("Avionics (FC, RX, GPS) (g)", 20, 500, 120,   5)
        payload_g_mass = st.slider("Payload (g)", 0, 2000, 250, 25)
        servo_g        = st.slider("Servos total (g)", 10, 300,  60,   5)
        misc_g         = st.slider("Misc / wiring (g)", 10, 300,  80,   5)

    total_g  = (airframe_g + battery_g + motor_g_mass + esc_g +
                avionics_g + payload_g_mass + servo_g + misc_g)
    total_kg = total_g / 1000
    st.session_state["total_mass_kg"] = total_kg
    st.metric("Total MTOW", f"{total_g:.0f} g  ({total_kg:.3f} kg)")

    st.divider()
    st.subheader("Component Placement")
    st.caption("Adjust x positions (from nose, in metres).")

    gd_now = st.session_state.get("gd")
    fuse_L = st.session_state.get("fuselage_length", 0.80)
    config = configuration

    if gd_now:
        default_items = default_mass_items(
            config, airframe_g, motor_g_mass, battery_g,
            avionics_g, payload_g_mass, esc_g, servo_g, fuse_L)
        default_items.append(MassItem("Misc", "misc", misc_g,
                                       x=fuse_L * 0.5, fixed=False))

        items = []
        for item in default_items:
            col_a, _ = st.columns([3, 1])
            with col_a:
                x_val = st.slider(
                    f"{item.name}  ({item.mass_g:.0f} g)  {'🔒' if item.fixed else ''}",
                    min_value=0.0, max_value=fuse_L,
                    value=float(min(item.x, fuse_L)),
                    step=0.01,
                    key=f"x_{item.name}",
                    disabled=item.fixed)
            item.x = x_val
            items.append(item)

        st.session_state["mass_items"] = items

        total_m, cg_x, cg_y = compute_cg(items)
        mac_le_x = fuse_L * 0.30
        cg_lo, cg_hi = target_cg_range(gd_now, fuse_L, mac_le_x)
        pkg_score_val, pkg_issues = packaging_score(items, cg_x, cg_lo, cg_hi, fuse_L)
        st.session_state.update({
            "cg_x": cg_x, "cg_lo": cg_lo, "cg_hi": cg_hi,
            "pkg_score": pkg_score_val, "pkg_issues": pkg_issues,
        })

        if st.button("🎯 Auto-place battery for best CG"):
            bx, msg = auto_place_battery(items, cg_lo, cg_hi, config)
            st.info(f"Suggested battery position: **{bx:.3f} m**  — {msg}")

        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CG from nose",    f"{cg_x*100:.1f} cm")
        col2.metric("Target band",     f"{cg_lo*100:.1f}-{cg_hi*100:.1f} cm")
        col3.metric("Packaging score", f"{pkg_score_val:.0f} / 100")
        col4.metric("MTOW",            f"{total_g:.0f} g")

        st.divider()
        st.subheader("Top-View Mass Layout")
        span_now   = st.session_state.get("span", 1.4)
        chord_now  = st.session_state.get("root_chord", 0.20)
        taper_now  = st.session_state.get("taper", 0.70)
        sweep_now  = st.session_state.get("sweep_deg", 0.0)

        top_fig = viz.plot_mass_layout(
            items, gd_now, config, fuse_L,
            cg_x, cg_lo, cg_hi,
            span_now, chord_now, taper_now, sweep_now)
        st.pyplot(top_fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(viz.plot_moment_diagram(items, cg_x, cg_lo, cg_hi),
                      use_container_width=True)
        with c2:
            st.pyplot(viz.plot_mass_pie(items), use_container_width=True)
    else:
        st.info("Define geometry first in the Geometry tab.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

with tab_perf:
    st.header("Performance Estimates")
    st.caption("All values are first-order estimates. Clearly approximate — not simulation output.")

    gd_now = st.session_state.get("gd")
    pd_now = st.session_state.get("pd")

    if gd_now and pd_now:
        airfoil_now = st.session_state.get("airfoil",      "NACA 4412")
        extra_load  = st.session_state.get("extra_load_w",  13.0)
        total_mass  = st.session_state.get("total_mass_kg",  1.9)
        rho         = st.slider("Air density (kg/m3)", 0.90, 1.225, 1.225, 0.005, key="perf_rho")
        n_max       = st.slider("Max load factor n+",  1.5, 8.0,  3.0, 0.1, key="perf_nmax")
        n_min       = st.slider("Min load factor n-", -4.0, -0.5, -1.5, 0.1, key="perf_nmin")

        perf = compute_performance(
            total_mass, gd_now, pd_now,
            airfoil_now, configuration, extra_load, rho)
        st.session_state["perf"] = perf

        st.subheader("Key Performance Metrics")
        cols = st.columns(4)
        cols[0].metric("Stall Speed",    f"{perf['V_stall']:.1f} m/s",
                        f"{perf['V_stall']*3.6:.0f} km/h")
        cols[1].metric("Max Speed",      f"{perf['V_max']:.1f} m/s",
                        f"{perf['V_max']*3.6:.0f} km/h")
        cols[2].metric("Best Endurance", f"{perf['end_max_min']:.0f} min",
                        f"@ {perf['V_end_max']:.1f} m/s")
        cols[3].metric("Max ROC",        f"{perf['ROC_max']:.2f} m/s",
                        f"{perf['ROC_max']*196.85:.0f} ft/min")

        cols2 = st.columns(4)
        cols2[0].metric("L/D Max",      f"{perf['LD_max']:.1f}")
        cols2[1].metric("Cruise Power", f"{perf['P_cruise_w']:.0f} W")
        cols2[2].metric("V best range", f"{perf['V_br']:.1f} m/s")
        cols2[3].metric("V best end.",  f"{perf['V_be']:.1f} m/s")

        st.divider()
        p_tab1, p_tab2, p_tab3, p_tab4 = st.tabs(
            ["Drag Polar & L/D", "Thrust Curves", "Rate of Climb", "Endurance"])

        with p_tab1:
            st.pyplot(viz.plot_drag_polar(perf), use_container_width=True)
        with p_tab2:
            st.pyplot(viz.plot_thrust_curves(perf), use_container_width=True)
        with p_tab3:
            st.pyplot(viz.plot_roc(perf), use_container_width=True)
        with p_tab4:
            st.pyplot(viz.plot_endurance(perf), use_container_width=True)
    else:
        st.info("Complete the Geometry and Propulsion tabs first.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — WARNINGS
# ─────────────────────────────────────────────────────────────────────────────

with tab_warn:
    st.header("Engineering Warnings & Suggestions")
    st.caption("Based on a senior-designer review of your current concept.")

    gd_now   = st.session_state.get("gd")
    pd_now   = st.session_state.get("pd")
    perf_now = st.session_state.get("perf")
    mission  = st.session_state.get("mission")

    if gd_now and pd_now and perf_now and mission:
        cg_x   = ss_get("cg_x",      0.35)
        cg_lo  = ss_get("cg_lo",     0.28)
        cg_hi  = ss_get("cg_hi",     0.40)
        pkg_s  = ss_get("pkg_score", 100)
        pkg_i  = ss_get("pkg_issues", [])
        total_m= ss_get("total_mass_kg", 1.9)

        all_warnings = full_sanity_check(
            configuration, mission, gd_now, pd_now, perf_now,
            total_m, cg_x, cg_lo, cg_hi, pkg_s, pkg_i)

        if not all_warnings:
            st.success("✅  No significant issues found. Design looks feasible.")
        else:
            sev_colors = {
                "CRITICAL": ("warning-critical", "🔴"),
                "WARNING":  ("warning-warning",  "🟠"),
                "CAUTION":  ("warning-caution",  "🟡"),
                "INFO":     ("warning-info",      "🔵"),
            }
            for w in all_warnings:
                cls, icon = sev_colors.get(w["severity"], ("warning-info", "ℹ️"))
                st.markdown(
                    f'<div class="{cls}">'
                    f'<strong>{icon} {w["severity"]}</strong> — {w["message"]}<br>'
                    f'<em>→ {w["fix"]}</em>'
                    f'</div>',
                    unsafe_allow_html=True)
    else:
        st.info("Fill in all tabs to run the full check.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — COMPARE
# ─────────────────────────────────────────────────────────────────────────────

with tab_compare:
    st.header("Concept Comparison")
    st.caption("Save snapshots of the current concept and compare side-by-side.")

    if "saved_concepts" not in st.session_state:
        st.session_state["saved_concepts"] = {}

    perf_now = st.session_state.get("perf")
    gd_now   = st.session_state.get("gd")

    if perf_now and gd_now:
        col_save, _ = st.columns([1, 3])
        with col_save:
            save_name = st.text_input("Snapshot name", value=concept_name)
            if st.button("💾 Save current concept"):
                st.session_state["saved_concepts"][save_name] = {
                    "configuration": configuration,
                    "MTOW_g":    ss_get("total_mass_kg", 0) * 1000,
                    "AR":        gd_now.AR,
                    "S":         gd_now.S,
                    "V_stall":   perf_now["V_stall"],
                    "V_max":     perf_now["V_max"],
                    "end_max":   perf_now["end_max_min"],
                    "ROC_max":   perf_now["ROC_max"],
                    "LD_max":    perf_now["LD_max"],
                    "pkg_score": ss_get("pkg_score", 0),
                }
                st.success(f"Saved: {save_name}")

    concepts = st.session_state["saved_concepts"]
    if len(concepts) >= 2:
        st.divider()
        st.subheader("Saved Concepts")
        import pandas as pd
        df = pd.DataFrame(concepts).T
        df.index.name = "Concept"
        df.columns    = ["Config", "MTOW (g)", "AR", "S (m2)",
                          "V_stall (m/s)", "V_max (m/s)",
                          "Endurance (min)", "ROC_max (m/s)", "L/D max", "Pkg Score"]
        df = df.round(2)
        st.dataframe(df, use_container_width=True)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = ["Endurance (min)", "L/D max", "V_max (m/s)", "ROC_max (m/s)", "Pkg Score"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        fig.patch.set_facecolor("#0d1117")
        colors = ["#58a6ff","#3fb950","#ffa657","#d2a8ff","#f78166","#79c0ff","#56d364"]
        for ax, metric in zip(axes, metrics):
            ax.set_facecolor("#161b22")
            for sp in ax.spines.values():
                sp.set_edgecolor("#30363d")
            vals = df[metric].values.astype(float)
            nms  = df.index.tolist()
            ax.bar(nms, vals, color=colors[:len(nms)], alpha=0.85)
            ax.set_title(metric, color="#e6edf3", fontsize=9, fontweight="bold")
            ax.tick_params(axis='x', colors="#8b949e", labelsize=7, rotation=20)
            ax.tick_params(axis='y', colors="#8b949e", labelsize=7)
            ax.grid(True, color="#21262d", linestyle="--", linewidth=0.5)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)

    elif len(concepts) == 1:
        st.info("Save at least one more concept to compare.")
    else:
        st.info("No concepts saved yet.")
