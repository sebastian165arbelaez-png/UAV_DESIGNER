"""
config_engine.py
================
Transparent rule-based scoring for UAV configuration selection.
Returns ranked configurations with scores, bonuses, penalties, and explanations.
"""

from models.data_models import MissionRequirements, ConfigurationScore
from typing import List, Dict


CONFIGURATIONS = [
    "Conventional Tractor",
    "Conventional Pusher",
    "Twin-Boom Pusher",
    "Flying Wing",
    "Canard",
    "Tandem Wing",
]

def strip_tail(name: str) -> str:
    return (name.replace(" (H-tail)","").replace(" (V-tail)","")
                .replace(" (Inverted V-tail)",""))

CONFIG_DESCRIPTIONS = {
    "Conventional Tractor":
        "Classic layout: motor at nose, tail at rear. Best-understood aerodynamics, "
        "easy CG management, good prop clearance.",
    "Conventional Pusher":
        "Motor at rear. Unobstructed forward view, clean nose for payload, "
        "but prop clearance at landing is a concern.",
    "Twin-Boom Pusher":
        "Twin tail booms with rear motor. Excellent forward payload access, "
        "good stability, more complex to build.",
    "Flying Wing":
        "No fuselage or tail — wing carries everything. Aerodynamically efficient, "
        "compact, but limited payload volume and tight CG range.",
    "Canard":
        "Small forward wing replaces conventional horizontal tail. "
        "Stall-resistant, good visibility, tricky to trim.",
    "Tandem Wing":
        "Two wings of similar size fore and aft. Large CG range, high payload "
        "volume, unusual but surprisingly practical for heavy-lift UAVs.",
}

DIFFICULTY = {
    "Conventional Tractor": "Easy",
    "Conventional Pusher":  "Easy",
    "Twin-Boom Pusher":     "Medium",
    "Flying Wing":          "Medium",
    "Canard":               "Hard",
    "Tandem Wing":          "Hard",
}


def score_configurations(m: MissionRequirements,
                          pressures: Dict[str, float],
                          selected: str = "") -> List[ConfigurationScore]:
    """
    Score every candidate configuration against the mission.
    Returns a list sorted best → worst.
    """
    results = []

    for name in CONFIGURATIONS:
        score = 50.0   # neutral baseline
        bonuses: List[str] = []
        penalties: List[str] = []

        # ── Endurance pressure ───────────────────────────────────────────
        ep = pressures.get("endurance", 0)
        if name == "Flying Wing" and ep >= 0.6:
            score += 12
            bonuses.append("High aerodynamic efficiency suits endurance missions.")
        if name in ("Conventional Tractor", "Conventional Pusher") and ep >= 0.5:
            score += 6
            bonuses.append("Proven efficiency and easy drag reduction for endurance.")
        if name == "Canard" and ep >= 0.6:
            score -= 5
            penalties.append("Canard trim drag reduces cruise efficiency slightly.")

        # ── Speed pressure ───────────────────────────────────────────────
        sp = pressures.get("speed", 0)
        if name == "Flying Wing" and sp >= 0.6:
            score += 8
            bonuses.append("Clean aerodynamics suit higher cruise speeds.")
        if name == "Twin-Boom Pusher" and sp >= 0.6:
            score -= 4
            penalties.append("Twin-boom drag increases at higher speeds.")

        # ── Payload / forward view ────────────────────────────────────────
        if m.payload_needs_fwd_view:
            if name in ("Conventional Pusher", "Twin-Boom Pusher"):
                score += 15
                bonuses.append("Pusher layout gives unobstructed forward payload view.")
            if name == "Conventional Tractor":
                score -= 10
                penalties.append("Propeller obstructs forward view in tractor layout.")
            if name == "Flying Wing":
                score -= 12
                penalties.append("Flying wing fuselage volume limits payload integration and visibility.")

        if m.payload_mass >= 400:
            if name in ("Conventional Tractor", "Conventional Pusher", "Tandem Wing"):
                score += 8
                bonuses.append("Good payload volume and CG flexibility for heavy payload.")
            if name == "Flying Wing":
                score -= 10
                penalties.append("Flying wing has limited fuselage volume and tight CG for heavy payloads.")

        # ── Hand launch ───────────────────────────────────────────────────
        if m.launch_method == "Hand launch":
            if name in ("Conventional Tractor", "Flying Wing"):
                score += 8
                bonuses.append("Compact and easily hand-launched.")
            if name == "Twin-Boom Pusher":
                score -= 5
                penalties.append("Twin-boom layout is wider and awkward to hand-launch.")
            if name == "Tandem Wing":
                score -= 8
                penalties.append("Tandem wing is complex to grip and launch safely.")

        # ── Belly landing ─────────────────────────────────────────────────
        if m.recovery_method == "Belly land":
            if name == "Conventional Pusher":
                score -= 12
                penalties.append("Pusher prop is highly vulnerable in belly landings.")
            if name == "Flying Wing":
                score -= 6
                penalties.append("Flying wing belly-lands on wing surface — limit sensitive underside components.")
            if name == "Conventional Tractor":
                score += 5
                bonuses.append("Tractor prop is protected in belly landings.")

        # ── Wingspan cap ──────────────────────────────────────────────────
        if 0 < m.wingspan_cap < 1.2:
            if name == "Flying Wing":
                score += 5
                bonuses.append("Flying wing makes efficient use of limited span.")
            if name == "Tandem Wing":
                score += 4
                bonuses.append("Tandem wing distributes lift across two shorter-span surfaces.")
            if name == "Twin-Boom Pusher":
                score -= 5
                penalties.append("Twin-boom layout adds structural width beyond wing span.")

        # ── Simplicity ────────────────────────────────────────────────────
        diff = DIFFICULTY[name]
        if diff == "Easy":
            score += 5
            bonuses.append("Simple build — well-documented design patterns available.")
        elif diff == "Hard":
            score -= 8
            penalties.append("Complex build — requires careful aerodynamic tuning.")

        # ── Flying wing special flag for trim sensitivity ─────────────────
        if name == "Flying Wing" and m.payload_mass > 300:
            score -= 6
            penalties.append("Flying wing CG range is tight; heavy payload complicates trimming.")

        results.append(ConfigurationScore(
            name=name,
            score=round(score, 1),
            bonuses=bonuses,
            penalties=penalties,
            difficulty=diff,
            description=CONFIG_DESCRIPTIONS[name],
        ))

    results.sort(key=lambda c: c.score, reverse=True)
    return results
