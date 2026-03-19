"""
scenarios.py — Generate all experiment scenarios.

Experiment design matrix
========================
Models (4):
  - HK       (group, continuous)
  - Deffuant  (pairwise, continuous)
  - Galam    (group, discrete)
  - Voter    (pairwise, discrete)

Recommendation system:
  - Random               → all 4 models
  - StructureM9          → all 4 models  (90 % structure + 10 % random)
  - OpinionM9            → HK & Deffuant only  (90 % opinion + 10 % random)

Alpha / q condition (Influence vs RewiringRate):
  - a_gt_q  : Influence=0.50, RewiringRate=0.05   (α > q)
  - q_gt_a  : Influence=0.05, RewiringRate=0.50   (q > α)

Repost rate (p):
  - p25 : RepostRate=0.25
  - p0  : RepostRate=0.00

Repetitions: 40

Network: Random, 500 nodes, ~15 follows each.
Tolerance (continuous models only): 0.45
"""

from __future__ import annotations
from typing import List, Dict, Any

# region Fixed simulation parameters
NODE_COUNT = 500
NODE_FOLLOW_COUNT = 15
POST_RETAIN_COUNT = 3
RECSYS_COUNT = 10
MAX_SIMULATION_STEP = 5000
REPETITIONS = 40
REPETITIONS_VOTER = 10  # Voter model is slower, so we can do fewer repetitions
TOLERANCE = 0.45  # for HK and Deffuant
# endregion

# region Parameter sweep axes
ALPHA_Q_CONDITIONS: Dict[str, Dict[str, float]] = {
    "a_gt_q": {"Influence": 0.50, "RewiringRate": 0.05},  # α > q
    "q_gt_a": {"Influence": 0.05, "RewiringRate": 0.50},  # q > α
}

REPOST_CONDITIONS: Dict[str, float] = {
    "p25": 0.25,
    "p0": 0.00,
}

# Recsys available per model class
RECSYS_CONTINUOUS = ["Random", "StructureM9", "OpinionM9"]
RECSYS_DISCRETE = ["Random", "StructureM9"]
# endregion


# region Scenario builder
def _make_scenario(
    dynamics: str,
    recsys: str,
    aq_key: str,
    p_key: str,
    rep: int,
) -> Dict[str, Any]:
    aq = ALPHA_Q_CONDITIONS[aq_key]
    influence = aq["Influence"]
    rewiring = aq["RewiringRate"]
    repost_rate = REPOST_CONDITIONS[p_key]

    # Normalise recsys name for use in unique name
    recsys_tag = recsys.lower().replace("m9", "_m9")

    unique_name = f"{dynamics.lower()}-{recsys_tag}-{aq_key}-{p_key}-r{rep:02d}"

    scenario: Dict[str, Any] = {
        "UniqueName": unique_name,
        "DynamicsType": dynamics,
        "RecsysFactoryType": recsys,
        "NetworkType": "Random",
        "NodeCount": NODE_COUNT,
        "NodeFollowCount": NODE_FOLLOW_COUNT,
        "PostRetainCount": POST_RETAIN_COUNT,
        "RecsysCount": RECSYS_COUNT,
        "MaxSimulationStep": MAX_SIMULATION_STEP,
        # Collect opinion time-series and rewiring counts
        "AgentNumber": True,
        "OpinionSum": True,
        "RewiringEvent": True,
    }

    if dynamics == "HK":
        scenario["HKParams"] = {
            "Tolerance": TOLERANCE,
            "Influence": influence,
            "RewiringRate": rewiring,
            "RepostRate": repost_rate,
        }
    elif dynamics == "Deffuant":
        scenario["DeffuantParams"] = {
            "Tolerance": TOLERANCE,
            "Influence": influence,
            "RewiringRate": rewiring,
            "RepostRate": repost_rate,
        }
    elif dynamics == "Galam":
        scenario["GalamParams"] = {
            "Influence": influence,
            "RewiringRate": rewiring,
            "RepostRate": repost_rate,
        }
    elif dynamics == "Voter":
        scenario["VoterParams"] = {
            "Influence": influence,
            "RewiringRate": rewiring,
            "RepostRate": repost_rate,
        }
        scenario["RecSysParams"] = {
            "UseCache": False,
        }
    else:
        raise ValueError(f"Unknown dynamics: {dynamics!r}")

    return scenario


# endregion


def generate_scenarios() -> List[Dict[str, Any]]:
    """Return the full list of scenario dicts for the experiment."""
    scenarios: List[Dict[str, Any]] = []

    for dynamics, recsys_list in [
        ("HK", RECSYS_CONTINUOUS),
        ("Deffuant", RECSYS_CONTINUOUS),
        ("Galam", RECSYS_DISCRETE),
        ("Voter", RECSYS_DISCRETE),
    ]:
        for recsys in recsys_list:
            for aq_key in ALPHA_Q_CONDITIONS:
                for p_key in REPOST_CONDITIONS:
                    repetitions = (
                        REPETITIONS_VOTER if dynamics == "Voter" else REPETITIONS
                    )
                    for rep in range(repetitions):
                        scenarios.append(
                            _make_scenario(dynamics, recsys, aq_key, p_key, rep)
                        )

    return scenarios


def print_summary() -> None:
    scenarios = generate_scenarios()
    from collections import Counter

    dynamics_counts = Counter(s["DynamicsType"] for s in scenarios)
    recsys_counts = Counter(s["RecsysFactoryType"] for s in scenarios)
    print(f"Total scenarios : {len(scenarios)}")
    print(f"By dynamics     : {dict(dynamics_counts)}")
    print(f"By recsys       : {dict(recsys_counts)}")


if __name__ == "__main__":
    print_summary()
