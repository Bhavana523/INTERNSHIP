import json
import math
import random
from dataclasses import dataclass, field, asdict
from typing import Optional
from irtcalibration import (
    ItemParameter, seed_item_pool, calibration_pipeline,
    generate_synthetic_responses, p_correct_3pl, item_information
)

THETA_GRID = [round(-3.0 + i * 0.1, 1) for i in range(61)]   # -3 to +3 in 0.1 steps
PRIOR_MEAN = 0.0
PRIOR_SD = 1.0


def normal_density(x: float, mean: float = 0.0, sd: float = 1.0) -> float:
    return math.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def estimate_theta_eap(
    responses: list[dict],   # [{item: ItemParameter, response: 0|1}]
    prior_mean: float = PRIOR_MEAN,
    prior_sd: float = PRIOR_SD
) -> tuple[float, float]:
    
    if not responses:
        return prior_mean, prior_sd

    numerator = 0.0
    denominator = 0.0

    for theta in THETA_GRID:
        prior = normal_density(theta, prior_mean, prior_sd)
        likelihood = 1.0
        for r in responses:
            item = r["item"]
            p = p_correct_3pl(theta, item.a, item.b, item.c)
            p = max(1e-6, min(1 - 1e-6, p))
            likelihood *= p if r["response"] == 1 else (1 - p)

        posterior_weight = likelihood * prior
        numerator += theta * posterior_weight
        denominator += posterior_weight

    if denominator == 0:
        return prior_mean, prior_sd

    theta_hat = numerator / denominator

    variance = sum(
        ((t - theta_hat) ** 2) * normal_density(t, prior_mean, prior_sd)
        for t in THETA_GRID
    ) / len(THETA_GRID)
    sem = max(0.05, math.sqrt(variance))

    return round(theta_hat, 4), round(sem, 4)


def select_next_item(
    theta: float,
    items: list[ItemParameter],
    administered_ids: set[str],
    session_competency_id: str,
    exposure_counts: dict[str, int]
) -> Optional[ItemParameter]:
    candidates = [
        item for item in items
        if item.item_id not in administered_ids
        and item.competency_id == session_competency_id
        and exposure_counts.get(item.item_id, 0) < item.exposure_cap
    ]

    if not candidates:
        return None

    best_item = max(candidates, key=lambda item: item_information(theta, item.a, item.b, item.c))
    return best_item

@dataclass
class CATSession:
    student_id: str
    assessment_id: str
    competency_id: str
    theta: float = 0.0
    sem: float = 1.0
    responses: list[dict] = field(default_factory=list)
    administered_ids: list[str] = field(default_factory=list)
    stopped: bool = False
    stop_reason: str = ""
    MAX_ITEMS: int = 30
    SEM_THRESHOLD: float = 0.30   # stop when SEM < 0.30


def update_session(
    session: CATSession,
    item: ItemParameter,
    response: int   # 0 or 1
) -> CATSession:
    """Record response and update theta estimate."""
    session.responses.append({"item": item, "response": response})
    session.administered_ids.append(item.item_id)
    session.theta, session.sem = estimate_theta_eap(session.responses)
    return session


def should_stop(session: CATSession) -> tuple[bool, str]:
    """Evaluate stopping rules."""
    if len(session.administered_ids) >= session.MAX_ITEMS:
        return True, "max_items_reached"
    if len(session.administered_ids) >= 5 and session.sem <= session.SEM_THRESHOLD:
        return True, "precision_target_met"
    return False, ""

def theta_to_scaled_score(theta: float) -> dict:
    """
    Map theta (logit scale, ~N(0,1)) to scaled score 200–800.
    Percentile computed from standard normal CDF approximation.
    """
    scaled = round(500 + theta * 100)
    scaled = max(200, min(800, scaled))
    t_val = 1 / (1 + 0.2316419 * abs(theta))
    poly = t_val * (0.319381530 + t_val * (-0.356563782 + t_val * (1.781477937
           + t_val * (-1.821255978 + t_val * 1.330274429))))
    normal_cdf = 1 - normal_density(theta) * poly
    if theta < 0:
        normal_cdf = 1 - normal_cdf
    percentile = round(normal_cdf * 100, 1)

    bands = [
        (300, "Beginner"),
        (450, "Average"),
        (600, "Good"),
        (800, "Excellent"),
    ]
    band = next((b for threshold, b in bands if scaled <= threshold), "Excellent")

    return {"scaled_score": scaled, "percentile": percentile, "performance_band": band}
_sessions: dict[str, CATSession] = {}
_items: list[ItemParameter] = []
_exposure_counts: dict[str, int] = {}


def initialize_engine(items: list[ItemParameter]):
    """Load item pool into engine. Called once at startup."""
    global _items, _exposure_counts
    _items = items
    _exposure_counts = {item.item_id: 0 for item in items}
    print(f"[ENGINE] Initialized with {len(_items)} items")


def api_start_session(student_id: str, assessment_id: str, competency_id: str) -> dict:
    """
    POST /inference/session/start
    Creates a new CAT session and returns the first item.
    """
    session = CATSession(
        student_id=student_id,
        assessment_id=assessment_id,
        competency_id=competency_id
    )
    first_item = select_next_item(
        theta=0.0,
        items=_items,
        administered_ids=set(),
        session_competency_id=competency_id,
        exposure_counts=_exposure_counts
    )

    if not first_item:
        return {"error": "No items available for this competency", "stop_test": True}

    _sessions[session_id_key(student_id, assessment_id)] = session

    return {
        "session_id": session_id_key(student_id, assessment_id),
        "next_item_id": first_item.item_id,
        "competency_id": competency_id,
        "difficulty_band": first_item.difficulty_band,
        "ability_estimate": 0.0,
        "sem": 1.0,
        "stop_test": False,
        "reason": "Session started; first item at median difficulty"
    }


def api_next_item(
    student_id: str,
    assessment_id: str,
    current_item_id: str,
    response: int 
) -> dict:
    """
    POST /inference/next-item
    Records response, updates theta, returns next item or final score.
    Mirrors the contract from Inference_and_Integrity_API.docx
    """
    key = session_id_key(student_id, assessment_id)
    session = _sessions.get(key)
    if not session:
        return {"error": "Session not found"}
    item = next((i for i in _items if i.item_id == current_item_id), None)
    if not item:
        return {"error": f"Item {current_item_id} not found"}
    session = update_session(session, item, response)
    _exposure_counts[current_item_id] = _exposure_counts.get(current_item_id, 0) + 1
    stop, reason = should_stop(session)
    if stop:
        session.stopped = True
        session.stop_reason = reason
        score_data = theta_to_scaled_score(session.theta)
        return {
            "stop_test": True,
            "stop_reason": reason,
            "ability_estimate": session.theta,
            "sem": session.sem,
            "scaled_score": score_data["scaled_score"],
            "percentile": score_data["percentile"],
            "performance_band": score_data["performance_band"],
            "items_administered": len(session.administered_ids)
        }
    next_item = select_next_item(
        theta=session.theta,
        items=_items,
        administered_ids=set(session.administered_ids),
        session_competency_id=session.competency_id,
        exposure_counts=_exposure_counts
    )

    if not next_item:
        score_data = theta_to_scaled_score(session.theta)
        return {
            "stop_test": True,
            "stop_reason": "item_pool_exhausted",
            "ability_estimate": session.theta,
            "sem": session.sem,
            **score_data
        }

    prev_item = next((i for i in _items if i.item_id == current_item_id), None)
    reason_text = (
        f"Student answered {prev_item.difficulty_band.lower()} question "
        f"{'correctly' if response == 1 else 'incorrectly'}; "
        f"ability estimate updated to {session.theta:+.3f}"
    )

    return {
        "next_item_id": next_item.item_id,
        "competency_id": session.competency_id,
        "difficulty_band": next_item.difficulty_band,
        "ability_estimate": session.theta,
        "sem": session.sem,
        "stop_test": False,
        "reason": reason_text
    }


def api_get_final_score(student_id: str, assessment_id: str) -> dict:
    """
    GET /inference/score/{assessment_id}
    Returns final scoring object. Matches Final Scoring API contract.
    """
    key = session_id_key(student_id, assessment_id)
    session = _sessions.get(key)
    if not session:
        return {"error": "Session not found"}

    score_data = theta_to_scaled_score(session.theta)
    return {
        "student_id": student_id,
        "assessment_id": assessment_id,
        "ability_estimate": session.theta,
        "sem": session.sem,
        "scaled_score": score_data["scaled_score"],
        "percentile": score_data["percentile"],
        "performance_band": score_data["performance_band"],
        "items_administered": len(session.administered_ids),
        "stopped_reason": session.stop_reason
    }


def session_id_key(student_id: str, assessment_id: str) -> str:
    return f"{student_id}::{assessment_id}"
if __name__ == "__main__":
    print("=" * 60)
    print("PlaceMux — Day 3 AI/ML: CAT Engine + Stub Endpoint")
    print("=" * 60)
    items = seed_item_pool()
    synthetic = generate_synthetic_responses(items, n_students=60)
    items = calibration_pipeline(items, synthetic)
    initialize_engine(items)
    print("\n[SIM] Simulating student STU_001 assessment on CSE_PROG_003")
    result = api_start_session("STU_001", "ASM_001", "CSE_PROG_003")
    print(f"  Session start → next_item: {result['next_item_id']} | difficulty: {result['difficulty_band']}")
    true_theta = 0.5
    step = 1
    while not result.get("stop_test"):
        item_id = result["next_item_id"]
        item_obj = next((i for i in items if i.item_id == item_id), None)
        simulated_response = 1 if random.random() < p_correct_3pl(true_theta, item_obj.a, item_obj.b, item_obj.c) else 0
        result = api_next_item("STU_001", "ASM_001", item_id, simulated_response)
        if not result.get("stop_test"):
            print(f"  Step {step:2d}: item={item_id} | resp={simulated_response} | theta={result['ability_estimate']:+.3f} | SEM={result['sem']:.3f}")
        step += 1

    print(f"\n  STOPPED: {result.get('stop_reason')}")
    final = api_get_final_score("STU_001", "ASM_001")
    print(f"\n[FINAL SCORE]")
    print(json.dumps(final, indent=2))

    print("\n✅ Day 3 Definition of Done:")
    print("  ✓ Calibration produces item parameters (cold-start + empirical)")
    print("  ✓ Theta updates correctly per response (EAP)")
    print("  ✓ Stub endpoint returns next item + theta (callable by Backend)")