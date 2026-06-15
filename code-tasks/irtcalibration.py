import json
import math
import os
import random
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional

IRT_MODEL_CHOICE = {
    "model": "3PL",
    "rationale": (
        "Item pool contains MCQ and multi-select types with non-trivial guessing probability. "
        "3PL (Three-Parameter Logistic) is selected to capture: "
        "(a) item difficulty b, (b) item discrimination a, (c) pseudo-guessing c. "
        "2PL is used as a fallback for coding/numeric items where c ≈ 0."
    ),
    "parameters": {
        "a": "discrimination: range 0.5–2.5, controls slope of ICC",
        "b": "difficulty: range -3 to +3, ability level at 50% correct (net of guessing)",
        "c": "guessing: fixed at 0.25 for 4-option MCQ, 0.20 for 5-option, 0.0 for coding/numeric"
    },
    "cold_start_strategy": "SME-seeded b values; a=1.0, c=item-type default until 30+ responses collected"
}


@dataclass
class ItemParameter:
    item_id: str
    competency_id: str
    sub_competency: str
    difficulty_band: str         
    bloom_level: str
    item_type: str                
    exposure_cap: int

  
    a: float = 1.0               
    b: float = 0.0               
    c: float = 0.25              
    calibrated: bool = False     
    response_count: int = 0      

def seed_item_pool() -> list[ItemParameter]:
    """
    Seed initial item pool for 2 priority competencies:
      - CSE_PROG_003 (Python Programming)
      - CSE_DSA_001  (Arrays)
    SME-assigned difficulty mapped to logit b values:
      Easy   → b ∈ [-1.5, -0.5]
      Medium → b ∈ [-0.5,  0.5]
      Hard   → b ∈ [ 0.5,  1.5]
    """
    random.seed(42)  

    def b_for_band(band: str) -> float:
        ranges = {"Easy": (-1.5, -0.5), "Medium": (-0.5, 0.5), "Hard": (0.5, 1.5)}
        lo, hi = ranges[band]
        return round(random.uniform(lo, hi), 3)

    def c_for_type(item_type: str) -> float:
        return 0.0 if item_type in ("Coding", "Numeric") else 0.25

    python_items = []
    py_sub = ["Variables & Types", "Control Flow", "Functions", "OOP", "List Comprehension",
              "Exception Handling", "File I/O", "Modules", "Decorators", "Generators"]
    bloom_cycle = ["Remember", "Understand", "Apply", "Analyze", "Apply", "Understand",
                   "Apply", "Remember", "Analyze", "Apply"]
    bands = ["Easy"] * 12 + ["Medium"] * 12 + ["Hard"] * 6

    for i, band in enumerate(bands):
        sub = py_sub[i % len(py_sub)]
        bloom = bloom_cycle[i % len(bloom_cycle)]
        itype = "Coding" if bloom == "Apply" and band == "Hard" else "MCQ"
        python_items.append(ItemParameter(
            item_id=f"CSE_PROG_003_{i+1:03d}",
            competency_id="CSE_PROG_003",
            sub_competency=sub,
            difficulty_band=band,
            bloom_level=bloom,
            item_type=itype,
            exposure_cap={"Easy": 500, "Medium": 300, "Hard": 200}[band],
            a=round(random.uniform(0.8, 1.8), 3),
            b=b_for_band(band),
            c=c_for_type(itype),
            calibrated=False,
        ))
    array_items = []
    arr_sub = ["1D Arrays", "2D Arrays", "Searching", "Sorting", "Two Pointers",
               "Sliding Window", "Prefix Sum", "Array Rotation", "Kadane's Algorithm", "Matrix"]
    bands2 = ["Easy"] * 12 + ["Medium"] * 12 + ["Hard"] * 6

    for i, band in enumerate(bands2):
        sub = arr_sub[i % len(arr_sub)]
        bloom = bloom_cycle[i % len(bloom_cycle)]
        itype = "Coding" if bloom in ("Apply", "Analyze") and band != "Easy" else "MCQ"
        array_items.append(ItemParameter(
            item_id=f"CSE_DSA_001_{i+1:03d}",
            competency_id="CSE_DSA_001",
            sub_competency=sub,
            difficulty_band=band,
            bloom_level=bloom,
            item_type=itype,
            exposure_cap={"Easy": 500, "Medium": 300, "Hard": 200}[band],
            a=round(random.uniform(0.8, 1.8), 3),
            b=b_for_band(band),
            c=c_for_type(itype),
            calibrated=False,
        ))

    all_items = python_items + array_items
    print(f"[SEED] Item pool seeded: {len(all_items)} items across 2 competencies")
    return all_items

def p_correct_3pl(theta: float, a: float, b: float, c: float) -> float:
    """P(correct | theta) under 3PL model."""
    return c + (1 - c) / (1 + math.exp(-1.702 * a * (theta - b)))


def item_information(theta: float, a: float, b: float, c: float) -> float:
    """Fisher information I(theta) for a 3PL item."""
    p = p_correct_3pl(theta, a, b, c)
    q = 1 - p
    if p <= c or q == 0:
        return 0.0
    numerator = (1.702 ** 2) * (a ** 2) * ((p - c) ** 2) * q
    denominator = ((1 - c) ** 2) * p
    return numerator / denominator if denominator > 0 else 0.0


RESPONSE_THRESHOLD = 30  

def calibration_pipeline(
    items: list[ItemParameter],
    response_matrix: list[dict]  
) -> list[ItemParameter]:
    """
    Cold-start calibration pipeline.
    - If item has < RESPONSE_THRESHOLD responses: use SME-seeded b; a=1.0
    - If item has >= RESPONSE_THRESHOLD responses: estimate b via MLE approximation
    Returns updated item list.
    """
    item_responses: dict[str, list[int]] = {}
    for r in response_matrix:
        iid = r["item_id"]
        item_responses.setdefault(iid, []).append(r["correct"])

    item_map = {item.item_id: item for item in items}

    for iid, responses in item_responses.items():
        if iid not in item_map:
            continue
        item = item_map[iid]
        item.response_count = len(responses)

        if len(responses) >= RESPONSE_THRESHOLD:
            p_obs = statistics.mean(responses)
            p_obs_clamped = max(item.c + 0.01, min(0.99, p_obs))
            try:
                logit_arg = (p_obs_clamped - item.c) / (1 - item.c)
                logit_arg = max(0.01, min(0.99, logit_arg))
                estimated_b = -math.log(logit_arg / (1 - logit_arg)) / (1.702 * item.a)
                item.b = round(max(-3.0, min(3.0, estimated_b)), 3)
                item.calibrated = True
            except (ValueError, ZeroDivisionError):
                pass  # keep seeded value if math fails
            print(f"[CALIBRATE] {iid}: n={len(responses)}, p={p_obs:.3f}, b_est={item.b:.3f}")
        else:
            print(f"[COLD-START] {iid}: n={len(responses)} < {RESPONSE_THRESHOLD}, using seeded b={item.b}")

    return items
def generate_synthetic_responses(
    items: list[ItemParameter],
    n_students: int = 50
) -> list[dict]:
    """Generate synthetic responses to test calibration pipeline."""
    responses = []
    theta_pool = [random.gauss(0, 1) for _ in range(n_students)]
    for item in items:
        sample_size = random.randint(10, 45)
        thetas = random.choices(theta_pool, k=sample_size)
        for sid, theta in enumerate(thetas):
            p = p_correct_3pl(theta, item.a, item.b, item.c)
            correct = 1 if random.random() < p else 0
            responses.append({
                "student_id": f"SYN_{sid:04d}",
                "item_id": item.item_id,
                "correct": correct
            })
    print(f"[SYNTHETIC] Generated {len(responses)} synthetic responses for {len(items)} items")
    return responses

SERVING_INFRA_REQUIREMENTS = {
    "inference_endpoint": {
        "latency_p99_ms": 200,
        "latency_p50_ms": 50,
        "throughput_rps": 500,
        "concurrency": 200,
    },
    "compute": {
        "phase1_cat": {"cpu_cores": 4, "ram_gb": 8, "gpu": "none — CPU sufficient for 3PL math"},
        "phase1_proctoring": {"cpu_cores": 8, "ram_gb": 16, "gpu": "1x NVIDIA T4 or equivalent"},
        "autoscaling": "horizontal pod autoscaler, min=2, max=10 replicas"
    },
    "storage": {
        "item_bank": "PostgreSQL (managed), ~10MB initial, scale to 1GB",
        "model_artifacts": "Object storage (S3-compatible), versioned",
        "proctoring_frames": "Encrypted object storage, 30-day retention, India region"
    },
    "deployment_target": "Container (Docker), Kubernetes managed runtime",
    "model_artifact_format": "Pickled Python object + JSON parameter file",
    "notes": "All proctoring data must remain in India region (DPDP compliance)"
}

if __name__ == "__main__":
    print("=" * 60)
    print("PlaceMux — Day 2 AI/ML: IRT Setup & Calibration")
    print("=" * 60)
    print("\n[1] IRT MODEL SELECTION")
    print(json.dumps(IRT_MODEL_CHOICE, indent=2))
    print("\n[2] SEEDING ITEM POOL")
    items = seed_item_pool()
    by_competency = {}
    for item in items:
        by_competency.setdefault(item.competency_id, []).append(item)
    for cid, citems in by_competency.items():
        bands = [i.difficulty_band for i in citems]
        print(f"  {cid}: {len(citems)} items | Easy={bands.count('Easy')} Med={bands.count('Medium')} Hard={bands.count('Hard')}")
    print("\n[3] CALIBRATION PIPELINE (synthetic data)")
    synthetic = generate_synthetic_responses(items, n_students=60)
    items = calibration_pipeline(items, synthetic)
    calibrated_count = sum(1 for i in items if i.calibrated)
    print(f"\n  Items calibrated from empirical data: {calibrated_count}/{len(items)}")
    print("\n[4] IRT MATH VERIFICATION")
    test_item = items[0]
    for theta in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        p = p_correct_3pl(theta, test_item.a, test_item.b, test_item.c)
        info = item_information(theta, test_item.a, test_item.b, test_item.c)
        print(f"  theta={theta:+.1f} → P(correct)={p:.3f}, I(theta)={info:.4f}")
    print("\n[5] SERVING INFRA REQUIREMENTS (→ DevOps)")
    print(json.dumps(SERVING_INFRA_REQUIREMENTS, indent=2))

    print("\n✅ Day 2 Definition of Done:")
    print("  ✓ Seeded pool ready: ≥30 items per competency (CSE_PROG_003, CSE_DSA_001)")
    print("  ✓ IRT model chosen: 3PL, documented with rationale")
    print("  ✓ Calibration skeleton runs on synthetic data")
    print("  ✓ Serving-infra requirements ready for DevOps handoff")
    pool_export = [asdict(i) for i in items]
    out_path = os.path.join(os.path.dirname(__file__), "item_pool_seed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pool_export, f, indent=2)
    print(f"\n  → item_pool_seed.json written for Backend ingestion: {out_path}")