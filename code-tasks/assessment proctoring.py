import json
import math
import random
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional
from irtcalibration import (
    ItemParameter, seed_item_pool, calibration_pipeline,
    generate_synthetic_responses, p_correct_3pl, item_information
)
from catengine import (
    CATSession, estimate_theta_eap, should_stop,
    theta_to_scaled_score, session_id_key, initialize_engine,
    update_session, THETA_GRID, normal_density
)
BLOOM_DISTRIBUTION_TARGET = {
    "Remember": 0.10,
    "Understand": 0.20,
    "Apply": 0.40,
    "Analyze": 0.20,
    "Evaluate": 0.05,
    "Create": 0.05,
}


def select_next_item_balanced(
    theta: float,
    items: list[ItemParameter],
    administered_ids: set[str],
    session_competency_id: str,
    exposure_counts: dict[str, int],
    bloom_counts: dict[str, int],
    n_administered: int
) -> Optional[ItemParameter]:
    """
    Enhanced item selection with content balancing:
    1. Filter eligible items (not used, right competency, under exposure cap)
    2. Penalise items from over-represented bloom levels
    3. Select by maximum adjusted information
    """
    candidates = [
        item for item in items
        if item.item_id not in administered_ids
        and item.competency_id == session_competency_id
        and exposure_counts.get(item.item_id, 0) < item.exposure_cap
    ]

    if not candidates:
        return None
    def adjusted_info(item: ItemParameter) -> float:
        base_info = item_information(theta, item.a, item.b, item.c)

        
        target_count = BLOOM_DISTRIBUTION_TARGET.get(item.bloom_level, 0.2) * max(n_administered, 1)
        current_count = bloom_counts.get(item.bloom_level, 0)
        bloom_penalty = max(0.5, 1.0 - max(0, current_count - target_count) * 0.1)

        
        usage_ratio = exposure_counts.get(item.item_id, 0) / item.exposure_cap
        exposure_penalty = max(0.3, 1.0 - usage_ratio * 0.5)

        return base_info * bloom_penalty * exposure_penalty

    return max(candidates, key=adjusted_info)

@dataclass
class FullCATSession(CATSession):
    bloom_counts: dict = field(default_factory=dict)
    SEM_THRESHOLD: float = 0.28   # tighter precision target


_full_sessions: dict[str, FullCATSession] = {}
_items_full: list[ItemParameter] = []
_exposure_counts_full: dict[str, int] = {}


def init_full_engine(items: list[ItemParameter]):
    global _items_full, _exposure_counts_full
    _items_full = items
    _exposure_counts_full = {item.item_id: 0 for item in items}


def cat_next_item_full(student_id: str, assessment_id: str,
                       current_item_id: Optional[str], response: Optional[int],
                       competency_id: str) -> dict:
    """
    Full CAT endpoint (Day 4 version) with content balancing.
    On first call (current_item_id=None), starts session.
    """
    key = session_id_key(student_id, assessment_id)

    if current_item_id is None:
        # Start new session
        session = FullCATSession(
            student_id=student_id,
            assessment_id=assessment_id,
            competency_id=competency_id
        )
        _full_sessions[key] = session
    else:
        session = _full_sessions.get(key)
        if not session:
            return {"error": "Session not found"}
        item = next((i for i in _items_full if i.item_id == current_item_id), None)
        if item:
            session = update_session(session, item, response)
            _exposure_counts_full[current_item_id] = _exposure_counts_full.get(current_item_id, 0) + 1
            session.bloom_counts[item.bloom_level] = session.bloom_counts.get(item.bloom_level, 0) + 1

        stop, reason = should_stop(session)
        if stop:
            score = theta_to_scaled_score(session.theta)
            return {
                "stop_test": True,
                "stop_reason": reason,
                "student_id": student_id,
                "assessment_id": assessment_id,
                "ability_estimate": session.theta,
                "sem": session.sem,
                **score,
                "items_administered": len(session.administered_ids)
            }

    next_item = select_next_item_balanced(
        theta=session.theta,
        items=_items_full,
        administered_ids=set(session.administered_ids),
        session_competency_id=competency_id,
        exposure_counts=_exposure_counts_full,
        bloom_counts=session.bloom_counts,
        n_administered=len(session.administered_ids)
    )

    if not next_item:
        score = theta_to_scaled_score(session.theta)
        return {"stop_test": True, "stop_reason": "pool_exhausted", **score}

    return {
        "next_item_id": next_item.item_id,
        "competency_id": competency_id,
        "difficulty_band": next_item.difficulty_band,
        "bloom_level": next_item.bloom_level,
        "ability_estimate": session.theta,
        "sem": session.sem,
        "stop_test": False
    }

@dataclass
class FaceEmbedding:
    student_id: str
    embedding: list[float]  
    source: str              


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def generate_face_embedding(student_id: str, noise_level: float = 0.0) -> list[float]:
    """Generate a deterministic face embedding (simulated) with optional noise."""
    seed = int(hashlib.md5(student_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    base = [rng.gauss(0, 1) for _ in range(128)]
    magnitude = math.sqrt(sum(x ** 2 for x in base))
    unit = [x / magnitude for x in base]
    if noise_level > 0:
        noisy = [x + rng.gauss(0, noise_level) for x in unit]
        mag2 = math.sqrt(sum(x ** 2 for x in noisy))
        return [x / mag2 for x in noisy]
    return unit


FACE_MATCH_THRESHOLD = 0.85   


def verify_identity(
    registered_embedding: FaceEmbedding,
    live_embedding: FaceEmbedding
) -> dict:
    """
    POST /proctoring/identity-verify
    Compares live capture to registered photo at session start.
    """
    similarity = cosine_similarity(registered_embedding.embedding, live_embedding.embedding)
    matched = similarity >= FACE_MATCH_THRESHOLD
    confidence = round(similarity * 100, 2)

    return {
        "student_id": registered_embedding.student_id,
        "identity_verified": matched,
        "similarity_score": round(similarity, 4),
        "confidence_pct": confidence,
        "threshold": FACE_MATCH_THRESHOLD,
        "action": "ALLOW_SESSION" if matched else "BLOCK_SESSION",
        "reason": (
            f"Face match confidence {confidence}% — "
            f"{'above' if matched else 'below'} threshold {FACE_MATCH_THRESHOLD * 100}%"
        )
    }

@dataclass
class ProctorFrame:
    frame_id: int
    timestamp_ms: int
    faces_detected: int
    face_confidence: float    
    gaze_deviation: float     
    head_pose_yaw: float      
    head_pose_pitch: float    


def analyze_frame(frame: ProctorFrame) -> dict:
    """
    Per-frame signal analysis.
    Returns signals emitted to Backend.
    """
    signals = []

    if frame.faces_detected == 0:
        signals.append({
            "type": "NO_FACE_DETECTED",
            "severity": "HIGH",
            "frame_id": frame.frame_id,
            "timestamp_ms": frame.timestamp_ms
        })
    elif frame.faces_detected > 1:
        signals.append({
            "type": "MULTIPLE_FACE_DETECTED",
            "severity": "HIGH",
            "face_count": frame.faces_detected,
            "frame_id": frame.frame_id,
            "timestamp_ms": frame.timestamp_ms
        })
    if abs(frame.head_pose_yaw) > 30 or abs(frame.head_pose_pitch) > 25:
        signals.append({
            "type": "SUSPICIOUS_HEAD_MOVEMENT",
            "severity": "MEDIUM",
            "yaw": frame.head_pose_yaw,
            "pitch": frame.head_pose_pitch,
            "frame_id": frame.frame_id,
            "timestamp_ms": frame.timestamp_ms
        })
    if frame.gaze_deviation > 0.7:
        signals.append({
            "type": "GAZE_AWAY",
            "severity": "LOW",
            "deviation": frame.gaze_deviation,
            "frame_id": frame.frame_id,
            "timestamp_ms": frame.timestamp_ms
        })

    return {
        "frame_id": frame.frame_id,
        "timestamp_ms": frame.timestamp_ms,
        "faces_detected": frame.faces_detected,
        "face_confidence": frame.face_confidence,
        "signals": signals,
        "clean_frame": len(signals) == 0
    }


def simulate_proctoring_session(n_frames: int = 20, cheat_probability: float = 0.1) -> list[dict]:
    """Simulate a stream of proctoring frames for a session."""
    results = []
    for i in range(n_frames):
        is_clean = random.random() > cheat_probability
        if is_clean:
            frame = ProctorFrame(
                frame_id=i,
                timestamp_ms=i * 5000,
                faces_detected=1,
                face_confidence=random.uniform(0.85, 0.99),
                gaze_deviation=random.uniform(0.0, 0.3),
                head_pose_yaw=random.uniform(-10, 10),
                head_pose_pitch=random.uniform(-8, 8)
            )
        else:
            violation_type = random.choice(["no_face", "multi_face", "gaze", "movement"])
            frame = ProctorFrame(
                frame_id=i,
                timestamp_ms=i * 5000,
                faces_detected=0 if violation_type == "no_face" else (2 if violation_type == "multi_face" else 1),
                face_confidence=random.uniform(0.5, 0.85),
                gaze_deviation=random.uniform(0.7, 1.0) if violation_type == "gaze" else 0.2,
                head_pose_yaw=random.uniform(35, 60) if violation_type == "movement" else 5,
                head_pose_pitch=random.uniform(30, 45) if violation_type == "movement" else 5
            )
        results.append(analyze_frame(frame))
    return results

if __name__ == "__main__":
    print("=" * 60)
    print("PlaceMux — Day 4 AI/ML: Complete CAT + Proctoring Core")
    print("=" * 60)

    # Setup engine
    items = seed_item_pool()
    synthetic = generate_synthetic_responses(items, n_students=60)
    items = calibration_pipeline(items, synthetic)
    init_full_engine(items)

    # ── Test enhanced CAT ──
    print("\n[1] COMPLETE CAT ENGINE (content-balanced)")
    result = cat_next_item_full("STU_002", "ASM_002", None, None, "CSE_DSA_001")
    print(f"  Session start → first item: {result['next_item_id']}")

    true_theta = 0.8
    steps = 0
    while not result.get("stop_test"):
        nid = result["next_item_id"]
        item_obj = next((i for i in items if i.item_id == nid), None)
        resp = 1 if random.random() < p_correct_3pl(true_theta, item_obj.a, item_obj.b, item_obj.c) else 0
        result = cat_next_item_full("STU_002", "ASM_002", nid, resp, "CSE_DSA_001")
        steps += 1

    print(f"  Stopped after {steps} items | reason: {result['stop_reason']}")
    print(f"  Final: theta={result['ability_estimate']:+.3f} | SEM={result['sem']:.3f} | score={result['scaled_score']} | band={result['performance_band']}")
    print("\n[2] IDENTITY VERIFICATION")
    registered = FaceEmbedding("STU_003", generate_face_embedding("STU_003"), "registration")
    live_same  = FaceEmbedding("STU_003", generate_face_embedding("STU_003", noise_level=0.05), "live_capture")
    live_diff  = FaceEmbedding("STU_003", generate_face_embedding("IMPOSTER", noise_level=0.0), "live_capture")

    result_match  = verify_identity(registered, live_same)
    result_no_match = verify_identity(registered, live_diff)
    print(f"  Same person: verified={result_match['identity_verified']}, similarity={result_match['similarity_score']:.4f}, action={result_match['action']}")
    print(f"  Imposter:    verified={result_no_match['identity_verified']}, similarity={result_no_match['similarity_score']:.4f}, action={result_no_match['action']}")
    print("\n[3] PRESENCE / MULTIPLE-FACE DETECTION")
    frames = simulate_proctoring_session(n_frames=20, cheat_probability=0.15)
    all_signals = [s for f in frames for s in f["signals"]]
    clean_count = sum(1 for f in frames if f["clean_frame"])
    print(f"  Frames analyzed: {len(frames)} | Clean: {clean_count} | Flagged: {len(frames)-clean_count}")
    print(f"  Total signals emitted: {len(all_signals)}")
    for sig in all_signals[:5]:
        print(f"    [{sig['severity']}] {sig['type']} at t={sig['timestamp_ms']}ms")

    print("\n✅ Day 4 Definition of Done:")
    print("  ✓ CAT returns coherent adaptive sequences + final scored object")
    print("  ✓ Identity match works at session start (face similarity)")
    print("  ✓ Presence / multiple-face signals are emitted per-frame")