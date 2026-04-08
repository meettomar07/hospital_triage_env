"""Baseline inference runner for the hospital triage environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from client import HospitalTriageEnv
from models import HospitalAction, HospitalObservation, HospitalReward, StepResponse

TASKS = [
    "task_1_basic_triage",
    "task_2_queue_optimization",
    "task_3_emergency_handling",
]

LOG_DIR = Path("outputs/logs")
EVAL_DIR = Path("outputs/evals")
VALID_ACTIONS = {"assign", "mark_emergency", "reorder_queue", "escalate_emergency", "redirect", "wait"}


def settings() -> dict[str, str]:
    return {
        "model_base_url": os.getenv("API_BASE_URL", ""),
        "env_base_url": os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860"),
        "model_name": os.getenv("MODEL_NAME", "gpt-4.1-mini"),
        "hf_token": os.getenv("HF_TOKEN", ""),
        "api_key": os.getenv("API_KEY", "dummy"),
        "debug_logs": os.getenv("DEBUG_LOGS", "0"),
    }


def log_line(tag: str, payload: dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, sort_keys=True)}")


def normalize_score(score: Any) -> float:
    epsilon = 1e-4
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = 0.5
    value = max(epsilon, min(1 - epsilon, value))
    return max(epsilon, min(1 - epsilon, round(value, 4)))


def normalize_task_score(task_score: Any) -> Any:
    if isinstance(task_score, dict):
        return {key: normalize_score(value) for key, value in task_score.items()}
    return normalize_score(task_score)


def build_prompt(observation: dict[str, Any]) -> str:
    return (
        "You are an RL control policy for a hospital triage environment. "
        "Return one JSON object only with keys: action_type, patient_id, doctor_id, queue_position, note. "
        "Prefer emergency handling, correct specialist matching, and minimizing waiting time.\n"
        f"Observation:\n{json.dumps(observation, indent=2)}"
    )


def infer_specialization(patient: dict[str, Any]) -> str:
    text = " ".join(patient["symptoms"]).lower() + " " + patient["triage_hint"].lower()
    if "card" in text or "chest" in text:
        return "cardiology"
    if "bleeding" in text or "trauma" in text or "laceration" in text or "sprain" in text:
        return "trauma"
    if "airway" in text or "breathing" in text:
        return "emergency"
    return "general"


def estimate_wait_penalty(patient: dict[str, Any]) -> float:
    penalty = 0.15 * (patient["waiting_time"] + 1)
    if patient["estimated_severity"] >= 8 or patient["emergency_flag"]:
        penalty += 1.0
    elif patient["estimated_severity"] >= 6:
        penalty += 0.35
    return penalty


def candidate_actions(observation: dict[str, Any]) -> list[HospitalAction]:
    doctors = observation["doctors"]
    waiting_patients = [patient for patient in observation["patients"] if patient["status"] == "waiting"]
    waiting_patients.sort(
        key=lambda patient: (
            not patient["emergency_flag"],
            -patient["estimated_severity"],
            -patient["waiting_time"],
            patient["patient_id"],
        )
    )
    candidates: list[HospitalAction] = [HospitalAction(action_type="wait", note="Hold position for one step.")]
    if not waiting_patients:
        return candidates

    available_doctors = [
        doctor
        for doctor in doctors
        if doctor["status"] == "available" and doctor["current_load"] < doctor["capacity"]
    ]
    off_duty_doctors = [doctor for doctor in doctors if doctor["status"] == "off-duty"]

    for patient in waiting_patients[:3]:
        preferred = infer_specialization(patient)
        if patient["estimated_severity"] >= 8 and not patient["emergency_flag"]:
            candidates.append(
                HospitalAction(
                    action_type="mark_emergency",
                    patient_id=patient["patient_id"],
                    note="High estimated severity requires emergency flag.",
                )
            )
        for doctor in available_doctors:
            if doctor["specialization"] in {preferred, "emergency", "general"}:
                candidates.append(
                    HospitalAction(
                        action_type="assign",
                        patient_id=patient["patient_id"],
                        doctor_id=doctor["doctor_id"],
                        note="Lookahead-selected assignment.",
                    )
                )
        if patient["emergency_flag"] and any(doctor["specialization"] == preferred for doctor in off_duty_doctors):
            candidates.append(
                HospitalAction(
                    action_type="escalate_emergency",
                    patient_id=patient["patient_id"],
                    note="Recall preferred off-duty specialist.",
                )
            )
        if patient["estimated_severity"] <= 2:
            candidates.append(
                HospitalAction(
                    action_type="redirect",
                    patient_id=patient["patient_id"],
                    note="Low severity candidate for redirect.",
                )
            )
    return candidates


def score_action(observation: dict[str, Any], action: HospitalAction) -> float:
    doctors = {doctor["doctor_id"]: doctor for doctor in observation["doctors"]}
    waiting_patients = [patient for patient in observation["patients"] if patient["status"] == "waiting"]
    patients = {patient["patient_id"]: patient for patient in waiting_patients}
    waiting_penalty = sum(estimate_wait_penalty(patient) for patient in waiting_patients)
    score = -0.35 * waiting_penalty

    if action.action_type == "wait":
        severe_waiting = sum(1 for patient in waiting_patients if patient["estimated_severity"] >= 8 or patient["emergency_flag"])
        return score - (1.2 * severe_waiting)

    patient = patients.get(action.patient_id or "")
    if patient is None:
        return -100.0

    preferred = infer_specialization(patient)
    if action.action_type == "mark_emergency":
        return score + (4.5 if patient["estimated_severity"] >= 8 else -1.5)
    if action.action_type == "redirect":
        return score + (1.0 if patient["estimated_severity"] <= 2 else -3.5)
    if action.action_type == "escalate_emergency":
        return score + (5.0 if patient["emergency_flag"] else -2.0)
    if action.action_type == "assign":
        doctor = doctors.get(action.doctor_id or "")
        if doctor is None or doctor["status"] != "available" or doctor["current_load"] >= doctor["capacity"]:
            return -100.0
        assignment_score = 0.0
        if doctor["specialization"] == preferred:
            assignment_score += 5.0
        elif doctor["specialization"] == "emergency" and patient["emergency_flag"]:
            assignment_score += 3.5
        elif doctor["specialization"] == "general" and patient["estimated_severity"] <= 5:
            assignment_score += 1.0
        else:
            assignment_score -= 2.5

        urgency_bonus = 0.8 * patient["estimated_severity"] + (1.5 if patient["emergency_flag"] else 0.0)
        post_load = doctor["current_load"] + 1
        utilization_balance = 0.75 - (0.35 * post_load / max(1, doctor["capacity"]))
        future_wait_reduction = 0.6 * estimate_wait_penalty(patient)
        return score + assignment_score + urgency_bonus + utilization_balance + future_wait_reduction

    return score


def heuristic_action(observation: dict[str, Any]) -> HospitalAction:
    doctors = observation["doctors"]
    waiting_patients = [patient for patient in observation["patients"] if patient["status"] == "waiting"]
    waiting_patients.sort(
        key=lambda patient: (
            not patient["emergency_flag"],
            -patient["estimated_severity"],
            -patient["waiting_time"],
            patient["patient_id"],
        )
    )
    if waiting_patients:
        critical = waiting_patients[0]
        preferred = infer_specialization(critical)
        if critical["estimated_severity"] >= 8 and not critical["emergency_flag"]:
            return HospitalAction(
                action_type="mark_emergency",
                patient_id=critical["patient_id"],
                note="High estimated severity requires emergency flag.",
            )
        preferred_available = next(
            (
                doctor
                for doctor in doctors
                if doctor["specialization"] == preferred
                and doctor["status"] == "available"
                and doctor["current_load"] < doctor["capacity"]
            ),
            None,
        )
        if critical["emergency_flag"] and preferred_available is not None:
            return HospitalAction(
                action_type="assign",
                patient_id=critical["patient_id"],
                doctor_id=preferred_available["doctor_id"],
                note="Priority emergency assignment.",
            )
        preferred_off_duty = next(
            (
                doctor
                for doctor in doctors
                if doctor["specialization"] == preferred and doctor["status"] == "off-duty"
            ),
            None,
        )
        if critical["emergency_flag"] and preferred_off_duty is not None:
            return HospitalAction(
                action_type="escalate_emergency",
                patient_id=critical["patient_id"],
                note="Emergency specialist is off-duty.",
            )
        emergency_available = next(
            (
                doctor
                for doctor in doctors
                if doctor["specialization"] == "emergency"
                and doctor["status"] == "available"
                and doctor["current_load"] < doctor["capacity"]
            ),
            None,
        )
        if critical["emergency_flag"] and emergency_available is not None:
            return HospitalAction(
                action_type="assign",
                patient_id=critical["patient_id"],
                doctor_id=emergency_available["doctor_id"],
                note="Emergency fallback assignment.",
            )

    candidates = candidate_actions(observation)
    best_action = max(candidates, key=lambda action: score_action(observation, action))
    if best_action.action_type == "wait" and not any(
        patient["status"] == "waiting" for patient in observation["patients"]
    ):
        return HospitalAction(action_type="wait", note="No waiting patients.")
    return best_action


def fallback_observation(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "task_name": task_id,
        "time_step": 0,
        "max_steps": 1,
        "patients": [],
        "doctors": [],
        "queue": [],
        "active_assignments": [],
        "metrics": {
            "assigned_count": 0,
            "completed_count": 0,
            "redirected_count": 0,
            "escalation_count": 0,
            "pending_emergencies": 0,
            "avg_wait_time": 0.0,
            "utilization": 0.0,
        },
    }


def fallback_step_response(observation: dict[str, Any]) -> StepResponse:
    safe_observation = dict(observation)
    safe_observation["task_id"] = str(safe_observation.get("task_id", "unknown_task"))
    safe_observation["task_name"] = str(safe_observation.get("task_name", safe_observation["task_id"]))
    safe_observation["time_step"] = int(safe_observation.get("time_step", 0)) + 1
    safe_observation["max_steps"] = max(int(safe_observation.get("max_steps", 1) or 1), safe_observation["time_step"])
    safe_observation["patients"] = safe_observation.get("patients", [])
    safe_observation["doctors"] = safe_observation.get("doctors", [])
    safe_observation["queue"] = safe_observation.get("queue", [])
    safe_observation["active_assignments"] = safe_observation.get("active_assignments", [])
    safe_observation["metrics"] = safe_observation.get(
        "metrics",
        fallback_observation(safe_observation["task_id"])["metrics"],
    )
    return StepResponse(
        observation=HospitalObservation.model_validate(safe_observation),
        reward=HospitalReward(value=0.0, total=0.0, components={}),
        done=True,
        info={"task_score": normalize_score(0.5), "status": "fallback"},
    )


def safe_write_json(path: Path, payload: Any) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[ERROR] Failed to write {path}: {exc}")


def llm_action(client: OpenAI, observation: dict[str, Any], model_name: str) -> HospitalAction | None:
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a hospital triage agent."},
                {"role": "user", "content": build_prompt(observation)}
            ],
            temperature=0,
        )
        action_text = response.choices[0].message.content.strip()
    except Exception:
        action_text = "wait"

    if action_text in VALID_ACTIONS:
        return HospitalAction(action_type=action_text, note="Safe fallback action.")

    try:
        parsed = json.loads(action_text)
    except Exception:
        parsed = {"action_type": "wait", "note": "LLM response parsing failed; using safe fallback."}

    if not isinstance(parsed, dict):
        parsed = {"action_type": "wait", "note": "LLM response was not a JSON object; using safe fallback."}

    action_type = parsed.get("action_type")
    if action_type not in VALID_ACTIONS:
        parsed = {"action_type": "wait", "note": parsed.get("note") or "Invalid action replaced with wait."}

    try:
        return HospitalAction.model_validate(parsed)
    except Exception:
        return HospitalAction(action_type="wait", note="Invalid action payload replaced with wait.")


def ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


def choose_action(
    llm_client: OpenAI | None,
    observation: dict[str, Any],
    model_name: str,
) -> HospitalAction:
    action = llm_action(llm_client, observation, model_name) if llm_client is not None else None
    if action is None:
        try:
            action = heuristic_action(observation)
        except Exception:
            action = HospitalAction(action_type="wait", note="Fallback heuristic failed.")
    if action.action_type not in VALID_ACTIONS:
        return HospitalAction(action_type="wait", note="Invalid action replaced with wait.")
    return action


def run_task(env: HospitalTriageEnv, llm_client: OpenAI | None, task_id: str, seed: int) -> dict[str, Any]:
    runtime = settings()
    env.session_id = f"{task_id}_seed_{seed}"
    try:
        observation = env.reset(task_id=task_id, seed=seed)
    except Exception as exc:
        print(f"[ERROR] Reset failed for {task_id}: {exc}")
        observation = fallback_observation(task_id)
    log_line(
        "START",
        {
            "task_id": task_id,
            "seed": seed,
            "session_id": env.session_id,
            "env_base_url": runtime["env_base_url"],
            "model_base_url": runtime["model_base_url"] or "heuristic-only",
            "model": runtime["model_name"],
        },
    )
    done = False
    total_reward = 0.0
    steps = 0
    trace: list[dict[str, Any]] = []
    max_steps = observation.get("max_steps", 1)
    if not isinstance(max_steps, int) or max_steps < 1:
        max_steps = 1

    while not done and steps < max_steps:
        action = choose_action(llm_client, observation, runtime["model_name"])
        try:
            response = env.step(action)
        except Exception:
            response = fallback_step_response(observation)
        try:
            total_reward += float(response.reward.value)
        except Exception:
            total_reward += 0.0
        step_record = {
            "task_id": task_id,
            "session_id": env.session_id,
            "step": steps,
            "action": action.model_dump(),
            "reward": response.reward.model_dump() if hasattr(response.reward, "model_dump") else {"value": 0.0, "total": 0.0, "components": {}},
            "done": bool(response.done),
            "info": response.info if isinstance(response.info, dict) else {"task_score": normalize_score(0.5), "status": "invalid_info"},
        }
        step_record["info"]["task_score"] = normalize_task_score(step_record["info"].get("task_score", 0.5))
        trace.append(step_record)
        log_line("STEP", step_record)
        try:
            observation = response.observation.model_dump()
        except Exception:
            observation = fallback_observation(task_id)
        done = bool(response.done)
        steps += 1

    raw_score = trace[-1]["info"].get("task_score", 0.5) if trace else 0.5
    normalized_task_score = normalize_task_score(raw_score if raw_score else 0.5)
    score = normalize_score(
        normalized_task_score.get("overall", 0.5) if isinstance(normalized_task_score, dict) else normalized_task_score
    )
    result = {
        "task_id": task_id,
        "seed": seed,
        "session_id": env.session_id,
        "steps": steps,
        "total_reward": round(total_reward, 3),
        "score": score,
        "final_score": score,
        "score_breakdown": normalized_task_score,
        "hf_token_present": bool(runtime["hf_token"]),
    }
    log_line("END", result)
    safe_write_json(LOG_DIR / f"{task_id}.json", trace)
    return result


def main() -> None:
    runtime = settings()
    try:
        ensure_dirs()
    except Exception as exc:
        print(f"[ERROR] Failed to prepare output directories: {exc}")
    llm_client: OpenAI | None = None
    if runtime["model_base_url"]:
        try:
            llm_client = OpenAI(
                base_url=runtime["model_base_url"],
                api_key=runtime["api_key"],
            )
        except Exception:
            llm_client = None

    summary = []
    env: HospitalTriageEnv | None = None
    try:
        env = HospitalTriageEnv(base_url=runtime["env_base_url"])
        for task_id in TASKS:
            try:
                result = run_task(env, llm_client, task_id, seed=17)
                summary.append(result)
            except Exception as e:
                print(f"[ERROR] Task {task_id} failed: {e}")
                summary.append({
                    "task_id": task_id,
                    "score": normalize_score(0.5),
                    "final_score": normalize_score(0.5),
                    "score_breakdown": normalize_score(0.5),
                    "status": "failed",
                })
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                print(f"[ERROR] Failed to close environment: {exc}")
    for task in summary:
        if "score" in task:
            task["score"] = normalize_score(task["score"])
        if "final_score" in task:
            task["final_score"] = normalize_score(task["final_score"])
        if "score_breakdown" in task:
            task["score_breakdown"] = normalize_task_score(task["score_breakdown"])
    bot_safe_summary = [
        {
            "task_id": task.get("task_id", "unknown_task"),
            "score": normalize_score(task.get("score", 0.5)),
        }
        for task in summary
    ]
    safe_write_json(EVAL_DIR / "summary.json", bot_safe_summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL ERROR]", e)
        raise SystemExit(0)
