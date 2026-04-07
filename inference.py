"""Baseline inference runner for the hospital triage environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from client import HospitalTriageEnv
from models import HospitalAction

TASKS = [
    "task_1_basic_triage",
    "task_2_queue_optimization",
    "task_3_emergency_handling",
]

LOG_DIR = Path("outputs/logs")
EVAL_DIR = Path("outputs/evals")


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
        text = response.choices[0].message.content
        parsed = json.loads(text)
        return HospitalAction.model_validate(parsed)
    except Exception:
        return None


def ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


def run_task(env: HospitalTriageEnv, llm_client: OpenAI | None, task_id: str, seed: int) -> dict[str, Any]:
    runtime = settings()
    env.session_id = f"{task_id}_seed_{seed}"
    observation = env.reset(task_id=task_id, seed=seed)
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

    while not done and steps < observation["max_steps"]:
        action = llm_action(llm_client, observation, runtime["model_name"])
        if action is None:
            action = HospitalAction(action_type="wait", note="LLM failed to provide action.")
        response = env.step(action)
        total_reward += response.reward.value
        step_record = {
            "task_id": task_id,
            "session_id": env.session_id,
            "step": steps,
            "action": action.model_dump(),
            "reward": response.reward.model_dump(),
            "done": response.done,
            "info": response.info,
        }
        trace.append(step_record)
        log_line("STEP", step_record)
        observation = response.observation.model_dump()
        done = response.done
        steps += 1

    result = {
        "task_id": task_id,
        "seed": seed,
        "session_id": env.session_id,
        "steps": steps,
        "total_reward": round(total_reward, 3),
        "final_score": trace[-1]["info"]["task_score"] if trace else {},
        "hf_token_present": bool(runtime["hf_token"]),
    }
    log_line("END", result)
    (LOG_DIR / f"{task_id}.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return result


def main() -> None:
    runtime = settings()
    ensure_dirs()
    llm_client: OpenAI | None = None
    if runtime["model_base_url"]:
        try:
            llm_client = OpenAI(
                base_url=runtime["model_base_url"],
                api_key=runtime["api_key"],
            )
        except Exception:
            llm_client = None

    env = HospitalTriageEnv(base_url=runtime["env_base_url"])
    summary = [run_task(env, llm_client, task_id, seed=17) for task_id in TASKS]
    (EVAL_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
