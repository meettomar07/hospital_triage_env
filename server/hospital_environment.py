"""Hospital triage simulation engine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from random import Random
from typing import Any

from models import (
    AssignmentRecord,
    HospitalAction,
    HospitalObservation,
    HospitalReward,
    MetricsSnapshot,
    VisibleDoctor,
    VisiblePatient,
)

SCORE_EPSILON = 0.01


def clamp_score(value: float) -> float:
    """Clamp score to strictly open interval (0, 1) as required by OpenEnv."""
    return max(0.01, min(0.99, float(value)))


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    name: str
    description: str
    max_steps: int
    doctors: list[dict[str, Any]]
    patients: list[dict[str, Any]]
    deterioration_wait_threshold: int = 3
    deterioration_step: int = 1
    emergency_delay_threshold: int = 2
    fatigue_recovery_per_step: float = 0.05


TASKS: dict[str, TaskConfig] = {
    "task_1_basic_triage": TaskConfig(
        task_id="task_1_basic_triage",
        name="Basic Triage",
        description="Match patients to the right specialist and prioritize the most severe cases first.",
        max_steps=8,
        deterioration_wait_threshold=4,
        emergency_delay_threshold=3,
        fatigue_recovery_per_step=0.07,
        doctors=[
            {"doctor_id": "dr_general_1", "specialization": "general", "status": "available", "capacity": 2},
            {"doctor_id": "dr_cardio_1", "specialization": "cardiology", "status": "available", "capacity": 1},
            {"doctor_id": "dr_trauma_1", "specialization": "trauma", "status": "available", "capacity": 1},
        ],
        patients=[
            {
                "patient_id": "p1",
                "symptoms": ["chest pain", "shortness of breath"],
                "symptom_summary": "Adult with chest pain radiating to arm.",
                "triage_hint": "possible cardiac event",
                "estimated_severity": 8,
                "true_severity": 9,
                "preferred_specialization": "cardiology",
                "arrival_time": 0,
                "emergency_flag": True,
                "service_time": 2,
            },
            {
                "patient_id": "p2",
                "symptoms": ["fever", "sore throat"],
                "symptom_summary": "Mild fever and throat pain for two days.",
                "triage_hint": "general consultation",
                "estimated_severity": 3,
                "true_severity": 3,
                "preferred_specialization": "general",
                "arrival_time": 0,
                "emergency_flag": False,
                "service_time": 1,
            },
            {
                "patient_id": "p3",
                "symptoms": ["head laceration", "dizziness"],
                "symptom_summary": "Forehead laceration after fall.",
                "triage_hint": "possible trauma evaluation",
                "estimated_severity": 6,
                "true_severity": 7,
                "preferred_specialization": "trauma",
                "arrival_time": 1,
                "emergency_flag": False,
                "service_time": 2,
            },
            {
                "patient_id": "p4",
                "symptoms": ["abdominal pain", "nausea"],
                "symptom_summary": "Progressive abdominal pain without bleeding.",
                "triage_hint": "general or gastro evaluation",
                "estimated_severity": 5,
                "true_severity": 5,
                "preferred_specialization": "general",
                "arrival_time": 2,
                "emergency_flag": False,
                "service_time": 2,
            },
        ],
    ),
    "task_2_queue_optimization": TaskConfig(
        task_id="task_2_queue_optimization",
        name="Queue Optimization",
        description="Reduce waiting time while keeping specialists busy and avoiding overload.",
        max_steps=12,
        deterioration_wait_threshold=3,
        emergency_delay_threshold=2,
        fatigue_recovery_per_step=0.05,
        doctors=[
            {"doctor_id": "dr_general_1", "specialization": "general", "status": "available", "capacity": 2},
            {"doctor_id": "dr_general_2", "specialization": "general", "status": "available", "capacity": 2},
            {"doctor_id": "dr_cardio_1", "specialization": "cardiology", "status": "available", "capacity": 1},
            {"doctor_id": "dr_trauma_1", "specialization": "trauma", "status": "available", "capacity": 1},
        ],
        patients=[
            {
                "patient_id": "p1",
                "symptoms": ["palpitations", "fatigue"],
                "symptom_summary": "New palpitations with lightheadedness.",
                "triage_hint": "cardiac review recommended",
                "estimated_severity": 7,
                "true_severity": 7,
                "preferred_specialization": "cardiology",
                "arrival_time": 0,
                "emergency_flag": False,
                "service_time": 2,
            },
            {
                "patient_id": "p2",
                "symptoms": ["cough", "fever"],
                "symptom_summary": "Persistent cough and moderate fever.",
                "triage_hint": "general consultation",
                "estimated_severity": 4,
                "true_severity": 4,
                "preferred_specialization": "general",
                "arrival_time": 0,
                "emergency_flag": False,
                "service_time": 1,
            },
            {
                "patient_id": "p3",
                "symptoms": ["sprained ankle", "swelling"],
                "symptom_summary": "Unable to bear weight comfortably.",
                "triage_hint": "trauma review helpful",
                "estimated_severity": 5,
                "true_severity": 5,
                "preferred_specialization": "trauma",
                "arrival_time": 1,
                "emergency_flag": False,
                "service_time": 2,
            },
            {
                "patient_id": "p4",
                "symptoms": ["migraine", "vomiting"],
                "symptom_summary": "Severe headache, no neuro deficits reported.",
                "triage_hint": "general high-priority consult",
                "estimated_severity": 6,
                "true_severity": 6,
                "preferred_specialization": "general",
                "arrival_time": 1,
                "emergency_flag": False,
                "service_time": 2,
            },
            {
                "patient_id": "p5",
                "symptoms": ["chest tightness", "sweating"],
                "symptom_summary": "Recurring chest tightness in diabetic patient.",
                "triage_hint": "cardiology priority",
                "estimated_severity": 8,
                "true_severity": 8,
                "preferred_specialization": "cardiology",
                "arrival_time": 2,
                "emergency_flag": True,
                "service_time": 2,
            },
            {
                "patient_id": "p6",
                "symptoms": ["rash", "itching"],
                "symptom_summary": "Widespread rash without airway compromise.",
                "triage_hint": "general consultation",
                "estimated_severity": 2,
                "true_severity": 2,
                "preferred_specialization": "general",
                "arrival_time": 3,
                "emergency_flag": False,
                "service_time": 1,
            },
        ],
    ),
    "task_3_emergency_handling": TaskConfig(
        task_id="task_3_emergency_handling",
        name="Emergency Handling",
        description="Detect emergencies quickly, escalate when specialists are off-duty, and balance conflicting urgent cases.",
        max_steps=14,
        deterioration_wait_threshold=2,
        emergency_delay_threshold=1,
        fatigue_recovery_per_step=0.04,
        doctors=[
            {"doctor_id": "dr_general_1", "specialization": "general", "status": "available", "capacity": 2},
            {"doctor_id": "dr_trauma_1", "specialization": "trauma", "status": "busy", "capacity": 1},
            {"doctor_id": "dr_cardio_1", "specialization": "cardiology", "status": "off-duty", "capacity": 1},
            {"doctor_id": "dr_emergency_1", "specialization": "emergency", "status": "available", "capacity": 1},
        ],
        patients=[
            {
                "patient_id": "p1",
                "symptoms": ["crushing chest pain", "sweating", "confusion"],
                "symptom_summary": "Probable acute coronary syndrome on arrival.",
                "triage_hint": "immediate specialist attention",
                "estimated_severity": 8,
                "true_severity": 10,
                "preferred_specialization": "cardiology",
                "arrival_time": 0,
                "emergency_flag": True,
                "service_time": 3,
            },
            {
                "patient_id": "p2",
                "symptoms": ["deep cut", "heavy bleeding"],
                "symptom_summary": "Uncontrolled bleeding from machinery injury.",
                "triage_hint": "trauma emergency",
                "estimated_severity": 9,
                "true_severity": 9,
                "preferred_specialization": "trauma",
                "arrival_time": 1,
                "emergency_flag": True,
                "service_time": 3,
            },
            {
                "patient_id": "p3",
                "symptoms": ["high fever", "dehydration"],
                "symptom_summary": "Stable but needs prompt review.",
                "triage_hint": "general consult",
                "estimated_severity": 5,
                "true_severity": 5,
                "preferred_specialization": "general",
                "arrival_time": 1,
                "emergency_flag": False,
                "service_time": 2,
            },
            {
                "patient_id": "p4",
                "symptoms": ["wheezing", "labored breathing"],
                "symptom_summary": "Respiratory distress worsening in lobby.",
                "triage_hint": "airway emergency",
                "estimated_severity": 7,
                "true_severity": 9,
                "preferred_specialization": "emergency",
                "arrival_time": 2,
                "emergency_flag": True,
                "service_time": 2,
            },
        ],
    ),
}


class HospitalTriageEnvironment:
    def __init__(self) -> None:
        self.rng = Random(7)
        self.reset()

    def reset(self, task_id: str = "task_1_basic_triage", seed: int = 7) -> HospitalObservation:
        if task_id not in TASKS:
            available = ", ".join(sorted(TASKS))
            raise ValueError(f"Unknown task_id '{task_id}'. Available tasks: {available}")

        task = TASKS[task_id]
        self.rng = Random(seed)
        self.task = task
        self.seed = seed
        self.time_step = 0
        self.max_steps = task.max_steps
        self.done = False
        self.patients: dict[str, dict[str, Any]] = {p["patient_id"]: deepcopy(p) for p in task.patients}
        self.doctors: dict[str, dict[str, Any]] = {d["doctor_id"]: deepcopy(d) for d in task.doctors}
        for patient in self.patients.values():
            patient["waiting_time"] = 0
            patient["status"] = "waiting"
            patient["assigned_doctor_id"] = None
            patient["escalated"] = False
            patient["manual_queue_rank"] = None
            patient["emergency_marked_by_agent"] = bool(patient["emergency_flag"])
            patient["emergency_mark_count"] = 0
            patient["last_action_time"] = None
        for doctor in self.doctors.values():
            doctor["current_load"] = 0
            doctor["fatigue"] = 0.0
        self.active_assignments: list[dict[str, Any]] = []
        self.completed_patients: list[str] = []
        self.redirected_patients: list[str] = []
        self.escalation_count = 0
        self.assignment_count = 0
        self.invalid_action_count = 0
        self.history: list[dict[str, Any]] = []
        self.event_log: list[dict[str, Any]] = []
        self.cumulative_reward = 0.0
        self._record_event("reset", {"task_id": task_id, "seed": seed})
        return self._observation()

    def state(self) -> dict[str, Any]:
        return {
            "task": {
                "task_id": self.task.task_id,
                "name": self.task.name,
                "description": self.task.description,
            },
            "seed": self.seed,
            "time_step": self.time_step,
            "max_steps": self.max_steps,
            "done": self.done,
            "patients": deepcopy(list(self.patients.values())),
            "doctors": deepcopy(list(self.doctors.values())),
            "active_assignments": deepcopy(self.active_assignments),
            "history": deepcopy(self.history),
            "event_log": deepcopy(self.event_log),
        }

    def step(self, action: HospitalAction) -> tuple[HospitalObservation, HospitalReward, bool, dict[str, Any]]:
        if self.done:
            task_score = self._task_score()
            safe_total = self.normalize_score(self.cumulative_reward if self.cumulative_reward > 0 else task_score["overall"])
            reward = HospitalReward(
                value=safe_total,
                total=safe_total,
                components={"episode_done": 0.0},
            )
            return self._observation(), reward, True, {
                "message": "Episode already finished.",
                "action_applied": action.model_dump(),
                "action_valid": False,
                "error": {
                    "code": "episode_done",
                    "message": "Episode already finished.",
                    "recoverable": False,
                },
                "metrics": self._metrics().model_dump(),
                "task_score": task_score["overall"],
                "score_breakdown": task_score,
                "raw_reward": {"value": 0.0, "total": 0.0, "components": {"episode_done": 0.0}},
                "debug": self._debug_snapshot(),
            }

        components = {
            "assignment": 0.0,
            "priority_bonus": 0.0,
            "emergency_bonus": 0.0,
            "escalation_bonus": 0.0,
            "waiting_penalty": 0.0,
            "overload_penalty": 0.0,
            "idle_penalty": 0.0,
            "routing_penalty": 0.0,
            "delay_penalty": 0.0,
            "redirect_adjustment": 0.0,
            "completion_bonus": 0.0,
            "fatigue_penalty": 0.0,
            "invalid_action_penalty": 0.0,
            "system_pressure_penalty": 0.0,
        }
        info: dict[str, Any] = {
            "action_applied": action.model_dump(),
            "action_valid": True,
            "error": None,
        }

        if action.action_type == "assign":
            self._apply_assignment(action, components, info)
        elif action.action_type == "mark_emergency":
            self._mark_emergency(action, components, info)
        elif action.action_type == "reorder_queue":
            self._reorder_queue(action, components, info)
        elif action.action_type == "escalate_emergency":
            self._escalate_emergency(action, components, info)
        elif action.action_type == "redirect":
            self._redirect_patient(action, components, info)
        elif action.action_type == "wait":
            info["message"] = "No direct intervention this step."
            self._record_event("wait", {"note": action.note})

        self._advance_time(components)
        raw_reward_value = round(sum(components.values()), 3)
        # Keep per-step rewards positive and bounded so any cumulative scoring logic
        # used by external evaluators remains strictly within the open interval.
        reward_value = self.normalize_score(
            self.normalize_score(raw_reward_value) / max(2, self.max_steps + 1)
        )
        self.cumulative_reward = self.normalize_score(self.cumulative_reward + reward_value)
        reward = HospitalReward(value=reward_value, total=self.cumulative_reward, components=components)
        self.done = self._check_done()
        info["metrics"] = self._metrics().model_dump()
        task_score = self._task_score()
        info["task_score"] = task_score["overall"]
        info["score_breakdown"] = task_score
        info["raw_reward"] = {"value": raw_reward_value, "total": raw_reward_value, "components": components}
        info["reward"] = reward.model_dump()
        info["reward_breakdown"] = reward.components
        info["debug"] = self._debug_snapshot()
        self.history.append(
            {
                "time_step": self.time_step,
                "action": action.model_dump(),
                "reward": reward.model_dump(),
                "done": self.done,
                "message": info.get("message", ""),
                "action_valid": info["action_valid"],
            }
        )
        return self._observation(), reward, self.done, info

    def _available_patients(self) -> list[dict[str, Any]]:
        patients = [
            p
            for p in self.patients.values()
            if p["status"] == "waiting" and p["arrival_time"] <= self.time_step
        ]
        patients.sort(
            key=lambda p: (
                p.get("manual_queue_rank") is None,
                p.get("manual_queue_rank", 999),
                not p["emergency_marked_by_agent"],
                -p["true_severity"],
                -p["waiting_time"],
                p["arrival_time"],
                p["patient_id"],
            )
        )
        return patients

    def _apply_assignment(self, action: HospitalAction, components: dict[str, float], info: dict[str, Any]) -> None:
        patient = self.patients.get(action.patient_id or "")
        doctor = self.doctors.get(action.doctor_id or "")
        if not patient or not doctor:
            self._mark_invalid_action(
                components,
                info,
                "Invalid patient or doctor reference.",
                penalty=2.0,
                code="invalid_reference",
            )
            return
        if patient["status"] != "waiting" or patient["arrival_time"] > self.time_step:
            self._mark_invalid_action(
                components,
                info,
                "Patient unavailable for assignment.",
                penalty=1.5,
                code="patient_unavailable",
            )
            return
        if doctor["status"] == "off-duty":
            self._mark_invalid_action(
                components,
                info,
                "Doctor is off-duty; escalate first.",
                penalty=3.0,
                code="doctor_off_duty",
            )
            return
        if doctor["current_load"] >= doctor["capacity"]:
            self._mark_invalid_action(
                components,
                info,
                "Doctor is overloaded.",
                penalty=3.5,
                code="doctor_overloaded",
            )
            return

        doctor["status"] = "busy"
        doctor["current_load"] += 1
        doctor["fatigue"] = min(1.0, round(doctor["fatigue"] + 0.15, 3))
        patient["status"] = "assigned"
        patient["assigned_doctor_id"] = doctor["doctor_id"]
        patient["last_action_time"] = self.time_step
        assignment = {
            "patient_id": patient["patient_id"],
            "doctor_id": doctor["doctor_id"],
            "remaining_service_time": patient["service_time"],
        }
        self.active_assignments.append(assignment)
        self.assignment_count += 1

        preferred = patient["preferred_specialization"]
        if doctor["specialization"] == preferred:
            components["assignment"] += 4.0
        elif doctor["specialization"] == "emergency" and patient["emergency_marked_by_agent"]:
            components["assignment"] += 3.0
        elif doctor["specialization"] == "general" and patient["true_severity"] <= 5:
            components["assignment"] += 1.5
        else:
            components["routing_penalty"] -= 2.5

        if patient["true_severity"] >= 8:
            components["priority_bonus"] += 2.0
        elif patient["true_severity"] >= 6:
            components["priority_bonus"] += 1.0
        if doctor["fatigue"] >= 0.85:
            components["fatigue_penalty"] -= 0.75
        info["message"] = f"Assigned {patient['patient_id']} to {doctor['doctor_id']}."
        self._record_event(
            "assign",
            {
                "patient_id": patient["patient_id"],
                "doctor_id": doctor["doctor_id"],
                "doctor_specialization": doctor["specialization"],
            },
        )

    def _mark_emergency(self, action: HospitalAction, components: dict[str, float], info: dict[str, Any]) -> None:
        patient = self.patients.get(action.patient_id or "")
        if not patient:
            self._mark_invalid_action(
                components,
                info,
                "Unknown patient for emergency mark.",
                penalty=1.0,
                code="invalid_patient_id",
            )
            return
        if patient["arrival_time"] > self.time_step or patient["status"] != "waiting":
            self._mark_invalid_action(
                components,
                info,
                "Patient unavailable for emergency marking.",
                penalty=1.0,
                code="patient_unavailable",
            )
            return
        already_marked = patient["emergency_marked_by_agent"]
        patient["emergency_mark_count"] += 1
        patient["emergency_marked_by_agent"] = True
        patient["last_action_time"] = self.time_step
        if already_marked:
            components["invalid_action_penalty"] -= 0.25
            info["message"] = f"Emergency was already flagged for {patient['patient_id']}."
        elif patient["true_severity"] >= 8 or patient["emergency_flag"]:
            components["emergency_bonus"] += 3.0
            info["message"] = f"Emergency correctly flagged for {patient['patient_id']}."
        else:
            components["routing_penalty"] -= 1.5
            info["message"] = f"False emergency flag for {patient['patient_id']}."
        self._record_event("mark_emergency", {"patient_id": patient["patient_id"], "already_marked": already_marked})

    def _reorder_queue(self, action: HospitalAction, components: dict[str, float], info: dict[str, Any]) -> None:
        available = self._available_patients()
        patient_ids = [patient["patient_id"] for patient in available]
        if (action.patient_id or "") not in patient_ids or action.queue_position is None:
            self._mark_invalid_action(
                components,
                info,
                "Invalid reorder request.",
                penalty=1.0,
                code="invalid_queue_update",
            )
            return
        patient_ids.remove(action.patient_id)
        new_position = min(action.queue_position, len(patient_ids))
        patient_ids.insert(new_position, action.patient_id)
        for rank, patient_id in enumerate(patient_ids):
            self.patients[patient_id]["manual_queue_rank"] = rank
        patient = self.patients[action.patient_id]
        patient["last_action_time"] = self.time_step
        if patient["true_severity"] >= 8 and new_position == 0:
            components["priority_bonus"] += 1.5
        elif patient["true_severity"] <= 3 and new_position > 0:
            components["priority_bonus"] += 0.5
        else:
            components["delay_penalty"] -= 0.5
        info["message"] = f"Moved {action.patient_id} to queue position {new_position}."
        self._record_event(
            "reorder_queue",
            {"patient_id": action.patient_id, "queue_position": new_position, "queue": patient_ids},
        )

    def _escalate_emergency(self, action: HospitalAction, components: dict[str, float], info: dict[str, Any]) -> None:
        patient = self.patients.get(action.patient_id or "")
        if not patient:
            self._mark_invalid_action(
                components,
                info,
                "Unknown patient for escalation.",
                penalty=1.5,
                code="invalid_patient_id",
            )
            return
        if patient["arrival_time"] > self.time_step or patient["status"] != "waiting":
            self._mark_invalid_action(
                components,
                info,
                "Patient unavailable for escalation.",
                penalty=1.0,
                code="patient_unavailable",
            )
            return
        if not patient["emergency_marked_by_agent"] and patient["true_severity"] < 8:
            self._mark_invalid_action(
                components,
                info,
                "Escalation is only useful for emergency patients.",
                penalty=1.5,
                code="impossible_action",
            )
            return
        if patient["escalated"]:
            components["invalid_action_penalty"] -= 0.5
            info["message"] = f"Patient {patient['patient_id']} was already escalated."
            self._record_event("escalate_duplicate", {"patient_id": patient["patient_id"]})
            return
        patient["escalated"] = True
        patient["last_action_time"] = self.time_step
        self.escalation_count += 1
        for doctor in self.doctors.values():
            if doctor["specialization"] == patient["preferred_specialization"] and doctor["status"] == "off-duty":
                doctor["status"] = "available"
                components["escalation_bonus"] += 4.0
                info["message"] = f"Escalated {patient['patient_id']} and recalled {doctor['doctor_id']}."
                self._record_event(
                    "escalate_emergency",
                    {"patient_id": patient["patient_id"], "doctor_recalled": doctor["doctor_id"]},
                )
                return
        components["escalation_bonus"] += 1.0
        info["message"] = f"Escalation logged for {patient['patient_id']}, no off-duty specialist recalled."
        self._record_event("escalate_emergency", {"patient_id": patient["patient_id"], "doctor_recalled": None})

    def _redirect_patient(self, action: HospitalAction, components: dict[str, float], info: dict[str, Any]) -> None:
        patient = self.patients.get(action.patient_id or "")
        if not patient or patient["status"] != "waiting" or patient["arrival_time"] > self.time_step:
            self._mark_invalid_action(
                components,
                info,
                "Invalid redirect request.",
                penalty=1.0,
                code="patient_unavailable",
            )
            return
        patient["status"] = "redirected"
        patient["last_action_time"] = self.time_step
        self.redirected_patients.append(patient["patient_id"])
        if patient["true_severity"] <= 2:
            components["redirect_adjustment"] += 0.5
        elif patient["true_severity"] <= 4 and not patient["emergency_marked_by_agent"]:
            components["redirect_adjustment"] -= 0.5
        else:
            components["redirect_adjustment"] -= 3.0
        info["message"] = f"Redirected {patient['patient_id']}."
        self._record_event("redirect", {"patient_id": patient["patient_id"]})

    def _advance_time(self, components: dict[str, float]) -> None:
        self.time_step += 1

        completed_now: list[dict[str, Any]] = []
        for assignment in self.active_assignments:
            assignment["remaining_service_time"] -= 1
            if assignment["remaining_service_time"] <= 0:
                completed_now.append(assignment)

        for assignment in completed_now:
            self.active_assignments.remove(assignment)
            patient = self.patients[assignment["patient_id"]]
            doctor = self.doctors[assignment["doctor_id"]]
            patient["status"] = "completed"
            self.completed_patients.append(patient["patient_id"])
            doctor["current_load"] = max(0, doctor["current_load"] - 1)
            if doctor["current_load"] == 0:
                doctor["status"] = "available"
            components["completion_bonus"] += 2.0
            self._record_event(
                "patient_completed",
                {"patient_id": patient["patient_id"], "doctor_id": doctor["doctor_id"]},
            )

        for doctor in self.doctors.values():
            if doctor["current_load"] == 0 and doctor["status"] != "off-duty":
                doctor["fatigue"] = max(0.0, round(doctor["fatigue"] - self.task.fatigue_recovery_per_step, 3))

        for patient in self.patients.values():
            if patient["status"] != "waiting":
                continue
            if patient["arrival_time"] <= self.time_step:
                patient["waiting_time"] += 1
                if (
                    patient["waiting_time"] >= self.task.deterioration_wait_threshold
                    and patient["true_severity"] >= 7
                    and (patient["waiting_time"] - self.task.deterioration_wait_threshold) % 2 == 0
                ):
                    patient["true_severity"] = min(10, patient["true_severity"] + self.task.deterioration_step)
                    patient["estimated_severity"] = min(10, patient["estimated_severity"] + 1)
                    patient["emergency_flag"] = patient["true_severity"] >= 8
                    self._record_event(
                        "patient_deteriorated",
                        {
                            "patient_id": patient["patient_id"],
                            "true_severity": patient["true_severity"],
                            "estimated_severity": patient["estimated_severity"],
                        },
                    )
                penalty = 0.15 * patient["waiting_time"]
                if patient["true_severity"] >= 8:
                    penalty += 1.0
                elif patient["true_severity"] >= 6:
                    penalty += 0.35
                components["waiting_penalty"] -= round(penalty, 3)
                if (
                    patient["true_severity"] >= 8
                    and not patient["escalated"]
                    and patient["waiting_time"] >= self.task.emergency_delay_threshold
                ):
                    components["delay_penalty"] -= 1.5

        waiting_patients = [p for p in self._available_patients() if p["status"] == "waiting"]
        available_doctors = [
            d
            for d in self.doctors.values()
            if d["status"] == "available" and d["current_load"] < d["capacity"]
        ]
        if waiting_patients and available_doctors:
            components["idle_penalty"] -= round(0.5 * len(available_doctors), 3)
        if waiting_patients and not available_doctors:
            pressure = round(0.35 * len(waiting_patients), 3)
            if any(patient["emergency_marked_by_agent"] for patient in waiting_patients):
                pressure += 0.65
            components["system_pressure_penalty"] -= pressure
        overloaded_doctors = [doctor for doctor in self.doctors.values() if doctor["current_load"] >= doctor["capacity"]]
        if overloaded_doctors:
            components["overload_penalty"] -= round(0.25 * len(overloaded_doctors), 3)

    def _check_done(self) -> bool:
        if self.time_step >= self.max_steps:
            return True
        return all(
            patient["status"] in {"completed", "redirected"}
            or patient["arrival_time"] > self.time_step
            for patient in self.patients.values()
        )

    def _metrics(self) -> MetricsSnapshot:
        arrived_patients = [p for p in self.patients.values() if p["arrival_time"] <= self.time_step]
        waits = [patient["waiting_time"] for patient in arrived_patients]
        total_capacity = sum(doctor["capacity"] for doctor in self.doctors.values())
        total_load = sum(doctor["current_load"] for doctor in self.doctors.values())
        pending_emergencies = len(
            [
                p
                for p in self.patients.values()
                if p["status"] == "waiting" and p["arrival_time"] <= self.time_step and p["emergency_marked_by_agent"]
            ]
        )
        return MetricsSnapshot(
            assigned_count=self.assignment_count,
            completed_count=len(self.completed_patients),
            redirected_count=len(self.redirected_patients),
            escalation_count=self.escalation_count,
            pending_emergencies=pending_emergencies,
            avg_wait_time=round(sum(waits) / len(waits), 3) if waits else 0.0,
            utilization=round(total_load / total_capacity, 3) if total_capacity else 0.0,
        )

    @staticmethod
    def normalize_score(score: float) -> float:
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.5
        return clamp_score(score)

    def _normalize_score_map(self, scores: dict[str, Any]) -> dict[str, float]:
        return {key: self.normalize_score(value) for key, value in scores.items()}

    def _task_score(self) -> dict[str, float]:
        metrics = self._metrics()

        completion_ratio = self.normalize_score(
            len(self.completed_patients) / len(self.patients) if self.patients else 0.5
        )
        emergency_cases = [p for p in self.patients.values() if p["true_severity"] >= 8 or p["emergency_flag"]]
        resolved_emergencies = [
            p
            for p in emergency_cases
            if p["status"] == "completed" and p["waiting_time"] <= max(2, self.task.emergency_delay_threshold + 1)
        ]
        emergency_score = self.normalize_score(len(resolved_emergencies) / len(emergency_cases) if emergency_cases else 1.0)
        wait_score = self.normalize_score(max(0.0, 1.0 - (metrics.avg_wait_time / 6.0)))
        utilization_score = self.normalize_score(min(1.0, metrics.utilization + 0.25))
        safety_score = self.normalize_score(
            max(
                0.0,
                1.0 - ((self.invalid_action_count * 0.08) + (len(self.redirected_patients) * 0.03)),
            )
        )
        overall = self.normalize_score(
            0.35 * completion_ratio
            + 0.25 * emergency_score
            + 0.2 * wait_score
            + 0.1 * utilization_score
            + 0.1 * safety_score
        )
        return self._normalize_score_map({
            "overall": overall,
            "completion_ratio": completion_ratio,
            "emergency_score": emergency_score,
            "wait_score": wait_score,
            "utilization_score": utilization_score,
            "safety_score": safety_score,
        })

    def _queue(self) -> list[str]:
        return [patient["patient_id"] for patient in self._available_patients()]

    def _observation(self) -> HospitalObservation:
        visible_patients = [
            VisiblePatient(
                patient_id=patient["patient_id"],
                symptoms=patient["symptoms"],
                symptom_summary=patient["symptom_summary"],
                triage_hint=patient["triage_hint"],
                estimated_severity=patient["estimated_severity"],
                waiting_time=patient["waiting_time"],
                arrival_time=patient["arrival_time"],
                emergency_flag=patient["emergency_marked_by_agent"],
                assigned_doctor_id=patient["assigned_doctor_id"],
                status=patient["status"],
            )
            for patient in self.patients.values()
            if patient["arrival_time"] <= self.time_step or patient["status"] != "waiting"
        ]
        visible_doctors = [
            VisibleDoctor(
                doctor_id=doctor["doctor_id"],
                specialization=doctor["specialization"],
                status=doctor["status"],
                capacity=doctor["capacity"],
                current_load=doctor["current_load"],
                fatigue=doctor["fatigue"],
            )
            for doctor in self.doctors.values()
        ]
        assignments = [AssignmentRecord(**assignment) for assignment in self.active_assignments]
        return HospitalObservation(
            task_id=self.task.task_id,
            task_name=self.task.name,
            time_step=self.time_step,
            max_steps=self.max_steps,
            patients=visible_patients,
            doctors=visible_doctors,
            queue=self._queue(),
            active_assignments=assignments,
            metrics=self._metrics(),
        )

    def _debug_snapshot(self) -> dict[str, Any]:
        waiting = [
            patient["patient_id"]
            for patient in self.patients.values()
            if patient["status"] == "waiting" and patient["arrival_time"] <= self.time_step
        ]
        available_doctors = [
            doctor["doctor_id"]
            for doctor in self.doctors.values()
            if doctor["status"] == "available" and doctor["current_load"] < doctor["capacity"]
        ]
        return {
            "seed": self.seed,
            "time_step": self.time_step,
            "waiting_patients": waiting,
            "available_doctors": available_doctors,
            "system_overloaded": bool(waiting and not available_doctors),
            "active_assignments": deepcopy(self.active_assignments),
            "invalid_action_count": self.invalid_action_count,
            "recent_events": deepcopy(self.event_log[-5:]),
        }

    def _mark_invalid_action(
        self,
        components: dict[str, float],
        info: dict[str, Any],
        message: str,
        penalty: float = 1.0,
        code: str = "invalid_action",
    ) -> None:
        self.invalid_action_count += 1
        components["invalid_action_penalty"] -= penalty
        info["action_valid"] = False
        info["message"] = message
        info["error"] = {
            "code": code,
            "message": message,
            "recoverable": True,
        }
        self._record_event("invalid_action", {"message": message, "penalty": penalty, "code": code})

    def _record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.event_log.append(
            {
                "time_step": self.time_step,
                "event_type": event_type,
                "payload": deepcopy(payload),
            }
        )
