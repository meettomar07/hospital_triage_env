"""Typed models for the hospital triage environment."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    Action = BaseModel
    Observation = BaseModel


class VisiblePatient(BaseModel):
    patient_id: str
    symptoms: list[str]
    symptom_summary: str
    triage_hint: str
    estimated_severity: int = Field(ge=1, le=10)
    waiting_time: int = Field(ge=0)
    arrival_time: int = Field(ge=0)
    emergency_flag: bool = False
    assigned_doctor_id: str | None = None
    status: Literal["waiting", "assigned", "completed", "redirected"] = "waiting"


class VisibleDoctor(BaseModel):
    doctor_id: str
    specialization: str
    status: Literal["available", "busy", "off-duty"]
    capacity: int = Field(ge=1)
    current_load: int = Field(ge=0)
    fatigue: float = Field(ge=0.0, le=1.0)


class AssignmentRecord(BaseModel):
    patient_id: str
    doctor_id: str
    remaining_service_time: int = Field(ge=0)


class MetricsSnapshot(BaseModel):
    assigned_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    redirected_count: int = Field(ge=0)
    escalation_count: int = Field(ge=0)
    pending_emergencies: int = Field(ge=0)
    avg_wait_time: float = Field(ge=0.0)
    utilization: float = Field(ge=0.0, le=1.0)


class HospitalObservation(Observation):
    task_id: str
    task_name: str
    time_step: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    patients: list[VisiblePatient]
    doctors: list[VisibleDoctor]
    queue: list[str]
    active_assignments: list[AssignmentRecord]
    metrics: MetricsSnapshot


class HospitalAction(Action):
    action_type: Literal[
        "assign",
        "mark_emergency",
        "reorder_queue",
        "escalate_emergency",
        "redirect",
        "wait",
    ]
    patient_id: str | None = None
    doctor_id: str | None = None
    queue_position: int | None = Field(default=None, ge=0)
    note: str | None = None
    session_id: str = Field(default="default", min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_action_fields(self) -> "HospitalAction":
        if self.action_type == "assign":
            if not self.patient_id or not self.doctor_id:
                raise ValueError("assign actions require both patient_id and doctor_id")
        elif self.action_type in {"mark_emergency", "escalate_emergency", "redirect"}:
            if not self.patient_id:
                raise ValueError(f"{self.action_type} actions require patient_id")
        elif self.action_type == "reorder_queue":
            if not self.patient_id or self.queue_position is None:
                raise ValueError("reorder_queue actions require patient_id and queue_position")
        return self


class HospitalReward(BaseModel):
    value: float
    total: float
    components: dict[str, float] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task_id: str = "task_1_basic_triage"
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    session_id: str = Field(default="default", min_length=1, max_length=128)


class StepRequest(HospitalAction):
    action_type: Literal[
        "assign",
        "mark_emergency",
        "reorder_queue",
        "escalate_emergency",
        "redirect",
        "wait",
    ] = "wait"
    session_id: str = Field(default="default", min_length=1, max_length=128)


class StepResponse(BaseModel):
    observation: HospitalObservation
    reward: HospitalReward
    done: bool
    info: dict[str, Any]
