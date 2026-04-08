"""FastAPI entrypoint for the hospital triage environment."""

from __future__ import annotations

from threading import Lock
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query

from models import ResetRequest, StepRequest, StepResponse
from server.hospital_environment import HospitalTriageEnvironment, TASKS

app = FastAPI(title="Hospital Triage OpenEnv", version="0.1.0")
environment_lock = Lock()
session_environments: dict[str, HospitalTriageEnvironment] = {"default": HospitalTriageEnvironment()}


def get_environment(session_id: str) -> HospitalTriageEnvironment:
    environment = session_environments.get(session_id)
    if environment is None:
        environment = HospitalTriageEnvironment()
        session_environments[session_id] = environment
    return environment


@app.get("/")
def root() -> dict[str, object]:
    return {
        "name": "hospital_triage_env",
        "tasks": list(TASKS.keys()),
        "methods": ["reset", "step", "state"],
        "multi_session": True,
    }


@app.get("/tasks")
def tasks() -> dict[str, object]:
    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "name": task.name,
                "description": task.description,
                "max_steps": task.max_steps,
            }
            for task in TASKS.values()
        ]
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(request: ResetRequest | None = None) -> dict[str, object]:
    request = request or ResetRequest()
    if request.task_id not in TASKS:
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"Unknown task_id '{request.task_id}'.",
                "available_tasks": sorted(TASKS),
            },
        )

    with environment_lock:
        observation = get_environment(request.session_id).reset(task_id=request.task_id, seed=request.seed)
        return observation.model_dump()


@app.post("/step")
def step(action: StepRequest | None = None) -> StepResponse:
    action = action or StepRequest()
    with environment_lock:
        environment = session_environments.get(action.session_id)
        if environment is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": f"Unknown session_id '{action.session_id}'. Call /reset first.",
                    "session_id": action.session_id,
                },
            )
        observation, reward, done, info = environment.step(action)
        info["session_id"] = action.session_id
        return StepResponse(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )


@app.get("/state")
def state(session_id: str = Query(default="default", min_length=1, max_length=128)) -> dict[str, object]:
    with environment_lock:
        environment = session_environments.get(session_id)
        if environment is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": f"Unknown session_id '{session_id}'. Call /reset first.",
                    "session_id": session_id,
                },
            )
        payload = environment.state()
        payload["session_id"] = session_id
        return payload


def _normalize_score(value: Any) -> float:
    epsilon = 1e-4
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.5
    return max(epsilon, min(1 - epsilon, round(parsed, 4)))


def _extract_score(item: dict[str, Any]) -> float:
    for key in ("score", "final_score", "task_score", "overall"):
        if key in item:
            return _normalize_score(item.get(key))
    return _normalize_score(0.5)


@app.post("/grader")
def grader(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    entries = payload.get("summary")
    if not isinstance(entries, list):
        entries = payload.get("task_scores")
    if not isinstance(entries, list):
        entries = []

    score_by_task: dict[str, float] = {task_id: _normalize_score(0.5) for task_id in TASKS}
    for item in entries:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id", ""))
        if task_id in score_by_task:
            score_by_task[task_id] = _extract_score(item)

    overall = _normalize_score(sum(score_by_task.values()) / max(1, len(score_by_task)))
    return {
        "task_scores": [{"task_id": task_id, "score": score} for task_id, score in score_by_task.items()],
        "overall_score": overall,
    }


@app.post("/baseline")
def baseline() -> dict[str, Any]:
    task_scores = [{"task_id": task_id, "score": _normalize_score(0.5)} for task_id in TASKS]
    return {
        "task_scores": task_scores,
        "overall_score": _normalize_score(0.5),
    }


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


def run() -> None:
    main()


if __name__ == "__main__":
    main()
