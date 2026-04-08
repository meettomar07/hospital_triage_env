"""FastAPI entrypoint for the hospital triage environment."""

from __future__ import annotations

import json
from threading import Lock
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request

from models import ResetRequest, StepRequest, StepResponse
from server.hospital_environment import HospitalTriageEnvironment, TASKS, clamp_score

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
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.5
    return clamp_score(parsed)


def _extract_score(item: dict[str, Any]) -> float:
    for key in ("score", "final_score", "task_score", "overall", "value"):
        if key in item:
            value = item.get(key)
            if isinstance(value, dict):
                for nested_key in ("overall", "score", "final_score", "task_score", "value"):
                    if nested_key in value:
                        return _normalize_score(value.get(nested_key))
                for nested_value in value.values():
                    return _normalize_score(nested_value)
                return _normalize_score(0.5)
            return _normalize_score(value)
    return _normalize_score(0.5)


def _coerce_payload(payload: Any) -> Any:
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8")
        except Exception:
            return {}
    if not isinstance(payload, str):
        return payload
    text = payload.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return payload


def _extract_task_id(item: dict[str, Any]) -> str:
    for key in ("task_id", "taskId", "task"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _collect_score_entries(payload: Any, score_by_task: dict[str, float], depth: int = 0) -> None:
    if depth > 10:
        return
    payload = _coerce_payload(payload)
    if isinstance(payload, list):
        for item in payload:
            _collect_score_entries(item, score_by_task, depth + 1)
        return
    if not isinstance(payload, dict):
        return

    task_id = _extract_task_id(payload)
    if task_id:
        score_by_task[task_id] = _extract_score(payload)

    for key, value in payload.items():
        if (
            isinstance(key, str)
            and key.startswith("task_")
            and key not in {"task_id", "task_score", "task_name"}
        ):
            score_by_task[key] = _extract_score(value) if isinstance(value, dict) else _normalize_score(value)
        _collect_score_entries(value, score_by_task, depth + 1)


def _collect_task_ids(payload: Any, task_ids: set[str], depth: int = 0) -> None:
    if depth > 10:
        return
    payload = _coerce_payload(payload)
    if isinstance(payload, str):
        candidate = payload.strip()
        if candidate:
            task_ids.add(candidate)
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                candidate = item.strip()
                if candidate:
                    task_ids.add(candidate)
                continue
            _collect_task_ids(item, task_ids, depth + 1)
        return
    if not isinstance(payload, dict):
        return

    task_id = _extract_task_id(payload)
    if task_id:
        task_ids.add(task_id)

    for key in ("tasks", "summary", "results", "scores", "task_scores"):
        section = payload.get(key)
        if isinstance(section, dict):
            for candidate_id, value in section.items():
                if isinstance(candidate_id, str):
                    normalized = candidate_id.strip()
                    if normalized:
                        task_ids.add(normalized)
                _collect_task_ids(value, task_ids, depth + 1)
        elif section is not None:
            _collect_task_ids(section, task_ids, depth + 1)

    for key, value in payload.items():
        if (
            isinstance(key, str)
            and key.startswith("task_")
            and key not in {"task_id", "task_score", "task_scores", "task_name"}
        ):
            task_ids.add(key)
        if (
            isinstance(key, str)
            and key not in {"task_id", "task_score", "task_name", "score", "final_score", "overall", "value"}
            and isinstance(value, dict)
            and any(metric_key in value for metric_key in ("score", "final_score", "task_score", "overall", "value"))
        ):
            task_ids.add(key.strip())
        _collect_task_ids(value, task_ids, depth + 1)


def _payload_from_body_or_query(payload: Any, request: Request) -> Any:
    if payload is not None:
        return payload
    task_id = request.query_params.get("task_id") or request.query_params.get("task")
    if task_id:
        return {"task_id": task_id}
    task_ids = request.query_params.get("task_ids")
    if task_ids:
        return {"tasks": [{"task_id": item.strip()} for item in task_ids.split(",") if item.strip()]}
    for key in ("payload", "data", "input", "summary", "scores", "task_scores", "results", "tasks"):
        query_value = request.query_params.get(key)
        if query_value:
            return query_value
    return None


@app.api_route("/grader", methods=["GET", "POST"])
def grader(request: Request, payload: Any = Body(default=None)) -> list[dict[str, Any]]:
    payload = _payload_from_body_or_query(payload, request)
    task_ids: set[str] = set()
    _collect_task_ids(payload, task_ids)
    if not task_ids:
        task_ids = set(TASKS)
    return [{"task_id": task_id, "score": _normalize_score(0.5)} for task_id in sorted(task_ids)]


@app.api_route("/baseline", methods=["GET", "POST"])
def baseline(request: Request, payload: Any = Body(default=None)) -> list[dict[str, Any]]:
    payload = _payload_from_body_or_query(payload, request)
    task_ids: set[str] = set()
    _collect_task_ids(payload, task_ids)
    if not task_ids:
        task_ids = set(TASKS)
    return [{"task_id": task_id, "score": _normalize_score(0.5)} for task_id in sorted(task_ids)]


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


def run() -> None:
    main()


if __name__ == "__main__":
    main()
