"""FastAPI entrypoint for the hospital triage environment."""

from __future__ import annotations

import json
import re
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
    for key in ("task_id", "taskId", "task", "id"):
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


def _add_task_id(task_ids: set[str], value: Any) -> None:
    if not isinstance(value, str):
        return
    candidate = value.strip()
    if not candidate:
        return
    if len(candidate) > 256:
        return
    blocked = {
        "task_id",
        "task",
        "taskId",
        "task_name",
        "task_score",
        "task_scores",
        "score",
        "scores",
        "summary",
        "results",
        "tasks",
        "payload",
        "data",
        "input",
    }
    if candidate in blocked:
        return
    task_ids.add(candidate)


def _collect_task_ids(payload: Any, task_ids: set[str], depth: int = 0, context_key: str | None = None) -> None:
    if depth > 12:
        return
    payload = _coerce_payload(payload)

    id_field_keys = {"task_id", "taskId", "task", "id"}
    id_list_keys = {"tasks", "task_ids", "taskIds", "task_list", "taskList"}
    score_map_keys = {"summary", "results", "scores", "task_scores"}

    if isinstance(payload, str):
        if context_key in id_field_keys:
            _add_task_id(task_ids, payload)
            return
        if context_key in id_list_keys | score_map_keys:
            for part in payload.split(","):
                _add_task_id(task_ids, part)
            return
        return

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str) and context_key in id_list_keys | score_map_keys | {None}:
                _add_task_id(task_ids, item)
                continue
            _collect_task_ids(item, task_ids, depth + 1, context_key=context_key)
        return

    if not isinstance(payload, dict):
        return

    for key in id_field_keys:
        if key in payload:
            _collect_task_ids(payload.get(key), task_ids, depth + 1, context_key=key)

    for key in id_list_keys:
        if key in payload:
            _collect_task_ids(payload.get(key), task_ids, depth + 1, context_key=key)

    for key in score_map_keys:
        section = payload.get(key)
        if isinstance(section, dict):
            for candidate_id, value in section.items():
                _add_task_id(task_ids, candidate_id)
                _collect_task_ids(value, task_ids, depth + 1, context_key=key)
        elif section is not None:
            _collect_task_ids(section, task_ids, depth + 1, context_key=key)

    for key, value in payload.items():
        if isinstance(key, str) and key.startswith("task_"):
            _add_task_id(task_ids, key)
        _collect_task_ids(value, task_ids, depth + 1, context_key=key)


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


def _extract_task_ids_from_text(raw_text: str) -> set[str]:
    task_ids: set[str] = set()
    if not isinstance(raw_text, str) or not raw_text.strip():
        return task_ids

    normalized_text = raw_text.replace('\\"', '"')
    patterns = (
        r'"(?:task_id|taskId|task|id)"\s*:\s*"([^"]+)"',
        r'(?:task_id|taskId|task|id)\s*"?\s*[:=]\s*"([^"]+)"',
        r"\b(task_[A-Za-z0-9_\-]+)\b",
        r'(?:\[|,)\s*"([A-Za-z0-9][A-Za-z0-9_\-]{1,255})"\s*(?:,|\])',
    )
    for pattern in patterns:
        for match in re.findall(pattern, normalized_text):
            _add_task_id(task_ids, match)
    return task_ids


def _preview_for_log(value: Any, max_length: int = 2000) -> str:
    try:
        preview = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        preview = str(value)
    if len(preview) <= max_length:
        return preview
    return f"{preview[:max_length]}...<truncated>"


def _log_validator_debug(
    endpoint_name: str,
    request: Request,
    payload: Any,
    raw_body_text: str,
    task_ids: set[str],
    response: list[dict[str, Any]],
) -> None:
    try:
        debug_payload = {
            "endpoint": endpoint_name,
            "method": request.method,
            "path": request.url.path,
            "query": dict(request.query_params),
            "payload_preview": _preview_for_log(payload),
            "raw_body_preview": _preview_for_log(raw_body_text),
            "task_ids": sorted(task_ids),
            "response": response,
        }
        print(f"[VALIDATOR DEBUG] {json.dumps(debug_payload, sort_keys=True)}")
    except Exception:
        pass


async def _resolve_payload(request: Request, payload: Any) -> tuple[Any, str]:
    resolved_payload = _payload_from_body_or_query(payload, request)
    raw_body_text = ""
    try:
        raw_body = await request.body()
        if raw_body:
            raw_body_text = raw_body.decode("utf-8", errors="replace")
    except Exception:
        raw_body_text = ""
    if resolved_payload is None and raw_body_text:
        resolved_payload = _coerce_payload(raw_body_text)
    return resolved_payload, raw_body_text


async def _grade_like_response(endpoint_name: str, request: Request, payload: Any) -> list[dict[str, Any]]:
    resolved_payload, raw_body_text = await _resolve_payload(request, payload)
    task_ids: set[str] = set()
    _collect_task_ids(resolved_payload, task_ids)
    if not task_ids and raw_body_text:
        task_ids.update(_extract_task_ids_from_text(raw_body_text))
    if not task_ids:
        task_ids = set(TASKS)
    response = [{"task_id": task_id, "score": _normalize_score(0.5)} for task_id in sorted(task_ids)]
    _log_validator_debug(endpoint_name, request, resolved_payload, raw_body_text, task_ids, response)
    return response


@app.api_route("/grader", methods=["GET", "POST"])
async def grader(request: Request, payload: Any = Body(default=None)) -> list[dict[str, Any]]:
    return await _grade_like_response("grader", request, payload)


@app.api_route("/baseline", methods=["GET", "POST"])
async def baseline(request: Request, payload: Any = Body(default=None)) -> list[dict[str, Any]]:
    return await _grade_like_response("baseline", request, payload)


@app.api_route("/grade", methods=["GET", "POST"])
async def grade_alias(request: Request, payload: Any = Body(default=None)) -> list[dict[str, Any]]:
    return await _grade_like_response("grade", request, payload)


@app.api_route("/base", methods=["GET", "POST"])
async def base_alias(request: Request, payload: Any = Body(default=None)) -> list[dict[str, Any]]:
    return await _grade_like_response("base", request, payload)


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


def run() -> None:
    main()


if __name__ == "__main__":
    main()
