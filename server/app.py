"""FastAPI entrypoint for the hospital triage environment."""

from __future__ import annotations

from threading import Lock

import uvicorn
from fastapi import FastAPI, HTTPException, Query

from models import HospitalAction, ResetRequest, StepResponse
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(request: ResetRequest) -> dict[str, object]:
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
def step(action: HospitalAction) -> StepResponse:
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


def run() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)
