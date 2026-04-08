"""HTTP client for the hospital triage environment."""

from __future__ import annotations

import time
from typing import Any

import requests
from requests import Response

from models import HospitalAction, ResetRequest, StepResponse


class HospitalTriageEnv:
    def __init__(self, base_url: str, session_id: str = "default", max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.max_retries = max_retries
        self.session = requests.Session()

    def reset(self, task_id: str = "task_1_basic_triage", seed: int = 7) -> dict[str, Any]:
        payload = ResetRequest(task_id=task_id, seed=seed, session_id=self.session_id).model_dump()
        response = self._request("POST", "/reset", json=payload)
        return response.json()

    def step(self, action: HospitalAction) -> StepResponse:
        payload = action.model_dump()
        payload["session_id"] = self.session_id
        response = self._request("POST", "/step", json=payload)
        return StepResponse.model_validate(response.json())

    def state(self) -> dict[str, Any]:
        response = self._request("GET", "/state", params={"session_id": self.session_id})
        return response.json()

    def health(self) -> dict[str, Any]:
        response = self._request("GET", "/health")
        return response.json()

    def tasks(self) -> dict[str, Any]:
        try:
            response = self._request("GET", "/tasks")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                response = self._request("GET", "/")
            else:
                raise
        return response.json()

    def close(self) -> None:
        self.session.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Response:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(method, f"{self.base_url}{path}", timeout=30, **kwargs)
                response.raise_for_status()
                return response
            except requests.ConnectionError as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(0.25 * (2**attempt))
            except requests.HTTPError:
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Request failed for {method} {path}")
