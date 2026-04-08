"""Regression tests for the hospital triage environment."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient
from pydantic import ValidationError

from inference import normalize_score, normalize_task_score
from models import HospitalAction
from server.app import app
from server.hospital_environment import HospitalTriageEnvironment


class HospitalEnvironmentTests(unittest.TestCase):
    def test_reset_is_deterministic_for_same_seed(self) -> None:
        env_a = HospitalTriageEnvironment()
        env_b = HospitalTriageEnvironment()

        first = env_a.reset(task_id="task_2_queue_optimization", seed=17).model_dump()
        second = env_b.reset(task_id="task_2_queue_optimization", seed=17).model_dump()

        self.assertEqual(first, second)

    def test_assign_requires_patient_and_doctor(self) -> None:
        with self.assertRaises(ValidationError):
            HospitalAction(action_type="assign", patient_id="p1")

    def test_invalid_escalation_is_reported_in_info(self) -> None:
        env = HospitalTriageEnvironment()
        env.reset(task_id="task_1_basic_triage", seed=7)

        observation, reward, done, info = env.step(
            HospitalAction(action_type="escalate_emergency", patient_id="p2")
        )

        self.assertFalse(done)
        self.assertFalse(info["action_valid"])
        self.assertLess(reward.components["invalid_action_penalty"], 0.0)
        self.assertIn("Escalation is only useful", info["message"])
        self.assertIn("debug", info)
        self.assertEqual(reward.value, reward.total)
        self.assertIn("reward_breakdown", info)
        self.assertEqual(observation.task_id, "task_1_basic_triage")

    def test_task_scores_are_strictly_within_open_interval(self) -> None:
        env = HospitalTriageEnvironment()
        env.reset(task_id="task_1_basic_triage", seed=7)

        initial_scores = env._task_score()
        for value in initial_scores.values():
            self.assertGreater(value, 0.0)
            self.assertLess(value, 1.0)

        _, _, _, info = env.step(HospitalAction(action_type="wait"))
        self.assertIsInstance(info["task_score"], float)
        self.assertGreater(info["task_score"], 0.0)
        self.assertLess(info["task_score"], 1.0)
        self.assertGreater(info["reward"]["value"], 0.0)
        self.assertLess(info["reward"]["value"], 1.0)
        self.assertGreater(info["reward"]["total"], 0.0)
        self.assertLess(info["reward"]["total"], 1.0)
        self.assertIn("score_breakdown", info)
        for value in info["score_breakdown"].values():
            self.assertGreater(value, 0.0)
            self.assertLess(value, 1.0)


class ScoreNormalizationTests(unittest.TestCase):
    def test_normalize_score_clamps_to_open_interval(self) -> None:
        self.assertGreater(normalize_score(0.0), 0.0)
        self.assertLess(normalize_score(0.0), 1.0)
        self.assertGreater(normalize_score(1.0), 0.0)
        self.assertLess(normalize_score(1.0), 1.0)

    def test_normalize_task_score_clamps_every_field(self) -> None:
        normalized = normalize_task_score({"overall": 0.0, "wait_score": 1.0, "safety_score": "bad"})

        for value in normalized.values():
            self.assertGreater(value, 0.0)
            self.assertLess(value, 1.0)


class HospitalApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()

    def test_invalid_task_returns_422(self) -> None:
        response = self.client.post("/reset", json={"task_id": "missing_task", "seed": 1})

        self.assertEqual(response.status_code, 422)
        payload = response.json()
        self.assertIn("available_tasks", payload["detail"])

    def test_state_includes_seed_and_event_log(self) -> None:
        self.client.post("/reset", json={"task_id": "task_1_basic_triage", "seed": 11})
        response = self.client.get("/state")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["seed"], 11)
        self.assertTrue(payload["event_log"])

    def test_sessions_are_isolated(self) -> None:
        self.client.post("/reset", json={"task_id": "task_1_basic_triage", "seed": 11, "session_id": "alpha"})
        self.client.post("/reset", json={"task_id": "task_2_queue_optimization", "seed": 17, "session_id": "beta"})

        alpha_state = self.client.get("/state", params={"session_id": "alpha"})
        beta_state = self.client.get("/state", params={"session_id": "beta"})

        self.assertEqual(alpha_state.status_code, 200)
        self.assertEqual(beta_state.status_code, 200)
        self.assertEqual(alpha_state.json()["task"]["task_id"], "task_1_basic_triage")
        self.assertEqual(beta_state.json()["task"]["task_id"], "task_2_queue_optimization")


if __name__ == "__main__":
    unittest.main()
