---
title: Hospital Triage Openenv
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: OpenEnv hospital triage RL environment
---
# Hospital Triage OpenEnv

## Overview

Hospital Triage OpenEnv is a production-style reinforcement learning environment for hospital intake and front-desk coordination. The agent acts as an AI receptionist: it reviews arriving patients, prioritizes urgent cases, assigns doctors, manages the waiting queue, and escalates emergencies when the right specialist is unavailable.

This project is built with FastAPI and exposed through an OpenEnv-compatible interface:

- `POST /reset`
- `POST /step`
- `GET /state`

It includes three benchmark tasks, dense reward shaping, deterministic seeded resets, and a baseline inference runner that works with OpenAI-compatible APIs or a deterministic heuristic fallback.

## Why This Matters

Hospital intake is a high-stakes decision problem with limited resources, changing urgency, and incomplete information. A good policy must do more than classify symptoms. It must coordinate people, timing, and constraints under pressure.

This environment is designed to evaluate that kind of operational reasoning:

- partial observability through estimated rather than fully revealed severity
- dynamic patient deterioration over time
- specialist routing and fallback behavior
- emergency escalation when doctors are off-duty
- queue management under competing priorities
- step-wise dense reward instead of sparse terminal-only feedback

## Architecture

The system is intentionally simple to run and easy to inspect.

```text
                +----------------------+
                |  Inference Runner    |
                |  inference.py        |
                +----------+-----------+
                           |
                           | HTTP
                           v
                +----------------------+
                |  FastAPI Server      |
                |  server/app.py       |
                +----------+-----------+
                           |
                           | environment calls
                           v
                +----------------------+
                |  RL Environment      |
                |  hospital_environment|
                +----------+-----------+
                           |
          +----------------+----------------+
          |                                 |
          v                                 v
 +--------------------+          +--------------------+
 |  Typed Models      |          |  HTTP Client       |
 |  models.py         |          |  client.py         |
 +--------------------+          +--------------------+
```

Core responsibilities:

- `server/app.py`: FastAPI entrypoint and OpenEnv-facing routes
- `server/hospital_environment.py`: task definitions, transitions, rewards, metrics, and debug info
- `models.py`: typed observations, actions, rewards, and request/response models
- `client.py`: simple HTTP client for environment interaction
- `inference.py`: baseline policy runner and evaluation logger

## Environment Design

The environment simulates a hospital reception desk coordinating patients and doctors over discrete time steps.

### Patients

Each patient includes:

- `patient_id`
- symptoms and triage summary
- `estimated_severity` in observations
- hidden `true_severity` in internal state
- `arrival_time`
- `waiting_time`
- emergency status
- preferred specialization
- service duration

### Doctors

Each doctor includes:

- `doctor_id`
- specialization such as `general`, `cardiology`, `trauma`, or `emergency`
- `status` of `available`, `busy`, or `off-duty`
- `capacity`
- `current_load`
- fatigue estimate

### Dynamics

- Time advances every step.
- Waiting patients accumulate penalties.
- Severe patients deteriorate if left waiting.
- Queue order can be changed manually.
- Emergency cases may require escalation to recall off-duty specialists.
- Busy doctors finish service after a fixed number of steps.
- Invalid actions are tracked explicitly for safer training and better debugging.

## Action Space

The agent can choose one of:

- `assign`: assign a patient to a doctor
- `mark_emergency`: explicitly flag a patient as emergency
- `reorder_queue`: move a patient within the waiting queue
- `escalate_emergency`: recall an off-duty specialist or log escalation
- `redirect`: send an appropriate low-severity patient elsewhere
- `wait`: take no direct action for one step

Example action:

```json
{
  "action_type": "assign",
  "patient_id": "p1",
  "doctor_id": "dr_cardio_1",
  "note": "Chest pain with high severity."
}
```

## Observation Space

Each observation contains:

- `task_id`
- `task_name`
- `time_step`
- `max_steps`
- visible patients with symptoms, waiting time, estimated severity, emergency flag, and status
- visible doctors with specialization, status, capacity, load, and fatigue
- current queue ordering
- active assignments
- metrics snapshot including wait time, utilization, escalations, and pending emergencies

The full hidden environment state can be inspected through `GET /state` for grading and debugging.

## Tasks

### Task 1: Basic Triage

Focus:

- correct specialist matching
- early prioritization of high-severity patients
- safe handling of obvious urgent cases

### Task 2: Queue Optimization

Focus:

- minimizing wait time
- keeping doctors productively utilized
- avoiding unnecessary idle capacity or overload

### Task 3: Emergency Handling

Focus:

- emergency response under resource constraints
- escalation when specialists are unavailable
- balancing urgent and non-urgent patient flow

The three tasks form a clean progression:

1. Learn correct routing.
2. Learn queue and utilization tradeoffs.
3. Learn escalation and multi-constraint emergency handling.

## Reward Design

The reward is dense and step-wise to make the environment trainable and diagnostic.

Positive components include:

- correct specialist assignment
- prioritizing severe patients
- correct emergency detection
- effective emergency escalation
- patient completion

Negative components include:

- waiting penalties that grow over time
- delayed handling of severe emergencies
- misrouting patients
- invalid actions
- overloading doctors
- leaving available doctors idle while patients wait
- inappropriate redirects
- fatigue-related pressure in overloaded workflows

This produces a more stable training signal than sparse terminal reward alone.

## Sample Results

Representative heuristic baseline results with seeded evaluation:

| Task | Description | Steps | Total Reward | Overall Score |
|---|---|---:|---:|---:|
| `task_1_basic_triage` | Basic specialist matching and prioritization | 5 | 20.35 | 87.50 |
| `task_2_queue_optimization` | Throughput and queue efficiency | 6 | 27.05 | 85.83 |
| `task_3_emergency_handling` | Escalation and urgent-case handling | 6 | 13.85 | 76.67 |

These results come from the included deterministic baseline flow and show stable end-to-end execution across all three tasks.

## OpenEnv Compliance

This project is designed to satisfy the standard OpenEnv interaction contract.

Checklist:

- [x] FastAPI app entrypoint is exposed in `openenv.yaml`
- [x] `reset(task_id, seed)` returns the initial observation
- [x] `step(action)` returns observation, reward, done, and info
- [x] `state()` exposes the full internal state for debugging and grading
- [x] observations are typed and serializable
- [x] actions are typed and validated
- [x] seeded resets support deterministic replay
- [x] app is container-ready for deployment

OpenEnv manifest:

```yaml
name: hospital_triage_env

entrypoint:
  module: server.app
  callable: app
```

## Project Structure

```text
hospital_triage_env/
+-- client.py
+-- inference.py
+-- models.py
+-- openenv.yaml
+-- pyproject.toml
+-- README.md
+-- Dockerfile
+-- outputs/
¦   +-- evals/
¦   +-- logs/
+-- server/
    +-- app.py
    +-- hospital_environment.py
```

## Setup & Running

### Local development

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Core dependency note:

- `openenv-core>=0.2.0` is included in `pyproject.toml`
- `uv.lock` can be regenerated with `uv lock`

Health check:

```bash
curl http://127.0.0.1:7860/health
```

Expected response:

```json
{"status":"ok"}
```

### API surface

- `GET /` returns environment metadata and available tasks
- `GET /health` returns service health
- `POST /reset` starts a seeded episode for a given task
- `POST /step` applies one environment action
- `GET /state` returns the full internal state

## Inference

Run the baseline evaluation:

```bash
python inference.py
```

By default the runner expects the environment at:

```bash
ENV_BASE_URL=http://127.0.0.1:7860
```

Optional environment variables:

```bash
export API_BASE_URL=https://your-openai-compatible-endpoint.example/v1
export ENV_BASE_URL=http://127.0.0.1:7860
export MODEL_NAME=gpt-4.1-mini
export HF_TOKEN=your_hf_token
export OPENAI_API_KEY=your_api_key
```

Inference behavior:

- uses an OpenAI-compatible responses API when `API_BASE_URL` is set
- falls back to a deterministic heuristic policy when model output is unavailable or invalid
- writes step traces to `outputs/logs/`
- writes evaluation summaries to `outputs/evals/summary.json`
- emits `[START]`, `[STEP]`, and `[END]` logs for easy inspection

## Deployment (Hugging Face Spaces)

This repository is ready for Docker-based Hugging Face Spaces deployment.

Steps:

1. Create a new Docker Space.
2. Upload this repository.
3. Ensure the Space exposes port `7860`.
4. Launch the app with `uvicorn server.app:app`.

The included `Dockerfile` installs the package and serves the FastAPI app on port `7860`.

## Reproducibility

The project is designed to be repeatable and judge-friendly.

- seeded resets produce deterministic task initialization
- fixed benchmark tasks keep evaluation comparable
- heuristic fallback is deterministic
- traces are written to disk for auditability
- summary scores are exported in machine-readable JSON
- `state()` exposes full internals for debugging and grading

## Future Improvements

- per-session episode isolation instead of one shared in-memory environment
- procedurally generated task variations to reduce overfitting
- richer doctor shift constraints and handoff delays
- more nuanced patient deterioration and uncertainty modeling
- benchmark harnesses for multi-seed aggregate scoring
- learned policies or fine-tuned controllers on top of the existing environment contract

