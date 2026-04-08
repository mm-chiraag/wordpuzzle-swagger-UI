---
title: WordPuzzle OpenEnv
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: mit
short_description: Wordle-style mini-game RL environment for LLM agents
---

# WordPuzzle — OpenEnv RL Environment

A Wordle-style mini-game reinforcement learning environment built on the OpenEnv framework.
The agent guesses a hidden word using letter-position feedback (G/Y/X), receiving rewards in [0.0, 1.0].

---

## Tasks

| Task ID | Word Length | Max Attempts | Difficulty |
|---|---|---|---|
| wordpuzzle-easy | 4 letters | 6 attempts | Easy |
| wordpuzzle-medium | 5 letters | 5 attempts | Medium |
| wordpuzzle-hard | 6 letters | 4 attempts | Hard |

---

## Action Space

| Field | Type | Description |
|---|---|---|
| action | string | The word guess (must match task word length) |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| feedback | list[str] | Per-guess feedback strings (G=correct, Y=wrong pos, X=absent) |
| attempts_used | int | Number of guesses made so far |
| max_attempts | int | Total allowed guesses for this task |
| task_level | int | 1=Easy, 2=Medium, 3=Hard |
| message | str | Human-readable status |
| solved | bool | True if word was correctly guessed |
| revealed_word | str or null | The target word (only shown when episode ends) |

---

## Reward Function

| Condition | Reward |
|---|---|
| Correct letter, correct position (G) | +1.0 / word_length |
| Correct letter, wrong position (Y) | +0.3 / word_length |
| Solve bonus (any attempt) | 0.5 + 0.5 × (remaining / max_attempts) |
| No match | 0.0 |

All rewards are capped to **[0.0, 1.0]**.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| /reset | POST | Start a new episode |
| /step | POST | Submit a guess |
| /state | GET | Get current episode state |
| /grader | POST | Score a guess (used by validator) |
| /tasks | GET | List all 3 tasks with grader info |
| /health | GET | Health check |
| /docs | GET | Swagger API docs |

---

## Quick Start (local)
bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "wordpuzzle-easy", "session_id": "test"}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "fire", "session_id": "test"}'

# Grader
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{"task_id": "wordpuzzle-easy", "guess": "ball", "target": "ball"}'

## Docker
bash
docker build -t wordpuzzle-env .
docker run -p 7860:7860 wordpuzzle-env

## Run Inference
bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-3.5-turbo
export HF_TOKEN=your_key
export ENV_URL=http://localhost:7860

python inference.py