"""
app.py — FastAPI server for WordPuzzle OpenEnv
Endpoints: /reset  /step  /state  /grader  /tasks  /health
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from run_standalone import (
    WordPuzzleEnvironment,
    WordPuzzleAction,
    LEVEL_CONFIG,
    WORDS,
)

app = FastAPI(title="WordPuzzle OpenEnv")

# ─── Session storage ──────────────────────────────────────────────────────────
env_sessions: dict = {}

TASK_LEVEL_MAP = {
    "wordpuzzle-easy":   1,
    "wordpuzzle-medium": 2,
    "wordpuzzle-hard":   3,
}


# ─── Request models ───────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: Optional[str] = "wordpuzzle-easy"
    session_id: Optional[str] = "default"


class StepRequest(BaseModel):
    action: str
    session_id: Optional[str] = "default"


class GraderRequest(BaseModel):
    task_id: str
    guess: str
    target: Optional[str] = None   # if omitted, grader tests against a fresh env


# ─── Helper ───────────────────────────────────────────────────────────────────
def _obs_dict(obs, reward=0.0, done=False):
    return {
        "observation": {
            "feedback":      obs.feedback,
            "attempts_used": obs.attempts_used,
            "max_attempts":  obs.max_attempts,
            "task_level":    obs.task_level,
            "message":       obs.message,
            "solved":        obs.solved,
            "revealed_word": obs.revealed_word,
        },
        "reward": reward,
        "done":   done,
    }


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html><head><meta charset="utf-8"><title>WordPuzzle OpenEnv</title>
    <style>
      body{background:#0d1117;color:#c9d1d9;font-family:monospace;padding:40px;max-width:800px;margin:0 auto;}
      h1{color:#58a6ff;}
      .badge{background:#238636;color:#fff;padding:4px 12px;border-radius:20px;margin:4px;display:inline-block;}
      pre{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;}
    </style></head>
    <body>
    <h1>WordPuzzle — OpenEnv RL Environment</h1>
    <p>A Wordle-style mini-game reinforcement learning environment.</p>
    <span class="badge">Runtime Correctness</span>
    <span class="badge">Interface Compliance</span>
    <span class="badge">Task Design</span>
    <span class="badge">Grading Logic</span>
    <pre>
Tasks     : wordpuzzle-easy | wordpuzzle-medium | wordpuzzle-hard
Interface : /reset  /step  /state  /grader  /tasks  /health
Reward    : 0.0 – 1.0 (partial credit + solve bonus)
    </pre>
    <p>See <a href="/docs" style="color:#58a6ff;">/docs</a> for full API reference.</p>
    </body></html>
    """


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    """List all tasks with grader info — required by the validator."""
    tasks = []
    for task_id, level in TASK_LEVEL_MAP.items():
        cfg = LEVEL_CONFIG[level]
        tasks.append({
            "id":          task_id,
            "name":        f"WordPuzzle {cfg['name']}",
            "description": f"Guess a {cfg['word_length']}-letter word in {cfg['max_attempts']} attempts.",
            "word_length": cfg["word_length"],
            "max_attempts":cfg["max_attempts"],
            "grader": {
                "endpoint":  "/grader",
                "field":     "score",
                "range":     [0.0, 1.0],
            },
            "action_schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": f"{cfg['word_length']}-letter word guess"}
                },
                "required": ["action"],
            },
        })
    return {"tasks": tasks}


@app.post("/reset")
def reset(req: ResetRequest):
    task_id    = req.task_id or "wordpuzzle-easy"
    session_id = req.session_id or "default"
    level      = TASK_LEVEL_MAP.get(task_id, 1)

    env = WordPuzzleEnvironment(task_level=level)
    obs = env.reset()
    env_sessions[session_id] = env

    result = _obs_dict(obs)
    result["task_id"] = task_id
    return result


@app.post("/step")
def step(req: StepRequest):
    session_id = req.session_id or "default"

    if session_id not in env_sessions:
        env = WordPuzzleEnvironment(task_level=1)
        env.reset()
        env_sessions[session_id] = env

    env    = env_sessions[session_id]
    action = WordPuzzleAction(guess=req.action)

    try:
        obs, reward, done = env.step(action)
    except Exception as e:
        return JSONResponse(status_code=200, content={"error": str(e), "done": True, "reward": 0.0})

    if done:
        env_sessions.pop(session_id, None)

    return _obs_dict(obs, reward, done)


@app.get("/state")
def state(session_id: str = "default"):
    if session_id not in env_sessions:
        return {"error": "No active session"}
    env = env_sessions[session_id]
    s   = env.state()
    return {
        "state": {
            "target_word":  s.target_word,
            "guesses":      s.guesses,
            "task_level":   s.task_level,
            "total_reward": s.total_reward,
            "done":         s.done,
        }
    }


@app.post("/grader")
def grader(req: GraderRequest):
    """
    Score a single guess for a given task.
    Returns score in [0.0, 1.0].
    The validator calls this endpoint to verify grader existence and range.
    """
    level = TASK_LEVEL_MAP.get(req.task_id, 1)
    env   = WordPuzzleEnvironment(task_level=level)

    # If caller provides a target, use it; otherwise pick one for grading
    if req.target:
        env._target_word = req.target.lower().strip()
    else:
        env.reset()

    guess = req.guess.lower().strip()

    # Validate length
    if len(guess) != env.word_length:
        return JSONResponse(status_code=200, content={
            "score":   0.0,
            "valid":   False,
            "reason":  f"Guess must be {env.word_length} letters (got {len(guess)}).",
            "task_id": req.task_id,
        })

    score   = env._compute_reward(guess, env._target_word, attempt_num=1)
    solved  = (guess == env._target_word)
    feedback = env._compute_feedback(guess, env._target_word)

    return {
        "score":    score,          # always in [0.0, 1.0]
        "valid":    True,
        "solved":   solved,
        "feedback": feedback,
        "task_id":  req.task_id,
        "guess":    guess,
    }


def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
