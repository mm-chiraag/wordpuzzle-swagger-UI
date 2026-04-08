"""
inference.py — WordPuzzle LLM Baseline Agent
Uses OpenAI client with structured stdout logs: [START] [STEP] [END]
Required env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN
"""
import os
import json
import time
import requests
import sys

from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-3.5-turbo")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",       API_BASE_URL)

TASKS = ["wordpuzzle-easy", "wordpuzzle-medium", "wordpuzzle-hard"]

client = OpenAI(
    api_key=HF_TOKEN or "dummy",
    base_url=API_BASE_URL if API_BASE_URL.endswith("/v1") else API_BASE_URL + "/v1"
    if "openai" not in API_BASE_URL else API_BASE_URL,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def env_reset(task_id: str, session_id: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset",
                      json={"task_id": task_id, "session_id": session_id},
                      timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: str, session_id: str) -> dict:
    r = requests.post(f"{ENV_URL}/step",
                      json={"action": action, "session_id": session_id},
                      timeout=30)
    r.raise_for_status()
    return r.json()


def build_prompt(observation: dict, task_id: str) -> str:
    word_len_map = {"wordpuzzle-easy": 4, "wordpuzzle-medium": 5, "wordpuzzle-hard": 6}
    word_len = word_len_map.get(task_id, 5)
    feedback = observation.get("feedback", [])
    attempts_used = observation.get("attempts_used", 0)
    max_attempts  = observation.get("max_attempts", 6)

    feedback_str = "\n".join(f"  Attempt {i+1}: {f}" for i, f in enumerate(feedback)) if feedback else "  (no guesses yet)"
    return f"""You are playing WordPuzzle, a Wordle-style game.
Guess a {word_len}-letter word. You have {max_attempts - attempts_used} attempt(s) remaining.

Feedback legend: G=correct position, Y=wrong position, X=not in word.

Previous guesses and feedback:
{feedback_str}

Reply with ONLY a single {word_len}-letter word (lowercase, no punctuation, nothing else).
Your guess:"""


def get_llm_guess(prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip().lower()
        # keep only alpha chars
        return "".join(c for c in raw if c.isalpha())
    except Exception as e:
        print(f"[LLM ERROR] {e}", file=sys.stderr)
        return "crane"   # fallback


# ─── Main loop ────────────────────────────────────────────────────────────────
def run_task(task_id: str) -> dict:
    session_id = f"inference-{task_id}-{int(time.time())}"

    result = env_reset(task_id, session_id)
    obs    = result.get("observation", {})

    total_reward = 0.0
    steps        = 0
    done         = False
    solved       = False

    print(json.dumps({
        "event":   "[START]",
        "task_id": task_id,
        "session_id": session_id,
        "max_attempts": obs.get("max_attempts"),
    }))

    while not done:
        prompt = build_prompt(obs, task_id)
        guess  = get_llm_guess(prompt)

        if not guess:
            guess = "crane"

        step_result   = env_step(guess, session_id)
        obs           = step_result.get("observation", {})
        reward        = step_result.get("reward", 0.0)
        done          = step_result.get("done", False)
        total_reward += reward
        steps        += 1
        solved        = obs.get("solved", False)

        print(json.dumps({
            "event":        "[STEP]",
            "task_id":      task_id,
            "step":         steps,
            "action":       guess,
            "reward":       reward,
            "total_reward": total_reward,
            "done":         done,
            "solved":       solved,
            "feedback":     obs.get("feedback", []),
            "message":      obs.get("message", ""),
        }))

    score = min(1.0, total_reward)

    print(json.dumps({
        "event":        "[END]",
        "task_id":      task_id,
        "session_id":   session_id,
        "total_steps":  steps,
        "total_reward": total_reward,
        "score":        score,
        "solved":       solved,
        "revealed_word": obs.get("revealed_word"),
    }))

    return {"task_id": task_id, "score": score, "solved": solved, "steps": steps}


def main():
    print(f"[INFO] ENV_URL={ENV_URL}  MODEL={MODEL_NAME}", file=sys.stderr)
    results = []
    for task_id in TASKS:
        try:
            r = run_task(task_id)
            results.append(r)
        except Exception as e:
            print(f"[ERROR] {task_id}: {e}", file=sys.stderr)
            results.append({"task_id": task_id, "score": 0.0, "solved": False, "error": str(e)})

    avg = sum(r["score"] for r in results) / len(results)
    print(json.dumps({
        "event":   "[SUMMARY]",
        "results": results,
        "average_score": round(avg, 4),
    }))


if __name__ == "__main__":
    main()
