"""
run_standalone.py — Core WordPuzzle RL environment logic
Self-contained; no OpenEnv dependency needed at runtime.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import random

# ─── Word banks per difficulty ────────────────────────────────────────────────
WORDS = {
    1: ["ball", "cake", "dark", "east", "fire", "gold", "help", "iron",
        "jump", "kite", "lamp", "mind", "nose", "open", "park", "quiz",
        "rain", "sail", "tree", "unit", "vine", "wake", "xray", "yard", "zero"],
    2: ["adobe", "brave", "cabin", "draft", "eagle", "flame", "grace", "haven",
        "input", "joker", "kneel", "lemon", "maple", "night", "olive", "plaza",
        "queen", "ridge", "stone", "tiger", "umbra", "vivid", "water", "xenon", "yacht"],
    3: ["bridge", "canyon", "crayon", "dragon", "engine", "fabric", "garden",
        "harbor", "meadow", "nectar", "output", "pillar", "quartz", "rabbit",
        "sunset", "temple", "vacuum", "walnut", "yellow", "zipper"],
}

# ─── Difficulty config ────────────────────────────────────────────────────────
LEVEL_CONFIG = {
    1: {"word_length": 4, "max_attempts": 6, "name": "Easy"},
    2: {"word_length": 5, "max_attempts": 5, "name": "Medium"},
    3: {"word_length": 6, "max_attempts": 4, "name": "Hard"},
}


# ─── Models ───────────────────────────────────────────────────────────────────
@dataclass
class WordPuzzleAction:
    guess: str


@dataclass
class WordPuzzleObservation:
    feedback: List[str]
    attempts_used: int
    max_attempts: int
    task_level: int
    message: str
    solved: bool
    revealed_word: Optional[str] = None


@dataclass
class WordPuzzleState:
    target_word: str
    guesses: List[str]
    task_level: int
    total_reward: float
    done: bool


# ─── Environment ──────────────────────────────────────────────────────────────
class WordPuzzleEnvironment:
    def __init__(self, task_level: int = 1):
        assert task_level in LEVEL_CONFIG, "task_level must be 1, 2, or 3"
        self.task_level = task_level
        cfg = LEVEL_CONFIG[task_level]
        self.word_length: int = cfg["word_length"]
        self.max_attempts: int = cfg["max_attempts"]

        self._target_word: str = ""
        self._guesses: List[str] = []
        self._feedback: List[str] = []
        self._done: bool = False
        self._solved: bool = False
        self._total_reward: float = 0.0

    def _pick_word(self) -> str:
        return random.choice(WORDS[self.task_level])

    def _compute_feedback(self, guess: str, target: str) -> str:
        result = []
        target_chars = list(target)
        marks = [None] * len(guess)

        for i, (g, t) in enumerate(zip(guess, target)):
            if g == t:
                marks[i] = "G"
                target_chars[i] = None

        for i, g in enumerate(guess):
            if marks[i] is not None:
                continue
            if g in target_chars:
                marks[i] = "Y"
                target_chars[target_chars.index(g)] = None
            else:
                marks[i] = "X"

        return "".join(marks)

    def _compute_reward(self, guess: str, target: str, attempt_num: int) -> float:
        if guess == target:
            remaining = self.max_attempts - attempt_num
            bonus = 0.5 + 0.5 * (remaining / self.max_attempts)
            return round(min(1.0, bonus), 4)

        target_chars = list(target)
        score = 0.0
        for i, g in enumerate(guess):
            if i < len(target) and g == target[i]:
                score += 1.0 / self.word_length
                target_chars[i] = None
            elif g in target_chars:
                score += 0.3 / self.word_length
                idx = target_chars.index(g)
                target_chars[idx] = None

        return round(min(0.9, score), 4)

    def reset(self) -> WordPuzzleObservation:
        self._target_word = self._pick_word()
        self._guesses = []
        self._feedback = []
        self._done = False
        self._solved = False
        self._total_reward = 0.0
        return WordPuzzleObservation(
            feedback=[],
            attempts_used=0,
            max_attempts=self.max_attempts,
            task_level=self.task_level,
            message=f"Guess the {self.word_length}-letter word! You have {self.max_attempts} attempts.",
            solved=False,
        )

    def step(self, action: WordPuzzleAction) -> Tuple[WordPuzzleObservation, float, bool]:
        if self._done:
            raise ValueError("Episode is done. Call reset() first.")

        guess = action.guess.lower().strip()

        if len(guess) != self.word_length:
            raise ValueError(f"Guess must be {self.word_length} letters. Got '{guess}'.")
        if not guess.isalpha():
            raise ValueError(f"Guess must contain only letters. Got '{guess}'.")

        self._guesses.append(guess)
        attempt_num = len(self._guesses)

        fb = self._compute_feedback(guess, self._target_word)
        self._feedback.append(fb)

        reward = self._compute_reward(guess, self._target_word, attempt_num)
        self._total_reward += reward

        self._solved = (guess == self._target_word)
        self._done = self._solved or (attempt_num >= self.max_attempts)

        if self._solved:
            msg = f"Correct! You guessed '{self._target_word}' in {attempt_num} attempt(s)!"
        elif self._done:
            msg = f"Out of attempts. The word was '{self._target_word}'."
        else:
            msg = f"Try again! {self.max_attempts - attempt_num} attempt(s) left."

        obs = WordPuzzleObservation(
            feedback=self._feedback,
            attempts_used=attempt_num,
            max_attempts=self.max_attempts,
            task_level=self.task_level,
            message=msg,
            solved=self._solved,
            revealed_word=self._target_word if self._done else None,
        )
        return obs, reward, self._done

    def state(self) -> WordPuzzleState:
        return WordPuzzleState(
            target_word=self._target_word,
            guesses=self._guesses,
            task_level=self.task_level,
            total_reward=self._total_reward,
            done=self._done,
        )
