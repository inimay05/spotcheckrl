"""
inference.py — Mandatory hackathon inference script for SpotCheckRL.
Located in ROOT directory (not inside app/).

Environment variables (never hardcoded):
  API_BASE_URL        LLM API base  (default: https://api.openai.com/v1)
  MODEL_NAME          Model ID      (default: gpt-4o-mini)
  HF_TOKEN            API key (checked first)
  OPENAI_API_KEY      API key (fallback)
  SPOTCHECKRL_TASK    Task to run   (default: easy)
  ENV_BASE_URL        Env server    (default: http://localhost:7860)
"""

import os

# ---------------------------------------------------------------------------
# Config — all from environment, no magic numbers in this file
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME   = os.getenv("MODEL_NAME", "<your-active-model>")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
TASK_NAME    = os.getenv("SPOTCHECKRL_TASK", "easy")
BASE_URL     = os.getenv("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS          = 100
SUCCESS_THRESHOLD  = 0.5


# ---------------------------------------------------------------------------
# Logging helpers — exact format required by hackathon spec
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()   # 'true' or 'false'
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards)
    success_val  = str(success).lower()
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} "
        f"rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(obs: dict) -> str:
    price         = obs.get("price", 0.0)
    bid           = obs.get("bid", 0.0)
    running       = obs.get("running", True)
    progress      = obs.get("progress", 0.0)
    last_ckpt     = obs.get("last_checkpoint", 0.0)
    exposure      = obs.get("exposure", round(progress - last_ckpt, 4))
    valid_actions = obs.get("valid_actions", ["CONTINUE"])

    return (
        f"You are scheduling a cloud compute job on a spot instance.\n\n"
        f"Current state:\n"
        f"  task:            {TASK_NAME}\n"
        f"  price:           {price:.4f}  (interruption threshold / bid: {bid:.2f})\n"
        f"  running:         {running}\n"
        f"  progress:        {progress:.4f}  (1.0 = complete)\n"
        f"  last_checkpoint: {last_ckpt:.4f}\n"
        f"  exposure:        {exposure:.4f}  (unsaved progress at risk)\n\n"
        f"Rules:\n"
        f"  - If price > bid the instance is interrupted and progress rolls back to last checkpoint.\n"
        f"  - CHECKPOINT saves current progress (small penalty).\n"
        f"  - TERMINATE pauses the instance voluntarily.\n"
        f"  - RESUME restarts a paused instance.\n"
        f"  - CONTINUE advances progress by 0.02.\n\n"
        f"Choose exactly one action from: {valid_actions}\n"
        f"Respond with ONLY the action word. Nothing else."
    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_action(text: str, valid_actions: list) -> str:
    text = text.strip().upper()
    for action in valid_actions:
        if action in text:
            return action
    return valid_actions[0]   # safe fallback


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

def main() -> None:
    from openai import OpenAI
    import requests

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    rewards: list[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=TASK_NAME, env="spotcheckrl", model=MODEL_NAME)

    try:
        # Reset environment
        resp = requests.post(
            f"{BASE_URL}/reset",
            params={"task_id": TASK_NAME, "seed": 42},
        )
        resp.raise_for_status()
        obs = resp.json()

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            valid    = obs.get("valid_actions", ["CONTINUE"])
            prompt   = build_prompt(obs)

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a cloud scheduling agent. Respond with exactly one action word and nothing else."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=20,
                    temperature=0.0,
                )
                raw        = completion.choices[0].message.content or ""
                action_str = parse_action(raw, valid)
                error_msg  = None

            except Exception as exc:
                action_str = valid[0]
                error_msg  = str(exc)[:50]

            step_resp = requests.post(
                f"{BASE_URL}/step",
                json={"action_type": action_str},
                headers={"Content-Type": "application/json"},
            ).json()

            reward      = step_resp.get("reward", {}).get("value", 0.0)
            done        = step_resp.get("done", False)
            obs         = step_resp.get("observation", obs)

            rewards.append(reward)
            steps_taken = step

            log_step(step, action_str, reward, done, error_msg)

            if done:
                break

        # Grade the completed episode
        try:
            grade_resp = requests.post(f"{BASE_URL}/grader").json()
            score      = grade_resp.get("score", 0.0)
        except Exception:
            score = 0.0

        success = score >= SUCCESS_THRESHOLD

    finally:
        log_end(success, steps_taken, score, rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not API_KEY:
        print("[ERROR] HF_TOKEN or OPENAI_API_KEY not set", flush=True)
        import sys
        sys.exit(1)

    main()
