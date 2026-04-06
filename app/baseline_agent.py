"""
Heuristic baseline agent for SpotCheckRL.

Uses the HTTP API (httpx, already in requirements.txt).
No LLM, no API key — pure rule-based logic.
"""

# httpx is already in requirements.txt (httpx==0.27.0)
import httpx

BASE_URL = "http://localhost:7860"


# ---------------------------------------------------------------------------
# Heuristic policy
# ---------------------------------------------------------------------------

def heuristic_action(obs: dict) -> str:
    """
    Rule-based action selection based on current observation dict.

    Priority (highest first):
      1. Not running  -> RESUME
      2. Price near bid AND exposed -> CHECKPOINT first to save progress
      3. Price near bid             -> TERMINATE (pause, too risky)
      4. Exposure too large         -> CHECKPOINT every 30% anyway
      5. Otherwise                  -> CONTINUE
    """
    price      = obs["price"]
    bid        = obs["bid"]
    running    = obs["running"]
    progress   = obs["progress"]
    last_ckpt  = obs["last_checkpoint"]

    if not running:
        return "RESUME"

    exposure = progress - last_ckpt

    if price > bid * 0.85 and exposure > 0.10:
        return "CHECKPOINT"   # protect progress before danger zone

    if price > bid * 0.85:
        return "TERMINATE"    # pause — too risky right now

    if exposure > 0.30:
        return "CHECKPOINT"   # periodic safety checkpoint

    return "CONTINUE"


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------

def run_task(task_id: str, seed: int = 42) -> dict:
    """Run one full episode for task_id and return a result summary."""
    client = httpx.Client(base_url=BASE_URL, timeout=30.0)

    # Reset
    resp = client.post("/reset", params={"task_id": task_id, "seed": seed})
    resp.raise_for_status()
    obs = resp.json()

    steps         = 0
    total_reward  = 0.0
    interruptions = 0
    done          = False

    # Episode loop
    while not done:
        action_str = heuristic_action(obs)

        resp = client.post(
            "/step",
            json={"action_type": action_str},
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

        obs           = data["observation"]
        total_reward += data["reward"]["value"]
        steps        += 1

        if data["info"].get("interrupted"):
            interruptions += 1

        done = data["done"]

    # Grade
    resp = client.post("/grader")
    resp.raise_for_status()
    grade_data = resp.json()

    client.close()

    return {
        "task_id":      task_id,
        "score":        grade_data["score"],
        "steps":        steps,
        "completed":    grade_data["episode_result"]["completed"],
        "interruptions": interruptions,
        "total_reward": round(total_reward, 2),
    }


# ---------------------------------------------------------------------------
# Run all three tasks
# ---------------------------------------------------------------------------

def run_all_baselines(seed: int = 42) -> dict:
    """Run the heuristic baseline on easy/medium/hard. Returns {task_id: score}."""
    results = {}
    for task_id in ("easy", "medium", "hard"):
        r = run_task(task_id, seed=seed)
        results[task_id] = r["score"]
    return results


# ---------------------------------------------------------------------------
# __main__ — pretty table
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print(f"{'Task':<10} {'Score':>7} {'Steps':>7} {'Completed':>11} {'Interruptions':>15} {'Total Reward':>14}")
    print("-" * 68)

    for task_id in ("easy", "medium", "hard"):
        try:
            r = run_task(task_id, seed=42)
            print(
                f"{r['task_id']:<10} "
                f"{r['score']:>7.4f} "
                f"{r['steps']:>7d} "
                f"{str(r['completed']):>11} "
                f"{r['interruptions']:>15d} "
                f"{r['total_reward']:>14.2f}"
            )
        except Exception as exc:
            print(f"{task_id:<10} ERROR: {exc}")

    print()
