"""
FastAPI application for SpotCheckRL.
All endpoints operate on sessions['default'].
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import MAX_STEPS, TASK_CONFIG
from app.environment import SpotSchedulingEnv
from app.graders import get_grader
from app.models import Action, Observation

logger = logging.getLogger("spotcheckrl")

app = FastAPI(
    title="SpotCheckRL",
    description="Cloud spot-instance job scheduling RL environment.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session store — single shared environment keyed by 'default'
# ---------------------------------------------------------------------------
sessions: dict[str, SpotSchedulingEnv] = {}


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("SpotCheckRL ready on port 7860")
    print("SpotCheckRL ready on port 7860")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_env() -> SpotSchedulingEnv:
    env = sessions.get("default")
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return env


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=Observation)
def reset(task_id: str = "easy", seed: int = 42) -> Observation:
    """Create (or recreate) the default environment and return initial observation."""
    if task_id not in TASK_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Choose: easy, medium, hard")
    env = SpotSchedulingEnv(task_id=task_id, seed=seed)
    sessions["default"] = env
    return env.reset()


@app.post("/step")
def step(action: Action):
    """Advance one timestep. Body: {\"action_type\": \"CONTINUE|CHECKPOINT|TERMINATE|RESUME\"}"""
    env = _get_env()
    return env.step(action)


@app.get("/state", response_model=Observation)
def state() -> Observation:
    """Return current observation without advancing the environment."""
    env = _get_env()
    return env.state()


@app.get("/tasks")
def tasks():
    """List all tasks with descriptions and full action/observation schemas."""
    task_list = [
        {
            "id": tid,
            "description": _task_description(tid),
            "difficulty": tid,
        }
        for tid in ("easy", "medium", "hard")
    ]
    return {
        "tasks": task_list,
        "action_schema": Action.model_json_schema(),
        "observation_schema": Observation.model_json_schema(),
    }


def _task_description(task_id: str) -> str:
    descriptions = {
        "easy":   "Complete a job under stable pricing with rare spikes.",
        "medium": "Balance execution and risk under moderate volatility.",
        "hard":   "Optimize execution under high volatility and frequent interruptions.",
    }
    return descriptions[task_id]


@app.post("/grader")
def grader():
    """Score a completed episode. Raises 400 if episode still in progress."""
    env = _get_env()
    episode_done = (env.progress >= 1.0) or (env.time_elapsed >= MAX_STEPS)
    if not episode_done:
        raise HTTPException(
            status_code=400,
            detail=f"Episode not complete yet (progress={env.progress:.3f}, steps={env.time_elapsed}/{MAX_STEPS})",
        )
    result = env.get_episode_result()
    grader_fn = get_grader(env.task_id)
    score = grader_fn(result)
    return {
        "task_id": env.task_id,
        "score": score,
        "episode_result": result.model_dump(),
    }


@app.get("/baseline")
def baseline():
    """Run the heuristic baseline agent on all tasks and return scores."""
    try:
        from app.baseline_agent import run_all_baselines
        return run_all_baselines(seed=42)
    except Exception as e:
        return {"easy": 0.0, "medium": 0.0, "hard": 0.0, "error": str(e)}


@app.get("/render")
def render() -> PlainTextResponse:
    """Return an ASCII display of the current environment state."""
    env = _get_env()
    return PlainTextResponse(env.render())


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok", "version": "1.0.0", "name": "SpotCheckRL"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, workers=1)
