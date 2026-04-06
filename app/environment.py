"""
Core RL environment for SpotCheckRL.
Simulates cloud spot-instance job scheduling under stochastic pricing.
All constants imported from app/config — no magic numbers here.
"""

from __future__ import annotations

from app.config import (
    IDEAL_STEPS,
    INVALID_ACTION_BEHAVIOR,
    MAX_STEPS,
    PROGRESS_INCREMENT,
    REWARD_PARAMS,
    TASK_CONFIG,
)
from app.models import (
    Action,
    EpisodeResult,
    Observation,
    Reward,
    RewardBreakdown,
    StepResponse,
)
from app.price_model import PriceModel

# Unpack reward params once for readability
_ALPHA  = REWARD_PARAMS["ALPHA"]   # progress reward scale
_GAMMA  = REWARD_PARAMS["GAMMA"]   # checkpoint penalty
_DELTA  = REWARD_PARAMS["DELTA"]   # rollback penalty scale
_ETA    = REWARD_PARAMS["ETA"]     # completion bonus
_LAMBDA = REWARD_PARAMS["LAMBDA"]  # per-step time penalty


class SpotSchedulingEnv:
    """Spot-instance job scheduling environment."""

    def __init__(self, task_id: str = "easy", seed: int = 42) -> None:
        self.task_id    = task_id
        self.seed       = seed
        self.price_model = PriceModel(task_id, seed)
        self._trajectory: list[StepResponse] = []
        self._total_reward = 0.0
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset episode state; price model RNG is NOT reset."""
        self.price_model.reset()

        self.progress           = 0.0
        self.last_checkpoint    = 0.0
        self.running            = True
        self.time_elapsed       = 0
        self._trajectory        = []
        self._total_reward      = 0.0
        self.checkpoint_count   = 0
        self.interruption_count = 0
        self.total_rollback_loss = 0.0

        # Advance price once so initial observation is not always exactly mu
        self.price_model.step()

        return self._build_obs()

    def step(self, action: Action) -> StepResponse:
        """Advance one time step and return (obs, reward, done, info)."""

        # --- Step 1: advance price ---
        new_price = self.price_model.step()

        # --- Step 2: check interruption ---
        interrupted   = self.running and (new_price > self.price_model.bid)
        rollback_loss = 0.0

        if interrupted:
            rollback_loss             = self.progress - self.last_checkpoint
            self.progress             = self.last_checkpoint
            self.interruption_count  += 1
            self.total_rollback_loss += rollback_loss

        # --- Step 3: action masking ---
        valid_when_running    = {"CONTINUE", "CHECKPOINT", "TERMINATE"}
        valid_when_stopped    = {"RESUME"}
        legal_set             = valid_when_running if self.running else valid_when_stopped
        action_was_valid      = action.action_type in legal_set

        # --- Step 4: apply action (only if NOT interrupted AND valid) ---
        prev_progress    = self.progress
        checkpoint_taken = False

        if not interrupted and action_was_valid:
            atype = action.action_type
            if atype == "CONTINUE" and self.running:
                self.progress = min(1.0, self.progress + PROGRESS_INCREMENT)

            elif atype == "CHECKPOINT" and self.running:
                self.last_checkpoint = self.progress
                self.checkpoint_count += 1
                checkpoint_taken = True

            elif atype == "TERMINATE" and self.running:
                self.running = False

            elif atype == "RESUME" and not self.running:
                self.running = True

            # invalid action behaviour is no_op (INVALID_ACTION_BEHAVIOR = 'no_op')
            # — no else branch needed; already validated above

        # --- Step 5: compute reward ---
        is_done_for_reward = (self.progress >= 1.0)
        reward = self._compute_reward(
            prev_progress    = prev_progress,
            checkpoint_taken = checkpoint_taken,
            rollback_loss    = rollback_loss,
            is_done          = is_done_for_reward,
        )

        # --- Step 6: advance time ---
        self.time_elapsed += 1

        # --- Step 7: episode termination ---
        done = (self.progress >= 1.0) or (self.time_elapsed >= MAX_STEPS)

        # --- Step 8: build response ---
        obs = self._build_obs()

        if done:
            reason = "completed" if self.progress >= 1.0 else "timeout"
        else:
            reason = "ongoing"

        info = {
            "interrupted":      interrupted,
            "action_was_valid": action_was_valid,
            "rollback_loss":    rollback_loss,
            "reason":           reason,
        }

        response = StepResponse(observation=obs, reward=reward, done=done, info=info)
        self._trajectory.append(response)
        self._total_reward += reward.value
        return response

    def state(self) -> Observation:
        return self._build_obs()

    def render(self) -> str:
        """Return an ASCII display of the current environment state."""
        def bar(fraction: float) -> str:
            filled = round(max(0.0, min(1.0, fraction)) * 10)
            return "█" * filled + "░" * (10 - filled)

        bid    = self.price_model.bid
        price  = self.price_model.current_price

        price_fraction = price / bid if bid > 0 else 0.0
        if price > bid:
            price_line = f"Price:    [{bar(price_fraction)}]  {price:.1f}  (bid: {bid:.1f})  ⚠ INTERRUPTED"
        else:
            price_line = f"Price:    [{bar(price_fraction)}]  {price:.1f}  (bid: {bid:.1f})  ← SAFE"

        exposure = round(self.progress - self.last_checkpoint, 4)
        exposure_suffix = "  ⚠ at risk" if exposure > 0.1 else ""

        status = "RUNNING" if self.running else "PAUSED"

        return (
            f"=== SpotCheckRL State ===\n"
            f"{price_line}\n"
            f"Progress: [{bar(self.progress)}]  {round(self.progress * 100):.0f}%\n"
            f"Saved:    [{bar(self.last_checkpoint)}]  {round(self.last_checkpoint * 100):.0f}%\n"
            f"Exposure: [{bar(exposure)}]  {round(exposure * 100):.0f}%{exposure_suffix}\n"
            f"Status:   {status}  |  Step: {self.time_elapsed}"
        )

    def get_episode_result(self) -> EpisodeResult:
        return EpisodeResult(
            task_id             = self.task_id,
            total_reward        = round(self._total_reward, 4),
            steps_taken         = self.time_elapsed,
            completed           = self.progress >= 1.0,
            interruption_count  = self.interruption_count,
            checkpoint_count    = self.checkpoint_count,
            final_progress      = round(self.progress, 4),
            total_rollback_loss = round(self.total_rollback_loss, 4),
            ideal_steps         = IDEAL_STEPS,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        prev_progress:    float,
        checkpoint_taken: bool,
        rollback_loss:    float,
        is_done:          bool,
    ) -> Reward:
        progress_reward    =  _ALPHA  * (self.progress - prev_progress)
        checkpoint_penalty = -_GAMMA  * (1.0 if checkpoint_taken else 0.0)
        rollback_penalty   = -_DELTA  * rollback_loss
        completion_bonus   =  _ETA    if (is_done and self.progress >= 1.0) else 0.0
        time_penalty       = -_LAMBDA * 1.0

        total = round(
            progress_reward + checkpoint_penalty
            + rollback_penalty + completion_bonus + time_penalty,
            6,
        )

        breakdown = RewardBreakdown(
            progress_reward    = progress_reward,
            checkpoint_penalty = checkpoint_penalty,
            rollback_penalty   = rollback_penalty,
            completion_bonus   = completion_bonus,
            time_penalty       = time_penalty,
            total              = total,
        )
        return Reward(value=total, breakdown=breakdown)

    def _build_obs(self) -> Observation:
        return Observation(
            price           = round(self.price_model.current_price, 6),
            progress        = round(self.progress, 6),
            last_checkpoint = round(self.last_checkpoint, 6),
            running         = self.running,
            bid             = self.price_model.bid,
            # valid_actions and exposure are set automatically by Pydantic validator
        )


# ---------------------------------------------------------------------------
# __main__ — heuristic agent test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def heuristic(obs: Observation) -> Action:
        """Simple price-aware heuristic."""
        if not obs.running:
            return Action(action_type="RESUME")
        price_threshold = obs.bid * 0.85
        if obs.price > price_threshold and obs.exposure > 0.05:
            return Action(action_type="CHECKPOINT")
        if obs.price > price_threshold:
            return Action(action_type="TERMINATE")
        return Action(action_type="CONTINUE")

    print(f"{'Task':<8} {'Done':<6} {'Steps':>6} {'Int':>5} {'Ckpt':>5} {'Reward':>10}")
    print("-" * 45)

    for task_id in ("easy", "medium", "hard"):
        env  = SpotSchedulingEnv(task_id=task_id, seed=42)
        obs  = env.reset()
        done = False

        while not done:
            action   = heuristic(obs)
            response = env.step(action)
            obs      = response.observation
            done     = response.done

        result = env.get_episode_result()
        print(
            f"{result.task_id:<8} {str(result.completed):<6} "
            f"{result.steps_taken:>6} {result.interruption_count:>5} "
            f"{result.checkpoint_count:>5} {result.total_reward:>10.2f}"
        )

    print()
    # Assertion: easy should always complete with this heuristic
    env  = SpotSchedulingEnv(task_id="easy", seed=42)
    obs  = env.reset()
    done = False
    while not done:
        response = env.step(heuristic(obs))
        obs, done = response.observation, response.done
    result = env.get_episode_result()
    assert result.completed, f"FAIL: easy did not complete (progress={result.final_progress})"
    print("Assertion PASS: easy task completed.")
