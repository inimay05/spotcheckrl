"""
Stochastic spot-price model for SpotCheckRL.
Ornstein-Uhlenbeck mean reversion + Poisson jump (spike).
All parameters sourced from app/config — no magic numbers here.
"""

from __future__ import annotations

from typing import List

import numpy as np

from app.config import HISTORY_LENGTH, PRICE_MU, PRICE_THETA, TASK_CONFIG


class PriceModel:
    """Mean-reverting jump-diffusion price process."""

    def __init__(self, task_id: str, seed: int = 42) -> None:
        config = TASK_CONFIG[task_id]

        self.mu        = PRICE_MU
        self.theta     = PRICE_THETA
        self.sigma     = config["sigma"]
        self.p_spike   = config["spike_prob"]
        self.spike_min = config["spike_min"]
        self.spike_max = config["spike_max"]
        self._bid      = float(config["bid"])

        # Seeded RNG — reproducible episodes
        self.rng = np.random.RandomState(seed)

        self.price   = float(self.mu)
        self.history: List[float] = [float(self.mu)] * HISTORY_LENGTH

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def step(self) -> float:
        """Advance one time step; return new price."""
        reversion = self.theta * (self.mu - self.price)
        noise     = self.sigma * self.rng.standard_normal()
        spike     = (
            self.rng.uniform(self.spike_min, self.spike_max)
            if self.rng.random() < self.p_spike
            else 0.0
        )

        new_price    = max(0.1, self.price + reversion + noise + spike)
        self.price   = new_price
        self.history.append(new_price)
        self.history = self.history[-HISTORY_LENGTH:]
        return new_price

    def reset(self) -> None:
        """Reset price to mu without touching the RNG."""
        self.price   = float(self.mu)
        self.history = [float(self.mu)] * HISTORY_LENGTH

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def bid(self) -> float:
        return self._bid

    @property
    def current_price(self) -> float:
        return self.price

    @property
    def price_history(self) -> List[float]:
        return list(self.history)


# ---------------------------------------------------------------------------
# __main__ — sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    STEPS = 200

    # Wider bands than point targets to absorb 200-step statistical variance.
    # Design targets (long-run): easy ~0–5%, medium ~10–30%, hard ~30–60%.
    EXPECTED_RANGES = {
        "easy":   (0.00, 0.08),
        "medium": (0.05, 0.40),
        "hard":   (0.25, 0.70),
    }

    print(f"{'Task':<8} {'Min':>7} {'Max':>7} {'Avg':>7} {'Int%':>8}  Sanity")
    print("-" * 52)

    all_pass = True
    for task_id in ("easy", "medium", "hard"):
        model = PriceModel(task_id, seed=42)
        prices = [model.step() for _ in range(STEPS)]

        min_p  = min(prices)
        max_p  = max(prices)
        avg_p  = sum(prices) / STEPS
        int_rate = sum(1 for p in prices if p > model.bid) / STEPS

        lo, hi = EXPECTED_RANGES[task_id]
        ok = lo <= int_rate <= hi
        if not ok:
            all_pass = False

        label = "PASS" if ok else f"FAIL (expected {lo*100:.0f}–{hi*100:.0f}%)"
        print(
            f"{task_id:<8} {min_p:>7.2f} {max_p:>7.2f} {avg_p:>7.2f}"
            f" {int_rate*100:>7.1f}%  {label}"
        )

    print()
    print("Overall:", "PASS" if all_pass else "FAIL")

    # ------------------------------------------------------------------
    # Reproducibility check — same seed → same first 5 prices
    # ------------------------------------------------------------------
    print()
    print("=== Reproducibility check (seed=42, easy, first 5 prices) ===")
    runs = []
    for _ in range(2):
        m = PriceModel("easy", seed=42)
        runs.append([round(m.step(), 6) for _ in range(5)])

    for i, run in enumerate(runs, 1):
        print(f"  Run {i}: {run}")

    match = runs[0] == runs[1]
    print(f"  Identical: {'PASS' if match else 'FAIL'}")
