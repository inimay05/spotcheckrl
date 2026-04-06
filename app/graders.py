"""
Scoring functions for SpotCheckRL — 0.0 to 1.0.
Pure deterministic functions. No randomness. No state.
All imported from app/models — no magic numbers here.
"""

from app.models import EpisodeResult


# ---------------------------------------------------------------------------
# Sub-metric helpers
# ---------------------------------------------------------------------------

def compute_completion(result: EpisodeResult) -> float:
    """1.0 if job completed, else fractional progress (partial credit)."""
    return 1.0 if result.completed else result.final_progress


def compute_efficiency(result: EpisodeResult) -> float:
    """
    1.0 = finished in exactly ideal_steps (50).
    Lower = took more steps than ideal. 0.0 if not completed.
    """
    if not result.completed:
        return 0.0
    ideal = result.ideal_steps  # always 50, from config via EpisodeResult
    efficiency = ideal / max(result.steps_taken, ideal)
    return round(min(1.0, efficiency), 4)


def compute_rollback_score(result: EpisodeResult) -> float:
    """
    1.0 = no rollback losses at all.
    0.0 = rolled back everything that was ever completed.
    rollback_ratio = total_rollback_loss / final_progress
    """
    rollback_ratio = result.total_rollback_loss / max(result.final_progress, 0.001)
    return round(max(0.0, 1.0 - rollback_ratio), 4)


def compute_checkpoint_score(result: EpisodeResult) -> float:
    """
    0.0 if not completed.
    Otherwise: ideal rate = 1 checkpoint per 0.25 progress = 4 checkpoints/unit.
    score = min(rate / 4.0, 1.0)
    """
    if not result.completed:
        return 0.0
    rate = result.checkpoint_count / max(result.final_progress, 0.001)
    return round(min(1.0, rate / 4.0), 4)


# ---------------------------------------------------------------------------
# Graders — weights from design notes
# ---------------------------------------------------------------------------

def grade_easy(result: EpisodeResult) -> float:
    """
    Easy: completion matters most, efficiency and rollback secondary.
    Weights: completion=0.5, efficiency=0.2, rollback=0.2, checkpoint=0.1
    """
    score = (
        0.5 * compute_completion(result)
        + 0.2 * compute_efficiency(result)
        + 0.2 * compute_rollback_score(result)
        + 0.1 * compute_checkpoint_score(result)
    )
    return round(min(1.0, max(0.0, score)), 4)


def grade_medium(result: EpisodeResult) -> float:
    """
    Medium: balanced across all four metrics.
    Weights: completion=0.4, efficiency=0.2, rollback=0.2, checkpoint=0.2
    """
    score = (
        0.4 * compute_completion(result)
        + 0.2 * compute_efficiency(result)
        + 0.2 * compute_rollback_score(result)
        + 0.2 * compute_checkpoint_score(result)
    )
    return round(min(1.0, max(0.0, score)), 4)


def grade_hard(result: EpisodeResult) -> float:
    """
    Hard: rollback management and completion weighted highest.
    Weights: completion=0.3, efficiency=0.2, rollback=0.3, checkpoint=0.2
    """
    score = (
        0.3 * compute_completion(result)
        + 0.2 * compute_efficiency(result)
        + 0.3 * compute_rollback_score(result)
        + 0.2 * compute_checkpoint_score(result)
    )
    return round(min(1.0, max(0.0, score)), 4)


def get_grader(task_id: str):
    """Return the grader function for a given task_id."""
    return {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}[task_id]


# ---------------------------------------------------------------------------
# __main__ — determinism and ordering checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from app.config import IDEAL_STEPS

    def make_result(task_id, completed, steps, interruptions, rollback, checkpoints,
                    final_progress=None):
        fp = 1.0 if completed else (final_progress or 0.5)
        return EpisodeResult(
            task_id             = task_id,
            total_reward        = 0.0,   # not used by graders
            steps_taken         = steps,
            completed           = completed,
            interruption_count  = interruptions,
            checkpoint_count    = checkpoints,
            final_progress      = fp,
            total_rollback_loss = rollback,
            ideal_steps         = IDEAL_STEPS,
        )

    scenarios = {
        "perfect": make_result("x", completed=True,  steps=50,  interruptions=0,
                               rollback=0.0, checkpoints=4),
        "good":    make_result("x", completed=True,  steps=80,  interruptions=1,
                               rollback=0.1, checkpoints=2),
        "bad":     make_result("x", completed=False, steps=150, interruptions=5,
                               rollback=0.4, checkpoints=0, final_progress=0.5),
    }

    def run_table():
        header = f"{'Scenario':<10} {'easy':>8} {'medium':>8} {'hard':>8}"
        print(header)
        print("-" * len(header))
        rows = {}
        for name, result in scenarios.items():
            e = grade_easy(result)
            m = grade_medium(result)
            h = grade_hard(result)
            rows[name] = (e, m, h)
            print(f"{name:<10} {e:>8.4f} {m:>8.4f} {h:>8.4f}")
        return rows

    print("=== Run 1 ===")
    rows1 = run_table()
    print()
    print("=== Run 2 ===")
    rows2 = run_table()

    # Verify ordering: perfect > good > bad
    print()
    print("=== Verification ===")
    all_pass = True
    for i, task in enumerate(("easy", "medium", "hard")):
        p, g, b = rows1["perfect"][i], rows1["good"][i], rows1["bad"][i]
        order_ok  = p > g > b
        range_ok  = all(0.0 <= v <= 1.0 for v in (p, g, b))
        repro_ok  = rows1["perfect"][i] == rows2["perfect"][i]
        status = "PASS" if (order_ok and range_ok and repro_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {task:<8} order(p>g>b)={order_ok}  range=[0,1]={range_ok}"
              f"  reproducible={repro_ok}  => {status}")

    print()
    print("Overall:", "PASS" if all_pass else "FAIL")
