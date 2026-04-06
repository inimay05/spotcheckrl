# All parameters in one place. No magic numbers anywhere else.

# ---------------------------------------------------------------------------
# Reward parameters
# ---------------------------------------------------------------------------
REWARD_PARAMS = {
    "ALPHA": 100,   # reward scale for job completion
    "GAMMA": 3,     # penalty scale for interruptions
    "DELTA": 100,   # reward for clean finish
    "ETA": 150,     # bonus for finishing under IDEAL_STEPS
    "LAMBDA": 2,    # cost multiplier for expensive price windows
}

# ---------------------------------------------------------------------------
# Step / progress
# ---------------------------------------------------------------------------
PROGRESS_INCREMENT = 0.02  # fixed, same across all tasks

IDEAL_STEPS = 50   # 1.0 / 0.02

MAX_STEPS = 300

HISTORY_LENGTH = 10

# ---------------------------------------------------------------------------
# Price model (Ornstein-Uhlenbeck + jump)
# ---------------------------------------------------------------------------
PRICE_MU = 10.0    # long-run mean for all tasks

PRICE_THETA = 0.15  # mean-reversion speed

# ---------------------------------------------------------------------------
# Task configuration
#
# BID_PRINCIPLE: bid = mu + k * sigma
#   Bids calibrated so long-run interruption rates hit design targets:
#     easy   ~0–5%   | k=16  bid=18  (bid >> mu, almost never interrupted)
#     medium ~10–30% | k=4   bid=16  (moderate margin, spikes cause interruptions)
#     hard   ~30–60% | k=4   bid=22  (wide sigma; bid=22 = 10 + 4*3.0 — still frequently exceeded)
#   Note: k decreases easy→hard even though absolute bid rises, because sigma grows
#   faster. Hard is hard because sigma=3 makes the price distribution very wide.
# ---------------------------------------------------------------------------
TASK_CONFIG = {
    "easy": {
        "sigma": 0.5,
        "spike_prob": 0.02,
        "spike_min": 2,
        "spike_max": 5,
        "bid": 18,
    },
    "medium": {
        "sigma": 1.5,
        "spike_prob": 0.05,
        "spike_min": 6,
        "spike_max": 10,
        "bid": 16,
    },
    "hard": {
        "sigma": 3.0,
        "spike_prob": 0.10,
        "spike_min": 12,
        "spike_max": 20,
        "bid": 22,
    },
}

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
INVALID_ACTION_BEHAVIOR = "no_op"
