---
title: SpotCheckRL
emoji: ☁️
colorFrom: purple
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - cloud-computing
  - spot-instances
---

# SpotCheckRL

## Quickstart

```bash
# 1. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1
# 2. Reset an episode
curl -X POST "http://localhost:7860/reset?task_id=easy&seed=42"
# 3. Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "CONTINUE"}'
```

---

## Setup Instructions

### Local

```bash
git clone https://huggingface.co/spaces/yamyam05/spotcheckr1
cd spotcheckrl
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1
```

### Docker

```bash
docker build -t spotcheckrl .
docker run -p 7860:7860 spotcheckrl &

sleep 3

curl http://localhost:7860/health
```

Expected output:
```json
{"status": "ok", "version": "1.0.0", "name": "SpotCheckRL"}
```

```bash
curl -X POST "http://localhost:7860/reset?task_id=easy"
```

Expected output (truncated):
```json
{
  "price": 9.930868,
  "progress": 0.0,
  "last_checkpoint": 0.0,
  "running": true,
  "bid": 18.0,
  "valid_actions": ["CONTINUE", "CHECKPOINT", "TERMINATE"],
  "exposure": 0.0
}
```

---

## Description

Cloud spot instances offer dramatically lower compute costs — often 60–90% cheaper than on-demand pricing — but can be interrupted at any time when the market price exceeds your bid. The core problem: when should an agent checkpoint progress, and when should it pause — to minimise total cost from rollbacks, overhead, and wasted time?

SpotCheckRL formalises this as a reinforcement learning environment centred on checkpoint timing. An agent manages a single long-running job on a spot instance whose price evolves according to a mean-reverting jump-diffusion process. Each step the agent chooses to continue executing, checkpoint progress to durable storage, voluntarily terminate the instance, or resume after a pause. Interruptions do not end the episode — progress rolls back to the last checkpoint and execution continues. The agent may voluntarily TERMINATE to pause during high-price windows and RESUME when prices stabilise.

Three difficulty tiers (easy, medium, hard) vary price volatility and interruption frequency, requiring the agent to learn increasingly sophisticated risk-aware checkpoint strategies.

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `price` | `float >= 0` | Current spot market price |
| `progress` | `float [0,1]` | Job completion fraction |
| `last_checkpoint` | `float [0,1]` | Progress saved to durable storage |
| `running` | `bool` | Whether the instance is currently active |
| `bid` | `float >= 0` | Interruption threshold — price above this triggers rollback |
| `valid_actions` | `List[str]` | Action mask: legal actions in the current state |
| `exposure` | `float [0,1]` | Computed field: `progress - last_checkpoint` (unsaved progress at risk) |

> `task_id` is available as episode metadata via `/reset` and `/state` but is not part of the agent's observation.

---

## Action Space

| Action | Valid when | Effect |
|---|---|---|
| `CONTINUE` | `running=True` | Advance progress by 0.02 |
| `CHECKPOINT` | `running=True` | Save current progress (`last_checkpoint = progress`) |
| `TERMINATE` | `running=True` | Voluntarily pause the instance |
| `RESUME` | `running=False` | Restart a paused instance |

Invalid actions are **no-ops** — state is unchanged — but the time penalty still applies. This reflects the real-world cost of time: the instance continues running regardless of whether a valid action was taken.

---

## Design Constants

### Progress Increment

The job is modeled as a fixed-length task:

```
progress_increment = 1 / ideal_steps
```

With `ideal_steps = 50`, this gives `progress_increment = 0.02`. This balances decision complexity and training stability, allows the agent to observe multiple price fluctuations per episode, and keeps task length constant and reproducible across all difficulty tiers.

### Bid (Interruption Threshold)

An interruption occurs when:

```
price > bid
```

Higher bid → fewer interruptions (easier). Lower bid → more frequent interruptions (harder). Difficulty scaling is achieved by varying `bid` alongside price volatility, not by changing job length.

### Difficulty Scaling Principle

All tasks share the same `progress_increment` and reward function. They differ only in environment stochasticity:

| Parameter | Easy | Medium | Hard |
|---|---|---|---|
| `σ` (volatility) | 0.5 | 1.5 | 3.0 |
| `spike_prob` | 2% | 5% | 10% |
| `bid` | 18 | 16 | 22 |

Higher `σ` → more unstable prices. Higher `spike_prob` → more frequent interruptions.

Bids are calibrated using `bid = μ + k × σ`, where μ = 10.0 is the long-run price mean, σ is the task volatility, and k is a scaling factor. Although hard has the highest absolute bid (22), its wide price distribution (σ=3.0) means the price frequently exceeds it — hard is hard because of volatility, not a low bid. The effective interruption rates are approximately 0–5% on easy, 10–30% on medium, and 30–60% on hard.

---

## Task Descriptions

### Easy

- **Objective**: Complete the job (progress = 1.0) with minimal rollback.
- **Environment**: sigma=0.5, spike_prob=2%, spike_size=2–5, bid=18. Prices are highly stable and rarely approach the bid threshold. Interruptions are uncommon.
- **Grading weights**: 50% completion + 20% efficiency + 20% rollback score + 10% checkpoint score.
- **Expected behavior**: A competent agent should almost always complete the job near the ideal 50-step count with few or no interruptions.

### Medium

- **Objective**: Balance execution speed against interruption risk under moderate volatility.
- **Environment**: sigma=1.5, spike_prob=5%, spike_size=6–10, bid=16. Prices fluctuate meaningfully and occasional spikes will breach the bid. Checkpointing strategy begins to matter.
- **Grading weights**: 40% completion + 20% efficiency + 20% rollback score + 20% checkpoint score.
- **Expected behavior**: The agent needs to recognize elevated-price windows and either checkpoint before risk or temporarily terminate, accepting extra steps in exchange for lower rollback exposure.

### Hard

- **Objective**: Optimize execution under high volatility and frequent interruptions.
- **Environment**: sigma=3.0, spike_prob=10%, spike_size=12–20, bid=22. Even with a high bid, the wide price distribution causes frequent interruptions. Large spikes are common.
- **Grading weights**: 30% completion + 20% efficiency + 30% rollback score + 20% checkpoint score.
- **Expected behavior**: Rollback minimization is the dominant concern. An agent that checkpoints aggressively and pauses during high-price periods will score significantly better than one that runs continuously.

---

## Reward Function

| Term | Formula | Weight | Purpose |
|---|---|---|---|
| Progress reward | `+ALPHA * Δprogress` | 100 | Directly incentivises making job progress |
| Checkpoint penalty | `-GAMMA * [1 if CHECKPOINT else 0]` | 3 | Discourages excessive checkpointing (it costs time) |
| Rollback penalty | `-DELTA * rollback_loss` | 100 | Heavy penalty for losing unsaved work to interruption |
| Completion bonus | `+ETA * [1 if done and progress=1.0]` | 150 | One-time bonus for finishing the full job |
| Time penalty | `-LAMBDA` | 2 | Flat per-step cost to encourage efficiency over stalling |

These weights scale competing objectives (progress, risk, efficiency) and are chosen to ensure no single term dominates the reward.

**Efficiency interpretation:** Ideal completion is 50 steps. Efficiency is implicitly measured by how close actual steps are to this ideal — the flat `LAMBDA` penalty accumulates for every extra step taken beyond the optimum.

The reward weights are chosen to reflect real cloud tradeoffs: useful compute progress is valuable, rollback of unsaved work is equally costly, checkpointing incurs small overhead, and prolonged execution is penalized to encourage efficient completion.

---

## Price Model

Prices follow a **mean-reverting jump-diffusion** process combining two components:

**Ornstein-Uhlenbeck (mean reversion)**
```
dP = θ(μ - P)dt + σ dW
```
The price is always pulled back toward the long-run mean `μ = 10.0` at rate `θ = 0.15`. This models the competitive equilibrium dynamic of spot markets — prices rarely stay extremely high or low for long.

**Poisson jumps (spikes)**
At each step, with probability `spike_prob`, a sudden upward jump of magnitude `Uniform(spike_min, spike_max)` is added. This models demand surges or capacity events that temporarily spike prices well above the bid threshold.

The combination produces realistic spot price behavior: mostly stable near the mean, with occasional sharp spikes that create interruption risk. Task difficulty is controlled by increasing `σ` (background volatility) and `spike_prob` (spike frequency) from easy to hard.

| Parameter | Role |
|---|---|
| `σ` (sigma) | Controls magnitude of random price fluctuations — higher σ means more unstable prices |
| `spike_prob` | Probability of a sudden price spike per step — higher means more frequent interruptions |
| `spike_size` | Magnitude of a spike, sampled from `Uniform(spike_min, spike_max)` |

### Why this price model?

A simple random walk doesn't capture mean reversion — in real spot markets, prices are competitively anchored and rarely drift without bound. Pure Ornstein-Uhlenbeck without jumps misses sudden demand surges that are the primary cause of real-world preemptions. The combination produces realistic behaviour: stable most of the time, with sharp spikes the agent must learn to anticipate and respond to.

---

## Design Decisions

**(a) Academically grounded price model.** The OU + jump-diffusion process is rooted in stochastic finance (Ornstein-Uhlenbeck, 1930; Merton, 1976), capturing two key real-world dynamics: mean reversion (prices are competitively anchored and rarely drift without bound) and sudden demand spikes (Poisson jumps that temporarily exceed the bid threshold). Together they produce realistic spot market behaviour that makes checkpoint timing genuinely non-trivial.

**(b) `exposure` as a first-class observation.** The field `exposure = progress - last_checkpoint` directly encodes the agent's current risk position — the amount of progress that would be lost to an interruption right now. Making this explicit removes the need for the agent to compute it from raw fields and keeps the observation semantically rich.

**(c) CHECKPOINT and TERMINATE are separate actions.** Most save-state environments conflate saving progress with stopping execution. Separating them creates a strictly richer decision space: the agent can checkpoint and keep running (accepting overhead but staying live), terminate without checkpointing (pausing during a spike with nothing saved), or any combination — each with distinct risk/reward tradeoffs.

**(d) Difficulty scales volatility, not job length.** All three tasks use the same `progress_increment` and reward function. Only `σ`, `spike_prob`, and `bid` vary. This ensures scores are directly comparable across tiers — differences in agent performance reflect learned strategy, not structural asymmetry in task definition.

**(e) `time_elapsed` is not exposed to the agent.** The episode has a maximum step budget enforced internally, but the agent is not told how many steps have elapsed. This is deliberate — agents must learn to pace themselves through the reward signal alone (the flat `-LAMBDA` penalty per step), rather than explicitly counting down to a deadline. This keeps the observation space minimal and tests whether the agent internalises efficient behaviour rather than just reacting to a visible timer.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset?task_id=easy&seed=42` | Create a new episode. Returns initial `Observation`. |
| `POST` | `/step` | Advance one timestep. Body: `{"action_type": "CONTINUE"}`. Returns `StepResponse`. |
| `GET` | `/state` | Return current observation without advancing the environment. |
| `GET` | `/tasks` | List all tasks with full action and observation JSON schemas. |
| `POST` | `/grader` | Score a completed episode. Returns `score` in [0, 1] and full `EpisodeResult`. |
| `GET` | `/baseline` | Run the heuristic baseline agent on all three tasks. Returns `{easy, medium, hard}` scores. |
| `GET` | `/render` | Returns ASCII visualisation of current environment state. |

Full interactive documentation available at `/docs` (Swagger UI) once the server is running.

---

## Baseline Scores

Heuristic agent (rule-based, no LLM, seed=42):

| Task | Score | Steps | Completed |
|---|---|---|---|
| Easy | 0.9637 | 53 | True |
| Medium | 0.5920 | 238 | True |
| Hard | 0.4838 | 296 | True |

The heuristic checkpoints when `price > bid * 0.85 and exposure > 0.10`, terminates when `price > bid * 0.85`, and checkpoints every 30% of progress. An LLM agent reasoning about current price relative to bid and exposure should be able to improve on these scores, particularly on medium and hard.

---

## Model Interface

The environment supports any OpenAI-compatible model interface. Model configuration is controlled via environment variables:

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | Base URL for the model API |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | — | Auth token (Hugging Face or OpenAI key) |

The inference script is configured with MAX_STEPS = 100 per task to stay within the 20-minute runtime limit on standard evaluation hardware.

The model interacts with the environment through a standardized agent interface:

```
action = model(observation)
```

This design allows plug-and-play evaluation of different models, compatibility with both OpenAI API and Hugging Face-hosted models, and consistent benchmarking across agents.

---

## Limitations and Assumptions

Price dynamics are synthetic and not fitted to real-world spot pricing traces — this is a deliberate design choice for reproducibility and controlled benchmarking. The environment models a single job on a single instance; multi-job scheduling is out of scope.

---

## Citations

Price model based on mean-reverting jump-diffusion:
- Ornstein, L.S. and Uhlenbeck, G.E. (1930). On the Theory of Brownian Motion. *Physical Review*, 36, 823.
- Merton, R.C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1-2), 125-144.
