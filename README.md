# Mval Frontier Scoring Update: Cross-Floor Evaluation Report

This document summarizes the cross-floor evaluation before and after updating the frontier scoring rule to include a distance-aware term.

## 1) Change Summary

- **Goal**: Update frontier scoring to prefer semantically promising frontiers while still favoring near-term reachable ones.
- **New formula**:

  \[
  M_{val}(F_i) = M_{ss}(F_i)\cdot\left(1 + \exp(-d_i)\cdot\mathbf{1}[d_i \le d_{\theta}]\right) + \epsilon
  \]

- **Hyperparameters**:
  - \(d_{\theta} = 3.0\) meters (distance bonus is zero when \(d_i > d_{\theta}\))
  - \(\epsilon = 0.01\)
- **Distance definition**:
  - \(d_i\) is the real-time Euclidean distance from robot current position to frontier \(F_i\), recomputed every step for every frontier.

## 2) Code Locations

- Updated frontier ranking logic: `ascent/llm_planner.py`
  - `_get_best_frontier_with_llm(...)`: pass `robot_xy` into frontier sorting
  - `_sort_frontiers_by_value(...)`: apply Mval formula and sort by Mval descending
- Added Mval constants in planner:
  - `_MVAL_D_THETA = 3.0`
  - `_MVAL_EPSILON = 0.01`

## 3) Evaluation Setup

- Task: MP3D cross-floor ObjectNav evaluation
- Config: `experiments/eval_ascent_mp3d_cross_floor.yaml`
- Episodes: 50
- VLM server ports: `13181-13186`
- Runtime mode: tmux detached session for SSH-safe long run

## 4) Before vs After Results

| Metric | Before (2026-03-07) | After (2026-03-08) | Delta |
|---|---:|---:|---:|
| Success | 4.00% (2/50) | 4.00% (2/50) | 0.00 |
| SPL | 1.77% | 1.88% | +0.11 |
| Soft SPL | 6.48% | 7.62% | +1.14 |
| Avg Distance-to-Goal | 18.7274 | 18.1573 | -0.5701 |
| Avg Reward | -0.2919 | 0.2622 | +0.5541 |
| Traveled Stairs | 0.2200 | 0.2200 | 0.0000 |
| Target Detected | 0.2600 | 0.2400 | -0.0200 |
| Stop Called | 0.2600 | 0.2200 | -0.0400 |
| Avg Num Steps | 399.02 | 415.08 | +16.06 |

## 5) Failure-Type Comparison

| Failure Type | Before | After | Delta |
|---|---:|---:|---:|
| `never_saw_target_did_not_travel_stairs_likely_infeasible` | 27 | 27 | 0 |
| `false_positive` | 11 | 10 | -1 |
| `never_saw_target_traveled_stairs_likely_infeasible` | 8 | 9 | +1 |
| `never_saw_target_traveled_stairs_feasible` | 2 | 2 | 0 |

## 6) Interpretation

- The Mval update improved trajectory quality indicators (`SPL`, `Soft SPL`, `Avg DTG`, `Avg Reward`).
- Overall success did not increase in this run (still 2/50), so the main bottleneck remains cross-floor target discovery and stair-transition success.
- The updated scoring appears to make movement decisions more useful locally, but not yet sufficient to solve global multi-floor search failures.

## 7) Reproduce / Monitor

Use `ascent_nav` environment:

```bash
cd /home/user/12/ascent
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate ascent_nav
python -m ascent.run --config-name eval_ascent_mp3d_cross_floor 2>&1 | tee eval_cross_floor.log
```

SSH-safe background run (tmux):

```bash
tmux new-session -d -s eval_cross_floor
tmux send-keys -t eval_cross_floor "cd /home/user/12/ascent && source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate ascent_nav && python -m ascent.run --config-name eval_ascent_mp3d_cross_floor 2>&1 | tee eval_cross_floor.log" C-m
```

Check progress:

```bash
tmux attach-session -t eval_cross_floor
# or
tail -f /home/user/12/ascent/eval_cross_floor.log
```
