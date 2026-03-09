# ASCENT Improved: Distance-Aware Frontier Scoring for Cross-Floor ObjectNav

> 基于 [ASCENT](https://github.com/Zeying-Gong/ascent) 的改进版本，通过引入距离感知的 frontier 评分机制，显著提升跨楼层目标导航中的路径质量与探索效率。

## Core Improvement

原始 ASCENT 的 frontier 评分仅依赖语义匹配分数 $M_{ss}$，在跨楼层场景中容易反复探索远处低价值区域，导致路径冗余。

本项目提出 **距离感知的 frontier 评分公式**，在保留语义信息的同时引入距离衰减奖励，使 agent 优先探索近距离高语义价值区域：

$$M_{val}(F_i) = M_{ss}(F_i) \cdot \left(1 + \exp(-d_i) \cdot \mathbf{1}[d_i \le d_{\theta}]\right) + \epsilon$$

其中：
- $d_i$：机器人当前位置到 frontier $F_i$ 的实时欧氏距离（每步重新计算）
- $d_{\theta} = 3.0\ \text{m}$：距离奖励阈值，超过该距离不再给予近距离加成
- $\epsilon = 0.01$：平滑项，防止零分

**设计思路**：当 frontier 距离较近（$d_i \le d_{\theta}$）时，$\exp(-d_i)$ 提供额外正向加成；距离越近加成越大，鼓励 agent 先完成对近距离区域的探索，再向远处推进，从而减少无效往返。

## Evaluation Results

在 MP3D 跨楼层 ObjectNav 任务（50 episodes）上进行对比评测：

| Metric | Before | After | Improvement |
|---|---:|---:|---:|
| **Avg Reward** | -0.2919 | **+0.2622** | +0.5541 |
| **Soft SPL** | 6.48% | **7.62%** | +1.14 (+17.6%) |
| **Avg Distance-to-Goal** | 18.73 m | **18.16 m** | -0.57 m |
| **SPL** | 1.77% | **1.88%** | +0.11 |

### Key Takeaways

- **Average Reward 从负转正**：改进前 agent 平均获得负奖励（-0.29），说明探索策略整体低效；改进后提升至 +0.26，表明距离感知评分有效提升了每步决策的质量。
- **Soft SPL 提升 17.6%**：路径效率显著改善，agent 能更高效地向目标方向推进。
- **Distance-to-Goal 降低 0.57m**：agent 在有限步数内更接近目标位置。

## Quick Start

### Run VLM Servers

```bash
./scripts/launch_vlm_servers_ascent.sh
```

### Run Cross-Floor Evaluation

```bash
python -u -m ascent.run --config-name=eval_ascent_mp3d_cross_floor
```

### Run Standard MP3D Evaluation

```bash
python -u -m ascent.run --config-name=eval_ascent_mp3d
```

## Code Changes

- `ascent/llm_planner.py`：实现距离感知的 frontier 排序逻辑
- `experiments/eval_ascent_mp3d_cross_floor.yaml`：跨楼层评测配置
- `experiments/eval_ascent_mp3d_long_distance.yaml`：长距离评测配置
- `scripts/filter_episodes.py`：episode 筛选工具

## Acknowledgments

This project is built upon [ASCENT](https://github.com/Zeying-Gong/ascent). Thanks to the original authors for their excellent work.
