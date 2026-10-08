# Agent Self-Evolution — Experiment Results

> 实验记录：截至 2026-10-08 已完成的 ALFWorld 实验。本文仅记录实际已运行的结果；待开展实验明确标注为 TODO。

## 1. 实验目标

检验训练自由（Training-Free）的经验记忆是否能改善 ReAct Agent 在 ALFWorld 上的任务成功率与决策效率，并通过逐级消融评估：

1. 普通失败轨迹反思（Ordinary Memory）的作用；
2. 关键步骤分析和未验证反事实建议（Unverified Memory）的额外作用；
3. 真实环境反事实验证（CF-Verified Memory）的额外作用。

所有经验从训练任务的失败轨迹构建，held-out 任务不用于经验构建。所有方法保持同一模型、任务 manifest 和评估协议。

## 2. 数据、模型与协议

| 设置 | 数值或说明 |
|---|---|
| 环境 | ALFWorld 文本任务 |
| LLM | `qwen3.8-chat` |
| 固定 seed | `42` |
| 训练任务 | 1000 |
| `valid_seen` | 140 |
| `valid_unseen` | 134 |
| Agent 决策上限 | 50 |
| 环境交互上限 | 50 |
| 经验检索 | BM25 对 `source_task` 评分 |
| Top-K | 3 |
| 去重 | 完全相同 `lesson` 文本去重 |

Manifest：

```text
outputs/manifests/train_seed42_1000.json
outputs/manifests/valid_seen_seed42_140.json
outputs/manifests/valid_unseen_seed42_134.json
```

`valid_seen` / `valid_unseen` 为不同 held-out split；两者的任务类型组成不同，不应仅依据原始成功率差异推断哪个 split 必然更难。

## 3. 训练集 ReAct Baseline

### 3.1 整体结果

| Metric | Result |
|---|---:|
| Episodes | 1000 |
| Successes | 867 |
| Success rate | 86.70% |
| Failures | 133 |
| Truncated rate | 1.90% |
| Avg decision steps | 20.26 |
| Avg environment steps | 20.19 |
| Format errors | 67 |
| Inadmissible actions | 670 |
| Format valid rate | 99.67% |
| Admissible action rate | 96.68% |
| Valid action rate | 96.36% |

### 3.2 按任务类型

| Task type | Successes / Episodes | Success rate |
|---|---:|---:|
| `look_at_obj_in_light` | 80/88 | 90.91% |
| `pick_and_place_simple` | 205/208 | 98.56% |
| `pick_clean_then_place_in_recep` | 149/189 | 78.84% |
| `pick_cool_then_place_in_recep` | 105/138 | 76.09% |
| `pick_heat_then_place_in_recep` | 97/130 | 74.62% |
| `pick_two_obj_and_place` | 231/247 | 93.52% |

133 个失败中，`clean`、`cool`、`heat` 合计 106 个，约占 79.7%。这些任务是后续失败分析和经验提炼的重要来源。

## 4. 失败分析与反事实验证

### 4.1 失败分析

输入：133 条失败轨迹。LLM 输出：

- `critical_step`
- `failure_type`
- `failure_reason`
- `original_action`
- `counterfactual_actions`
- `expected_effect`

输出目录：

```text
outputs/failure_analysis_qwen3.8-chat_train_seed42_1000/
```

### 4.2 环境反事实验证

对候选动作在对应 ALFWorld 任务中重放历史步骤、替换关键动作并继续执行，记录候选后续成功与否。

| Metric | Result |
|---|---:|
| Failed episodes analyzed | 133 |
| Counterfactual candidates | 371 |
| Successful candidate continuations | 197 |
| Failed candidate continuations | 174 |
| Candidate continuation success rate | 53.10% |
| Episodes with ≥1 successful candidate | 91/133 (68.42%) |
| Episodes with no successful candidate | 42/133 (31.58%) |
| Average candidates per failed episode | 2.79 |

输出目录：

```text
outputs/counterfactual_qwen3.8-chat_train_seed42_1000/
```

**解释边界**：这些统计描述在重放协议下观察到的候选继续执行结果，不能直接等同于对某一步动作的严格因果效应估计。

## 5. 三类经验记忆

| Method | Extraction input | Raw | Unique | Removed duplicates |
|---|---|---:|---:|---:|
| Ordinary | 完整失败轨迹 | 133 | 132 | 1 |
| Unverified | 失败分析、关键步骤、反事实候选与预期效果 | 133 | 132 | 1 |
| Verified | 失败分析、反事实环境执行结果与后续轨迹 | 133 | 130 | 3 |

共同点：经验包含 `failure_type`、`lesson`、`source_task`；使用 exact-string lesson 去重；通过 `BM25(source_task)` 检索 Top-3 lessons；经验在初始 ReAct Prompt 中注入。不同方法的提炼 Prompt 输入并不相同，这正是各消融方法的定义。

经验文件：

```text
outputs/memory/ordinary_experience_store_qwen3.8-chat_train_seed42_1000.json
outputs/memory/unverified_experience_store_qwen3.8-chat_train_seed42_1000.json
outputs/memory/experience_store_qwen3.8-chat_train_seed42_1000.json
```

## 6. Held-out 整体结果

### 6.1 成功率和平均决策步数

| Method | Seen successes | Seen success rate | Seen avg steps | Unseen successes | Unseen success rate | Unseen avg steps |
|---|---:|---:|---:|---:|---:|---:|
| ReAct | 120/140 | 85.71% | 19.97 | 122/134 | 91.04% | 18.24 |
| Ordinary | 122/140 | 87.14% | 18.59 | 127/134 | 94.78% | 15.32 |
| Unverified | 127/140 | 90.71% | 17.01 | 129/134 | 96.27% | 14.55 |
| **Verified (Ours)** | **128/140** | **91.43%** | **15.54** | **131/134** | **97.76%** | **13.22** |

### 6.2 逐级差异（百分点）

| Comparison | Seen Δ success rate | Unseen Δ success rate |
|---|---:|---:|
| ReAct → Ordinary | +1.43 pp | +3.74 pp |
| Ordinary → Unverified | +3.57 pp | +1.49 pp |
| Unverified → Verified | +0.72 pp | +1.49 pp |
| ReAct → Verified | +5.72 pp | +6.72 pp |

相对 ReAct，Verified 平均决策步数下降约 22.2%（seen）和 27.5%（unseen）。相对 Unverified，Verified 平均决策步数进一步下降约 8.6%（seen）和 9.1%（unseen）。

## 7. 按任务类型的成功率

### 7.1 `valid_seen`（140 episodes）

| Task type | ReAct | Ordinary | Unverified | Verified |
|---|---:|---:|---:|---:|
| `look_at_obj_in_light` | 12/13 | 10/13 | 11/13 | 12/13 |
| `pick_and_place_simple` | 35/35 | 35/35 | 35/35 | 35/35 |
| `pick_clean_then_place_in_recep` | 19/27 | 22/27 | 21/27 | 22/27 |
| `pick_cool_then_place_in_recep` | 19/25 | 19/25 | 21/25 | 21/25 |
| `pick_heat_then_place_in_recep` | 12/16 | 13/16 | 15/16 | 15/16 |
| `pick_two_obj_and_place` | 23/24 | 23/24 | 24/24 | 23/24 |

### 7.2 `valid_unseen`（134 episodes）

| Task type | ReAct | Ordinary | Unverified | Verified |
|---|---:|---:|---:|---:|
| `look_at_obj_in_light` | 18/18 | 17/18 | 18/18 | 18/18 |
| `pick_and_place_simple` | 24/24 | 24/24 | 24/24 | 24/24 |
| `pick_clean_then_place_in_recep` | 24/31 | 27/31 | 29/31 | 29/31 |
| `pick_cool_then_place_in_recep` | 18/21 | 21/21 | 18/21 | 20/21 |
| `pick_heat_then_place_in_recep` | 21/23 | 22/23 | 23/23 | 23/23 |
| `pick_two_obj_and_place` | 17/17 | 16/17 | 17/17 | 17/17 |

**观察**：各任务类型并非单调改善。例如 unseen `cool` 上 Ordinary 为 21/21，Verified 为 20/21；seen `look` 上 Ordinary 低于 ReAct。不能据总体增益推断每类任务均获益。

## 8. Episode-level 配对分析

配对单位为相同 split、相同 episode 文件。`fail → success` 表示后一个方法修复了前一个方法的失败；`success → fail` 表示出现回退。净增益为两者之差。

### 8.1 `valid_seen`

| Comparison | Fail→Fail | Fail→Success | Success→Fail | Success→Success | Net gain |
|---|---:|---:|---:|---:|---:|
| ReAct → Ordinary | 10 | 10 | 8 | 112 | +2 |
| Ordinary → Unverified | 9 | 9 | 4 | 118 | +5 |
| Unverified → Verified | 8 | 5 | 4 | 123 | +1 |
| ReAct → Verified | 7 | 13 | 5 | 115 | +8 |

### 8.2 `valid_unseen`

| Comparison | Fail→Fail | Fail→Success | Success→Fail | Success→Success | Net gain |
|---|---:|---:|---:|---:|---:|
| ReAct → Ordinary | 1 | 11 | 6 | 116 | +5 |
| Ordinary → Unverified | 1 | 6 | 4 | 123 | +2 |
| Unverified → Verified | 1 | 4 | 2 | 127 | +2 |
| ReAct → Verified | 1 | 11 | 2 | 120 | +9 |

### 8.3 解释

Verified 相对 Unverified：

- `valid_seen`：修复 5 个 episode，回退 4 个，净增 1 个成功任务。
- `valid_unseen`：修复 4 个 episode，回退 2 个，净增 2 个成功任务。

Verified 相对 ReAct：

- `valid_seen`：修复 13 个 episode，回退 5 个，净增 8 个。
- `valid_unseen`：修复 11 个 episode，回退 2 个，净增 9 个。

因此，真实环境验证表现出方向一致的净收益，但**没有消除负迁移**。仅凭目前的配对计数不能宣称统计显著性。后续计划使用 exact McNemar test 并结合 case study 分析。

## 9. 动作格式与有效性（已记录的摘要）

| Split / Method | Format errors | Inadmissible actions | Truncated rate |
|---|---:|---:|---:|
| seen / ReAct | 5 | 100 | 2.86% |
| seen / Ordinary | 0 | 28 | 0.00% |
| seen / Unverified | 0 | 27 | 0.00% |
| seen / Verified | 3 | 73 | 0.71% |
| unseen / ReAct | 0 | 45 | 0.00% |
| unseen / Ordinary | 10 | 81 | 0.75% |
| unseen / Unverified | 0 | 24 | 0.00% |
| unseen / Verified | 0 | 22 | 0.00% |

这组指标表明记忆方法的动作格式/合法性表现并不一致；例如 Ordinary 在 unseen 上的格式错误和不可执行动作多于 Verified。应在 episode 级别检查原因，避免仅凭聚合指标作机制归因。

## 10. 实验文件与复现

### 10.1 训练阶段

```text
outputs/baseline_qwen3.8-chat_train_seed42_1000/
outputs/failure_analysis_qwen3.8-chat_train_seed42_1000/
outputs/counterfactual_qwen3.8-chat_train_seed42_1000/
outputs/ordinary_experiences_qwen3.8-chat_train_seed42_1000/
outputs/unverified_experiences_qwen3.8-chat_train_seed42_1000/
outputs/experiences_qwen3.8-chat_train_seed42_1000/
```

### 10.2 测试阶段

```text
outputs/baseline_qwen3.8-chat_valid_seen_seed42_140/
outputs/baseline_qwen3.8-chat_valid_unseen_seed42_134/
outputs/ordinary_memory_qwen3.8-chat_valid_seen_seed42_140/
outputs/ordinary_memory_qwen3.8-chat_valid_unseen_seed42_134/
outputs/unverified_memory_qwen3.8-chat_valid_seen_seed42_140/
outputs/unverified_memory_qwen3.8-chat_valid_unseen_seed42_134/
outputs/memory_qwen3.8-chat_valid_seen_seed42_140/
outputs/memory_qwen3.8-chat_valid_unseen_seed42_134/
```

### 10.3 配对比较

```bash
uv run python -m scripts.compare_methods
```

脚本读取各方法的 `episode_*.json`，根据顶层 `success` 字段按 episode 文件名对齐，输出四种方法的配对转移计数。此脚本目前**尚未计算 McNemar p-value**。

## 11. 当前结论与证据边界

1. **经验记忆具有正向总体效果**：Ordinary 在两个 held-out split 上均优于无记忆 ReAct。
2. **结构化失败分析具有进一步收益**：Unverified 在两个 split 上均优于 Ordinary。
3. **环境反事实验证提供初步额外收益**：Verified 在两个 split 上的成功率和平均决策步数均优于 Unverified，但净成功数分别仅多 1 和 2。
4. **负迁移仍然存在**：配对比较显示部分原本成功的任务会在引入或更换经验记忆后失败。

这些观察尚不足以证明各阶段的性能差异均具有统计显著性；不同经验提炼输入与生成内容可能同时影响结果。不能把所有最终收益单独归因于验证机制。

## 12. 下一步实验（TODO）

- [ ] Exact McNemar paired test、置信区间与必要的多次重复实验。
- [ ] 选取 `fail→success` 和 `success→fail` 的 episode，检查检索经验、决策轨迹和负迁移原因。
- [ ] Reflexion baseline（保持模型、任务清单和评估协议一致）。
- [ ] 经验库规模/积累阶段对成功率的影响曲线。
- [ ] API 调用次数、输入输出 token 与实际成本统计。
- [ ] BM25 `source_task` 与其他经验检索字段/策略的消融。

## 13. 复现注意事项

- 当前脚本主要通过文件顶部常量配置路径，执行前核对 manifest、模型和输出目录。
- 生成失败分析、反事实验证和经验提炼会调用 LLM API，成本较高；已有结果可复用，不必重跑。
- 对比时必须使用相同 split 和相同 episode 文件集合。
- 输出目录包含历史实验记录，避免误删或覆盖。
- 目前尚无完整 API 成本指标，不应在报告中填写估计值作为实测结果。

