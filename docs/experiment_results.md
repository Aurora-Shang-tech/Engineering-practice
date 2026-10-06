# Agent Self-Evolution 实验结果

本文档记录 Agent Self-Evolution 项目当前已经完成的实验及结果。实验基于 ALFWorld 文本交互环境，核心方法为：

**Training-Free Self-Evolution + Counterfactual Credit Assignment + Experience Memory**

在不更新大语言模型参数的情况下，从失败轨迹中分析关键决策，通过真实环境中的反事实重放验证替代动作，并将验证结果提炼为可复用经验。后续任务通过 BM25 检索相关经验并注入 ReAct Agent 的上下文。

---

## 1. 实验设置

### 1.1 模型

主模型：

```text
qwen3.8-chat
```

Agent：

```text
ReAct
```

每个 episode 最大决策步数：

```text
50
```

### 1.2 数据划分

ALFWorld 当前可用任务：

| Split | 可用任务数 | 用途 |
|---|---:|---|
| train | 3553 | Experience Acquisition |
| valid_seen | 140 | Seen-scene Evaluation |
| valid_unseen | 134 | Unseen-scene Evaluation |

从 train split 中使用固定随机种子：

```text
seed = 42
```

采样 1000 个任务用于经验构建：

```text
outputs/manifests/train_seed42_1000.json
```

最终评测使用：

```text
outputs/manifests/valid_seen_seed42_140.json
outputs/manifests/valid_unseen_seed42_134.json
```

经验构建任务与最终评测任务分离。

---

## 2. Train Baseline

在 1000 个 train tasks 上运行 qwen3.8-chat ReAct Agent。

### 2.1 Overall Results

| Metric | Result |
|---|---:|
| Episodes | 1000 |
| Successes | 867 |
| Success Rate | **86.70%** |
| Failures | 133 |
| Truncated Rate | 1.90% |
| Average Decision Steps | 20.26 |
| Average Environment Steps | 20.19 |
| Format Errors | 67 |
| Inadmissible Actions | 670 |
| Format Valid Rate | 99.67% |
| Admissible Action Rate | 96.68% |
| Valid Action Rate | 96.36% |

### 2.2 Task-Type Results

| Task Type | Success |
|---|---:|
| look_at_obj_in_light | 80/88 = 90.91% |
| pick_and_place_simple | 205/208 = 98.56% |
| pick_clean_then_place_in_recep | 149/189 = 78.84% |
| pick_cool_then_place_in_recep | 105/138 = 76.09% |
| pick_heat_then_place_in_recep | 97/130 = 74.62% |
| pick_two_obj_and_place | 231/247 = 93.52% |

133 个失败任务中，clean / cool / heat 三类任务共占 106 个，约占全部失败的 79.7%。

---

## 3. Failure Analysis

对全部 133 个失败轨迹进行 LLM failure analysis。

分析内容包括：

- critical step
- failure type
- failure reason
- original action
- counterfactual action candidates
- expected effect

最终完成：

```text
133 / 133 failure trajectories
```

输出目录：

```text
outputs/failure_analysis_qwen3.8-chat_train_seed42_1000
```

---

## 4. Counterfactual Verification

对于 Failure Analyzer 生成的 counterfactual actions，从原始轨迹的 critical step 恢复环境状态。

每个候选动作独立执行：

```text
Original trajectory
        ↓
Replay to critical step
        ↓
Replace original action
        ↓
Execute counterfactual action
        ↓
Continue with the same ReAct Agent
        ↓
Observe final task outcome
```

### 4.1 Results

| Metric | Result |
|---|---:|
| Failed trajectories | 133 |
| Counterfactual candidates | 371 |
| Successful candidates | 197 |
| Failed candidates | 174 |
| Candidate Success Rate | **53.10%** |
| Failures with ≥1 successful candidate | 91 |
| Failures with all candidates failed | 42 |

因此：

```text
91 / 133 = 68.42%
```

的原始失败任务至少存在一个由 Failure Analyzer 提出的替代动作，使 Agent 从 critical state 继续执行后最终成功。

该结果表示候选动作经过真实环境执行验证，而不是仅依赖 LLM 对反事实结果的主观判断。

---

## 5. Experience Extraction

根据以下信息提炼可迁移经验：

```text
Failure trajectory
+ Failure analysis
+ Counterfactual candidates
+ Counterfactual verification outcome
+ Continuation actions
```

共生成：

```text
133 experiences
```

Exact deduplication 后：

```text
130 unique experiences
```

仅删除完全相同的 lesson，不进行 LLM semantic merging，以避免过度抽象或修改经过环境验证的经验。

最终 Experience Store：

```text
outputs/memory/experience_store_qwen3.8-chat_train_seed42_1000.json
```

---

## 6. Experience Retrieval

当前采用：

**BM25 Task-Similarity Experience Retrieval**

对于当前任务 `q` 和历史经验 `e_i`：

```text
BM25(current_task, experience.source_task)
```

按照 BM25 score 排序后检索：

```text
Top-K = 3
```

将对应经验的 `lesson` 注入 ReAct Agent 初始 prompt。

当前第一版方法仅使用 `source_task` 进行检索，不使用 `failure_type` 作为硬过滤条件。

---

## 7. Held-Out Evaluation

经验库仅由 train 1000 tasks 构建。

最终评测分别在：

```text
valid_seen
valid_unseen
```

进行。

比较：

```text
ReAct
vs.
ReAct + Counterfactual Experience Memory
```

模型、环境、manifest、最大步数和 Agent 主循环保持一致，主要差异为是否检索并注入历史经验。

---

## 8. Valid Seen Results

### 8.1 Overall

| Metric | ReAct | + CF Experience Memory |
|---|---:|---:|
| Successes | 120/140 | **128/140** |
| Success Rate | 85.71% | **91.43%** |
| Truncated Rate | 2.86% | **0.71%** |
| Avg. Decision Steps | 19.97 | **15.54** |
| Avg. Environment Steps | 19.94 | **15.51** |
| Format Errors | 5 | **3** |
| Inadmissible Actions | 100 | **73** |
| Format Valid Rate | 99.82% | **99.86%** |
| Admissible Action Rate | 96.42% | **96.64%** |
| Valid Action Rate | 96.24% | **96.51%** |

Success rate：

```text
85.71% → 91.43%
```

绝对提升：

```text
+5.72 percentage points
```

平均决策步数：

```text
19.97 → 15.54
```

下降约：

```text
22.2%
```

### 8.2 Task-Type Results

| Task Type | ReAct | + Memory | Δ |
|---|---:|---:|---:|
| look_at_obj_in_light | 92.31% | 92.31% | 0.00 pp |
| pick_and_place_simple | 100.00% | 100.00% | 0.00 pp |
| pick_clean_then_place_in_recep | 70.37% | **81.48%** | **+11.11 pp** |
| pick_cool_then_place_in_recep | 76.00% | **84.00%** | **+8.00 pp** |
| pick_heat_then_place_in_recep | 75.00% | **93.75%** | **+18.75 pp** |
| pick_two_obj_and_place | 95.83% | 95.83% | 0.00 pp |

---

## 9. Valid Unseen Results

### 9.1 Overall

| Metric | ReAct | + CF Experience Memory |
|---|---:|---:|
| Successes | 122/134 | **131/134** |
| Success Rate | 91.04% | **97.76%** |
| Truncated Rate | 0.00% | 0.00% |
| Avg. Decision Steps | 18.24 | **13.22** |
| Avg. Environment Steps | 18.24 | **13.22** |
| Format Errors | 0 | 0 |
| Inadmissible Actions | 45 | **22** |
| Format Valid Rate | 100.00% | 100.00% |
| Admissible Action Rate | 98.16% | **98.76%** |
| Valid Action Rate | 98.16% | **98.76%** |

Success rate：

```text
91.04% → 97.76%
```

绝对提升：

```text
+6.72 percentage points
```

平均决策步数：

```text
18.24 → 13.22
```

下降约：

```text
27.5%
```

Baseline 失败任务数从：

```text
12 → 3
```

减少 75%。

### 9.2 Task-Type Results

| Task Type | ReAct | + Memory | Δ |
|---|---:|---:|---:|
| look_at_obj_in_light | 100.00% | 100.00% | 0.00 pp |
| pick_and_place_simple | 100.00% | 100.00% | 0.00 pp |
| pick_clean_then_place_in_recep | 77.42% | **93.55%** | **+16.13 pp** |
| pick_cool_then_place_in_recep | 85.71% | **95.24%** | **+9.53 pp** |
| pick_heat_then_place_in_recep | 91.30% | **100.00%** | **+8.70 pp** |
| pick_two_obj_and_place | 100.00% | 100.00% | 0.00 pp |

---

## 10. Current Findings

目前实验结果表明，从训练任务失败轨迹中构建的 Experience Memory 能够迁移到 held-out ALFWorld tasks。

在两个 evaluation split 上均观察到：

1. 任务成功率提高；
2. 平均决策步数下降；
3. clean / cool / heat 等复杂状态操作任务提升最明显；
4. valid_unseen 上仍然获得明显提升，说明经验表现出跨未见场景迁移能力；
5. 当前未观察到 Memory 对已经接近或达到 100% 的简单任务造成明显性能下降。

当前结果支持 **Experience Memory 有效**，但尚不能将全部性能提升单独归因于 Counterfactual Verification。

原因是当前完整方法同时包含：

```text
Failure Analysis
+ Counterfactual Generation
+ Environment Verification
+ Experience Extraction
+ BM25 Retrieval
+ Memory Injection
```

因此仍需要进一步消融实验确定各模块贡献。

---

## 11. Planned Ablations

后续计划比较：

| Method | valid_seen | valid_unseen |
|---|---:|---:|
| ReAct | **85.71%** | **91.04%** |
| Reflexion | TBD | TBD |
| Ordinary Experience Memory | TBD | TBD |
| LLM Critical-Step Memory (Unverified) | TBD | TBD |
| Counterfactual-Verified Experience Memory (Ours) | **91.43%** | **97.76%** |

其中最关键的消融为：

```text
LLM Critical-Step Memory
vs.
Counterfactual-Verified Experience Memory
```

用于分析真实环境中的 counterfactual verification 是否能够提高经验质量。

后续还可以比较 Experience Retrieval：

```text
No Memory
Random Experience
BM25(source_task)
BM25(lesson)
BM25(source_task + lesson)
```

当前正式方法使用：

```text
BM25(source_task), Top-K = 3
```

---

## 12. Current Status

已完成：

```text
[✓] 1000-task ReAct baseline
[✓] 133 failure trajectory analyses
[✓] 371 counterfactual candidate verifications
[✓] 133 experience extractions
[✓] Exact deduplication → 130 experiences
[✓] ExperienceStore
[✓] BM25 task-similarity retrieval
[✓] Experience-Augmented ReAct
[✓] valid_seen ReAct evaluation
[✓] valid_seen Memory evaluation
[✓] valid_unseen ReAct evaluation
[✓] valid_unseen Memory evaluation
```

待完成：

```text
[ ] LLM Critical-Step Memory ablation
[ ] Ordinary Experience Memory baseline
[ ] Reflexion baseline
[ ] Retrieval ablations
[ ] API/token cost statistics
[ ] Final statistical analysis
[ ] Final report
```
