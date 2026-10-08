# Agent Self-Evolution

**Training-Free LLM Agent Self-Evolution via Counterfactual Verification and Experience Memory**

本项目研究如何在**不更新大语言模型参数**的条件下，让 LLM Agent 从 ALFWorld 任务执行的失败反馈中提炼经验，并利用真实环境中的反事实验证提高经验的可靠性。Agent 在后续任务中通过检索长期经验记忆改善决策。

> 当前状态：已完成 ReAct 基线、失败分析、反事实验证、三种经验记忆构建，以及 `valid_seen` / `valid_unseen` 上四种方法的配对评估。统计显著性检验、Reflexion 对照、经验增长曲线和 API 成本统计尚未完成。

## 1. 研究问题

常规 ReAct Agent 每次执行任务时主要依赖当前上下文，难以持续利用历史失败经验。本项目关注三个问题：

1. 从失败轨迹直接提炼的经验，能否帮助后续任务？
2. 对失败轨迹进行关键步骤分析和反事实动作推理，能否改善经验质量？
3. 将反事实候选动作放回**真实环境重放并验证**，是否能在未经验证的 LLM 经验之上提供额外收益？

这里的 *self-evolution* 是指经验库及其辅助决策能力随历史反馈改进，**不是** PPO、GRPO 或其他模型参数训练。

## 2. 方法概览

```text
ALFWorld tasks
    │
    ▼
ReAct Agent ───────────────► Episode trajectories
                                  │
                                  ▼
                           Failed trajectories
                                  │
                         Failure analysis
                    (critical step / cause /
                     counterfactual candidates)
                                  │
                                  ▼
                      Environment replay
                    + counterfactual validation
                                  │
                                  ▼
                         Experience extraction
                                  │
                                  ▼
                         ExperienceStore
                       (exact deduplication)
                                  │
                     BM25(source_task), Top-K=3
                                  │
                                  ▼
                     Retrieved lessons injected
                       into ReAct initial prompt
                                  │
                                  ▼
                        Held-out evaluation
```

### 2.1 失败分析

`FailureAnalyzer` 从失败轨迹中识别关键决策步骤、失败类型和原因，并生成替代动作候选及预期影响。

### 2.2 反事实验证

对于候选动作，验证器在相同任务环境中重放关键步骤之前的历史动作，替换关键动作，然后继续运行 Agent，记录后续任务是否成功。**候选动作由 LLM 提议，验证结果来自环境执行**。该过程提供实证反馈，但不等同于严格因果识别。

### 2.3 经验提炼与检索

从验证结果提炼可复用的通用 `lesson`，连同 `source_task` 和 `failure_type` 保存到经验库。经验使用精确 `lesson` 字符串去重；检索阶段以当前任务为查询，对历史 `source_task` 使用 BM25 排序，选取 Top-3 条经验注入 ReAct 初始提示词。`failure_type` 不作为当前检索过滤条件。

## 3. 项目结构

以下列出已使用的主要文件；并非仓库全部内容。

```text
src/
├── agents/react_agent.py             # ReAct 与经验注入
├── analysis/failure_analyzer.py      # 失败分析
├── core/
│   ├── protocol.py                  # Agent 消息和 Prompt
│   └── trajectory.py                # 轨迹数据结构及 JSON I/O
├── counterfactual/verifier.py        # 反事实环境重放
├── data/
│   ├── alfworld_data.py             # ALFWorld 任务发现
│   └── manifest.py                  # 固定任务清单
├── envs/environment.py              # ALFWorld 环境封装
├── llm/client.py                     # LLM API 客户端
└── memory/
    ├── experience.py                # 经验数据结构
    ├── store.py                     # BM25 经验库
    ├── extractor.py                 # 已验证经验提炼
    ├── unverified_extractor.py      # 未验证关键步骤经验
    └── ordinary_extractor.py        # 普通失败轨迹经验

scripts/
├── create_manifest.py
├── evaluate_baseline.py
├── analyze_failures.py
├── verify_counterfactuals.py
├── extract_experiences.py
├── extract_unverified_experiences.py
├── extract_ordinary_experiences.py
├── build_experience_store.py
├── build_unverified_experience_store.py
├── build_ordinary_experience_store.py
├── evaluate_valid_seen_memory.py
├── evaluate_valid_unseen_memory.py
├── evaluate_valid_seen_unverified_memory.py
├── evaluate_valid_unseen_unverified_memory.py
├── evaluate_valid_seen_ordinary_memory.py
├── evaluate_valid_unseen_ordinary_memory.py
└── compare_methods.py

docs/
└── experiment_results.md
```

## 4. 实验设置

- **环境**：ALFWorld 文本交互任务。
- **模型**：`qwen3.8-chat`（通过项目的 LLM API 客户端调用）。
- **任务清单**：固定随机种子 `42`，通过 manifest 固定各 split 的 episode。
- **步数上限**：每个 episode 最多 50 个 Agent 决策步骤，并限制环境交互步数。
- **经验来源**：训练集 1000 个 episode 的失败轨迹；测试集不参与经验构建。
- **检索**：`BM25(source_task)`，`Top-K=3`。
- **经验去重**：完全相同的 `lesson` 字符串去重。

| Split | Episodes | 用途 |
|---|---:|---|
| `train` | 1000 | 运行基线，收集失败并构建经验库 |
| `valid_seen` | 140 | Held-out evaluation |
| `valid_unseen` | 134 | Held-out evaluation（未见场景） |

`140` 和 `134` 是当前任务发现与过滤规则下可用的对应 split 任务数。任务 manifest 保证四种方法在同一批 episode 上比较。

## 5. 对照方法

| 方法 | 经验来源 | 关键步骤分析 | 反事实环境验证 |
|---|---|---|---|
| **ReAct** | 不使用长期经验 | 否 | 否 |
| **Ordinary Memory** | 直接总结完整失败轨迹 | 否 | 否 |
| **Unverified Memory** | 失败分析 + LLM 反事实候选 | 是 | 否 |
| **CF-Verified Memory (Ours)** | 失败分析 + 环境验证结果 | 是 | 是 |

三个记忆方法使用相同的经验库格式、精确文本去重规则、BM25 任务相似度检索和 Top-K 设置。各经验提炼 Prompt 的输入内容随方法不同，这是对照方法定义的一部分。

## 6. 主要实验结果

### 6.1 成功率与平均决策步数

| Method | valid_seen | valid_unseen | Seen Avg Steps | Unseen Avg Steps |
|---|---:|---:|---:|---:|
| ReAct | 120/140 (85.71%) | 122/134 (91.04%) | 19.97 | 18.24 |
| Ordinary Memory | 122/140 (87.14%) | 127/134 (94.78%) | 18.59 | 15.32 |
| Unverified Memory | 127/140 (90.71%) | 129/134 (96.27%) | 17.01 | 14.55 |
| **CF-Verified Memory (Ours)** | **128/140 (91.43%)** | **131/134 (97.76%)** | **15.54** | **13.22** |

与 ReAct 相比，CF-Verified Memory 在 `valid_seen` / `valid_unseen` 上分别提高 **5.72 / 6.72 个百分点**，平均决策步数分别下降约 **22.2% / 27.5%**。

### 6.2 相同 episode 的配对比较

| Split | Comparison | Fail → Success | Success → Fail | Net Gain |
|---|---|---:|---:|---:|
| seen | ReAct → Verified | 13 | 5 | +8 |
| unseen | ReAct → Verified | 11 | 2 | +9 |
| seen | Unverified → Verified | 5 | 4 | +1 |
| unseen | Unverified → Verified | 4 | 2 | +2 |

配对分析显示 Verified 方法既能修复部分失败任务，也会使少量原本成功的任务失败。反事实验证相对于 Unverified 的额外成功率收益较小；**目前不能声称具有统计显著性**。

完整数据、任务类型分析及实验局限见 [实验结果报告](docs/experiment_results.md)。

## 7. 运行与复现

项目使用 Python 3.11、`uv` 管理的虚拟环境，以及已安装的 ALFWorld 资源。需要先按本地 LLM 客户端配置可用的模型名称和 API 访问凭据；不要将密钥提交到 Git。

以下命令对应当前已有脚本，部分脚本在源码中使用固定路径常量；运行前请核对脚本顶部的输入/输出目录，避免覆盖或混用实验结果。

```bash
# 基线（训练集；已有结果无需重复运行）
uv run python -m scripts.evaluate_baseline

# 失败分析与反事实验证（计算/API 成本较高）
uv run python -m scripts.analyze_failures
uv run python -m scripts.verify_counterfactuals

# 三类经验提炼
uv run python -m scripts.extract_ordinary_experiences
uv run python -m scripts.extract_unverified_experiences
uv run python -m scripts.extract_experiences

# 构建三类经验库
uv run python -m scripts.build_ordinary_experience_store
uv run python -m scripts.build_unverified_experience_store
uv run python -m scripts.build_experience_store

# Held-out 评估（脚本内路径需与目标 split 匹配）
uv run python -m scripts.evaluate_valid_seen_ordinary_memory
uv run python -m scripts.evaluate_valid_unseen_ordinary_memory
uv run python -m scripts.evaluate_valid_seen_unverified_memory
uv run python -m scripts.evaluate_valid_unseen_unverified_memory
uv run python -m scripts.evaluate_valid_seen_memory
uv run python -m scripts.evaluate_valid_unseen_memory

# Episode-level 配对统计（无需 API 调用）
uv run python -m scripts.compare_methods
```

ReAct 的 held-out 基线结果保存在 `outputs/baseline_qwen3.8-chat_valid_seen_seed42_140` 和 `outputs/baseline_qwen3.8-chat_valid_unseen_seed42_134`。当前 `scripts/evaluate_baseline.py` 的默认常量可能仍指向训练集；如需重新运行 held-out 基线，应先确认相应 manifest 与输出路径。

### 7.1 主要输出

```text
outputs/
├── manifests/
│   ├── train_seed42_1000.json
│   ├── valid_seen_seed42_140.json
│   └── valid_unseen_seed42_134.json
├── baseline_qwen3.8-chat_train_seed42_1000/
├── failure_analysis_qwen3.8-chat_train_seed42_1000/
├── counterfactual_qwen3.8-chat_train_seed42_1000/
├── ordinary_experiences_qwen3.8-chat_train_seed42_1000/
├── unverified_experiences_qwen3.8-chat_train_seed42_1000/
├── experiences_qwen3.8-chat_train_seed42_1000/
└── memory/
    ├── ordinary_experience_store_qwen3.8-chat_train_seed42_1000.json
    ├── unverified_experience_store_qwen3.8-chat_train_seed42_1000.json
    └── experience_store_qwen3.8-chat_train_seed42_1000.json
```

## 8. 已完成与后续工作

**已完成**：ALFWorld ReAct 基线、1000 条训练轨迹、133 条失败分析、371 个反事实候选的验证、三类经验提炼与经验库、seen/unseen 四方法评估、episode-level 配对计数。

**待完成**：

- Exact McNemar 配对检验、置信区间及必要的重复实验。
- 对修复和回退任务进行 case study，分析 negative transfer。
- Reflexion 对照实验。
- 随经验积累数量变化的学习曲线。
- API 调用次数、token 使用量及成本统计。
- 可选的检索策略消融（`source_task`、`lesson`、组合检索等）。

## 9. 局限性

当前结果基于单一模型和固定任务清单；虽然四种方法在同一批任务上评估，但尚未完成统计显著性检验或跨模型验证。Verified 相比 Unverified 的净成功任务数较小，不能据此断言反事实验证具有普遍或显著的优势。实际环境重放提供了可观测的验证反馈，但并不自动构成严格的因果归因。经验库也可能产生负迁移。

## 10. 研究定位

本项目的重点是 **Training-Free Self-Evolution + Counterfactual Verification + Experience Memory**：通过环境反馈和经验积累增强 Agent，而非更新基础模型参数。

