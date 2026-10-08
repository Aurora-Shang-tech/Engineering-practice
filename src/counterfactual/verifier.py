"""ALFWorld反事实验证"""

from dataclasses import dataclass

from src.core.protocol import environment_feedback, trajectory_prefix_messages
from src.core.trajectory import Trajectory
from src.counterfactual.replay import replay_to_step


@dataclass(frozen=True)
class CounterfactualResult:
    """一次完整反事实验证的结果"""

    critical_step: int
    original_action: str
    counterfactual_action: str

    success: bool
    continuation_steps: int
    trajectory: Trajectory | None = None

@dataclass(frozen=True)
class CounterfactualBatchResult:
    """多个反事实候选的验证结果"""

    critical_step: int
    original_action: str
    results: tuple[CounterfactualResult, ...]


def verify_counterfactual(
    trajectory: Trajectory,
    critical_step: int,
    counterfactual_action: str,
    agent,
    *,
    max_steps: int = 50,
) -> CounterfactualResult:
    """替换关键动作，并让Agent从反事实状态继续运行"""

    # 1. 检查critical_step
    if critical_step < 0 or critical_step >= len(trajectory.steps):
        raise IndexError(
            f"critical_step out of range: {critical_step}"
        )

    # 2. 原轨迹中的关键步骤
    original_step = trajectory.steps[critical_step]

    # 3. Replay到关键动作执行之前
    env, replayed = replay_to_step(
        trajectory=trajectory,
        step_index=critical_step,
        max_steps=max_steps,
    )

    try:
        # 4. 检查反事实动作是否合法
        if counterfactual_action not in replayed.admissible_actions:
            raise ValueError(
                "Counterfactual action is not admissible: "
                f"{counterfactual_action}"
            )

        # 5. 恢复关键动作之前的完整ReAct历史
        messages = trajectory_prefix_messages(
            trajectory=trajectory,
            step_index=critical_step,
        )

        # 6. 用反事实动作替换原来的关键动作
        messages.append(
            {
                "role": "assistant",
                "content": (
                    "Thought: Try an alternative action from the critical state.\n"
                    f"Action: {counterfactual_action}"
                ),
            }
        )

        # 7. 在真实恢复后的环境中执行反事实动作
        counterfactual_step = env.step(counterfactual_action)

        # 反事实动作本身已经直接完成任务
        if counterfactual_step.won:
            return CounterfactualResult(
                critical_step=critical_step,
                original_action=original_step.action,
                counterfactual_action=counterfactual_action,
                success=True,
                continuation_steps=0,
            )

        # 反事实动作导致episode结束，但任务失败
        if counterfactual_step.done:
            return CounterfactualResult(
                critical_step=critical_step,
                original_action=original_step.action,
                counterfactual_action=counterfactual_action,
                success=False,
                continuation_steps=0,
            )

        # 8. 把反事实动作产生的真实环境反馈加入历史
        feedback = environment_feedback(
            step=critical_step + 1,
            observation=counterfactual_step.observation,
            admissible_actions=counterfactual_step.admissible_actions,
            valid_format=True,
            admissible=counterfactual_step.admissible,
            done=counterfactual_step.done,
            won=counterfactual_step.won,
        )

        messages.append(
            {
                "role": "user",
                "content": feedback,
            }
        )

        # 9. 保持与baseline相同的总decision budget
        remaining_steps = max_steps - (critical_step + 1)

        if remaining_steps <= 0:
            return CounterfactualResult(
                critical_step=critical_step,
                original_action=original_step.action,
                counterfactual_action=counterfactual_action,
                success=False,
                continuation_steps=0,
            )

        # 10. 从反事实状态继续让ReAct Agent自主运行
        continuation = agent.continue_from(
            env,
            task=trajectory.task,
            gamefile=trajectory.gamefile,
            observation=counterfactual_step.observation,
            admissible_actions=counterfactual_step.admissible_actions,
            messages=messages,
            max_decisions=remaining_steps,
        )

        # 11. 返回完整反事实验证结果
        return CounterfactualResult(
            critical_step=critical_step,
            original_action=original_step.action,
            counterfactual_action=counterfactual_action,
            success=continuation.success,
            continuation_steps=len(continuation),
            trajectory=continuation,
        )

    finally:
        env.close()


def verify_counterfactual_candidates(
    trajectory: Trajectory,
    critical_step: int,
    counterfactual_actions: tuple[str, ...],
    agent,
    *,
    max_steps: int = 50,
) -> CounterfactualBatchResult:
    """逐个验证多个反事实候选动作"""

    if critical_step < 0 or critical_step >= len(trajectory.steps):
        raise IndexError(f"critical_step out of range: {critical_step}")

    original_step = trajectory.steps[critical_step]

    results = []

    for action in counterfactual_actions:
        result = verify_counterfactual(
            trajectory=trajectory,
            critical_step=critical_step,
            counterfactual_action=action,
            agent=agent,
            max_steps=max_steps,
        )
        results.append(result)

    return CounterfactualBatchResult(
        critical_step=critical_step,
        original_action=original_step.action,
        results=tuple(results),
    )
