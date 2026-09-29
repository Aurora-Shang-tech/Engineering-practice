"""ALFWorld轨迹状态重放"""

from dataclasses import dataclass

from src.core.trajectory import Trajectory
from src.envs.environment import ALFWorldEnvironment

@dataclass(frozen=True)
class ReplayState:
    """重放到指定决策步骤后的环境状态"""

    observation: str
    admissible_actions: tuple[str, ...]

def replay_to_step(
    trajectory: Trajectory,
    step_index: int,
    max_steps: int = 50,
) -> tuple[ALFWorldEnvironment, ReplayState]:
    """重放轨迹，恢复到指定决策步骤执行之前的环境状态"""

    if step_index < 0 or step_index >= len(trajectory.steps):
        raise IndexError(f"Invalid step index: {step_index}")

    env = ALFWorldEnvironment(
        gamefile=trajectory.gamefile,
        max_steps=max_steps,
    )

    state = env.reset()

    observation = state.observation
    admissible_actions = state.admissible_actions

    try:
        # 只执行目标步骤之前的动作
        for index in range(step_index):
            step = trajectory.steps[index]

            if not step.valid_format:
                continue

            result = env.step(step.action)

            observation = result.observation
            admissible_actions = result.admissible_actions

        expected = trajectory.steps[step_index]

        if observation.strip() != expected.observation.strip():
            raise RuntimeError(f"Replay observation mismatch at step {step_index}")

        if tuple(admissible_actions) != tuple(expected.admissible_actions):
            raise RuntimeError(f"Replay admissible actions mismatch at step {step_index}")

        return env, ReplayState(observation=observation, admissible_actions=tuple(admissible_actions))

    except Exception:
        env.close()
        raise
