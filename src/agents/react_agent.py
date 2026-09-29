"""基于ReAct的ALFWorld Agent"""

from src.core.protocol import environment_feedback, initial_messages, parse_assistant
from src.core.trajectory import Trajectory, TrajectoryStep
from src.envs.environment import ALFWorldEnvironment

class ReActAgent:
    """使用LLM在ALFWorld中执行任务"""

    def __init__(self, llm, max_steps: int = 50):
        self.llm = llm
        self.max_steps = max_steps

    def run(self, env: ALFWorldEnvironment) -> Trajectory:
        """在一个ALFWorld game中运行完整episode"""

        state = env.reset()
        trajectory = Trajectory(
            task=state.task,
            gamefile=state.gamefile
        )

        # messages同时承担完整交互历史
        messages = initial_messages(
            task=state.task,
            observation=state.observation,
            admissible_actions=state.admissible_actions,
        )

        return self._run_loop(
            env=env,
            trajectory=trajectory,
            messages=messages,
            observation=state.observation,
            admissible_actions=state.admissible_actions,
            max_decisions=self.max_steps,
        )


    def continue_from(
        self,
        env: ALFWorldEnvironment,
        *,
        task: str,
        gamefile: str,
        observation: str,
        admissible_actions: tuple[str, ...],
        messages: list[dict],
        max_decisions: int,
    ) -> Trajectory:
        """从已经恢复好的环境状态继续运行Agent"""

        trajectory = Trajectory(
            task=task,
            gamefile=gamefile,
        )

        return self._run_loop(
            env=env,
            trajectory=trajectory,
            messages=list(messages),
            observation=observation,
            admissible_actions=admissible_actions,
            max_decisions=max_decisions,
        )

    def _run_loop(
        self,
        env: ALFWorldEnvironment,
        *,
        trajectory: Trajectory,
        messages: list[dict],
        observation: str,
        admissible_actions: tuple[str, ...],
        max_decisions: int,
    ) -> Trajectory:
        """执行ReAct决策循环"""

        for step_index in range(max_decisions):
            response = self.llm.chat(messages)
            parsed = parse_assistant(response)

            # 把模型原始回答加入历史
            messages.append({
                "role": "assistant",
                "content": response,
            })

            if not parsed.valid_format:
                # 格式错误没有真正执行环境动作任记录为一次Agent决策
                trajectory.add_step(
                    TrajectoryStep(
                        observation=observation,
                        admissible_actions=list(admissible_actions),
                        thought=parsed.reasoning,
                        action=parsed.action or "",
                        valid_format=False,
                        admissible=False,
                        next_observation=observation,
                        score=0.0,
                        done=False,
                        won=False,
                    )
                )
                feedback = environment_feedback(
                    step = step_index + 1,
                    observation=observation,
                    admissible_actions=admissible_actions,
                    valid_format=False,
                    admissible=False,
                    done=False,
                    won=False,
                )
                messages.append({
                    "role": "user",
                    "content": feedback,
                })

                continue

            assert parsed.action is not None

            action = parsed.action

            env_step = env.step(action)

            trajectory.add_step(
                TrajectoryStep(
                    observation=observation,
                    admissible_actions=list(admissible_actions),
                    thought=parsed.reasoning,
                    action=action,
                    valid_format=True,
                    admissible=env_step.admissible,
                    next_observation=env_step.observation,
                    score=env_step.score,
                    done=env_step.done,
                    won=env_step.won,
                )
            )

            if env_step.won:
                trajectory.success = True
                break

            if env_step.done:
                break

            observation = env_step.observation
            admissible_actions = env_step.admissible_actions

            feedback = environment_feedback(
                step=step_index + 1,
                observation=observation,
                admissible_actions=admissible_actions,
                valid_format=True,
                admissible=env_step.admissible,
                done=env_step.done,
                won=env_step.won,
            )
            messages.append({
                "role": "user",
                "content": feedback,
            })
        else:
            trajectory.truncated = True

        return trajectory


