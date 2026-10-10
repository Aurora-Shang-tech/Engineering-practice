"""Reflexion Agent: 通过失败反思改进同一任务的后续尝试"""

from dataclasses import dataclass, field

from src.agents.react_agent import ReActAgent
from src.core.protocol import initial_messages
from src.core.trajectory import Trajectory
from src.envs.environment import ALFWorldEnvironment

@dataclass
class ReflexionResult:
    """记录同一任务的所有尝试及反思。"""

    task: str
    gamefile: str
    attempts: list[Trajectory] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return any(t.success for t in self.attempts)

    @property
    def total_steps(self) -> int:
        return sum(len(t.steps) for t in self.attempts)

class ReflexionAgent:
    """失败后生成语言反思,再尝试完成同一任务"""

    def __init__(
        self,
        llm,
        max_steps: int = 50,
        max_attempts: int = 2,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        self.llm = llm
        self.max_steps = max_steps
        self.max_attempts = max_attempts

        # 复用已有的ReAct决策循环
        self.react = ReActAgent(
            llm=llm,
            max_steps=max_steps,
        )

    def generate_reflection(
        self,
        trajectory: Trajectory,
    ) -> str:
        """根据失败轨迹生成下一次尝试的改进建议"""

        history = []

        for index, step in enumerate(trajectory.steps, start=1):
            history.append(
                f"Step {index}\n"
                f"Thought: {step.thought}\n"
                f"Action: {step.action}\n"
                f"Admissible: {step.admissible}\n"
                f"Observation: {step.next_observation}"
            )

        prompt = (
            "You are analyzing a failed ALFWorld task attempt.\n"
            "Identify the most important mistakes and explain "
            "how the agent should change its behavior next time.\n\n"
            f"Task: {trajectory.task}\n\n"
            "Failed trajectory:\n"
            + "\n\n".join(history)
            + "\n\n"
            "Write a concise reflection with actionable advice. "
            "Do not invent objects or actions. "
            "Pay attention to required state changes such as "
            "cleaning, cooling, and heating."
        )

        response = self.llm.chat([
            {
                "role": "system",
                "content": (
                    "You are an expert at reflecting on failed "
                    "ALFWorld agent trajectories."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ])

        return response.strip()

    def run(
        self,
        env: ALFWorldEnvironment,
    ) -> ReflexionResult:
        """执行最多max_attempts次尝试"""

        result = None
        reflections = []

        for attempt_index in range(self.max_attempts):

            # 第一次尝试直接使用原始 ReAct
            if attempt_index == 0:
                trajectory = self.react.run(env)

                result = ReflexionResult(
                    task=trajectory.task,
                    gamefile=trajectory.gamefile,
                )

            else:
                # reset() 将环境恢复到同一个 gamefile
                state = env.reset()

                if state.gamefile != result.gamefile:
                    raise RuntimeError(
                        "Reflexion retry loaded a different game."
                    )

                messages = initial_messages(
                    task=state.task,
                    observation=state.observation,
                    admissible_actions=state.admissible_actions,
                )

                # 把此前反思注入初始 Prompt
                reflection_text = "\n".join(
                    f"{i}. {reflection}"
                    for i, reflection in enumerate(
                        reflections, start=1
                    )
                )

                messages[1]["content"] += (
                    "\n\nReflections from previous failed "
                    "attempts on this same task:\n"
                    + reflection_text
                    + "\n\nUse these reflections to improve "
                    "your next attempt."
                )

                trajectory = self.react.continue_from(
                    env=env,
                    task=state.task,
                    gamefile=state.gamefile,
                    observation=state.observation,
                    admissible_actions=state.admissible_actions,
                    messages=messages,
                    max_decisions=self.max_steps,
                )

            result.attempts.append(trajectory)

            if trajectory.success:
                break

            # 最后一次失败后无需再生成反思
            if attempt_index + 1 < self.max_attempts:
                reflection = self.generate_reflection(trajectory)
                reflections.append(reflection)
                result.reflections.append(reflection)

        assert result is not None
        return result
