"""从完整失败轨迹中直接提炼普通经验"""

from src.core.trajectory import Trajectory
from src.memory.experience import Experience

class OrdinaryExperienceExtractor:
    """从失败轨迹中直接提炼可复用经验"""

    def __init__(self, llm):
        self.llm = llm

    def extract(
        self,
        trajectory: Trajectory,
    ) -> Experience:
        """从完整失败轨迹中直接总结一条经验"""

        prompt = self._build_prompt(trajectory)
        messages = [
            {
                "role": "system",
                "content": (
                    "You extract reusable lessons from failed ALFWorld "
                    "agent trajectories.\n\n"
                    "Analyze the failed trajectory and summarize one general "
                    "decision rule that can help solve similar future tasks.\n"
                    "Do not merely repeat a specific action, location, "
                    "step number, or object instance.\n"
                    "Do not invent counterfactual outcomes or claim that an "
                    "alternative action would have succeeded.\n\n"
                    "Return exactly this format:\n"
                    "Failure type: <short failure category>\n"
                    "Lesson: <one concise reusable lesson>"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        response = self.llm.chat(messages)
        failure_type, lesson = self._parse_response(response)

        return Experience(
            failure_type=failure_type,
            lesson=lesson,
            source_task=trajectory.task,
        )

    def _build_prompt(
        self,
        trajectory: Trajectory,
    ) -> str:
        """构造普通经验提炼prompt"""

        steps = []

        for index, step in enumerate(trajectory.steps, start=1):
            steps.append(
                f"Step {index}:\n"
                f"Observation: {step.observation}\n"
                f"Thought: {step.thought}\n"
                f"Action: {step.action}\n"
                f"Next observation: {step.next_observation}"
            )

        trajectory_text = "\n\n".join(steps)

        return (
            f"Task:\n{trajectory.task}\n\n"
            f"Task success:\n{trajectory.success}\n\n"
            f"Trajectory:\n{trajectory_text}"
        )

    @staticmethod
    def _parse_response(
        response: str,
    ) -> tuple[str, str]:
        """解析LLM生成的失败类型和经验"""

        failure_type = None
        lesson = None

        for line in response.splitlines():
            stripped = line.strip()

            if stripped.startswith("Failure type:"):
                failure_type = stripped[len("Failure type:"):].strip()

            elif stripped.startswith("Lesson:"):
                lesson = stripped[len("Lesson:"):].strip()

        if not failure_type:
            raise ValueError("Experience response is missing Failure type")

        if not lesson:
            raise ValueError("Experience response is missing Lesson")

        return failure_type, lesson
