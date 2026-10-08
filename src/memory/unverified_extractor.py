"""从失败分析中提炼未经环境验证的经验"""

from src.analysis.failure_analyzer import FailureAnalysis
from src.core.trajectory import Trajectory
from src.memory.experience import Experience

class UnverifiedExperienceExtractor:
    """从LLM失败分析中提炼未经反事实环境验证的经验"""

    def __init__(self, llm):
        self.llm = llm

    def extract(
        self,
        trajectory: Trajectory,
        analysis: FailureAnalysis,
    ) -> Experience:
        """提炼一条未经环境验证的可复用经验"""

        prompt = self._build_prompt(trajectory=trajectory, analysis=analysis)

        messages = [
            {
                "role": "system",
                "content": (
                    "You extract reusable lessons from failed ALFWorld "
                    "agent trajectories and LLM-generated counterfactual suggestions.\n\n"
                    "The lesson should describe a general decision rule "
                    "that can help solve future tasks.\n"
                    "Do not merely repeat a specific action, location, "
                    "step number, or object instance.\n"
                    "The counterfactual suggestions have NOT been verified "
                    "through environment execution. Do not claim that they succeeded.\n\n"
                    "Return exactly this format:\n"
                    "Lesson: <one concise reusable lesson>"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        response = self.llm.chat(messages)
        lesson = self._parse_response(response)

        return Experience(
            failure_type=analysis.failure_type,
            lesson=lesson,
            source_task=trajectory.task,
        )
    
    def _build_prompt(
        self,
        trajectory: Trajectory,
        analysis: FailureAnalysis,
    ) -> str:
        """构造未经环境验证的经验提炼Prompt"""

        candidates_text = "\n".join(f"- {action}" for action in analysis.counterfactual_actions)

        return (
            f"Task:\n{trajectory.task}\n\n"
            f"Failure type:\n{analysis.failure_type}\n\n"
            f"Failure reason:\n{analysis.failure_reason}\n\n"
            f"Original action:\n{analysis.original_action}\n\n"
            f"Counterfactual action candidates (unverified):\n"
            f"{candidates_text}\n\n"
            f"Expected effect:\n{analysis.expected_effect}"
        )

    @staticmethod
    def _parse_response(response: str) -> str:
        """解析 LLM 生成的经验"""

        for line in response.splitlines():
            stripped = line.strip()

            if stripped.startswith("Lesson:"):
                lesson = stripped[len("Lesson:"):].strip()

                if not lesson:
                    raise ValueError("Experience lesson is empty")

                return lesson

        raise ValueError("Experience response is missing Lesson")
