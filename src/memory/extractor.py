"""从失败分析与反事实验证结果中提炼经验"""

from src.analysis.failure_analyzer import FailureAnalysis
from src.core.trajectory import Trajectory
from src.counterfactual.verifier import CounterfactualBatchResult
from src.memory.experience import Experience

class ExperienceExtractor:
    """将一次失败及其反验证结果提炼出可复用经验"""

    def __init__(self, llm):
        self.llm = llm

    def extract(
        self,
        trajectory: Trajectory,
        analysis: FailureAnalysis,
        verification: CounterfactualBatchResult,
    ) -> Experience:
        """提炼一条可复用经验"""
        
        prompt = self._build_prompt(
            trajectory=trajectory,
            analysis=analysis,
            verification=verification,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You extract reusable lessons from failed ALFWorld "
                    "agent trajectories and counterfactual experiments.\n\n"
                    "The lesson should describe a general decision rule "
                    "that can help solve future tasks.\n"
                    "Do not merely repeat a specific action, location, "
                    "step number, or object instance.\n"
                    "Do not claim that a counterfactual action succeeded "
                    "when the verification result says it failed.\n\n"
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
        verification: CounterfactualBatchResult,
    ) -> str:
        """构造经验提炼Prompt"""

        results = []

        for index, result in enumerate(verification.results, start=1):
            continuation_actions = []

            if result.trajectory is not None:
                continuation_actions = [
                    step.action
                    for step in result.trajectory.steps
                ]

            if continuation_actions:
                actions_text = "\n".join(f"- {action}" for action in continuation_actions)

            else:
                actions_text = "(no continuation)"

            results.append(
                f"Candidate {index}:\n"
                f"Action: {result.counterfactual_action}\n"
                f"Success: {result.success}\n"
                f"Continuation steps: {result.continuation_steps}\n"
                f"Continuation actions:\n"
                f"{actions_text}"
            )

        results_text = "\n\n".join(results)

        return (
            f"Task:\n{trajectory.task}\n\n"
            f"Failure type:\n{analysis.failure_type}\n\n"
            f"Failure reason:\n{analysis.failure_reason}\n\n"
            f"Original action:\n{analysis.original_action}\n\n"
            f"Counterfactual verification results:\n"
            f"{results_text}"
        )

    @staticmethod
    def _parse_response(response: str) -> str:
        """解析LLM生成的经验"""

        for line in response.splitlines():
            stripped = line.strip()

            if stripped.startswith("Lesson:"):
                lesson = stripped[len("Lesson:"):].strip()

                if not lesson:
                    raise ValueError("Experience lesson is empty")

                return lesson
        raise ValueError("Experience response is missing Lesson")
