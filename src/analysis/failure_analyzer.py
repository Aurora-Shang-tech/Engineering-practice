"""使用LLM分析ALFWorld失败轨迹"""

from dataclasses import dataclass

from src.core.trajectory import Trajectory

@dataclass(frozen=True)
class FailureAnalysis:
    """一次失败轨迹分析结果"""

    critical_step: int
    failure_type: str
    failure_reason: str

    original_action: str
    counterfactual_action: str
    expected_effect: str


class FailureAnalyzer:
    """分析失败轨迹并提出一个可验证的反事实动作"""

    def __init__(self, llm):
        self.llm = llm

    def analyze(self, trajectory: Trajectory) -> FailureAnalysis:
        """分析失败轨迹"""

        if trajectory.success:
            raise ValueError("Cannot analyze a successful trajectory")

        prompt = self._build_prompt(trajectory)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are analyzing a failed ALFWorld agent trajectory.\n"
                    "Your goal is to identify the earliest important decision "
                    "that caused or strongly contributed to the failure, and "
                    "propose one better counterfactual action.\n\n"

                    "ALFWorld uses symbolic object types. "
                    "Do not assume semantically similar object types are "
                    "interchangeable. For example, a mug is not necessarily "
                    "a cup.\n\n"

                    "Important rules:\n"
                    "1. Choose exactly one critical step.\n"
                    "2. Prefer the earliest decision whose replacement could "
                    "reasonably prevent the failure.\n"
                    "3. The counterfactual action MUST be copied exactly from "
                    "the admissible actions at that critical step.\n"
                    "4. The counterfactual action MUST differ from the "
                    "original action.\n"
                    "5. Do not use observations that were only available "
                    "after the critical step to justify the decision.\n"
                    "6. Do not assume the counterfactual succeeds. It will be "
                    "tested in the real environment later.\n\n"

                    "Return exactly this format:\n"
                    "Critical step: <integer>\n"
                    "Failure type: <short description>\n"
                    "Failure reason: <description>\n"
                    "Original action: <action>\n"
                    "Counterfactual action: <action>\n"
                    "Expected effect: <description>"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        response = self.llm.chat(messages)

        analysis = self._parse_response(response)

        self._validate_analysis(
            trajectory=trajectory,
            analysis=analysis,
        )

        return analysis

    def _build_prompt(self, trajectory: Trajectory) -> str:
        """把失败轨迹转换成分析Prompt"""

        parts = [
            f"Task:\n{trajectory.task}",
            (
                "Final result:\n"
                f"Success: {trajectory.success}\n"
                f"Truncated: {trajectory.truncated}"
            ),
            "Trajectory:",
        ]

        for index, step in enumerate(trajectory.steps):
            actions_text = "\n".join(f" - {action}" for action in step.admissible_actions)

            parts.append(
                f"\nStep {index}\n"
                f"Observation:\n{step.observation}\n\n"
                f"Thought:\n{step.thought}\n\n"
                f"Action:\n{step.action}\n\n"
                f"Valid format: {step.valid_format}\n"
                f"Admissible: {step.admissible}\n"
                f"Done: {step.done}\n"
                f"Won: {step.won}\n\n"
                f"Next observation:\n{step.next_observation}\n\n"
                f"Admissible actions before this decision:\n"
                f"{actions_text}"
            )

        return "\n\n".join(parts)

    @staticmethod
    def _parse_response(response: str) -> FailureAnalysis:
        """解析LLM返回的结构化失败分析"""

        fields = {}

        prefixes = {
            "Critical step:": "critical_step",
            "Failure type:": "failure_type",
            "Failure reason:": "failure_reason",
            "Original action:": "original_action",
            "Counterfactual action:": "counterfactual_action",
            "Expected effect:": "expected_effect",
        }

        for line in response.splitlines():
            stripped = line.strip()

            for prefix, field_name in prefixes.items():
                if stripped.startswith(prefix):
                    fields[field_name] = stripped[len(prefix):].strip()
                    break

        required = set(prefixes.values())
        missing = required - fields.keys()

        if missing:
            raise ValueError(f"Failure analysis response is missing fields:{sorted(missing)}")

        try:
            critical_step = int(fields["critical_step"])
        except ValueError as exc:
            raise ValueError(f"Critical step must be an integer {fields['critical_step']!r}") from exc

        return FailureAnalysis(
            critical_step=critical_step,
            failure_type=fields["failure_type"],
            failure_reason=fields["failure_reason"],
            original_action=fields["original_action"],
            counterfactual_action=fields["counterfactual_action"],
            expected_effect=fields["expected_effect"],
        )


    @staticmethod
    def _validate_analysis(
        trajectory: Trajectory,
        analysis: FailureAnalysis,
    ) -> None:
        """检查LLM提出的反事实是否与真实轨迹一致"""

        if not 0 <= analysis.critical_step < len(trajectory.steps):
            raise ValueError(f"Critical step is outside trajectory: {analysis.critical_step}")

        step =  trajectory.steps[analysis.critical_step]

        if analysis.original_action != step.action:
            raise ValueError(f"Original action does not match trajectory")

        if analysis.counterfactual_action == step.action:
            raise ValueError("Counterfactual action is identical to original action")

        if analysis.counterfactual_action not in step.admissible_actions:
            raise ValueError("Counterfactual action is not admissible")




