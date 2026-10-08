"""测试经验提炼Prompt"""

from pathlib import Path

from src.analysis.failure_analyzer import FailureAnalysis
from src.core.trajectory import Trajectory
from src.counterfactual.verifier import CounterfactualBatchResult, CounterfactualResult
from src.llm.client import LLMClient
from src.memory.extractor import ExperienceExtractor

TRAJECTORY_PATH = Path("outputs/baseline_qwen3.8-chat_train_seed42_50/episode_005.json")

def main():
    trajectory = Trajectory.load_json(str(TRAJECTORY_PATH))

    analysis = FailureAnalysis(
        critical_step=12,
        failure_type="Wrong object type",
        failure_reason=(
            'The task asks for a "cup", but the agent treated '
            'a "mug" as a cup.'
        ),
        original_action="take mug 1 from countertop 1",
        counterfactual_actions=(
            "go to countertop 2",
            "go to countertop 3",
            "go to cabinet 4",
        ),
        expected_effect="Continue searching for the required object type.",
    )

    verification = CounterfactualBatchResult(
        critical_step=12,
        original_action="take mug 1 from countertop 1",
        results=(
            CounterfactualResult(
                critical_step=12,
                original_action="take mug 1 from countertop 1",
                counterfactual_action="go to countertop 2",
                success=False,
                continuation_steps=37,
            ),
            CounterfactualResult(
                critical_step=12,
                original_action="take mug 1 from countertop 1",
                counterfactual_action="go to countertop 3",
                success=False,
                continuation_steps=37,
            ),
            CounterfactualResult(
                critical_step=12,
                original_action="take mug 1 from countertop 1",
                counterfactual_action="go to cabinet 4",
                success=False,
                continuation_steps=37,
            ),
        ),
    )
    llm = LLMClient()

    extractor = ExperienceExtractor(llm)

    experience = extractor.extract(
        trajectory=trajectory,
        analysis=analysis,
        verification=verification,
    )

    print("=" * 60)
    print("Extracted Experience")
    print("=" * 60)
    print(f"Failure type: {experience.failure_type}")
    print(f"Lesson: {experience.lesson}")
    print(f"Source task: {experience.source_task}")


if __name__ == "__main__":
    main()
