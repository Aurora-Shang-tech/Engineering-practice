"""测试FailureAnalyzer"""

from pathlib import Path
from src.analysis.failure_analyzer import FailureAnalyzer
from src.core.trajectory import Trajectory
from src.llm.client import LLMClient

TRAJECTORY_PATH = Path("outputs/baseline_qwen3.8-chat_train_seed42_50/episode_005.json")

def main():
    trajectory = Trajectory.load_json(TRAJECTORY_PATH)

    llm = LLMClient()
    analyzer = FailureAnalyzer(llm)

    analysis = analyzer.analyze(trajectory)

    print("=" * 60)
    print("Failure Analysis")
    print("=" * 60)

    print(f"Critical step: {analysis.critical_step}")
    print(f"Failure type: {analysis.failure_type}")
    print(f"Failure reason: {analysis.failure_reason}")

    print()
    print(f"Original action: {analysis.original_action}")
    print("Counterfactual actions:")
    for index, action in enumerate(analysis.counterfactual_actions, start=1):
        print(f" {index}. {action}")

    print()
    print(f"Expected effect: {analysis.expected_effect}")

if __name__ == "__main__":
    main()
