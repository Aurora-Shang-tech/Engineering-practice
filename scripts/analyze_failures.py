
"""批量分析baseline中的失败轨迹"""

import json
from pathlib import Path

from src.analysis.failure_analyzer import FailureAnalyzer
from src.core.trajectory import Trajectory
from src.llm.client import LLMClient

TRAJECTORY_DIR = Path("outputs/baseline_qwen3.8-chat_train_seed42_1000")
OUTPUT_DIR = Path("outputs/failure_analysis_qwen3.8-chat_train_seed42_1000")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    llm = LLMClient()
    analyzer = FailureAnalyzer(llm)

    trajectory_paths = sorted(TRAJECTORY_DIR.glob("episode_*.json"))

    print(f"Found {len(trajectory_paths)} trajectories")

    failure_count = 0
    analyzed_count = 0
    
    for trajectory_path in trajectory_paths:
        trajectory = Trajectory.load_json(trajectory_path)

        # 成功轨迹不进行错误分析
        if trajectory.success:
            continue

        failure_count += 1

        output_path = OUTPUT_DIR / trajectory_path.name

        print()
        print("=" * 60)
        print(f"Failure {failure_count}: {trajectory_path.name}")
        print(f"Task: {trajectory.task}")
        print("=" * 60)

        # 已分析失败直接跳过，支持断点续跑
        if output_path.exists():
            print("Already analyzed, skipping")
            continue

        analysis = analyzer.analyze(trajectory)

        result = {
            "trajectory_path": str(trajectory_path),
            "task": trajectory.task,
            "critical_step": analysis.critical_step,
            "failure_type": analysis.failure_type,
            "failure_reason": analysis.failure_reason,
            "original_action": analysis.original_action,
            "counterfactual_actions": list(analysis.counterfactual_actions),
            "expected_effect": analysis.expected_effect,
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        analyzed_count += 1

        print(f"Critical step: {analysis.critical_step}")
        print(f"Failure type: {analysis.failure_type}")
        print(f"Saved: {output_path}")

    print()
    print("=" * 60)
    print("Failure Analysis Summary")
    print("=" * 60)
    print(f"Total trajectories: {len(trajectory_paths)}")
    print(f"Failures: {failure_count}")
    print(f"Newly analyzed: {analyzed_count}")


if __name__ == "__main__":
    main()
