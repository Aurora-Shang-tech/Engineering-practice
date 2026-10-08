"""批量从失败分析和未经反事实验证结果中提炼经验"""

import json
from pathlib import Path

from src.analysis.failure_analyzer import FailureAnalysis
from src.core.trajectory import Trajectory
from src.llm.client import LLMClient
from src.memory.unverified_extractor import UnverifiedExperienceExtractor

TRAJECTORY_DIR = Path("outputs/baseline_qwen3.8-chat_train_seed42_1000")
ANALYSIS_DIR = Path("outputs/failure_analysis_qwen3.8-chat_train_seed42_1000")
OUTPUT_DIR = Path("outputs/unverified_experiences_qwen3.8-chat_train_seed42_1000")

def load_analysis(path: Path) -> FailureAnalysis:
    """从JSON恢复FailureAnalysis"""

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return FailureAnalysis(
        critical_step=data["critical_step"],
        failure_type=data["failure_type"],
        failure_reason=data["failure_reason"],
        original_action=data["original_action"],
        counterfactual_actions=tuple(
            data["counterfactual_actions"]
        ),
        expected_effect=data["expected_effect"],
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    llm = LLMClient()
    extractor = UnverifiedExperienceExtractor(llm)

    analysis_paths = sorted(ANALYSIS_DIR.glob("episode_*.json"))

    print(f"Found {len(analysis_paths)} failure analyses")

    extracted_count = 0

    for index, analysis_path in enumerate(analysis_paths, start=1):
        episode_name = analysis_path.stem

        trajectory_path = TRAJECTORY_DIR / f"{episode_name}.json"
        output_path = OUTPUT_DIR / f"{episode_name}.json"

        print()
        print("=" * 60)
        print(
            f"Experience {index}/{len(analysis_paths)}: "
            f"{episode_name}"
        )
        print("=" * 60)

        # 已提炼的经验直接跳过，支持断点续跑
        if output_path.exists():
            print("Already extracted, skipping.")
            continue

        trajectory = Trajectory.load_json(str(trajectory_path))
        analysis = load_analysis(analysis_path)

        experience = extractor.extract(
            trajectory=trajectory,
            analysis=analysis,
        )

        result = {
            "source_episode": episode_name,
            "source_task": experience.source_task,
            "failure_type": experience.failure_type,
            "lesson": experience.lesson,
            "critical_step": analysis.critical_step,
            "counterfactual_candidates": list(
                analysis.counterfactual_actions
            ),
            "expected_effect": analysis.expected_effect,
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        extracted_count += 1

        print(f"Failure type: {experience.failure_type}")
        print(f"Lesson: {experience.lesson}")
        print(f"Saved: {output_path}")

    print()
    print("=" * 60)
    print("Experience Extraction Summary")
    print("=" * 60)
    print(f"Failures: {len(analysis_paths)}")
    print(f"Newly extracted: {extracted_count}")


if __name__ == "__main__":
    main()

