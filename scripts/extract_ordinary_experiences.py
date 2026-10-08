"""批量从失败轨迹结果中提炼经验"""

import json
from pathlib import Path

from src.core.trajectory import Trajectory
from src.llm.client import LLMClient
from src.memory.ordinary_extractor import OrdinaryExperienceExtractor

TRAJECTORY_DIR = Path("outputs/baseline_qwen3.8-chat_train_seed42_1000")
OUTPUT_DIR = Path("outputs/ordinary_experiences_qwen3.8-chat_train_seed42_1000")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    llm = LLMClient()
    extractor = OrdinaryExperienceExtractor(llm)

    trajectory_paths = sorted(TRAJECTORY_DIR.glob("episode_*.json"))

    failure_paths = []

    for trajectory_path in trajectory_paths:
        trajectory = Trajectory.load_json(str(trajectory_path))

        if not trajectory.success:
            failure_paths.append(trajectory_path)

    print(f"Found {len(failure_paths)} failed trajectories")

    extracted_count = 0

    for index, trajectory_path in enumerate(failure_paths, start=1):
        episode_name = trajectory_path.stem
        output_path = OUTPUT_DIR / f"{episode_name}.json"

        print()
        print("=" * 60)
        print(
            f"Experience {index}/{len(failure_paths)}: "
            f"{episode_name}"
        )
        print("=" * 60)

        # 已提炼的经验直接跳过，支持断点续跑
        if output_path.exists():
            print("Already extracted, skipping.")
            continue

        trajectory = Trajectory.load_json(str(trajectory_path))

        experience = extractor.extract(
            trajectory=trajectory,
        )

        result = {
            "source_episode": episode_name,
            "source_task": experience.source_task,
            "failure_type": experience.failure_type,
            "lesson": experience.lesson,
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        extracted_count += 1

        print(f"Failure type: {experience.failure_type}")
        print(f"Lesson: {experience.lesson}")
        print(f"Saved: {output_path}")

    print()
    print("=" * 60)
    print("Ordinary Experience Extraction Summary")
    print("=" * 60)
    print(f"Failures: {len(failure_paths)}")
    print(f"Newly extracted: {extracted_count}")


if __name__ == "__main__":
    main()

