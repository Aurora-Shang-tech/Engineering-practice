"""批量验证失败分析中的反事实动作"""

import json
from pathlib import Path

from src.agents.react_agent import ReActAgent
from src.core.trajectory import Trajectory
from src.counterfactual.verifier import verify_counterfactual_candidates
from src.llm.client import LLMClient

MAX_STEPS = 50
TRAJECTORY_DIR = Path("outputs/baseline_qwen3.8-chat_train_seed42_1000")
ANALYSIS_DIR = Path("outputs/failure_analysis_qwen3.8-chat_train_seed42_1000")
OUTPUT_DIR = Path("outputs/counterfactual_qwen3.8-chat_train_seed42_1000")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    llm = LLMClient()
    agent = ReActAgent(
        llm=llm,
        max_steps=MAX_STEPS,
    )

    analysis_paths = sorted(ANALYSIS_DIR.glob("episode_*.json"))
    print(f"Found {len(analysis_paths)} failure analysed")

    verified_count = 0
    candidate_count = 0
    success_count = 0

    for index, analysis_path in enumerate(analysis_paths, start=1):
        episode_name = analysis_path.stem

        trajectory_path = TRAJECTORY_DIR / f"{episode_name}.json"
        output_path = OUTPUT_DIR / f"{episode_name}.json"

        print()
        print("=" * 60)
        print(
            f"Failure {index}/{len(analysis_paths)}: "
            f"{episode_name}"
        )
        print("=" * 60)

        # 已验证的失败直接跳过，支持断点续跑
        if output_path.exists():
            print("Already verified, skipping.")
            continue

        trajectory = Trajectory.load_json(str(trajectory_path))

        with analysis_path.open("r", encoding="utf-8") as f:
            analysis = json.load(f)

        critical_step = analysis["critical_step"]
        counterfactual_actions = tuple(analysis["counterfactual_actions"])

        print(f"Task: {trajectory.task}")
        print(f"Critical step: {critical_step}")
        print(
            f"Counterfactual candidates: "
            f"{len(counterfactual_actions)}"
        )

        result = verify_counterfactual_candidates(
            trajectory=trajectory,
            critical_step=critical_step,
            counterfactual_actions=counterfactual_actions,
            agent=agent,
            max_steps=MAX_STEPS,
        )

        candidate_results = []

        for candidate_index, candidate in enumerate(result.results, start=1):
            candidate_count += 1
            success_count += int(candidate.success)

            trajectory_output_path = None

            if candidate.trajectory is not None:
                trajectory_output_path = OUTPUT_DIR / (f"{episode_name}_candidate_{candidate_index}.json")

                candidate.trajectory.save_json(str(trajectory_output_path))

            candidate_results.append(
                {
                    "action": candidate.counterfactual_action,
                    "success": candidate.success,
                    "continuation_steps": candidate.continuation_steps,
                    "trajectory_path": str(trajectory_output_path) if trajectory_output_path is not None else None,
                }
            )

            print(
                f"Candidate {candidate_index}: "
                f"{candidate.counterfactual_action}"
            )
            print(f"  Success: {candidate.success}")
            print(
                f"  Continuation steps: "
                f"{candidate.continuation_steps}"
            )

        output = {
            "trajectory_path": str(trajectory_path),
            "analysis_path": str(analysis_path),
            "task": trajectory.task,
            "critical_step": result.critical_step,
            "original_action": result.original_action,
            "candidates": candidate_results,
        }

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                output,
                f,
                ensure_ascii=False,
                indent=2,
            )

        verified_count += 1

        print(f"Saved: {output_path}")

    print()
    print("=" * 60)
    print("Counterfactual Verification Summary")
    print("=" * 60)
    print(f"Failures: {len(analysis_paths)}")
    print(f"Newly verified: {verified_count}")
    print(f"Candidates tested: {candidate_count}")
    print(f"Successful candidates: {success_count}")


if __name__ == "__main__":
    main()

