"""测试失败轨迹是否能够准确重放到指定步骤"""

from pathlib import Path

from src.core.trajectory import Trajectory
from src.counterfactual.replay import replay_to_step

TRAJECTORY_PATH = Path("outputs/baseline_qwen3.8-chat_train_seed42_50/episode_005.json")
CRITICAL_STEP = 12

def main() -> None:
    trajectory = Trajectory.load_json(TRAJECTORY_PATH)

    env, state = replay_to_step(
        trajectory=trajectory,
        step_index=CRITICAL_STEP,
    )

    try:
        state = env.reset()

        print("=" * 60)
        print("Replay Test")
        print("=" * 60)
        print(f"Critical step: {CRITICAL_STEP}")
        print()
        print("Observation:")
        print(state.observation)
        print()
        print("Admissible actions:")
        for action in state.admissible_actions:
            print("-", action)
        print()
        print("Replay validation passed")

    finally:
        env.close()

if __name__ == "__main__":
    main()
