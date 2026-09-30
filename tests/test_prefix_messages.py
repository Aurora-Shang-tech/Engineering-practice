"""测试从Trajectory恢复ReAct对话历史"""

from pathlib import Path

from src.core.protocol import trajectory_prefix_messages
from src.core.trajectory import Trajectory

TRAJECTORY_PATH = Path("outputs/baseline_qwen3.8-chat_train_seed42_50/episode_005.json")
CRITICAL_STEP = 12

def main():
    trajectory = Trajectory.load_json(TRAJECTORY_PATH)

    messages = trajectory_prefix_messages(
        trajectory=trajectory,
        step_index=CRITICAL_STEP,
    )

    print("=" * 60)
    print("Trajectory Prefix Messages")
    print("=" * 60)

    print(f"Critical step: {CRITICAL_STEP}")
    print(f"Messages: {len(messages)}")

    print()
    print("=" * 60)
    print("Last messages")
    print("=" * 60)

    # 最后4条就够我们检查边界。
    for message in messages[-4:]:
        print()
        print(f"[{message['role'].upper()}]")
        print(message["content"])


if __name__ == "__main__":
    main()
