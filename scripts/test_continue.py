"""测试ReActAgent从replay后的中间状态继续运行"""

from pathlib import Path

from src.agents.react_agent import ReActAgent
from src.core.protocol import trajectory_prefix_messages
from src.core.trajectory import Trajectory
from src.counterfactual.replay import replay_to_step
from src.envs.environment import ALFWorldEnvironment
from src.llm.client import LLMClient

TRAJECTORY_PATH = Path("outputs/baseline_qwen3.8-chat_train_seed42_50/episode_005.json")
CRITICAL_STEP = 12
MAX_DECISIONS = 20

def main():
    # 1.读取失败轨迹
    original = Trajectory.load_json(TRAJECTORY_PATH)

    # 2.创建同一个ALFWorld环境
    env = ALFWorldEnvironment(
        gamefile=original.gamefile,
        max_steps=50,
    )

    try:
        # 3. replay到critical step 执行之前
        replayed = replay_to_step(
            env=env,
            trajectory=original,
            step_index=CRITICAL_STEP,
        )
        print("=" * 60)
        print("Replayed Critical State")
        print("=" * 60)

        print("Observation:")
        print(replayed.observation)

        print()
        print("Admissible actions:")
        for action in replayed.admissible_actions:
            print(f" - {action}")

        # 4.重建critical step之前的ReAct对话历史
        messages = trajectory_prefix_messages(
            trajectory=original,
            step_index=CRITICAL_STEP,
        )

        # 5.从当前状态继续运行
        llm = LLMClient()
        agent = ReActAgent(
            llm=llm,
            max_steps=50,
        )

        continued = agent.continue_from(
            env=env,
            task=original.task,
            gamefile=original.gamefile,
            observation=replayed.observation,
            admissible_actions=replayed.admissible_actions,
            messages=messages,
            max_decisions=MAX_DECISIONS,
        )

        print()
        print("=" * 60)
        print("Continued Trajectory")
        print("=" * 60)

        print(f"Success: {continued.success}")
        print(f"Truncated: {continued.truncated}")
        print(f"Steps: {len(continued)}")

        for i, step in enumerate(continued.steps):
            print()
            print("-" * 60)
            print(f"Continuation Step {i + 1}")
            print("-" * 60)
            print(f"Thought: {step.thought}")
            print(f"Action: {step.action}")
            print(f"Admissible: {step.admissible}")
            print(f"Observation: {step.next_observation}")
            print(f"Won: {step.won}")

    finally:
        env.close()


if __name__ == "__main__":
    main()


