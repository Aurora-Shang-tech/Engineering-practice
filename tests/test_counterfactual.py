"""测试完整反事实验证"""

from pathlib import Path
from src.agents.react_agent import ReActAgent
from src.core.trajectory import Trajectory
from src.counterfactual.verifier import verify_counterfactual
from src.llm.client import LLMClient

TRAJECTORY_PATH = Path("outputs/baseline_qwen3.8-chat_train_seed42_50/episode_005.json")
CRITICAL_STEP = 12
COUNTERFACTUAL_ACTION = "go to countertop 2"
MAX_STEPS = 50

def main():
    # 1.加载失败轨迹
    trajectory = Trajectory.load_json(str(TRAJECTORY_PATH))

    print("=" * 60)
    print("Counterfactual Verification")
    print("=" * 60)

    print(f"Task: {trajectory.task}")
    print(f"Original success: {trajectory.success}")
    print(f"Critical step: {CRITICAL_STEP}")
    print(
        "Original action: "
        f"{trajectory.steps[CRITICAL_STEP].action}"
    )
    print(
        "Counterfactual action: "
        f"{COUNTERFACTUAL_ACTION}"
    )

    # 2.创建与baseline相同的LLM
    llm = LLMClient()
    agent = ReActAgent(
        llm=llm,
        max_steps=MAX_STEPS,
    )

    # 3.真正执行反事实验证
    result = verify_counterfactual(
        trajectory=trajectory,
        critical_step=CRITICAL_STEP,
        counterfactual_action=COUNTERFACTUAL_ACTION,
        agent=agent,
        max_steps=MAX_STEPS,
    )

    # 4. 输出结果
    print()
    print("=" * 60)
    print("Counterfactual Result")
    print("=" * 60)

    print(f"Critical step: {result.critical_step}")
    print(f"Original action: {result.original_action}")
    print(
        "Counterfactual action: "
        f"{result.counterfactual_action}"
    )
    print(f"Success: {result.success}")
    print(
        "Continuation steps: "
        f"{result.continuation_steps}"
    )


if __name__ == "__main__":
    main()

