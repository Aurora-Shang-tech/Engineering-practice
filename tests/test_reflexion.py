"""测试 Reflexion 的重试与反思流程。"""

from unittest.mock import MagicMock

from src.agents.reflexion_agent import ReflexionAgent
from src.core.trajectory import Trajectory


def test_reflexion_retry():
    llm = MagicMock()
    llm.chat.return_value = "Clean the object before placing it."

    agent = ReflexionAgent(
        llm=llm,
        max_steps=50,
        max_attempts=2,
    )

    failed = Trajectory(
        task="put a clean egg in microwave.",
        gamefile="test_game",
    )
    failed.success = False

    succeeded = Trajectory(
        task="put a clean egg in microwave.",
        gamefile="test_game",
    )
    succeeded.success = True

    # 模拟第一次失败、第二次成功
    agent.react.run = MagicMock(return_value=failed)
    agent.react.continue_from = MagicMock(return_value=succeeded)

    env = MagicMock()
    env.reset.return_value = MagicMock(
        task=failed.task,
        gamefile=failed.gamefile,
        observation="Initial observation",
        admissible_actions=("look",),
    )

    result = agent.run(env)

    assert result.success is True
    assert len(result.attempts) == 2
    assert len(result.reflections) == 1

    # 检查反思是否进入第二次尝试的 Prompt
    kwargs = agent.react.continue_from.call_args.kwargs
    messages = kwargs["messages"]

    assert "Clean the object before placing it." in (
        messages[1]["content"]
    )

    print("Reflexion retry test passed.")


if __name__ == "__main__":
    test_reflexion_retry()
