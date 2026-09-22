from src.agents.react_agent import ReActAgent
from src.core.trajectory import Trajectory
from src.envs.alfworld_env import ALFWorldEnv
from src.llm.client import LLMClient

def main():
    # 初始化环境和ReAct Agnet
    env = ALFWorldEnv()
    llm = LLMClient()
    agent = ReActAgent(llm)

    # 开始新的Episode
    observation, actions, won, gamefile = env.reset()

    task = observation.split("Your task is to:", 1)[-1].strip()
    trajectory = Trajectory(
            gamefile=gamefile,
            task=task,
            agent_name="react",
    )
    # 初始化当前Episode的ReAct短期记忆
    agent.reset(task)
    done = False

    while not done:
        # LLM 根据当前observation进行推理并选择动作
        thought, action = agent.act(observation=observation,admissible_actions=actions)
        print(f"\n===== Step {len(trajectory) + 1} =====")
        print("Thought:", thought)
        print("Action:", action)

        if action not in actions:
            print("Failed to produce a valid action")
            break

        # 在ALFWorld执行动作
        result = env.step(action)

        # 将环境反馈加入Agent短期历史
        agent.update(
                observation=observation,
                thought=thought,
                action=action,
                next_observation=result.observation,
        )

        # 保存完整的ReAct轨迹
        trajectory.add_step(
            observation=observation,
            thought=thought,
            action=action,
            next_observation=result.observation,
            score=result.score,
            done=result.done,
            won=result.won,
        )

        print("Observation:", result.observation)
        print("Score:", result.score)
        print("Done:", result.done)
        print("Won:", result.won)

        # 更新下一轮状态
        observation = result.observation
        actions = result.admissible_actions
        done = result.done

    print("\n===== Episode Finished =====")
    print("Task:", trajectory.task)
    print("Total steps:", len(trajectory))
    print("Success:", trajectory.success)

    trajectory.save_json(
        "outputs/trajectories/react_episode.json"
    )

    print("Trajectory saved.")


if __name__ == "__main__":
    main()
