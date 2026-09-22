from src.agents.random_agent import RandomAgent
from src.core.trajectory import Trajectory
from src.envs.alfworld_env import ALFWorldEnv

def main():
    # 初始化环境和Agent
    env = ALFWorldEnv()
    agent = RandomAgent()

    # 开始一个新的Episode
    observation, actions, won, gamefile = env.reset()
    task = observation.split("Your task is to:", 1)[-1].strip()
    trajectory = Trajectory(gamefile=gamefile, task=task, agent_name="random")
    done = False
    while not done:
        # Agent根据当前状态选择动作
        action = agent.act(observation=observation, admissible_actions=actions)
        # 在环境中执行动作
        result = env.step(action)
        # 保存这一步的轨迹
        trajectory.add_step(
                observation=observation,
                action=action,
                next_observation=result.observation,
                score=result.score,
                done=result.done,
                won=result.won,
        )
        print(f"\n==== Step {len(trajectory)} ====")
        print("Action:", action)
        print("Observation:", result.observation)
        print("Score:", result.score)
        print("Done:", result.done)
        print("Won:", result.won)

        # 更新状态，进入下一轮
        observation = result.observation
        actions = result.admissible_actions
        done = result.done

    print("\n===== Episode Finished =====")
    print("Task:", trajectory.task)
    print("Total steps:", len(trajectory))
    print("Won:", trajectory.success)
    
    trajectory.save_json("outputs/trajectories/random_episode.json")
    print("Trajectory saved.")
if __name__ == "__main__":
    main()
