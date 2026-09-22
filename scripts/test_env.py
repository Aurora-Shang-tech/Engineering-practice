from src.core.trajectory import Trajectory
from src.envs.alfworld_env import ALFWorldEnv


def main():
    # 创建环境并开始一个新任务
    env = ALFWorldEnv()
    observation, actions, won, gamefile = env.reset()

    # 为当前任务创建一条轨迹
    trajectory = Trajectory(gamefile=gamefile)

    print("\n==== Observation ====")
    print(observation)

    # 暂时选择第一个合法动作
    action = actions[0]

    print("\n==== Action ====")
    print(action)

    # 在环境中执行动作
    result = env.step(action)

    # 记录这一步交互
    trajectory.add_step(
        observation=observation,
        action=action,
        next_observation=result.observation,
        score=result.score,
        done=result.done,
        won=result.won,
    )

    print("\n==== New Observation ====")
    print(result.observation)

    print("\n==== Trajectory ====")
    print("steps:", len(trajectory))
    print(trajectory.steps[0])


if __name__ == "__main__":
    main()
