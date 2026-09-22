from dataclasses import dataclass
import yaml
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

@dataclass
class StepResult:
    """保存Agent执行一次action后的环境结果"""
    observation: str
    score: float
    done: bool
    won: bool
    admissible_actions: list[str]


class ALFWorldEnv:
    def __init__(
            self,
            config_path: str = "configs/base_config.yaml",
            split: str = "train",
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        self.env_manager = AlfredTWEnv(config=config, train_eval=split) # ALFWorld的任务管理层
        self.env = self.env_manager.init_env(batch_size=1) # 真正交互的TextWorld环境

    def reset(self):
        """开启一个新的 ALFWorld 任务。"""
        observations, infos = self.env.reset()
        
        # 去掉batch维度，只返回当前任务的数据
        observation = observations[0]
        admissible_actions = infos["admissible_commands"][0]
        won = infos["won"][0]
        gamefile = infos["extra.gamefile"][0]
        
        return observation, admissible_actions, won, gamefile

    def step(self, action: str) -> StepResult:
        """执行一个action，并返回执行后的环境状态"""
        
        # ALFWorld 使用batch API，所以把action包装成列表
        observations, scores, dones, infos = self.env.step([action])

        return StepResult(
                observation=observations[0],
                score=scores[0],
                done=dones[0],
                won=infos["won"][0],
                admissible_actions=infos["admissible_commands"][0],
        )
