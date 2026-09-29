"""ALFWorld 环境封装"""

from dataclasses import dataclass
from pathlib import Path
import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
from src.core.protocol import canonical_action

@dataclass(frozen=True)
class EnvironmentState:
    """reset()后得到的环境初始状态"""

    observation: str
    task: str
    admissible_actions: tuple[str, ...]
    gamefile: str

@dataclass(frozen=True)
class EnvironmentStep:
    """执行一次action后得到的环境结果"""

    observation: str
    score: float
    done: bool
    won: bool
    admissible: bool # 执行的action是否合法
    admissible_actions: tuple[str, ...] # 执行后新状态的合法动作


class ALFWorldEnvironment:
    """单个ALFWorld TextWorld环境"""

    def __init__(
        self,
        gamefile: str,
        max_steps: int = 50,
    ):
        self.gamefile = str(Path(gamefile).expanduser().resolve())
        self.max_steps = max_steps
        self._admissible_actions: tuple[str, ...] = tuple()
        self._env = self._build_env()

    def _build_env(self):
        """创建只包含指定game的TextWorld环境"""

        request_infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            extras=["gamefile"],
        )

        # 与ALFWorld TextWorld环境保持一致
        wrappers = [
            AlfredDemangler(shuffle=False),
            AlfredInfos,
        ]

        env_id = textworld.gym.register_games(
            [self.gamefile],
            request_infos,
            batch_size=1,
            asynchronous=True,
            auto_reset=False,
            max_episode_steps=self.max_steps,
            wrappers=wrappers,
        )

        return textworld.gym.make(env_id)

    def reset(self) -> EnvironmentState:
        """把环境重置到当前game的初始状态"""

        observations, infos = self._env.reset()
        observation = str(observations[0])
        self._admissible_actions = self._normalize_actions(infos["admissible_commands"][0])

        loaded_gamefile = str(Path(infos["extra.gamefile"][0]).expanduser().resolve())
        if loaded_gamefile != self.gamefile:
            raise RuntimeError("Loaded game does not match target game\nExpected:{self.gamefile}Actual:{loaded_gamefile}")

        task = self._extract_task(observation)

        return EnvironmentState(
            observation=observation,
            task=task,
            admissible_actions=self._admissible_actions,
            gamefile=self.gamefile,
        )

    def step(
        self,
        action: str,
    ) -> EnvironmentStep:
        """执行一个ALFWorld action"""

        normalized_action = canonical_action(action)
        admissible = (normalized_action in self._admissible_actions)
        observations, scores, dones, infos = self._env.step([normalized_action])
        observation = str(observations[0])
        self._admissible_actions = self._normalize_actions(infos["admissible_commands"][0])

        return EnvironmentStep(
            observation=observation,
            score=float(scores[0]),
            done=bool(dones[0]),
            won=bool(infos["won"][0]),
            admissible=admissible,
            admissible_actions=self._admissible_actions,
        )

    @staticmethod
    def _normalize_actions(
        actions,
    ) -> tuple[str, ...]:
        """统一环境返回的admissible actions"""

        return tuple(
            canonical_action(str(action))
            for action in actions
            if canonical_action(str(action)) != "help"
        )

    @staticmethod
    def _extract_task(
        observation: str,
    ) -> str:
        """从ALFWorld初始observation中提取任务"""

        marker = "Your task is to:"
        index = observation.find(marker)

        if index == -1:
            raise ValueError("Could not find task description in initial observation")

        task = observation[index + len(marker):].strip()

        if not task:
            raise ValueError("ALFWorld task description is empty")

        return task

    def close(self) -> None:
        """释放TextWorld环境"""
        self._env.close()
       
