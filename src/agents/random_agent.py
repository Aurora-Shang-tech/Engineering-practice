import random

class RandomAgent:
    """从当前合法动作中随机选择一个动作。"""

    def act(
            self,
            observation: str,
            admissible_actions: list[str],
    ) -> str:
        return random.choice(admissible_actions)
