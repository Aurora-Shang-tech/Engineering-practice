from src.llm.client import LLMClient

class ReActAgent:
    """使用ReActcelve选择ALFWorld动作"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

        # 保存当前Episode的任务和交互历史
        self.task = ""
        self.history = []

    def reset(self, task: str) -> None:
        """开始新的Episode，并清空旧历史"""
        self.task = task
        self.history = []

    def act(
            self,
            observation: str,
            admissible_actions: list[str],
    ) -> tuple[str, str]:
        """根据当前观察选择动作，返回thought和action"""

        # 给每个合法动作分配编号，让LLM只选择编码
        actions_text = "\n".join(f"[{i}] {action}" for i, action in enumerate(admissible_actions))

        # 将之前的Thought/Action/Observation整理成文本
        history_text = self._format_history()
        messages = [
            {
                "role": "system",
                "content": (
                   "You are an agent interacting with the ALFWorld environment.\n"
                   "Your goal is to complete the given household task efficiently.\n\n"

                   "Important rules:\n"
                   "1. Use the previous interaction history to avoid repeating actions.\n"
                   "2. Choose exactly one action from the admissible actions.\n"
                   "3. Do not assume an object is clean, heated, or cooled unless you "
                   "have explicitly performed the corresponding action.\n"
                   "4. If the task requires a clean object, you must use a clean action.\n"
                   "5. If the task requires a heated object, you must use a heat action.\n"
                   "6. If the task requires a cooled object, you must use a cool action.\n"
                   "7. Do not assume the task is complete based only on your reasoning. "
                   "Continue acting until the environment indicates success.\n"
                   "8. Avoid using 'help' unless it is genuinely necessary.\n\n"
 
                   "Respond using exactly this format:\n"
                   "Thought: <your reasoning>\n"
                   "Action: <action number>"           
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{self.task}\n\n"
                    f"Previous interaction history:\n"
                    f"{history_text}\n\n"
                    f"Current observation:\n"
                    f"{observation}\n\n"
                    f"Admissible actions:\n"
                    f"{actions_text}"
                ),
            },
        ]
        
        max_retries = 3
        for attempt in range(max_retries):
            response = self.llm.chat(messages)

            try:
                # 尝试解析LLM返回动作编号
                thought, action_id = self._parse_response(response)

                # 编号必须落在合法动作范围内
                if 0 <= action_id < len(admissible_actions):
                    action = admissible_actions[action_id]
                    return thought, action

                error_message = (
                    f"Action ID {action_id} is out of range."
                    f"Choose a number from 0 to {len(admissible_actions) - 1}."
                )

            except ValueError:
                error_message = ("Your Action must be an integer action number, not the action text")

                # 把格式错误反馈给LLM，让它重新选择
                messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"{error_message}\n\n"
                            "Choose again using exactly this format:\n"
                            "Thought: <your reasoning>\n"
                            "Action: <action number>"
                        ),
                    }
                )

            raise ValueError(f"LLM failed to produce a valid action after {max_retries}attempts.")
                
    
    def update(
            self,
            observation: str,
            thought: str,
            action: str,
            next_observation: str,
    ) -> None:
        """记录刚刚完成的一步交互"""

        self.history.append(
            {
                "observation": observation,
                "thought": thought,
                "action": action,
                "next_observation": next_observation,
            }
        )
    
    def retry_action(
            self,
            observation: str,
            invalid_action: str,
            admissible_actions: list[str],
    ) -> tuple[str, str]:
        """当LLM生成非法动作时，根据合法动作重新决策"""
        
        actions_text = "\n".join(admissible_actions)

        messages = [
            {
            "role": "system",
            "content": (
                "You are an agent interacting with the ALFWorld environment.\n"
                "Your previous action was invalid.\n"
                "Choose exactly one action from the admissible actions below.\n\n"
                "Respond using exactly this format:\n"
                "Thought: <your reasoning>\n"
                "Action: <one admissible action>"
                ),
            },
            {
            "role": "user",
            "content": (
                f"Task:\n{self.task}\n\n"
                f"Current observation:\n{observation}\n\n"
                f"Invalid action:\n{invalid_action}\n\n"
                f"Admissible actions:\n{actions_text}"
                ),
            },
        ]

        response = self.llm.chat(messages)

        return self._parse_response(response)

    def _format_history(self) -> str:
        """将历史轨迹转成适合放入Prompt文本"""
        
        if not self.history:
            return "No previous interaction"

        parts = []

        for i, step in enumerate(self.history):
            parts.append(
                    f"Step {i + 1}:\n"
                f"Observation: {step['observation']}\n"
                f"Thought: {step['thought']}\n"
                f"Action: {step['action']}\n"
                f"Result: {step['next_observation']}"
            )

        return "\n\n".join(parts)

    def _parse_response(self, response: str) -> tuple[str, int]:
        """解析模型返回的 Thought 和 Action ID"""

        thought = ""
        action_id = None

        for line in response.splitlines():
            if line.startswith("Thought:"):
                thought = line.removeprefix("Thought:").strip()
            elif line.startswith("Action:"):
                action_text = line.removeprefix("Action:").strip()

                try:
                    action_id = int(action_text)
                except ValueError:
                    raise ValueError(f"LLM返回的action不是整数:\n{response}")

        if action_id is None:
            raise ValueError(f"LLM没有返回Action ID:\n{response}")

        return thought, action_id
