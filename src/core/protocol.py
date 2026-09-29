"""ALFWorld Agent的交互协议"""

from dataclasses import dataclass
from typing import Sequence

SYSTEM_PROMPT = """
You are an autonomous agent in the text-only ALFWorld environment.

Complete the household task using as few valid steps as possible.

At every turn:

1. Reason from the task, complete interaction history,
   current observation, and available actions.

2. Choose exactly one action.

3. The action must be copied exactly from the available actions.

4. Object and receptacle names include instance numbers.
   For example, "apple 1" and "fridge 1" are complete identifiers.
   Preserve them exactly.

5. Never invent an object, receptacle, or action.

6. Execute only one action at a time and wait for the next
   environment observation before choosing another action.

7. Do not claim that the task is complete yourself.
   The environment decides whether the task is completed.

Respond using exactly this format:

Thought: <your reasoning>
Action: <one available action>
""".strip()

@dataclass(frozen=True)
class ParsedAction:
    """一次LLM输出的解析结果"""

    valid_format: bool
    action: str | None
    reasoning: str


def canonical_action(action: str) -> str:
    """规范化ALFWorld action，用于与admissible actions比较"""
    return " ".join(action.strip().lower().split())


def parse_assistant(text: str) -> ParsedAction:
    """解析thought，action"""

    thought = ""
    action = None

    for line in text.splitlines():
        if line.startswith("Thought"):
            thought = line.removeprefix("Thought:").strip()

        elif line.startswith("Action:"):
            action = line.removeprefix("Action:").strip()

    if not action:
        return ParsedAction(
            valid_format=False,
            action=None,
            reasoning=text.strip(),
        )

    action = canonical_action(action)

    if not action:
        return ParsedAction(
            valid_format=False,
            action=None,
            reasoning=text.strip()
        )

    return ParsedAction(
        valid_format=True,
        action=action,
        reasoning=thought,
    )


def available_actions(
    admissible_actions: Sequence[str],
) -> list[str]:
    """规范化环境提供的合法动作"""

    return [
        canonical_action(action)
        for action in admissible_actions
        if canonical_action(action) != "help"
    ]

def initial_user_prompt(
    *,
    task: str,
    observation: str,
    admissible_actions: Sequence[str],
) -> str:
    """构造episode第一步的user prompt"""

    actions = available_actions(admissible_actions)
    lines = [
        f"Task: {task}",
        "",
        "Current step: 1",
        "",
        "Current observation:",
        observation,
        "",
        "Available actions (copy one exactly):",
    ]
    lines.extend(action for action in actions)
    lines.extend([
        "",
        "Choose exactly one available action",
    ])

    return "\n".join(lines)

def initial_messages(
    *,
    task: str,
    observation: str,
    admissible_actions: Sequence[str],
) -> list[dict]:
    """创建一个episode的初始对话"""

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": initial_user_prompt(
                task=task,
                observation=observation,
                admissible_actions=admissible_actions,
                ),
         },
    ]

def environment_feedback(
    *,
    step: int,
    observation: str,
    admissible_actions: Sequence[str],
    valid_format: bool,
    admissible: bool,
    done: bool,
    won: bool,
) -> str:
    """将一次环境执行结果转换成下一轮LLM能看到的反馈"""

    if won:
        status = "Task completed successfully."

    elif done:
        status = "Episode ended without success."

    elif not valid_format:
        status = "Invalid response format.The environment state did not advance."

    elif not admissible:
        status = "The action was not admissible.Choose one of the available actions."

    else:
        status = "Action executed."

    lines = [
        f"ALFWorld step {step} result: {status}",
        "",
        "Current observation:",
        observation,
    ]

    # episode 没结束时，把下一状态的合法动作继续给模型
    if not done:
        actions = available_actions(admissible_actions)

        lines.extend([
            "",
            "Available actions (copy one exactly):",
        ])
        lines.extend(
            action
            for action in actions
        )

    return "\n".join(lines)

def trajectory_prefix_messages(
    trajectory,
    step_index: int,
) -> list[dict]:
    """根据已有轨迹重建指定步骤执行之前的ReAct对话历史"""

    if step_index < 0 or step_index >= len(trajectory.steps):
        raise IndexError(f"Invalid step index: {step_index}")

    first_step = trajectory.steps[0]

    # 先恢复episode最开始的prompt
    messages = initial_messages(
        task=trajectory.task,
        observation=first_step.observation,
        admissible_actions=first_step.admissible_actions,
    )

    # 只恢复critical step之前的历史
    for index in range(step_index):
        step = trajectory.steps[index]

        # 恢复当时LLM的回答
        if step.valid_format:
            assistant_content=(
                f"Thought: {step.thought}\n"
                f"Action: {step.action}"
            )
        else:
            assistant_content = step.action

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
            }
        )

        # 恢复环境返回给Agent的feedback
        feedback = environment_feedback(
            step=index + 1,
            observation=step.next_observation,
            admissible_actions=(
                trajectory.steps[index + 1].admissible_actions
                if index + 1 < len(trajectory.steps)
                else ()
            ),
            valid_format=step.valid_format,
            admissible=step.admissible,
            done=step.done,
            won=step.won,
        )
        messages.append(
            {
                "role": "user",
                "content": feedback,
            }
        )

    return messages
