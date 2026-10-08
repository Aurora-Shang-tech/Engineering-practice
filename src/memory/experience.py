"""Agent可复用经验的数据结构"""

from dataclasses import dataclass

@dataclass(frozen=True)
class Experience:
    """从失败与反事实验证中提炼出的可复用经验"""

    failure_type: str
    lesson: str
    source_task: str
