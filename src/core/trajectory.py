"""ALFWorld Agent轨迹数据结构"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

@dataclass
class TrajectoryStep:
    """Agent与ALFWorld的一次交互"""

    observation: str                 # 执行动作之前的环境状态
    admissible_actions: list[str]    # 当时环境允许的动作
    thought: str                     # LLM的推理
    action: str                      # Agent最终选择的动作
    valid_format: bool                # LLM输出格式是否正确
    admissible: bool                 # action是否属于当前合法动作集合
    next_observation: str            # 执行动作之后的环境反馈
    score: float
    done: bool
    won: bool

@dataclass
class Trajectory:
    """一个完整ALFWorld episode的轨迹"""

    task: str
    gamefile: str
    success: bool = False
    truncated: bool = False # 是否因为达到最大步数而被截断
    steps: list[TrajectoryStep] = field(default_factory=list)

    def add_step(
        self,
        step: TrajectoryStep,
    ) -> None:
        """记录一步交互"""

        self.steps.append(step)

    def __len__(self) -> int:
        """返回episode已执行步数"""

        return len(self.steps)

    def save_json(
        self,
        path: str,
    ) -> None:
        """将轨迹保存成JSON"""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(
        cls,
        path: str,
    ) -> "Trajectory":
        """从JSON文件恢复轨迹"""

        input_path = Path(path)
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        steps = [TrajectoryStep(**step) for step in data.get("steps", [])]

        return cls(
            task = data["task"],
            gamefile=data["gamefile"],
            success=data.get("success", False),
            truncated=data.get("truncated", False),
            steps=steps,
        )
