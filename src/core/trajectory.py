from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

@dataclass
class TrajectoryStep:
    """记录Agent与环境的一次交互"""

    step_id: int
    observation: str
    thought: str
    action: str
    next_observation: str
    score: float
    done: bool
    won: bool

@dataclass
class Trajectory:
    """记录一个完整 Episode 的交互轨迹"""

    gamefile: str
    task: str
    agent_name: str

    steps: list[TrajectoryStep] = field(default_factory=list)

    def add_step(
            self,
            observation: str,
            thought: str,
            action: str,
            next_observation: str,
            score: float,
            done: bool,
            won: bool,
    ) -> None:
        """向当前轨迹添加一条交互记录"""
        step = TrajectoryStep(
                step_id=len(self.steps),
                observation=observation,
                thought=thought,
                action=action,
                next_observation=next_observation,
                score=score,
                done=done,
                won=won,
        )
        self.steps.append(step)

    def __len__(self) -> int:
        """返回当前轨迹包含步数"""
        
        return len(self.steps)
    
    @property
    def success(self) -> bool:
        "返回当前Episode是否成功"

        if not self.steps:
            return False

        return self.steps[-1].won
    

    def save_json(self, path: str) -> None:
        """将完整轨迹保存为JSON文件"""
        
        output_path = Path(path)
        
        # 如果父目录不存在就自动创建
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # dataclass -> dict -> JSON
        data = asdict(self)
        
        # 添加Episode级统计信息
        data["success"] = self.success
        data["total_steps"] = len(self)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

