"""可复用经验的存储与读取"""

import json
from pathlib import Path
import re
from rank_bm25 import BM25Okapi

from src.memory.experience import Experience

class ExperienceStore:
    """管理Agent积累的可复用经验"""

    def __init__(self):
        self.experiences: list[Experience] = []

    def add(self, experience: Experience) -> None:
        """添加一条经验"""

        self.experiences.append(experience)

    def __len__(self) -> int:
        return len(self.experiences)

    def save(self, path: str | Path) -> None:
        """将经验库保存为JSON"""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "failure_type": experience.failure_type,
                "lesson": experience.lesson,
                "source_task": experience.source_task,
            }
            for experience in self.experiences
        ]

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path) -> None:
        """从JSON加载经验库"""

        input_path = Path(path)

        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.experiences = [
            Experience(
                failure_type=item["failure_type"],
                lesson=item["lesson"],
                source_task=item["source_task"],
            )
            for item in data
        ]

    def retrieve(
        self,
        task: str,
        top_k: int = 3,
    ) -> list[Experience]:
        """使用BM25检索与当前任务最相关的历史经验"""

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        if not self.experiences:
            return []

        corpus = [
            self._tokenize(experience.source_task)
            for experience in self.experiences
        ]

        bm25 = BM25Okapi(corpus)

        query = self._tokenize(task)
        scores = bm25.get_scores(query)

        ranked_indices = sorted(
            range(len(self.experiences)),
            key=lambda index: scores[index],
            reverse=True,
        )

        return [
            self.experiences[index]
            for index in ranked_indices[:top_k]
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """将文本规范化为BM25使用的token"""

        return re.findall(r"[a-z0-9]+", text.lower())
