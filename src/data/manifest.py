"""ALFWorld实验任务清单"""

import json
from pathlib import Path

from src.data.alfworld_data import GameExample

def save_manifest(
    games: list[GameExample],
    path: str | Path,
    *,
    seed: int,
) -> None:
    """将一次实验所用游戏集合保存到JSON"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "seed": seed,
        "num_games": len(games),
        "games": [
            {
                "id": game.id,
                "split": game.split,
                "game_file": str(game.game_file),
                "task_type": game.task_type,
            }
            for game in games
        ],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_manifest(
    path: str | Path,
) -> list[GameExample]:
    """从JSON恢复实验任务集合"""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    games = []
    for item in data["games"]:
        games.append(
            GameExample(
                id=item["id"],
                split=item["split"],
                game_file=Path(item["game_file"]),
                task_type=item["task_type"],
            )
        )

    return games
