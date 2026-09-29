"""发现并固定顺序读取ALFWorld TextWorld游戏"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = Path.home() / ".cache" / "alfworld" / "json_2.1.1"
SPLIT_DIRECTORIES = {
    "train": "train",
    "valid_seen": "valid_seen",
    "valid_unseen": "valid_unseen",
}

SUPPORTED_TASK_TYPES = frozenset(
    {
        "pick_and_place_simple",
        "look_at_obj_in_light",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
    }
)

@dataclass(frozen=True)
class GameExample:
    """一局可复现的ALFWorld游戏"""

    id: str
    split: str
    game_file: Path
    task_type: str

def default_data_root() -> Path:
    """返回默认ALFWorld数据目录"""

    return DEFAULT_DATA_ROOT.resolve()

def split_directory(
    data_root: str | Path,
    split: str,
) -> Path:
    """获取某个split对应的数据目录"""

    if split not in SPLIT_DIRECTORIES:
        choices = ",".join(SPLIT_DIRECTORIES)

        raise ValueError(f"Unknown split: {split!r};choices:{choices}")

    root = Path(data_root).expanduser().resolve()

    if root.name in SPLIT_DIRECTORIES.values():
        return root

    nested = root / "json_2.1.1"
    if nested.is_dir():
        root = nested

    return root / SPLIT_DIRECTORIES[split]

def _load_json(path: Path) -> dict:
    """读取JSON，并确保顶层是字典"""

    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)

    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")

    return value

def discover_games(
    data_root: str | Path,
    split: str,
    *,
    task_types: set[str] | frozenset[str] | None = None,
) -> list[GameExample]:
    """发现指定split中所有支持且solvable的游戏"""

    directory = split_directory(data_root, split)
    if not directory.is_dir():
        raise FileNotFoundError(f"ALFWorld data directory not found: {directory}")

    allowed = (SUPPORTED_TASK_TYPES if task_types is None else frozenset(task_types))

    examples: list[GameExample] = []

    for game_file in sorted(directory.rglob("game.tw-pddl")):
        path_text = str(game_file)

        if "movable" in path_text or "Sliced" in path_text:
            continue

        trajectory_file = game_file.with_name("traj_data.json")

        if not trajectory_file.is_file():
            continue

        trajectory = _load_json(trajectory_file)
        task_type = str(trajectory.get("task_type", ""))

        if task_type not in allowed:
            continue

        game_data = _load_json(game_file)

        if not bool(game_data.get("solvable", False)):
            continue

        relative = game_file.relative_to(directory)

        examples.append(
            GameExample(
                id=f"{split}:{relative.parent.as_posix()}",
                split=split,
                game_file=game_file.resolve(),
                task_type=task_type,
            )
        )
    if not examples:
        raise ValueError(f"No usable ALFWorld games found under {directory}")

    return examples

def shuffled_games(
    data_root: str | Path,
    split: str,
    seed: int,
    *,
    max_games: int = 0,
) -> list[GameExample]:
    """使用固定随机种子打乱游戏"""

    games = discover_games(
        data_root,
        split,
    )

    random.Random(seed).shuffle(games)

    if max_games > 0:
        games = games[:max_games]

    return games

def take_batch(
    games: list[GameExample],
    start: int,
    batch_size: int,
) -> list[GameExample]:
    """从固定游戏序列中循环取一个batch"""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if not games:
        return []

    return [games[(start + offset) % len(games)] for offset in range(batch_size)]
                         
