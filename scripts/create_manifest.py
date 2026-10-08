"""创建固定的 ALFWorld 实验任务清单"""

from pathlib import Path

from src.data.alfworld_data import default_data_root, shuffled_games
from src.data.manifest import save_manifest


SPLIT = "valid_unseen"
SEED = 42
NUM_GAMES = 134

OUTPUT_PATH = Path(
    f"outputs/manifests/{SPLIT}_seed{SEED}_{NUM_GAMES}.json"
)


def main() -> None:
    games = shuffled_games(
        data_root=default_data_root(),
        split=SPLIT,
        seed=SEED,
        max_games=NUM_GAMES,
    )

    save_manifest(
        games=games,
        path=OUTPUT_PATH,
        seed=SEED,
    )

    print("=" * 60)
    print("Manifest Created")
    print("=" * 60)
    print(f"Split: {SPLIT}")
    print(f"Seed: {SEED}")
    print(f"Games: {len(games)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
