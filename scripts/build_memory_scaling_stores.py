import argparse
import json
import random
from pathlib import Path

from src.memory.store import ExperienceStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/memory/experience_store_qwen3.8-chat_train_seed42_1000.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    full_store = ExperienceStore()
    full_store.load(args.input)

    total = len(full_store)
    if total == 0:
        raise ValueError("Experience store is empty")

    indices = list(range(total))
    random.Random(args.seed).shuffle(indices)

    output_dir = Path("outputs/memory_scaling/stores")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source": str(args.input),
        "seed": args.seed,
        "total_experiences": total,
        "groups": {},
    }

    for percentage in (0, 25, 50, 75, 100):
        count = round(total * percentage / 100)
        selected = sorted(indices[:count])

        subset = ExperienceStore()
        for index in selected:
            subset.add(full_store.experiences[index])
         
        path = output_dir / f"memory_{percentage}.json"
        subset.save(path)

        metadata["groups"][str(percentage)] = {
            "count": len(subset),
            "path": str(path),
        }

        print(f"M{percentage}: {len(subset)}/{total} -> {path}")

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
        
