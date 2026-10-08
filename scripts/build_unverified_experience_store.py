"""从提炼结果构建去重后的经验库"""

import json
from pathlib import Path

from src.memory.experience import Experience
from src.memory.store import ExperienceStore

EXPERIENCE_DIR = Path("outputs/unverified_experiences_qwen3.8-chat_train_seed42_1000")
OUTPUT_PATH = Path("outputs/memory/unverified_experience_store_qwen3.8-chat_train_seed42_1000.json")

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = ExperienceStore()
    seen_lessons = set()

    paths = sorted(EXPERIENCE_DIR.glob("episode_*.json"))

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        lesson = data["lesson"].strip()

        # 完全相同经验只保留一条
        if lesson in seen_lessons:
            continue

        seen_lessons.add(lesson)

        store.add(
            Experience(
                failure_type=data["failure_type"],
                lesson=lesson,
                source_task=data["source_task"],
            )
        )

    store.save(OUTPUT_PATH)

    print(f"Raw experiences: {len(paths)}")
    print(f"Unique experiences: {len(store)}")
    print(f"Duplicates removed: {len(paths) - len(store)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
