"""测试ExperienceStore的保存与加载"""

from pathlib import Path

from src.memory.experience import Experience
from src.memory.store import ExperienceStore

OUTPUT_PATH = Path("outputs/test_experience_store.json")

def main():
    experience = Experience(
        failure_type="Wrong object type",
        lesson=(
            'Distinguish between "cup" and "mug" as distinct object types; '
            "do not substitute a mug for a cup when the task explicitly "
            "requires a cup."
        ),
        source_task="heat some cup and put it in countertop.",
    )

    # 1.创建经验库并添加经验
    store = ExperienceStore()
    store.add(experience)

    print("=" * 60)
    print("Before Save")
    print("=" * 60)
    print(f"Experiences: {len(store)}")
    print(f"Lesson: {store.experiences[0].lesson}")

    # 2.保存为JSON
    store.save(OUTPUT_PATH)

    # 3.创建一个新的空经验库
    loaded_store = ExperienceStore()

    print()
    print(f"Before load: {len(loaded_store)}")

    # 4. 从JSON恢复
    loaded_store.load(OUTPUT_PATH)

    print()
    print("=" * 60)
    print("After Load")
    print("=" * 60)
    print(f"Experiences: {len(loaded_store)}")
    print(f"Failure type: {loaded_store.experiences[0].failure_type}")
    print(f"Lesson: {loaded_store.experiences[0].lesson}")
    print(f"Source task: {loaded_store.experiences[0].source_task}")


if __name__ == "__main__":
    main()
