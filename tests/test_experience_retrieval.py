"""测试BM25历史经验检索"""

from src.memory.experience import Experience
from src.memory.store import ExperienceStore

def main():
    store = ExperienceStore()

    # 1. 构造三条不同任务的历史经验
    store.add(
        Experience(
            failure_type="Wrong object type",
            lesson=(
                "Distinguish between cup and mug as distinct object types; "
                "do not substitute a mug for a cup."
            ),
            source_task="heat some cup and put it in countertop.",
        )
    )

    store.add(
        Experience(
            failure_type="Task execution error",
            lesson=(
                "When a task requires a clean object, make sure the "
                "correct object is cleaned before placing it."
            ),
            source_task="put a clean pan in stoveburner.",
        )
    )

    store.add(
        Experience(
            failure_type="Task execution error",
            lesson=(
                "When a task requires a cooled object, cool the correct "
                "object before placing it in the target receptacle."
            ),
            source_task="put a cool apple in fridge.",
        )
    )
    
    # 2. 用一个新的相似任务进行检索
    query = "heat some cup and put it in fridge"

    results = store.retrieve(
        task=query,
        top_k=3,
    )

    # 3. 输出BM25排序结果
    print("=" * 60)
    print("BM25 Experience Retrieval")
    print("=" * 60)
    print(f"Query: {query}")

    for index, experience in enumerate(results, start=1):
        print()
        print(f"Rank {index}")
        print(f"Source task: {experience.source_task}")
        print(f"Failure type: {experience.failure_type}")
        print(f"Lesson: {experience.lesson}")


if __name__ == "__main__":
    main()
