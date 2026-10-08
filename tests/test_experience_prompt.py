"""测试历史经验是否正确注入ReAct初始化Prompt"""

from src.core.protocol import initial_messages
from src.memory.experience import Experience
from src.memory.store import ExperienceStore

def main():
    store = ExperienceStore()

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
                "When a task requires a cooled object, cool the correct "
                "object before placing it."
            ),
            source_task="put a cool apple in fridge.",
        )
    )

    task = "heat some cup and put it in fridge."

    retrieved = store.retrieve(
        task=task,
        top_k=1,
    )

    experiences = [
        experience.lesson
        for experience in retrieved
    ]

    messages = initial_messages(
        task=task,
        observation="You are in the middle of a room.",
        admissible_actions=(
            "go to countertop 1",
            "go to fridge 1",
            "look",
        ),
        experiences=experiences,
    )

    user_prompt = messages[1]["content"]

    print("=" * 60)
    print("Experience-Augmented Prompt")
    print("=" * 60)
    print(user_prompt)

    assert "Relevant experience from previous tasks:" in user_prompt
    assert experiences[0] in user_prompt

    print()
    print("Experience prompt injection passed")


if __name__ == "__main__":
    main()
