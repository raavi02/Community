import random


def ability_distribution(
    num_abilities: int, seed, player_id, global_random
) -> list[int]:
    local_random_ability = random.Random(seed + player_id)

    avg_ability = local_random_ability.randint(1, 9)
    spot = local_random_ability.randint(0, num_abilities - 1)

    abilities = [
        avg_ability if idx == spot else 10 - avg_ability for idx in range(num_abilities)
    ]

    return abilities


def task_difficulty_distribution(
    num_abilities: int, seed, task_generation_id, global_random
) -> list[int]:
    local_random_ability = random.Random(seed + task_generation_id)

    avg_ability = local_random_ability.randint(1, 7)
    spot = local_random_ability.randint(0, num_abilities - 1)

    abilities = [
        avg_ability + 1 if idx == spot else 10 - avg_ability + 1
        for idx in range(num_abilities)
    ]

    return abilities


if __name__ == "__main__":
    num = 3

    abilities = []
    for seed in range(10_000):
        abs = ability_distribution(num, seed, 0, 0)
        # print(abs)
        abilities.extend(abs)

    print(f"avg: {sum(abilities) / len(abilities)}")
    print(f"min: {min(abilities)}, max: {max(abilities)}")

    tasks = []
    for seed in range(10_000):
        task = task_difficulty_distribution(num, seed, 0, 0)
        # print(abs)
        tasks.extend(task)

    print(f"avg: {sum(tasks) / len(tasks)}")
    print(f"min: {min(tasks)}, max: {max(tasks)}")
