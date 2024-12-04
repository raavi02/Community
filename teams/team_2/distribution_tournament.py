import math
import random


def ability_distribution(
    num_abilities: int, seed, player_id, global_random
) -> list[int]:
    # 0.7 is a bit less easy
    # lam = 0.7
    lam = 1
    local_random_ability = random.Random(seed + player_id)
    # Max is to prevent 0 or negative values
    return [
        max(1, math.ceil(10 - local_random_ability.expovariate(lam)))
        for _ in range(num_abilities)
    ]


def task_difficulty_distribution(
    num_abilities: int, seed, task_generation_id, global_random
) -> list[int]:
    # 0.7 is a bit less easy
    # lam = 0.7
    lam = 1
    local_random_ability = random.Random(seed + task_generation_id)
    # Max is to prevent 0 or negative values
    return [
        max(1, min(5, math.floor(local_random_ability.expovariate(lam))))
        for _ in range(num_abilities)
    ]


if __name__ == "__main__":
    num = 3

    abilities = []
    for seed in range(100_000):
        abs = ability_distribution(num, seed, 0, 0)
        # print(abs)
        abilities.extend(abs)

    print(f"avg: {sum(abilities) / len(abilities)}")
    print(f"min: {min(abilities)}, max: {max(abilities)}")

    tasks = []
    for seed in range(100_000):
        task = task_difficulty_distribution(num, seed, 0, 0)
        # print(abs)
        tasks.extend(task)

    print(f"avg: {sum(tasks) / len(tasks)}")
    print(f"min: {min(tasks)}, max: {max(tasks)}")
