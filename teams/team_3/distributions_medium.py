import random


def ability_distribution(
    num_abilities: int, seed, player_id, global_random
) -> list[int]:
    local_random_ability = random.Random(seed + player_id)
    return [
        int(local_random_ability.betavariate(alpha=4, beta=8) * 10)
        for _ in range(num_abilities)
    ]


def task_difficulty_distribution(
    num_abilities: int, seed, task_generation_id, global_random
) -> list[int]:
    local_random_task = random.Random(seed + task_generation_id)
    difficulties = [0] * num_abilities
    for i in range(num_abilities):
        difficulties[i] = int(
            min(max(local_random_task.normalvariate(mu=4, sigma=1.5), 0), 10)
        )
    return difficulties
