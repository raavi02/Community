import random


print("med")


def ability_distribution(num_abilities: int, seed, player_id, global_random) -> list[int]:
    """
    Players have uniform random skill scores for all abilities ranging from 0 to 10.
    """
    local_random_ability = random.Random(seed + player_id)
    return [local_random_ability.randint(0, 10) for _ in range(num_abilities)]

def task_difficulty_distribution(num_abilities: int, seed, task_generation_id, global_random) -> list[int]:
    """
    Tasks have random difficulties for a subset of abilities ranging from 3 to 10.
    Only a few abilities will have non-zero difficulty. (num_abilities//3 + 1)

    Optimal strategies would minimize the energy expended through appropriate task assignment.
    """
    local_random_task = random.Random(seed + task_generation_id)
    difficulties = [0] * num_abilities
    difficulty_dimensions = num_abilities//3 + 1
    random_integers = local_random_task.sample(range(num_abilities), difficulty_dimensions)
    for i in random_integers:
        difficulties[i] = local_random_task.randint(3, 10)
    return difficulties
