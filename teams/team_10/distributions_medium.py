import random
def ability_distribution(num_abilities: int, seed, player_id, global_random) -> list[int]:
    local_random = random.Random(seed + player_id)
    return [local_random.randint(5, 8) for _ in range(num_abilities)]


def task_difficulty_distribution(num_abilities: int, seed, task_generation_id, global_random) -> list[int]:
    """
    Generate task difficulties with specified probabilities:
    - 1/20 chance of a super-hard task (10 10 10 10 10)
    - 1/20 chance of a hard task (9 9 9 9 9)
    - Otherwise, random normal difficulties between 5 and 8
    """
    local_random_task = random.Random(seed + task_generation_id)
    roll = local_random_task.randint(1, 20)

    if roll ==1:  # 1/20 chance of super-hard task
        return [10] * num_abilities
    elif roll == 2:  # 1/20 chance of hard task
        return [9] * num_abilities
    else:  # 10/20 chance of normal task
        return [local_random_task.randint(5, 8) for _ in range(num_abilities)]