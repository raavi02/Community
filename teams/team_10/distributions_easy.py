import random
def ability_distribution(num_abilities: int, seed, player_id, global_random) -> list[int]:
    """
    Generate ability difficulties (random between 7 and 10)
    """
    random = random.Random(seed + player_id)
    return [random.randint(7, 10) for _ in range(num_abilities)]


def task_difficulty_distribution(num_abilities: int, seed, task_generation_id, global_random) -> list[int]:
    """
    Generate task difficulties (random between 5 and 8)
    """
    local_random_task = random.Random(seed + task_generation_id)
    return [local_random_task.randint(5, 8) for _ in range(num_abilities)]