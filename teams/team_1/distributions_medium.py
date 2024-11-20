import random
def ability_distribution(num_abilities: int, seed, player_id, global_random) -> list[int]:
    local_random_ability = random.Random(seed + player_id)
    return [local_random_ability.randint(0, 10) for _ in range(num_abilities)]

def task_difficulty_distribution(num_abilities: int, seed, task_generation_id, global_random) -> list[int]:
    local_random_task = random.Random(seed + task_generation_id)

    random_choice = local_random_task.random()
    if random_choice > 0.7:
        difficulties = [7 for _ in range(num_abilities)]    # hard task
    elif random_choice > 0.4:
        difficulties = [4 for _ in range(num_abilities)]    # medium task
    else:
        difficulties = [1 for _ in range(num_abilities)]    # easy task

    return difficulties