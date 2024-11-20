import random

def ability_distribution(num_abilities: int, seed, player_id, global_random) -> list[int]:
    local_random_ability = random.Random(seed + player_id)
    abilities =  [local_random_ability.randint(8, 10) for _ in range(num_abilities)]
    return abilities

def task_difficulty_distribution(num_abilities: int, seed, task_generation_id, global_random) -> list[int]:
    local_random_task = random.Random(seed + task_generation_id)
    difficulties = [0] * num_abilities
    random_integers = local_random_task.sample(range(num_abilities), 2)
    for i in random_integers:
        difficulties[i] = local_random_task.randint(0, 3)
    return difficulties