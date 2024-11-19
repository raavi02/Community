import random
def ability_distribution(num_abilities: int, seed, player_id, global_random) -> list[int]:
    # 0.7 is a bit less easy
    # lam = 0.7
    lam = 1
    local_random_ability = random.Random(seed + player_id)
    # Max is to prevent 0 or negative values
    return [max(1, local_random_ability.expovariate(lam)) for _ in range(num_abilities)]

def task_difficulty_distribution(num_abilities: int, seed, task_generation_id, global_random) -> list[int]:
    local_random_task = random.Random(seed + task_generation_id)
    difficulties = [0] * num_abilities
    random_integers = local_random_task.sample(range(num_abilities), 3)
    for i in random_integers:
        difficulties[i] = local_random_task.randint(2, 10)
    return difficulties