import random

def ability_distribution(num_abilities: int, seed, player_id, global_random) -> list[int]:
    random.seed(seed + player_id)
    abilities = []

    for i in range(num_abilities):
        ability = int(random.gauss(3, 1))
        ability = max(0, min(10, ability))
        abilities.append(ability)

    return abilities

def task_difficulty_distribution(num_abilities: int, seed, task_generation_id, global_random) -> list[int]:
    random.seed(seed + task_generation_id)
    difficulties = []

    for i in range(num_abilities):
        difficulty = int(random.gauss(12, 2))
        difficulty = max(0, min(10, difficulty))
        difficulties.append(difficulty)

    return difficulties