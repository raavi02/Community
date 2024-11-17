import random
def ability_distribution(num_abilities: int, seed, player_id, global_random) -> list[int]:
    mu = 5.5 
    sigma = 1.5
    return [round(random.gauss(mu, sigma)) for _ in num_abilities]

def task_difficulty_distribution(num_abilities: int, seed, task_generation_id, global_random) -> list[int]:
    mu = 5.5 
    sigma = 1.5
    return [round(random.gauss(mu, sigma)) for _ in num_abilities]