import random

def ability_distribution(num_abilities: int, seed, player_id) -> list[int]:
    """
    Creates list of abilities for single player
    Returns:
        abilities (List): List of integers the length of the number of abilities
    """
    min_ability = 5
    max_ability = 10
    return get_uniform_dist(seed, player_id, num_abilities, min_ability, max_ability)

def task_difficulty_distribution(num_abilities: int, seed, task_generation_id) -> list[int]:
    local_random_task = random.Random(seed + task_generation_id)

    difficulties = [0] * num_abilities

    random_integers = local_random_task.sample(range(num_abilities), 3)
    for i in random_integers:
        difficulties[i] = local_random_task.randint(2, 10)
    return difficulties


def get_uniform_dist(seed, id, num_abilities, low, high):
    random_generator = random.Random(seed + id)
    return [random_generator.randint(low, high) for _ in range(num_abilities)]