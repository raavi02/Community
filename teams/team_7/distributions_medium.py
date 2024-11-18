import random

# todo: create a list of abilities and tasks where everyone is a specialist and every task is specialized (only 0 and single value between 6 and 10)

def ability_distribution(num_abilities: int, seed, player_id) -> list[int]:
    """
    Creates list of abilities for single player
    Returns:
        abilities (List): List of integers the length of the number of abilities
    """
    min_ability = 0
    max_ability = 4
    return get_uniform_dist(seed, player_id, num_abilities, min_ability, max_ability)


def task_difficulty_distribution(num_abilities: int, seed, task_generation_id) -> list[int]:

    min_difficulty = 6
    max_difficulty = 10
    return get_uniform_dist(seed, task_generation_id, num_abilities, min_difficulty, max_difficulty)


def get_uniform_dist(seed, id, num_abilities, low, high):
    random_generator = random.Random(seed + id)
    return [random_generator.randint(low, high) for _ in range(num_abilities)]