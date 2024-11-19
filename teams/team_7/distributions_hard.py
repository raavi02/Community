import random

# Specialists: each player and task require level 10 in a single ability.

def ability_distribution(num_abilities: int, seed, player_id, global_random) -> list[int]:
    """
    Creates list of abilities for single player
    Returns:
        abilities (List): List of integers the length of the number of abilities
    """
    return get_specialist_dist(seed, player_id, num_abilities)


def task_difficulty_distribution(num_abilities: int, seed, task_generation_id, global_random) -> list[int]:
    return get_specialist_dist(seed, task_generation_id, num_abilities)


def get_specialist_dist(seed, id, num_abilities):
    # Initialize abilities
    abilities = [0] * num_abilities

    # Select random ability to require specialist level requirements
    random_generator = random.Random(seed + id)
    specialist_ability_index = random_generator.randint(0, num_abilities - 1)
    abilities[specialist_ability_index] = 10

    return abilities