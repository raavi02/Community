import random


print("hard")


MAX_SKILL_SCORE = 10


def ability_distribution(num_abilities: int, seed, player_id, global_random) -> list[int]:
    """
    Players can have 0 non-zero skill scores across all abilities. Or they can have all non-zero skill scores. And anything in between.
    The skill score is capped at MAX_SKILL_SCORE-3, thereby ensuring that appropriate pairing of players required for optimal solution.

    The rationale for this ability distribution is to maximize variety and randomness.
    (e.g., instead of each player having a specified number of non-zero abilities, the number of non-zero abilities is random). 
    There are players who are naturally good at all things, players who are good at a few things, and players who are good at nothing. 
    This distribution tests the strategy's ability to adapt to different player types.
    """
    local_random_ability = random.Random(seed + player_id)
    random_integers = local_random_ability.sample(
        range(num_abilities), 
        local_random_ability.randint(0, num_abilities),
    )
    abilities = [0] * num_abilities
    for i in random_integers:
        abilities[i] = local_random_ability.randint(0, MAX_SKILL_SCORE-3)
    return abilities

def task_difficulty_distribution(num_abilities: int, seed, task_generation_id, global_random) -> list[int]:
    """
    Tasks have random difficulties for a subset of abilities ranging from 5 to 10.
    Only a few abilities will have zero difficulty. (random; can be 0 to 2 fields that have 0 difficulty)

    The difficulty is relatively higher than player ability distribution, thereby ensuring that effective phase I and II strategy is required to reach optimal solution.
    """
    local_random_task = random.Random(seed + task_generation_id)
    difficulties = [0] * num_abilities
    random_integers = local_random_task.sample(
        range(num_abilities), 
        local_random_task.randint(num_abilities-2, num_abilities),
    )
    for i in random_integers:
        difficulties[i] = local_random_task.randint(5, 10)
    return difficulties
