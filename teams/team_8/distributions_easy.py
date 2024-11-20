import random
print("easy")

MAX_SKILL_SCORE = 10


def ability_distribution(num_abilities: int, seed, player_id, global_random) -> list[int]:
    """
    All players have max skill score for all abilities.
    """
    return [MAX_SKILL_SCORE for _ in range(num_abilities)]

def task_difficulty_distribution(num_abilities: int, seed, task_generation_id, global_random) -> list[int]:
    """
    All tasks will have difficulty less than MAX_SKILL_SCORE. Therefore, any player can solve any task individually.
    The only thing this distribution checks for is whether the strategy correctly prioritizes players to solve tasks 
    individually when appropriate (to maximize the number of tasks solved).
    """
    local_random_task = random.Random(seed + task_generation_id)
    difficulties = [0] * num_abilities
    random_integers = local_random_task.sample(range(num_abilities), num_abilities//2)
    for i in random_integers:
        difficulties[i] = local_random_task.randint(1, MAX_SKILL_SCORE//2)
    return difficulties