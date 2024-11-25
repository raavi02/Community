import numpy as np
import random

# Constants for difficulty level (currently set to 'hard')
ABILITY_P = 0.3  # Probability for abilities (hard difficulty)
TASK_P = 0.7     # Probability for task difficulties (hard difficulty)
N = 10           # Binomial distribution's n parameter (values range from 0 to 10)

def ability_distribution(num_abilities: int, seed, player_id, global_random) -> list[int]:
    """
    Generates a list of abilities following a binomial distribution.
    
    Args:
        num_abilities (int): Number of abilities to generate.
        seed (int): Seed for random number generation.
        player_id (int): Unique player ID for creating a reproducible sequence.
        global_random (random.Random): Global random instance (not used here).
        
    Returns:
        list[int]: List of abilities.
    """
    local_random_ability = random.Random(seed + player_id)
    np_random = np.random.default_rng(local_random_ability.randint(0, 10**6))  # Reproducible numpy RNG
    abilities = np_random.binomial(N, ABILITY_P, num_abilities).tolist()  # Generate abilities
    return abilities

def task_difficulty_distribution(num_abilities: int, seed, task_generation_id, global_random) -> list[int]:
    """
    Generates a list of task difficulties following a binomial distribution.
    
    Args:
        num_abilities (int): Number of task difficulties to generate.
        seed (int): Seed for random number generation.
        task_generation_id (int): Unique task generation ID for creating a reproducible sequence.
        global_random (random.Random): Global random instance (not used here).
        
    Returns:
        list[int]: List of task difficulties.
    """
    local_random_task = random.Random(seed + task_generation_id)
    np_random = np.random.default_rng(local_random_task.randint(0, 10**6))  # Reproducible numpy RNG
    difficulties = np_random.binomial(N, TASK_P, num_abilities).tolist()  # Generate difficulties
    return difficulties