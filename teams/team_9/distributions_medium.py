import random

def ability_distribution(num_abilities: int, seed, player_id, global_random) -> list[int]:
    local_random_ability = random.Random(seed + player_id)
    return [max(0, min(10, local_random_ability.randint(3, 7))) for _ in range(num_abilities)]

def task_difficulty_distribution(num_abilities: int, seed, task_generation_id, global_random) -> list[int]:
    local_random_task = random.Random(seed + task_generation_id)
    difficulties = [0] * num_abilities
    
    # Make 4 abilities non-zero instead of 3
    non_zero_indices = local_random_task.sample(range(num_abilities), 4)
    for i in non_zero_indices:
        # Difficulty centered around 3-8 for medium level
        difficulties[i] = local_random_task.randint(3, 8)
    
    return difficulties