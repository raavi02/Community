import random
def ability_distribution(num_abilities: int, seed, player_id, global_random) -> list[int]:
    local_random_ability = random.Random(seed + player_id)
    abilities = []
    for _ in range(num_abilities):
        if local_random_ability.random() < 0.8:
            abilities.append(local_random_ability.randint(0, 5))
        else:
            abilities.append(local_random_ability.randint(6, 8))
    return abilities

def task_difficulty_distribution(num_abilities: int, seed, task_generation_id, global_random) -> list[int]:
    local_random_task = random.Random(seed + task_generation_id)

    # difficulty buckets as ranges
    easy_range = (1, 3)
    medium_range = (4, 6)
    hard_range = (7, 10)

    random_choice = local_random_task.random()
     # 75% chance for hard
    if random_choice < 0.75:
        difficulties = [local_random_task.randint(*hard_range) for _ in range(num_abilities)]
    # 15% chance for medium
    elif random_choice < 0.9:
        difficulties = [local_random_task.randint(*medium_range) for _ in range(num_abilities)]
    # 10% chance for easy
    else:
        difficulties = [local_random_task.randint(*easy_range) for _ in range(num_abilities)]

    # print(difficulties)
    return difficulties