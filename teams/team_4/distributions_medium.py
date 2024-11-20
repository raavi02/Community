import random
def ability_distribution(num_abilities: int, seed, player_id, global_random) -> list[int]:
    # local_random_ability = random.Random(seed + player_id)
    # return [local_random_ability.randint(0, 10) for _ in range(num_abilities)]

    # fill ability with a gaussian distribution between 0, 10 and the curve at 8
    random.seed(seed + player_id)
    abilities = []
    for i in range(num_abilities):
        # print(random.gauss(8, 2))
        abilities.append(int(random.gauss(5, 1)))
        if abilities[i] < 0:
            abilities[i] = 0
        if abilities[i] > 10:
            abilities[i] = 10
    
    return abilities

def task_difficulty_distribution(num_abilities: int, seed, task_generation_id, global_random) -> list[int]:
    # local_random_task = random.Random(seed + task_generation_id)
    random.seed(seed + task_generation_id)
    difficulties = [0] * num_abilities

    random_integers = random.sample(range(num_abilities), max(3, num_abilities // 2 + 2))
    for i in random_integers:
        difficulties[i] = int(random.gauss(8, 2))
        if difficulties[i] < 0:
            difficulties[i] = 0
        if difficulties[i] > 10:
            difficulties[i] = 10

    return difficulties