import random
print("hard")

#[0,0,1,0,0,3,4]
def ability_distribution(num_abilities: int, seed, player_id, global_random) -> list[int]:
    local_random_ability = random.Random(seed + player_id)
    #initialize everything to 0
    ability_dist= [0] * num_abilities
    #get random indices(0-5)
    random_indices = local_random_ability.sample(range(num_abilities), local_random_ability.randint(0, 2))
    for i in random_indices:
        #assign small random values to the random indices
        ability_dist[i] = local_random_ability.randint(1, 4)
    return ability_dist

#[8,10,9,9,10,10,10]
def task_difficulty_distribution(num_abilities: int, seed, task_generation_id, global_random) -> list[int]:
    local_random_task = random.Random(seed + task_generation_id)
    difficulties = [10] * num_abilities
    random_integers = local_random_task.sample(range(num_abilities), local_random_task.randint(3, 4))
    for i in random_integers:
        difficulties[i] = local_random_task.randint(8, 9)
    return difficulties