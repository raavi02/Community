import random
print("med")
#small set [00000]
#small set of 10 10 10 task
def ability_distribution(num_abilities: int, seed, player_id, global_random) -> list[int]:
    local_random_ability = random.Random(seed + player_id)
    #generate a value
    random_integer = local_random_ability.randint(0,20) #5% chance that player will be [0,0,0,0,0,0,0,0,0,0]
    #if the randomly chosen value is 0, then return a list of 0s
    if random_integer ==0:
        return [0] * num_abilities
    #otherwise, return a list of random integers from 
    #return medium to high val of abilities
    ability_dist=[]
    for i in range(num_abilities):
        ability_dist.append(local_random_ability.randint(5,10))
    return ability_dist

def task_difficulty_distribution(num_abilities: int, seed, task_generation_id, global_random) -> list[int]:
    local_random_task = random.Random(seed + task_generation_id)
    #generate a value
    random_integer = local_random_task.randint(0,10) #5% chance that player will be [0,0,0,0,0,0,0,0,0,0]
    #if the randomly chosen value is 0, then return a list of 0s
    if random_integer ==0:
        print("0")
        return [10] * num_abilities
    #otherwise, return a list of random integers from 
    #return medium to high val of abilities
    difficulties=[]
    for i in range(num_abilities):
        difficulties.append(local_random_task.randint(0,7))

    return difficulties