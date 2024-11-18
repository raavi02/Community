import numpy as np
from community import Community

def phaseIpreferences(player, community: Community, global_random):
    '''Return a list of task index and the partner id for the particular player. 
    The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list]
    and the second index as the partner id'''
    # list_choices = []
    # if player.energy < 0:
    #     return list_choices
    
    # num_members = len(community.members)
    # partner_id = num_members - player.id - 1
    # list_choices.append([0, partner_id])
    
    zero_loss_pairs = zero_energy_loss_pairs(player, community.tasks, community.members)
    return zero_loss_pairs


def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually'''
    bids = []
    if player.energy < 0:
        return bids
    num_abilities = len(player.abilities)
    for i, task in enumerate(community.tasks):
        energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(num_abilities)])
        if energy_cost >= 10:
            continue
        bids.append(i)
    return bids

def tasks_we_can_complete_alone(our_abilities, tasks) -> list[int]:
    return []
    alone_tasks = []
    our_abilities_np = np.array(our_abilities)
    
    for task_id, task in enumerate(tasks): 
        task_np = np.array(task)
        diff = our_abilities_np - task_np
        negative_sum = np.sum(diff[diff < 0])
        
        if negative_sum >= 0:
            alone_tasks.append(task_id)
    return alone_tasks

def zero_energy_loss_pairs(player, tasks, members) -> list[int]:
    # [Task_ID, other_player_id ]
    task_player_pairs = []
    our_abilities_np = np.array(player.abilities)
    our_id = player.id
    for task_id, task in enumerate(tasks):
        task_np = np.array(task)
        
        for other_person in members: 
            if other_person.id == our_id:
                continue
            
            others_abilities_np = np.array(other_person.abilities)
            combined_abilties = np.maximum(our_abilities_np, others_abilities_np)
            
            diff = combined_abilties - task_np
            negative_sum = np.sum(diff[diff < 0])
            if negative_sum >= 0:
                task_player_pairs.append([task_id, other_person.id])
    
    return task_player_pairs
        
# For example, suppose n=4 and that an individual has skill levels (8,6,4,2), and that the task has difficulty vector (5,5,5,5). Then the individual would use up 1+3=4 units of energy to perform the task.
        
