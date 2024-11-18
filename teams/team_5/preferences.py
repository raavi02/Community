import numpy as np
import scipy.optimize as opt
import random
import math
# from scipy.optimize import linear_sum_assignment

def phaseIpreferences(player, community, global_random):
    '''Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id'''
    # print("THIS IS THE LIST OF TASKS: ", community.tasks)
    exhausted = True if player.energy < 0 else False # Am I exhuasted?

    list_choices = []
    # if player.energy < -10:
    #     return list_choices 
    
    cost0_tasks = set()

    #If any member can complete the task using 0 energy, then exclude it from our list
    for i, task in enumerate(community.tasks):
        for member in community.members:
            energy_cost = sum([max(task[j] - member.abilities[j], 0) for j in range(len(member.abilities))])

            if energy_cost == 0:
                cost0_tasks.add(i)
                break
    print("THESE ARE THE FREE TASKS: ", cost0_tasks)

    # We're prioritizing tasks based on energy cost
    task_priorities = sorted(
        [(i, sum([max(task[j] - player.abilities[j], 0) for j in range(len(task))]))
         for i, task in enumerate(community.tasks)],
        key=lambda x: x[1]
    )

    exhausted_penalty = 0.5
    spent_energy = 0
    # Finding compatible partners for top priority tasks
    for task_id, _ in task_priorities:
        if task_id in cost0_tasks:
            continue

        best_partner = None
        min_cost = float('inf')
        task_energy = -1
        for partner in community.members:
            if partner.id == player.id:
                continue
            combined_abilities = [max(player.abilities[j],  partner.abilities[j]) for j in range(len(player.abilities))]
            energy_cost = round((sum([max(community.tasks[task_id][j] - combined_abilities[j], 0) for j in range(len(player.abilities))]))/2, 2)
            added_cost = exhausted_penalty if partner.energy < 0 else 0 #We prioritize active workers instead of exhasuted workers

            comparison_metric = energy_cost + added_cost

            if comparison_metric < min_cost and (partner.energy - energy_cost) > -10 and (player.energy - energy_cost) > -10: # Make sure ourself and our partner will not become incapacitated after performing the task
                min_cost = comparison_metric
                best_partner = partner.id
                task_energy = energy_cost

        spent_energy += task_energy
        
        if (player.energy - spent_energy) <= -10:
            print(f"Member {player.id} has too much work load, ending partner search now.")
            print(f"Member {player.id}'s partner choices, ", list_choices)
            return list_choices

        if best_partner is not None:
            list_choices.append([task_id, best_partner])
        
    print(f"Member {player.id}'s partner choices, ", list_choices)
        
    return list_choices
          
       
def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually'''
    bids = []
    # if player.energy < 0:
    #     return bids
    num_abilities = len(player.abilities)
    spent_energy = 0
    # task_scores = []

    cost0_tasks = set()

    #If any member can complete the task using 0 energy, then exclude it from our list
    for i, task in enumerate(community.tasks):
        energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(len(player.abilities))])
        if energy_cost == 0:
            bids.append(i)
        
        for member in community.members:
            if member.id == player.id: #Skip over self
                continue

            energy_cost = sum([max(task[j] - member.abilities[j], 0) for j in range(len(member.abilities))])

            if energy_cost == 0:
                # print(f"IT COSTS Member{member.id} Nothing to perform Task{i}")
                cost0_tasks.add(i)
                break

    # print("THESE ARE THE FREE TASKS PHASE II: ", cost0_tasks)

    task_priorities = sorted(
        [(i, sum([max(task[j] - player.abilities[j], 0) for j in range(len(task))])) #Tuple of taskid and energy cost 
            for i, task in enumerate(community.tasks) if i not in cost0_tasks], 
        key=lambda x: x[1] #Sort key
    )
    # print("PHASE II Task Prios: ", task_priorities)
    
    print(f"For Member {player.id}")
    print(f"Tasks that need to be completed: {task_priorities}")

    for task_id, energy_cost in task_priorities:
        print(f"Task {task_id} costs {energy_cost}")

        if energy_cost <= 0: #If it doesn't cost energy, do it.
            bids.append(task_id)
            continue
        
        if player.energy - (energy_cost + spent_energy) > -10: #If I am alive, I am gonna work.
            spent_energy += energy_cost
            bids.append(task_id)
            continue

        # Tasks with higher unmet abilities and lower energy cost are prioritized
        # benefit = sum(task) - energy_cost
        # if energy_cost <= player.energy:  # Only considering tasks within energy limits
            # task_scores.append((i, benefit / (energy_cost + 1)))

    # Sorting by benefit-to-energy ratio and returning top tasks
    # task_scores.sort(key=lambda x: -x[1])
    # bids = [task_id for task_id, _ in task_scores[:10]]  # Bidding for top 10 tasks
    print(f"Member {player.id}'s individual choices, ", bids)

    return bids
