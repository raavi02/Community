import numpy as np
import scipy.optimize as opt
import random
import math
from scipy.optimize import linear_sum_assignment
def phaseIpreferences(player, community, global_random):
    '''Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id'''
    list_choices = []
    if player.energy < 0:
        return list_choices
    
    # We're prioritizing tasks based on energy cost
    task_priorities = sorted(
        [(i, sum([max(task[j] - player.abilities[j], 0) for j in range(len(task))]))
         for i, task in enumerate(community.tasks)],
        key=lambda x: x[1]
    )

    # Finding compatible partners for top priority tasks
    for task_id, _ in task_priorities[:2]:
        best_partner = None
        min_cost = float('inf')
        for partner in community.members:
            if partner.id == player.id or partner.energy <= 0:
                continue
            combined_abilities = [player.abilities[j] + partner.abilities[j] for j in range(len(player.abilities))]
            energy_cost = sum([max(community.tasks[task_id][j] - combined_abilities[j], 0) for j in range(len(player.abilities))])
            if energy_cost < min_cost:
                min_cost = energy_cost
                best_partner = partner.id
        if best_partner is not None:
            list_choices.append([task_id, best_partner])
    return list_choices
          
       


def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually'''
    bids = []
    if player.energy < 0:
        return bids
    
    num_abilities = len(player.abilities)
    task_scores = []
    for i, task in enumerate(community.tasks):
        energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(num_abilities)])
        # Tasks with higher unmet abilities and lower energy cost are prioritized
        benefit = sum(task) - energy_cost
        if energy_cost <= player.energy:  # Only considering tasks within energy limits
            task_scores.append((i, benefit / (energy_cost + 1)))

    # Sorting by benefit-to-energy ratio and returning top tasks
    task_scores.sort(key=lambda x: -x[1])
    bids = [task_id for task_id, _ in task_scores[:10]]  # Bidding for top 10 tasks
    return bids
