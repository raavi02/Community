import numpy as np
from scipy.optimize import linear_sum_assignment


def create_cost_matrix(player, community):
    cost_matrix = []
    for task in community.tasks:
        task_costs = []
        for member in community.members:
            # Compute the element-wise maximum of abilities
            max_abilities = [max(i, j) if member.energy >= 0 else float("inf") for i, j in zip(player.abilities, member.abilities)]
            # Compute the delta and absolute values
            delta = [abs(max_val - req) for max_val, req in zip(max_abilities, task)]
            # Total cost is the sum of deltas
            total_cost = sum(delta)
            task_costs.append(total_cost)
        cost_matrix.append(task_costs)
    cost_matrix = np.array(cost_matrix)
    return cost_matrix



def best_partner(task: np.ndarray):
    for partner_id in range(len(task)):
        if task[partner_id] == task.min():
            return partner_id

    raise Exception("All arrays have a minimum value")


def phaseIpreferences(player, community, global_random):
    '''Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id'''
    list_choices = []
    if player.energy < 0:
        return list_choices
    # num_members = len(community.members)
    # partner_id = num_members - player.id - 1
    # list_choices.append([0, partner_id])
    # if len(community.tasks) > 1:
    #     list_choices.append([1, partner_id])
    # list_choices.append([0, min(partner_id + 1, num_members - 1)])

    cost_matrix = create_cost_matrix(player, community)

    best_partner_for_task = [(task_id, best_partner(cost_matrix[task_id]), cost_matrix[task_id].min()) for task_id in range(len(cost_matrix))]
    best_partner_for_task.sort(key=lambda x: x[2])

    requested_partners = []

    # to incentivize players to not request pairing up with the best member in the community, 
    # we require that they at least request 5 different partners
    PARTNER_REQUEST_AMOUNT = 5
    potential_partners = set()
    curr_idx = 0
    while len(potential_partners) < PARTNER_REQUEST_AMOUNT and curr_idx < len(best_partner_for_task):
        task_id, partner_id, cost = best_partner_for_task[curr_idx]
        if partner_id not in potential_partners:
            requested_partners.append([task_id, partner_id])
            potential_partners.add(partner_id)

        curr_idx += 1

    return requested_partners




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
        bids.append((i, energy_cost))

    bids.sort(key=lambda x: (x[1], -sum(community.tasks[x[0]])))
    return [b[0] for b in bids[:5]]
