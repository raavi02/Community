import numpy as np
from scipy.optimize import linear_sum_assignment


def create_cost_matrix(player, community):
    cost_matrix = []
    for task in community.tasks:
        task_costs = []
        for member in community.members:
            # Compute the element-wise maximum of abilities
            max_abilities = [max(i, j) for i, j in zip(player.abilities, member.abilities)]
            # Compute the delta and absolute values
            delta = [abs(max_val - req) for max_val, req in zip(max_abilities, task)]
            # Total cost is the sum of deltas
            total_cost = sum(delta)
            task_costs.append(total_cost)
        cost_matrix.append(task_costs)
    cost_matrix = np.array(cost_matrix)
    return cost_matrix


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
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    list_choices = [(row, col) for row, col in zip(row_ind, col_ind)]
    total_cost = cost_matrix[row_ind, col_ind].sum()
    return list_choices




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
