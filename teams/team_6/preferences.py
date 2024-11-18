from scipy.optimize import linear_sum_assignment
import numpy as np


def phaseIpreferences(player, community, global_random):
    """Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id
    """
    list_choices = []
    if player.energy < 0:
        return list_choices
    num_members = len(community.members)
    partner_id = num_members - player.id - 1
    list_choices.append([0, partner_id])
    if len(community.tasks) > 1:
        list_choices.append([1, partner_id])
    list_choices.append([0, min(partner_id + 1, num_members - 1)])
    return list_choices


def phaseIIpreferences(player, community, global_random):
    """Return a list of tasks for the particular player to do individually"""
    bids = []
    if player.energy < 0:
        return bids

    try:
        player_index = community.members.index(player)
        assignments, total_cost = optimal_assignment(community.tasks, community.members)

        print(total_cost)
        best_task = assignments.get(player_index)

        if best_task is not None:
            return [best_task]
        else:
            return []
    except Exception as e:
        print(e)
        return bids


def optimal_assignment(tasks, members):
    num_tasks = len(tasks)
    num_members = len(members)
    num_abilities = len(members[0].abilities)

    cost_matrix = np.zeros((num_tasks, num_members))

    for i, task in enumerate(tasks):
        for j, member in enumerate(members):
            cost_matrix[i][j] = sum(
                [max(task[k] - member.abilities[k], 0) for k in range(num_abilities)]
            )
            if member.energy - cost_matrix[i][j] < 0:
                cost_matrix[i][j] += 1e9

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    assignments = {player: task for task, player in zip(row_indices, col_indices)}
    total_cost = sum(
        cost_matrix[row_indices[i], col_indices[i]] for i in range(len(row_indices))
    )

    return assignments, total_cost
