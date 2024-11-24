from scipy.optimize import linear_sum_assignment
import numpy as np


def phaseIpreferences(player, community, global_random):
    """Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id
    """
    list_choices = []

    assignments, total_cost = assign_phase1(community.tasks, community.members)

    return list_choices


def phaseIIpreferences(player, community, global_random):
    """Return a list of tasks for the particular player to do individually"""
    bids = []
    if player.energy < 0:
        return bids

    try:
        wait_energy_threshold = 0
        player_index = community.members.index(player)
        assignments, total_cost = assign_phase2(community.tasks, community.members)

        best_task = assignments.get(player_index)
        if best_task is None:
            return []

        best_task_cost = loss_func(
            community.tasks[best_task], player.abilities, player.energy
        )
        if player.energy - best_task_cost < wait_energy_threshold:
            return []

        return [best_task]
    except Exception as e:
        print(e)
        return bids


def assign_phase1(tasks, members):
    num_tasks = len(tasks)
    num_members = len(members)

    partnerships = []
    for i in range(num_members):
        for j in range(i + 1, num_members):
            partnerships.append((i, j))
    num_partnerships = len(partnerships)

    cost_matrix = np.zeros((num_tasks, num_partnerships))

    for i, task in enumerate(tasks):
        for j, (member1_idx, member2_idx) in enumerate(partnerships):
            member1 = members[member1_idx]
            member2 = members[member2_idx]
            cost_matrix[i][j] = loss_func(
                task, np.maximum(member1.abilities, member2.abilities), 0
            )

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    assignments = {}
    for task_idx, partnership_idx in zip(row_indices, col_indices):
        member1_idx, member2_idx = partnerships[partnership_idx]
        task = tasks[task_idx]
        member1 = members[member1_idx]
        member2 = members[member2_idx]
        assignments[member1_idx] = (task, member1, member2)

    total_cost = sum(
        cost_matrix[row_indices[i], col_indices[i]] for i in range(len(row_indices))
    )

    return assignments, total_cost


def assign_phase2(tasks, members):
    num_tasks = len(tasks)
    num_members = len(members)

    cost_matrix = np.zeros((num_tasks, num_members))

    for i, task in enumerate(tasks):
        for j, member in enumerate(members):
            cost_matrix[i][j] = loss_func(task, member.abilities, member.energy)

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    assignments = {player: task for task, player in zip(row_indices, col_indices)}
    total_cost = sum(
        cost_matrix[row_indices[i], col_indices[i]] for i in range(len(row_indices))
    )

    return assignments, total_cost


def loss_phase1(player1, player2, task):
    return loss_func(task, np.maximum(player1.abilities, player2.abilities), 0)


def loss_func(task, abilities, current_energy):
    cost = sum([max(task[k] - abilities[k], 0) for k in range(len(abilities))])
    cost += max(0, cost - current_energy)
    return cost
