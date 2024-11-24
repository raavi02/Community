from scipy.optimize import linear_sum_assignment
import numpy as np


PHASE_1_ASSIGNMENTS = False


def phaseIpreferences(player, community, global_random):
    """Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id
    """
    list_choices = []

    if PHASE_1_ASSIGNMENTS:
        partner_loss_threshold = 0
        assignments, total_cost = assign_phase1(community.tasks, community.members)

        for assignment in assignments:
            if player.id in assignment[0] and assignment[2] <= partner_loss_threshold:
                for task in community.tasks:
                    if task == assignment[1]:
                        if player.id == assignment[0][0]:
                            list_choices.append(
                                [community.tasks.index(task), assignment[0][1]]
                            )
                        else:
                            list_choices.append(
                                [community.tasks.index(task), assignment[0][0]]
                            )

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

        best_task_cost = loss_phase2(
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
            cost_matrix[i][j] = loss_phase1(task, member1, member2)

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    assignments = []
    for task_idx, partnership_idx in zip(row_indices, col_indices):
        member1_idx, member2_idx = partnerships[partnership_idx]
        member1 = members[member1_idx]
        member2 = members[member2_idx]
        loss = cost_matrix[task_idx, partnership_idx]
        assignments.append(([member1.id, member2.id], tasks[task_idx], loss))

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
            cost_matrix[i][j] = loss_phase2(task, member.abilities, member.energy)

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    assignments = {player: task for task, player in zip(row_indices, col_indices)}
    total_cost = sum(
        cost_matrix[row_indices[i], col_indices[i]] for i in range(len(row_indices))
    )

    return assignments, total_cost


def loss_phase1(task, player1, player2):
    cost = sum(
        max(task[k] - max(player1.abilities[k], player2.abilities[k]), 0)
        for k in range(len(task))
    )
    cost += max(0, cost - player1.energy - player2.energy) / 2
    cost += sum(
        max(max(player1.abilities[k], player2.abilities[k]) - task[k], 0)
        for k in range(len(task))
    )
    cost += sum(
        abs(player1.abilities[k] - player2.abilities[k]) for k in range(len(task))
    )
    return cost


def loss_phase2(task, abilities, current_energy):
    cost = sum([max(task[k] - abilities[k], 0) for k in range(len(abilities))])
    cost += max(0, cost - current_energy)
    return cost
