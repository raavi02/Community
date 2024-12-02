from scipy.optimize import linear_sum_assignment
import numpy as np


PHASE_1_ASSIGNMENTS = False
PHASE_2_ASSIGNMENTS = False


def exists_good_match(
    tasks,
    player_abilities,
    each_difference=1,
    total_difference=None,
    penalty_tolerance=0,
    return_multiple_tasks=False,
):
    # go through all the tasks and check if there is a task whose difficulty in each category is less than the player's ability in that category by each_tolerance
    no_of_abilities = len(player_abilities)
    current_index = -1
    penalty_suffered = -1
    if total_difference == None:
        total_difference = no_of_abilities * each_difference

    returned_tasks = []
    for task in tasks:
        if (
            all(
                abs(player_abilities[i] - task[i]) <= each_difference
                for i in range(no_of_abilities)
            )
            and sum(
                [abs(player_abilities[i] - task[i]) for i in range(no_of_abilities)]
            )
            <= total_difference
        ):
            penalty = 0
            for i in range(no_of_abilities):
                if player_abilities[i] < task[i]:
                    penalty += task[i] - player_abilities[i]
            if penalty <= penalty_tolerance:
                returned_tasks.append(task)
                if penalty_suffered == -1 or penalty < penalty_suffered:
                    penalty_suffered = penalty
                    current_index = tasks.index(task)

    if current_index == -1:
        return False, penalty_suffered, current_index, []
    else:
        # print("TASKS: ", returned_tasks)
        return True, penalty_suffered, current_index, returned_tasks


def doable_tasks(player, community, allow_negative=True, max_allowed_loss=20):
    if allow_negative:
        remaining_energy = min(player.energy + 10, max_allowed_loss)
    else:
        remaining_energy = max(player.energy, 0)
    doable = []
    for task in community.tasks:
        energy_required = sum(
            max(0, task[i] - player.abilities[i]) for i in range(len(task))
        )
        if energy_required < remaining_energy or energy_required == 0:
            doable.append(task)
    return doable


def phaseIpreferences(player, community, global_random):
    """Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id
    """

    list_choices = []

    try:
        if PHASE_1_ASSIGNMENTS:
            assignments, total_cost = assign_phase1(community.tasks, community.members)
            for assignment in assignments:
                if len(assignment[0]) == 2 and player.id in assignment[0]:
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
        else:
            if (
                player.energy < 0
            ):  # If player has negative energy but can do at least one task with no energy loss, then don't partner up
                doable = doable_tasks(player, community, allow_negative=False)
                if len(doable) > 0:
                    return []
            else:  # If player has positive energy but can do at least one task with at most 50% energy loss, then don't partner up
                doable = doable_tasks(
                    player, community, max_allowed_loss=int(0.5 * player.energy)
                )
                if len(doable) > 0:
                    return []

            # perfect_match, _, _, _ = exists_good_match(
            # community.tasks,
            # player.abilities,
            # each_difference=1,
            # penalty_tolerance=2,
            # )
            # if perfect_match:
            #     # print("PERFECT MATCH")
            #     return []
            # else:

            for member in community.members:
                if (
                    member.energy < 0
                ):  # If member has negative energy but can do at least one task with no energy loss, then don't partner up
                    doable = doable_tasks(member, community, allow_negative=False)
                    if len(doable) > 0:
                        continue
                else:  # If member has positive energy but can do at least one task with at most 50% energy loss, then don't partner up
                    doable = doable_tasks(
                        member, community, max_allowed_loss=int(0.5 * member.energy)
                    )
                    if len(doable) > 0:
                        continue

                max_abilities = [
                    max(player.abilities[i], member.abilities[i])
                    for i in range(len(player.abilities))
                ]
                _, _, _, matching_tasks = exists_good_match(
                    community.tasks,
                    max_abilities,
                    each_difference=10,
                    penalty_tolerance=int(
                        0.5 * max(0, member.energy) + 0.5 * max(0, player.energy)
                    ),
                    return_multiple_tasks=True,
                )
                for matching_task in matching_tasks:
                    list_choices.append(
                        [community.tasks.index(matching_task), member.id]
                    )

            # threshold = 3
            # if len(list_choices) < threshold:
            #     for member in community.members:
            #         max_abilities = [
            #             max(player.abilities[i], member.abilities[i])
            #             for i in range(len(player.abilities))
            #         ]
            #         _, _, _, matching_tasks = exists_good_match(
            #             community.tasks,
            #             max_abilities,
            #             each_difference=4,
            #             penalty_tolerance=8,
            #             return_multiple_tasks=True,
            #         )
            #         for matching_task in matching_tasks:
            #             list_choices.append(
            #                 [community.tasks.index(matching_task), member.id]
            #             )

    except Exception as e:
        print(e)

    # print(list_choices)
    return list_choices


def phaseIIpreferences(player, community, global_random):
    """Return a list of tasks for the particular player to do individually"""

    weakest_player = weakest_member(player, community)
    # print("WEAKEST PLAYER: ", weakest_player)

    hard_tasks, impossible_tasks = find_impossible_tasks(community)

    # print("HARD TASKS: ", hard_tasks)
    # print("IMPOSSIBLE TASKS: ", impossible_tasks)

    if weakest_player == True and len(impossible_tasks) > 0:
        doable = doable_tasks(player, community)
        returned_tasks = []
        if len(doable) > 0:
            for task in doable:
                returned_tasks.append(community.tasks.index(task))
            return returned_tasks
        else:
            for task in impossible_tasks:
                returned_tasks.append(community.tasks.index(task))
            return returned_tasks

    bids = []
    if player.energy < 0:
        return bids

    try:
        if PHASE_2_ASSIGNMENTS:
            # Use cost matrix to assign tasks
            wait_energy_threshold = -9
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
        else:
            min_loss = float("inf")
            best_task = None

            for task in community.tasks:
                loss = loss_phase2(task, player.abilities, player.energy)
                resting_loss = loss_resting(task, player.abilities, player.energy)

                better_loss = min(loss, resting_loss)

                if better_loss < min_loss:
                    min_loss = better_loss

                    if loss < resting_loss:
                        best_task = community.tasks.index(task)
                    else:
                        best_task = None

            if best_task is not None:
                return [best_task]
            else:
                return []

    except Exception as e:
        print(e)
        return bids


def assign_phase1(tasks, members):
    num_tasks = len(tasks)
    num_members = len(members)

    # Generate all possible partnerships
    partnerships = []
    for i in range(num_members):
        for j in range(i + 1, num_members):
            partnerships.append((i, j))
    num_partnerships = len(partnerships)

    # Create cost matrix with columns for both partnerships and individual assignments
    cost_matrix = np.zeros((num_tasks, num_partnerships + num_members + num_members))

    # Fill partnership costs (num_partnerships columns)
    for i, task in enumerate(tasks):
        for j, (member1_idx, member2_idx) in enumerate(partnerships):
            member1 = members[member1_idx]
            member2 = members[member2_idx]
            cost_matrix[i][j] = loss_phase1(task, member1, member2)

    # Fill individual costs (num_members columns)
    for i, task in enumerate(tasks):
        for j, member in enumerate(members):
            cost_matrix[i][num_partnerships + j] = loss_phase2(
                task, member.abilities, member.energy
            )

    # Fill resting costs (num_tasks columns)
    for i, task in enumerate(tasks):
        for j, member in enumerate(members):
            # Resting cost is a function of current energy and some base resting penalty
            # This encourages rest when energy is low or no good tasks are available
            resting_cost = loss_resting(task, member.abilities, member.energy)

            # Column index for resting is the last set of columns
            cost_matrix[i][num_partnerships + num_members + j] = resting_cost

    # Solve assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Convert results to assignments
    assignments = []
    for task_idx, col_idx in zip(row_indices, col_indices):
        task = tasks[task_idx]
        # If column index is less than num_partnerships, it's a partnership
        if col_idx < num_partnerships:
            member1_idx, member2_idx = partnerships[col_idx]
            member1 = members[member1_idx]
            member2 = members[member2_idx]
            loss = cost_matrix[task_idx, col_idx]
            assignments.append(([member1.id, member2.id], task, loss))
            # print(f"Partnership -> cost: {loss}")
        elif col_idx < num_partnerships + num_members:
            # Individual assignment
            member_idx = col_idx - num_partnerships
            member = members[member_idx]
            loss = cost_matrix[task_idx, col_idx]
            assignments.append(([member.id], task, loss))
            # print(f"Individual -> cost: {loss}")
        else:
            # Resting
            loss = cost_matrix[task_idx, col_idx]
            assignments.append(([], task, loss))
            # print(f"Rest -> cost: {loss}")

    total_cost = sum(
        cost_matrix[row_indices[i], col_indices[i]] for i in range(len(row_indices))
    )

    return assignments, total_cost


def assign_phase2(tasks, members):
    num_tasks = len(tasks)
    num_members = len(members)

    cost_matrix = np.zeros((num_tasks, num_members + num_members))

    # Fill individual costs (num_members columns)
    for i, task in enumerate(tasks):
        for j, member in enumerate(members):
            cost_matrix[i][j] = loss_phase2(task, member.abilities, member.energy)

    # Fill resting costs (num_tasks columns)
    for i, task in enumerate(tasks):
        for j, member in enumerate(members):
            # Resting cost is a function of current energy and some base resting penalty
            # This encourages rest when energy is low or no good tasks are available
            resting_cost = loss_resting(task, member.abilities, member.energy)

            # Column index for resting is the last set of columns
            cost_matrix[i][num_members + j] = resting_cost

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    assignments = {}
    for task_idx, col_idx in zip(row_indices, col_indices):
        task = tasks[task_idx]
        if col_idx < num_members:
            member = members[col_idx]
            loss = cost_matrix[task_idx, col_idx]
            assignments[col_idx] = task_idx
        else:
            loss = cost_matrix[task_idx, col_idx]
            assignments[col_idx] = None

    total_cost = sum(
        cost_matrix[row_indices[i], col_indices[i]] for i in range(len(row_indices))
    )

    return assignments, total_cost


def loss_phase1(task, player1, player2):
    energy_used = sum(
        max(task[k] - max(player1.abilities[k], player2.abilities[k]), 0)
        for k in range(len(task))
    )
    negative_energy_compensation = (
        max(0, energy_used - player1.energy) + max(0, energy_used - player2.energy)
    ) / 2
    partnership_waste = sum(
        abs(player1.abilities[k] - player2.abilities[k]) for k in range(len(task))
    )
    skill_surplus = sum(
        max(max(player1.abilities[k], player2.abilities[k]) - task[k], 0)
        for k in range(len(task))
    )

    # cost = (
    #     energy_used + negative_energy_compensation + partnership_waste + skill_surplus
    # )
    cost = energy_used + negative_energy_compensation + skill_surplus

    scaling_factor = len(player1.abilities) * 10 * 3

    return cost


def loss_phase2(task, abilities, current_energy):
    energy_used = sum([max(task[k] - abilities[k], 0) for k in range(len(abilities))])
    negative_energy_compensation = max(0, energy_used - current_energy)
    skill_surplus = sum([abs(abilities[k] - task[k]) for k in range(len(abilities))])

    cost = energy_used + negative_energy_compensation + skill_surplus
    # cost = energy_used + negative_energy_compensation

    scaling_factor = len(abilities) * 10 * 3

    return cost / scaling_factor


def loss_resting(task, abilities, current_energy):
    energy_used = sum([max(task[k] - abilities[k], 0) for k in range(len(abilities))])
    cost = energy_used * max(0, current_energy)

    scaling_factor = len(abilities) * 10 * len(abilities) * 10

    return cost / scaling_factor


def weakest_member(player, community, top_n=1):
    all_skills = [sum(member.abilities) for member in community.members]
    player_skill = sum(player.abilities)

    if player_skill in sorted(all_skills)[:top_n]:
        return True
    else:
        return False


def find_impossible_tasks(community):
    impossible_tasks = []
    hard_tasks = []
    for task in community.tasks:
        best_case = 10**9
        for member in community.members:
            energy_deficit = sum(
                max(task[i] - member.abilities[i], 0) for i in range(len(task))
            )
            best_case = min(best_case, energy_deficit)
            if best_case <= 0:
                break
        if best_case >= 20:
            impossible_tasks.append(task)
        elif best_case >= 10:
            hard_tasks.append(task)

    return hard_tasks, impossible_tasks
