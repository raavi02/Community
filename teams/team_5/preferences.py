import numpy as np
import scipy.optimize as opt
import random
import math
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from teams.team_5.Hivemind import Hivemind
import sys

# List of things to implement (Feel free to add to it)
#TODO: Finding the weakest players (Dummy players only? Or we include weak players too)
#TODO: Determine when a player should rest
#TODO: Minimize the number of pairs a player volunteers for (Currently we are volunteering for a lot)

def phaseIpreferences(player, community, global_random):
    '''Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id'''

    list_choices = []
    surviving_workers = []

    num_of_workers = len(community.members)

    # Find out the remaining workers
    for i in range(num_of_workers):
        if community.members[i].energy > -10:
            # print(f"I CAN WORK, Says: {community.members[i].id}")
            surviving_workers.append(community.members[i])
    # If player is exhausted, return an empty list (no tasks)
    # if player.energy < 10:
    #     return list_choices

    free_tasks = set()

    # Check if the player can do individual tasks (no partner needed)
    for i, task in enumerate(community.tasks):
        for member in surviving_workers:
            energy_cost = sum([max(task[j] - member.abilities[j], 0) for j in range(len(member.abilities))])
            if energy_cost == 0:
           # print("Can take individual Task: ", task, " Player: ", player.abilities)
                free_tasks.add(i)
                break

    # Prioritize tasks based on energy cost and reward-to-energy ratio (score)
    # task_priorities = []

    # for i, task in enumerate(community.tasks):
    #     energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(len(player.abilities))])
    #     reward = sum([task[j] for j in range(len(task))])  # Sum of task difficulty (can be adjusted)
    #     score = reward / (energy_cost + 1)  # Reward-to-energy ratio (higher is better)
    #     task_priorities.append((i, score, energy_cost))

    # Sort tasks by score (reward-to-energy ratio, descending)
    # task_priorities.sort(key=lambda x: x[1], reverse=True)

    # Dynamic adjustment of max tasks to consider
    # max_tasks_to_consider = int(player.energy) # Adjust based on player’s energy (higher energy allows more tasks)

    # top_tasks = task_priorities[:max_tasks_to_consider]

    exhausted_penalty = 0.5 
    best_choice = []
    chosen_partner = set()
    # Find partners for the selected tasks based on combined abilities and energy constraints
    for task_id, task in enumerate(community.tasks):
        if task_id in free_tasks:
            continue

        best_partner = None
        min_metric = float('inf')
        min_cost = float('inf')

        # Iterate through community members to find the best partner
        for partner in surviving_workers:
            if partner.id == player.id:
                continue

            combined_abilities = [max(player.abilities[k], partner.abilities[k]) for k in range(len(player.abilities))]
            energy_cost = sum([max(community.tasks[task_id][j] - combined_abilities[j], 0) for j in range(len(player.abilities))])
            added_cost = exhausted_penalty if partner.energy < 0 else 0
            partner_lazy = -10 if partner.energy == 10 else 0
            lazy = -10 if player.energy == 10 else 0

            # Total energy cost considering player and partner's combined abilities
            comparison_metric = energy_cost + added_cost + lazy + partner_lazy

            if comparison_metric < min_metric and (partner.energy - (energy_cost / 2)) > -10 and (player.energy - (energy_cost / 2)) > -10 and partner.id not in chosen_partner:
                min_metric = comparison_metric
                best_partner = partner.id
                min_cost = energy_cost
                chosen_partner.add(best_partner)

        # If a compatible partner is found, assign the task to the player and partner
        if best_partner:
            best_choice = [task_id, best_partner, min_cost]
            # print(f"Worker {player.id} has chosen Worker {best_choice[1]} for {best_choice[0]}")

        if best_choice:
            list_choices.append(best_choice[:-1])

    return list_choices
          
def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually'''
    # global hivemind

    # if hivemind and hivemind.prev_task == len(community.tasks):
    #     bids = []
    #     bids.append(hivemind.phase2_optimal_pair[player.id])
    #     return bids

    bids = []
    
    surviving_workers = []

    num_of_workers = len(community.members)
    num_of_tasks = len(community.tasks)
    num_of_abilities = len(player.abilities)

    # Find out the remaining workers
    for i in range(num_of_workers):
        if community.members[i].energy > -10:
            # print(f"I CAN WORK, Says: {community.members[i].id}")
            surviving_workers.append(community.members[i])

    num_of_workers = len(surviving_workers) 

    cost_matrix  = [[np.inf for m in range(num_of_tasks)] for n in range(num_of_workers)] # n x m
    assert(len(cost_matrix) == num_of_workers and len(cost_matrix[0]) == num_of_tasks) # matrix check

    too_good = 1/((num_of_abilities * 10) + 1)
    tired_param = 2
    exhausted_param = 100000

    # Fill in matrix with value
    #TODO: Let the worst worker to be sacrificed first.
    for i in range(num_of_workers):
        for j in range(num_of_tasks):
            cost_matrix[i][j] = sum([max(community.tasks[j][k] - surviving_workers[i].abilities[k], 0) for k in range(num_of_abilities)])
            if (surviving_workers[i].energy - cost_matrix[i][j]) <= -10: # Worker will be sacrificed, if need be.
                cost_matrix[i][j] = exhausted_param
            elif (surviving_workers[i].energy - cost_matrix[i][j]) < 0:
                cost_matrix[i][j] *= tired_param
            elif cost_matrix[i][j] == 0:
                cost_matrix[i][j] += too_good * (sum([surviving_workers[i].abilities[k] - community.tasks[j][k] for k in range(num_of_abilities)]))

    # print("COST MATRIX:")
    # print(cost_matrix)

    # Solver
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")

    # Variables
    # x[i, j] is an array of 0-1 variables, which will be 1
    # if worker i is assigned to task j.
    x = {}
    for i in range(num_of_workers):
        for j in range(num_of_tasks):
            x[i, j] = solver.IntVar(0, 1, "")

    # Constraints
    if num_of_tasks > num_of_workers:
        # Each worker is assigned to at EXACTLY 1 task.
        for i in range(num_of_workers):
            solver.Add(solver.Sum([x[i, j] for j in range(num_of_tasks)]) == 1)

        # Each task is assigned to ONE OR LESS worker.
        for j in range(num_of_tasks):
            solver.Add(solver.Sum([x[i, j] for i in range(num_of_workers)]) <= 1)
    else:
        # Each worker is assigned to ONE OR LESS task.
        for i in range(num_of_workers):
            solver.Add(solver.Sum([x[i, j] for j in range(num_of_tasks)]) <= 1)

        # Each task is assigned to EXACTLY 1 worker.
        for j in range(num_of_tasks):
            solver.Add(solver.Sum([x[i, j] for i in range(num_of_workers)]) == 1)

    objective_terms = []
    for i in range(num_of_workers):
        for j in range(num_of_tasks):
            objective_terms.append(cost_matrix[i][j] * x[i, j])

    solver.Minimize(solver.Sum(objective_terms))

    # Solve
    # print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()

    phase2_optimal_pair = dict()

    # Print solution.
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        # print(f"Total cost = {solver.Objective().Value()}\n")
        for i in range(num_of_workers):
            for j in range(num_of_tasks):
                # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
                if x[i, j].solution_value() > 0.5:
                    # print(f"Worker {i} assigned to task {j}." + f" Cost: {cost_matrix[i][j]}")
                    sacrifice = True
                    if cost_matrix[i][j] == exhausted_param:
                        max_energy = 10

                        for k, worker_1 in enumerate(surviving_workers):
                            for l, worker_2 in enumerate(surviving_workers):    
                                if l <= k:
                                    continue
                            
                                combined_abilities = [max(worker_1.abilities[m], worker_2.abilities[m]) for m in range(len(player.abilities))]
                                energy_cost = sum([max(community.tasks[j][m] - combined_abilities[m], 0) for m in range(len(player.abilities))])

                                if (max_energy - (energy_cost / 2)) > -10:
                                    sacrifice = False
                                    # print(f"Worker {k} and Worker {l} can do Task {j} without dying")
                    if sacrifice:
                        phase2_optimal_pair[surviving_workers[i].id] = j
    else:
        print("No solution found.")
 
    hivemind = Hivemind(phase2_optimal_pair=phase2_optimal_pair, prev_task=len(community.tasks))

    if player.id in hivemind.phase2_optimal_pair:
        bids.append(hivemind.phase2_optimal_pair[player.id]) 

    return bids
