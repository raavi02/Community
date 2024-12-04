import pprint
from typing import List
import numpy as np
from community import Community, Member
import pandas as pd
import os
from teams.team_10.constants import *

# Task 1: Figure out global variables...
# Nicky 

# Task 2: NEW SACRIFICING STRATEGY -- IMPLEMENTED 
# Julianna 

# Task 3: Check if any player is capable of doing certains tasks... (Open ended) -- THINKING ON THIS
# Julianna

# Task 4:
# We want to make sure we don't voluenteer for tasks that an individual can complete (IN PROGRESS)
# Julianna 

# Task 5: 
# Only volenteering the best 5 or 10 tasks (Max)
# 1. We'll probably get a speed up from early returning
# 2. Don't really want to step on other peoples
# Akhil

# Task 6: Variable lower bounds.. Would be cool to adjust lower bounds gradually? 
# Akhil


# USE best pairs instead of just the first 8.


# Task 7: (Do later...)
# We could just play submissive.... 
# We see if we are playing with a bunch of other teams.
# Just volenteer for every task with every player from every other team
# - We don't something weird / worse than them. 
# - Reduces the complexity of the situation.


def phaseIpreferences(player, community: Community, global_random):
    """Return a list of task index and the partner id for the particular player.
    The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list]
    and the second index as the partner id"""
    
    if not hasattr(player, "data_store"): # Start tracking data
        player.data_store = {
            "turn": 1,  # Tracks the current turn number
            "tasks_at_turn": [], # Tracks the number of tasks on the board at each turn
            "acceptable_energy_level_at_turn": [], # Tracks the acceptable energy level at each turn
        }
    else:
        # Increment the turn
        player.data_store["turn"] += 1
    
    
    tasks_at_turn = player.data_store['tasks_at_turn']
    acceptable_energy_level = get_acceptable_energy_level(tasks_at_turn)
    
    # Update the data store
    player.data_store["tasks_at_turn"].append(len(community.tasks))
    player.data_store["acceptable_energy_level_at_turn"].append(acceptable_energy_level)

    task_pairs = find_pairs(
        player, community.tasks, community.members, acceptable_energy_level
    )
    
    if player.incapacitated:
        return []
    return task_pairs


def phaseIIpreferences(player: Member, community, global_random):
    """Return a list of tasks for the particular player to do individually"""
    # if player.data_store is not None: 
        # print("YAY")
        
    if player.incapacitated:
        return []
    
    tasks_at_turn = player.data_store['tasks_at_turn']
    acceptable_energy_level_at_turn = player.data_store['acceptable_energy_level_at_turn']
    
    NUM_TURNS_TO_WAIT_BEFORE_SACRIFICING = 5

    sacrifices = sacrifice(community.members, community.tasks)
    SACRIFICE_TIME = len(community.members) // 2

    if len(community.tasks) < SACRIFICE_TIME and sacrifices:
        to_be_sacrificed = find_weakest_agents(community.members, len(sacrifices))
        if player.id in to_be_sacrificed: 
            return sacrifices 
    START_SACRIFICING_YEAR = 20
    
    if len(tasks_at_turn) >= START_SACRIFICING_YEAR and all(
        x == LOW_ENERGY_LEVEL for x in acceptable_energy_level_at_turn[-NUM_TURNS_TO_WAIT_BEFORE_SACRIFICING:]
    ):
        # SACRIFICE THE ELDERS / CHILDREN
        num_weakest_palayers = 2
        weakest_players_id = find_weakest_agents(community.members, num_weakest_palayers)
        if player.id in weakest_players_id:
            return [task_id for task_id, _ in enumerate(community.tasks)]
        else:
            return []

    return tasks_we_can_complete_alone(player, player.abilities, community.tasks)
    

def get_acceptable_energy_level(tasks_at_turn: list[int]) -> int:
    
    TURNS_TO_LOOK_BACK = 3
    if len(tasks_at_turn) >= 3 and all(
        x == tasks_at_turn[-1] for x in tasks_at_turn[-TURNS_TO_LOOK_BACK:]
    ):
        acceptable_energy_level = LOW_ENERGY_LEVEL
    else:
        acceptable_energy_level = NORMAL_ENERGY_LEVEL
    
    return acceptable_energy_level

    
def tasks_we_can_complete_alone(player, our_abilities, tasks) -> list[int]:
    alone_tasks = []
    our_abilities_np = np.array(our_abilities)

    for task_id, task in enumerate(tasks):
        task_np = np.array(task)
        diff = our_abilities_np - task_np
        negative_sum = np.sum(diff[diff < 0])

        if player.energy + negative_sum > 0:
            alone_tasks.append(task_id)
    return alone_tasks


def find_pairs(player: Member, tasks, members, acceptable_energy_level) -> list[int]:
    # [Task_ID, other_player_id ]
    task_player_pairs = {}
    our_abilities_np = np.array(player.abilities)
    our_id = player.id
    for task_id, task in enumerate(tasks):
        task_np = np.array(task)
        task_player_pairs[task_id] = []
        for other_person in members:
            if other_person.id == our_id:
                continue

            others_abilities_np = np.array(other_person.abilities)
            combined_abilties = np.maximum(our_abilities_np, others_abilities_np)

            diff = combined_abilties - task_np
            negative_sum = np.sum(diff[diff < 0])
            player_above_acceptable_energy = (
                player.energy + negative_sum > acceptable_energy_level
            )
            other_person_above_acceptable_energy = (
                other_person.energy + negative_sum > acceptable_energy_level
            )
            # print(negative_sum)
            if player_above_acceptable_energy and other_person_above_acceptable_energy:
                task_player_pairs[task_id].append((abs(negative_sum), other_person.id))
    pairs = []
    for task in task_player_pairs:
        least_energy_pairs = sorted(task_player_pairs[task])
        for i in range(3):
            if i >= len(least_energy_pairs):
                break
            pairs.append([task, least_energy_pairs[i][1]])
    return pairs

# For example, suppose n=4 and that an individual has skill levels (8,6,4,2), and that the task has difficulty vector (5,5,5,5). Then the individual would use up 1+3=4 units of energy to perform the task.
"""
sacrifice(members, tasks)
- identifies if there are tasks that require sacrificing members
- returns a list of tasks that require sacrifices
"""
def sacrifice(members: List[Member], tasks: List) -> List[int]:
    exhausting_tasks = []
    for i in range(len(tasks)):
        task = tasks[i]
        task_np = np.array(task)
        not_exhausting = False
        for member in members:
            abilities_np = np.array(member.abilities)
            diff = abilities_np - task_np
            # if one member can complete it themselves without being sacrificed
            if np.sum(diff[diff < 0]) < (MAX_ENERGY_LEVEL - EXHAUSTED_ENERGY_LEVEL):
                not_exhausting = True
            # else, check pairs 
            else:
                for other in members:
                    other_np = np.array(other.abilities)
                    combined_np = np.maximum(abilities_np, other_np)
                    diff = combined_np - task_np
                    if np.sum(diff[diff < 0]) < (MAX_ENERGY_LEVEL - EXHAUSTED_ENERGY_LEVEL) * 2:
                        not_exhausting = True
                        break
            if not_exhausting:
                break
        if not not_exhausting:
            exhausting_tasks.append(i)
    return exhausting_tasks

"""
non_solo_tasks(community: Community)
- identifies which tasks can be completed with no/little energy depletion by one member of society
- returns a list of all indices of tasks that should (and can) be completed by pairs
"""
# def non_solo_tasks(community: Community):
#     tasks = []

# def all_energized(members: List[Member]) -> bool:
#     for member in members:
#         if not member.incapacitated and member.energy < MAX_ENERGY_LEVEL:
#             return False
#     return True

    
"""
original sacrificing strategy in phase II:
if len(tasks_at_turn) >= START_SACRIFICING_YEAR and all(
        x == LOW_ENERGY_LEVEL for x in acceptable_energy_level_at_turn[-NUM_TURNS_TO_WAIT_BEFORE_SACRIFICING:]
    ):
        # SACRIFICE THE ELDERS / CHILDREN
        elders_or_children = find_weakest_agents(community.members, 3)
        print(f"elders_or_children: {elders_or_children}")
        if player.id in elders_or_children:
            return [task_id for (task_id, _) in community.tasks]
        else:
            return []
"""

# Task 7: (Do later...)
# We could just play submissive.... 
# We see if we are playing with a bunch of other teams.
# Just volenteer for every task with every player from every other team
# - We don't something weird / worse than them. 
# - Reduces the complexity of the situation.


def find_weakest_agents(members, num_weakest) -> list[int]:
    """Return the id of the weakest agents in the community"""
    agents = [(member.id, sum(member.abilities)) for member in members if not member.incapacitated]

    three_weakest_agents = sorted(agents, key=lambda x: x[1])[:num_weakest]
    weakest_agent_ids = [agent[0] for agent in three_weakest_agents]
    return weakest_agent_ids
