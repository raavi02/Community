import pprint
from typing import List
import numpy as np
from community import Community, Member
import pandas as pd
from teams.team_10.constants import *

def phaseIpreferences(player, community: Community, global_random):
    """Return a list of task index and the partner id for the particular player.
    The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list]
    and the second index as the partner id"""
    
    if not hasattr(player, "data_store"): # Start tracking data
        player.data_store = {
            "turn": 1,  # Tracks the current turn number
            "tasks_at_turn": [], # Tracks the number of tasks on the board at each phase
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


def phaseIIpreferences(player: Member, community: Community, global_random):
    """Return a list of tasks for the particular player to do individually"""
        
    if player.incapacitated:
        return []
    
    tasks_at_turn = player.data_store['tasks_at_turn']
    acceptable_energy_level_at_turn = player.data_store['acceptable_energy_level_at_turn']
    members = community.members
    
    if len(tasks_at_turn) >= START_SACRIFICING_YEAR and all(
        x == LOW_ENERGY_LEVEL for x in acceptable_energy_level_at_turn[-NUM_TURNS_TO_WAIT_BEFORE_SACRIFICING:]):
        num_weakest_players = 2
        weakest_players_id = find_weakest_agents(community.members, num_weakest_players)
        if player.id in weakest_players_id:
            return [task_id for task_id, _ in enumerate(community.tasks)]
        else:
            return []

    return tasks_we_can_complete_alone(player, player.abilities, community.tasks)
    

def get_acceptable_energy_level(tasks_at_turn: list[int]) -> int:
    
    if stayed_constant(tasks_at_turn=tasks_at_turn):
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

def find_weakest_agents(members, num_weakest) -> list[int]:
    """Return the id of the weakest agents in the community"""
    agents = [(member.id, sum(member.abilities)) for member in members if not member.incapacitated]

    three_weakest_agents = sorted(agents, key=lambda x: x[1])[:num_weakest]
    weakest_agent_ids = [agent[0] for agent in three_weakest_agents]
    return weakest_agent_ids

def stayed_constant(tasks_at_turn: List[int]):
    if len(tasks_at_turn) >= TURNS_TO_LOOK_BACK and all(
        x == tasks_at_turn[-1] for x in tasks_at_turn[-TURNS_TO_LOOK_BACK:]
    ):
        return True
    return False

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