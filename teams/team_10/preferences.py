import numpy as np
from community import Community

tasks_at_turn = {}
acceptable_energy_level_at_turn = {}

START_SACRIFICING_YEAR = 30
LOW_ENERGY_LEVEL = -9
NORMAL_ENERGY_LEVEL = 0

def phaseIpreferences(player, community: Community, global_random):
    """Return a list of task index and the partner id for the particular player.
    The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list]
    and the second index as the partner id"""
    global tasks_at_turn
    global acceptable_energy_level_at_turn
    id_string = str(player.id)
    
    
    if id_string not in tasks_at_turn:
        print(f"Initializing tasks_at_turn for player {id_string}")
        tasks_at_turn[id_string] = []

    tasks_at_turn[id_string].append(len(community.tasks))
    
    if id_string not in acceptable_energy_level_at_turn:
        print(f"Initializing acceptable_energy_level_at_turn for player {id_string}")
        acceptable_energy_level_at_turn[id_string] = []
    acceptable_energy_level = get_acceptable_energy_level(player, tasks_at_turn)
    acceptable_energy_level_at_turn[id_string].append(acceptable_energy_level)
    if id_string == str(0):
        print(f"tasks_at_turn: {tasks_at_turn}")
        print(f"acceptable_energy_level_at_turn: {acceptable_energy_level_at_turn}")

    zero_loss_pairs = find_pairs(
        player, community.tasks, community.members, acceptable_energy_level
    )
    return zero_loss_pairs


tasks_at_turn2 = {}
acceptable_energy_level_at_turn2 = {}

def phaseIIpreferences(player, community, global_random):
    """Return a list of tasks for the particular player to do individually"""
    global tasks_at_turn2
    global acceptable_energy_level_at_turn2
    id_string = str(player.id)
    # Check if the player's ID is already in tasks_at_turn2
    if id_string not in tasks_at_turn2:
        print(f"Initializing tasks_at_turn2 for player phase 2 {id_string}")
        tasks_at_turn2[id_string] = []

    # Add the number of tasks for this turn
    tasks_at_turn2[id_string].append(len(community.tasks))

    # Check if the player's ID is already in acceptable_energy_level_at_turn2
    if id_string not in acceptable_energy_level_at_turn2:
        print(f"Initializing acceptable_energy_level_at_turn for player {id_string}")
        acceptable_energy_level_at_turn2[id_string] = []

    # Update acceptable energy level
    acceptable_energy_level = get_acceptable_energy_level(player, tasks_at_turn2)
    acceptable_energy_level_at_turn2[id_string].append(acceptable_energy_level)

    # Debugging: Print the current state
    if id_string == 0:
        print(f"tasks_at_turn2: {tasks_at_turn2}")
        print(f"acceptable_energy_level_at_turn2: {acceptable_energy_level_at_turn2}")
        
    if player.energy < -10:
        return []
    START_SACRIFICING_YEAR = 20
    
    if len(tasks_at_turn2[id_string]) >= START_SACRIFICING_YEAR and all(
        x == LOW_ENERGY_LEVEL for x in acceptable_energy_level_at_turn2[id_string][-10:]
    ):
        # SACRIFICE THE ELDERS / CHILDREN
        elders_or_children = find_weakest_agents(community.members)
        print(f"elders_or_children: {elders_or_children}")
        if id_string in elders_or_children:
            return [task_id for (task_id, _) in community.tasks]
        else:
            return []

    return tasks_we_can_complete_alone(player, player.abilities, community.tasks)
    # bids = []
    # if player.energy < 0:
    #     return bids
    # num_abilities = len(player.abilities)
    # for i, task in enumerate(community.tasks):
    #     energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(num_abilities)])
    #     if energy_cost >= 10:
    #         continue
    #     bids.append(i)
    # return bids


def get_acceptable_energy_level(player, tasks_at_turn):
    # tasks_at_turn[player.id].append(len(community.tasks))
    id_string = str(player.id)
    if len(tasks_at_turn[id_string]) >= 10 and all(
        x == tasks_at_turn[id_string][-1] for x in tasks_at_turn[id_string][-10:]
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


def find_pairs(player, tasks, members, acceptable_energy_level) -> list[int]:
    # [Task_ID, other_player_id ]
    task_player_pairs = []
    our_abilities_np = np.array(player.abilities)
    our_id = player.id
    for task_id, task in enumerate(tasks):
        task_np = np.array(task)

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
                task_player_pairs.append([task_id, other_person.id])
            if len(task_player_pairs) >= 8:
                return task_player_pairs

    return task_player_pairs


# For example, suppose n=4 and that an individual has skill levels (8,6,4,2), and that the task has difficulty vector (5,5,5,5). Then the individual would use up 1+3=4 units of energy to perform the task.

def find_weakest_agents(members):
    agents = [(id, sum(player.abilties)) for player in members]
    three_weakest_agents = sorted(agents, key=lambda x: x[1])[:3]
    return three_weakest_agents
