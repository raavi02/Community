import math
import matplotlib.pyplot as plt

turns = []
def getPainThreshold(community):
    '''Computes the pain threshold according to the average task difficulty and median member abilities'''
    
    # Only consider members who are not tired
    valid_members = [member for member in community.members if member.energy >= 0]

    if len(valid_members) == 0 or len(community.tasks) == 0:
        return -10
    
    # Find the maximum optimal energy pair expenditure across all tasks
    max_energy_across_all_tasks = -float("inf")
    for task in community.tasks:
        min_energy_expended = float("inf")
        for m1 in valid_members:
            for m2 in valid_members:
                if m1 == m2:
                    continue

                joint_abilities = [max(a1, a2) for a1, a2 in zip(m1.abilities, m2.abilities)]
                pair_energy_cost = sum([max(task[j] - joint_abilities[j], 0) for j in range(len(task))]) / 2

                if pair_energy_cost < min_energy_expended:
                    min_energy_expended = pair_energy_cost
        
        if min_energy_expended > max_energy_across_all_tasks:
            max_energy_across_all_tasks = min_energy_expended

    pain_threshold = 0 if max_energy_across_all_tasks < 10 else 10 - max_energy_across_all_tasks
    
    if pain_threshold <= -10:
        pain_threshold = -9     # prevent incapacitation

    return pain_threshold

def phaseIpreferences(player, community, global_random):
    '''Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id'''

    # Don't volunteer if tired
    if player.energy < 0:
        return []
    
    num_abilities = len(player.abilities)

    # Sort tasks in order of descending difficulty
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)
    pain_threshold = getPainThreshold(community)
    
    # Form partnerships based on the tasks
    partner_choices = []
    
    for task_id, task in sorted_tasks:
        solo_energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(num_abilities)])

        # Initialize params to find optimal partner for current task
        best_partner_id = None
        best_pair_energy_cost = float("inf")

        for partner in community.members:
            if partner.id == player.id or partner.energy < 0:
                continue
        
            joint_abilities = [max(a1, a2) for a1, a2 in zip(player.abilities, partner.abilities)]
            pair_energy_cost = sum([max(task[j] - joint_abilities[j], 0) for j in range(num_abilities)]) / 2

            # Skip partnership if energy cost will bring either member below the pain threshold
            if player.energy - pair_energy_cost < pain_threshold or partner.energy - pair_energy_cost < pain_threshold:
                continue
            
            # Better partnership found, update params
            if pair_energy_cost < best_pair_energy_cost:
                best_pair_energy_cost = pair_energy_cost
                best_partner_id = partner.id
        

        #if the everage energy of the community is greter than 5, we can push players to work solo
        avg_energy = getAvgEnergy(community)

        if (avg_energy >= 5): dynamic_threshold = 2
        else: dynamic_threshold = 1.2 # CHANGE TO MAKE DYNAMIC

        # Only bid for partnership if its more energy efficient than solo work
        if solo_energy_cost >= dynamic_threshold * best_pair_energy_cost:
            partner_choices.append([task_id, best_partner_id])

    return partner_choices

def getAvgEnergy(community):
    total_energy = 0
    for player in community.members:
        total_energy += player.energy
    return total_energy / len(community.members)
    
def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually'''
    
    # Don't volunteer if tired
    if player.energy < 0:
        return []
    
    bids = []

    for task_id, task in enumerate(community.tasks):
        energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(len(task))])
        
        # Volunteering logic
        if player.energy - energy_cost >= 0:
            bids.append(task_id)

    # # Handle extremely difficult tasks
    # diff_tasks = findDifficultTasks(community)
    # if len(diff_tasks) > 0:
    #     print("Difficult tasks:", len(diff_tasks))
    #     for task_id, _ in diff_tasks:
    #         # attempt to execute a sacrifice for these tasks
    #         sacrificer_id = executeSacrifice(community, community.tasks[task_id], len(player.abilities))
    #         if sacrificer_id is not None:
    #             print(f"Sacrificed player {sacrificer_id} to complete task {task_id}.")
    #             bids.append(task_id)

    return bids

def findDifficultTasks(community):
    """
    Identifies difficult tasks that cannot be completed by any individual player
    or through partnerships without causing energy exhaustion.
    """
    difficult_tasks = []
    num_abilities = len(community.members[0].abilities)

    for task_id, task in enumerate(community.tasks):
        # Check if any single player can complete the task
        # if a players' abilities exceed the task difficulties, the energy deficit is 0
        # if a players' abilities come short of the task difficulties, there is a non-zero energy deficit
            # if the non-zero energy deficit results in the overall energy going to -10 or below, that player is incapacitated
        individual_fail = all(
            player.energy - sum([max(task[j] - player.abilities[j], 0) for j in range(num_abilities)]) <= -10
            for player in community.members
        )

        partnership_fail = True
        for i, player1 in enumerate(community.members):
            for j, player2 in enumerate(community.members):
                if i >= j:  # Avoid self-pairing
                    continue

                # Joint abilities of the partnership
                joint_abilities = [max(player1.abilities[k], player2.abilities[k]) for k in range(num_abilities)]
                energy_cost = sum([max(task[l] - joint_abilities[l], 0) for l in range(num_abilities)]) / 2

                # Check if both players can handle their share of the energy cost
                if player1.energy - energy_cost > -10 and player2.energy - energy_cost > -10:
                    partnership_fail = False
                    break
            if not partnership_fail:
                break

        # if neither individuals nor partnerships can complete the task, it's "difficult"
        if individual_fail and partnership_fail:
            max_energy_loss = max([
                sum([max(task[j] - player.abilities[j], 0) for j in range(num_abilities)])
                for player in community.members
            ])
            difficult_tasks.append((task_id, max_energy_loss))

    return difficult_tasks

def isValidForPartnership(player, community, task):
    if player.energy <= 0:
        return False
    
    ### THIS CODE BELOW IS LIKE THE STRONG PLAYER LOGIC, DIDN'T HELP....

    # if task:
    #     energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(len(task))])
    #     if energy_cost == 0:
    #         return False
    # else:
    #     for task in community.tasks:
    #         energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(len(task))])
    #         if energy_cost == 0:
    #             return False

    # sacrificing strategy
    
    return True


def executeSacrifice(community, task, num_abilities):
    """
    Execute a sacrifice to complete a difficult task.
    - Chooses the player with the lowest abilities and lowest energy to sacrifice.
    - Ensures no valid individual solution exists (partnerships already ruled out in phase I).
    """
    # Sort members by (total ability, energy), ascending
    sorted_members = sorted(
        community.members,
        key=lambda member: (sum(member.abilities), member.energy)
    )

    return sorted_members[0]
