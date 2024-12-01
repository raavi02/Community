import math

def getPainThreshold(community):
    '''Computes the pain threshold according to the average task difficulty and median member abilities'''
    
    # Only consider members who are not overly tired
    valid_members = [member for member in community.members if member.energy >= -5]

    if len(valid_members) == 0 or len(community.tasks) == 0:
        return -10
    
    # Compute the average ability total across all valid members
    avg_ability_total = 0
    for member in valid_members:
        avg_ability_total += sum(member.abilities)
    avg_ability_total /= len(valid_members)

    # Compute the average task difficulty total across all remaining tasks
    avg_task_total = 0
    for task in community.tasks:
        avg_task_total += sum(task)
    avg_task_total /= len(community.tasks)

    # Find the pain threshold members should be willing to bear
    avg_energy_expended = max(avg_task_total - avg_ability_total, 0)
    pain_threshold = min(10 - (avg_energy_expended / 2), 0)

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

        dynamic_threshold = 1.5 # CHANGE TO MAKE DYNAMIC

        # Only bid for partnership if its more energy efficient than solo work
        if solo_energy_cost >= dynamic_threshold * best_pair_energy_cost:
            partner_choices.append([task_id, best_partner_id])

    return partner_choices

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

    # Handle extremely difficult tasks
    diff_tasks = findDifficultTasks(community)
    if len(diff_tasks) > 0:
        for task_id, _ in diff_tasks:
            # attempt to execute a sacrifice for these tasks
            sacrificer_id = executeSacrifice(community, community.tasks[task_id], len(player.abilities))
            if sacrificer_id is not None:
                print(f"Sacrificed player {sacrificer_id} to complete task {task_id}.")
                bids.append(task_id)

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
        individual_fail = all(
            sum([max(task[j] - player.abilities[j], 0) for j in range(num_abilities)]) > player.energy
            for player in community.members
        )

        # Check if any partnership can complete the task
        partnership_fail = True
        for i, player1 in enumerate(community.members):
            for j, player2 in enumerate(community.members):
                if i >= j:
                    continue
                
                joint_abilities = [max(player1.abilities[k], player2.abilities[k]) for k in range(num_abilities)]
                energy_cost = sum([max(task[l] - joint_abilities[l], 0) for l in range(num_abilities)]) / 2

                # Check if both players can handle the energy cost
                if player1.energy - energy_cost >= -10 and player2.energy - energy_cost >= -10:
                    partnership_fail = False
                    break
            if not partnership_fail:
                break

        # ff neither individuals nor partnerships can complete the task, it's "difficult"
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
    - identifies the player with the lowest energy who can complete the task
    - ensures no other valid partnerships or individual efforts can resolve the task
    - deducts the energy cost and marks the sacrificer as incapacitated
    """
    # check if the task requires a sacrifice (this might be redundant since this method is only called on difficult tasks anyway)
    individual_fail = all(
        sum([max(task[j] - player.abilities[j], 0) for j in range(num_abilities)]) > player.energy
        for player in community.members
    )

    partnership_fail = True
    for i, player1 in enumerate(community.members):
        for j, player2 in enumerate(community.members):
            if i >= j:
                continue
            
            joint_abilities = [max(player1.abilities[k], player2.abilities[k]) for k in range(num_abilities)]
            energy_cost = sum([max(task[l] - joint_abilities[l], 0) for l in range(num_abilities)]) / 2

            if player1.energy - energy_cost >= -10 and player2.energy - energy_cost >= -10:
                partnership_fail = False
                break
        if not partnership_fail:
            break

    # if no valid partnerships or individual efforts, proceed with sacrifice
    if individual_fail and partnership_fail:
        # identify sacrificer (player with the lowest energy)
        sacrificer = None
        lowest_energy = float('inf')

        for player in community.members:
            energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(num_abilities)])
            if energy_cost <= player.energy and player.energy < lowest_energy:
                lowest_energy = player.energy
                sacrificer = player

        # execute the sacrifice
        if sacrificer:
            energy_cost = sum([max(task[j] - sacrificer.abilities[j], 0) for j in range(num_abilities)])
            sacrificer.energy -= energy_cost
            print(f"Player {sacrificer.id} sacrificed for task, remaining energy: {sacrificer.energy}")
            
            if sacrificer.energy <= -10:
                sacrificer.incapacitated = True
                print(f"Player {sacrificer.id} is now incapacitated.")
            
            return sacrificer.id
    # no sacrifice made
    return None  
