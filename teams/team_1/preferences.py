def getPainThreshold(community):
    '''Computes the pain threshold according to the average task difficulty and median member abilities'''
    
    # Only consider members who are not tired
    valid_members = [member for member in community.members if member.energy >= 0]

    if len(valid_members) == 0 or len(community.tasks) == 0:
        return -9
    
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
        
        # If the average energy of the community is greater than 5, push players to work solo
        avg_energy = getAvgEnergy(community)
        dynamic_threshold = 2 if avg_energy >= 5 else 1.2

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
    bids = []

    # Handle impossible tasks with sacrifice
    impossible_tasks = findImpossibleTasks(community)
    if impossible_tasks:
        sacrificee_id = getWeakestMember(community)
        for task_id in impossible_tasks:
            # Execute a sacrifice if member is the weakest
            if player.id == sacrificee_id:
                bids.append(task_id)
        return bids
    
    # Don't volunteer if tired
    if player.energy < 0:
        return bids

    for task_id, task in enumerate(community.tasks):
        energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(len(task))])
        
        # Volunteering logic
        if player.energy - energy_cost >= 0:
            bids.append(task_id)
    
    return bids

def findImpossibleTasks(community):
    """
    Identifies difficult tasks that cannot be completed by any individual player
    or through partnerships without causing energy exhaustion.
    """
    impossible_tasks = []
    num_abilities = len(community.members[0].abilities)

    for task_id, task in enumerate(community.tasks):
        # Check whether any individual player can complete the task without incapacitation on full energy
        individual_fail = True
        for player in community.members:
            if player.incapacitated:
                continue
            energy_cost = sum(max(task[j] - player.abilities[j], 0) for j in range(num_abilities))
            if 10 - energy_cost > -10:
                individual_fail = False
                break

        # Check whether any partnership can complete the task without incapacitation on full energy
        partnership_fail = True
        for i, player1 in enumerate(community.members):
            if player1.incapacitated:
                continue
            for j, player2 in enumerate(community.members):
                if i >= j or player2.incapacitated:  # Avoid self-pairing
                    continue

                joint_abilities = [max(player1.abilities[k], player2.abilities[k]) for k in range(num_abilities)]
                energy_cost = sum(max(task[l] - joint_abilities[l], 0) for l in range(num_abilities)) / 2

                if 10 - energy_cost > -10:
                    partnership_fail = False
                    break
            
            if not partnership_fail:
                break

        # If neither individuals nor partnerships can complete the task, it's impossible
        if individual_fail and partnership_fail:
            impossible_tasks.append(task_id)

    return impossible_tasks

def getWeakestMember(community):
    """
    Finds the weakest member of the community for sacrifice on an impossible task.
    """
    # Find valid members who are not incapacitated
    valid_members = []
    for member in community.members:
        if not member.incapacitated:
            valid_members.append(member)

    # Sort valid members by (total ability, energy), ascending
    sorted_valid_members = sorted(
        valid_members,
        key=lambda member: (sum(member.abilities), member.energy)
    )

    return sorted_valid_members[0].id