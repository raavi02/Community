def phaseIpreferences(player, community, global_random):
    '''Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id'''

    all_players = set(range(len(community.members)))
    num_abilities = len(player.abilities)

    # Sort tasks in order of descending difficulty
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)

    if not isValidForPartnership(player, community, None):
        return []
    
    # Form partnerships based on the tasks
    partner_choices = []
    
    for task_id, task in sorted_tasks:
        best_partner_id = None
        best_final_energy = -float("inf")
        for partner_id in all_players:
            if partner_id == player.id:
                continue
        
            partner = community.members[partner_id]
            if isValidForPartnership(partner, community, task):
                joint_abilities = [max(a1, a2) for a1, a2 in zip(player.abilities, partner.abilities)]
                energy_cost = sum([max(task[j] - joint_abilities[j], 0) for j in range(num_abilities)]) / 2
                
                player_final_energy = player.energy - energy_cost
                partner_final_energy = partner.energy - energy_cost
                lowest_final_energy = min(player_final_energy, partner_final_energy)

                # Skip pairing if it will tire out either of the two players too much
                if lowest_final_energy < -1:
                    continue

                # Keep track of best partnership to preserve energy
                if lowest_final_energy > best_final_energy:
                    best_final_energy = lowest_final_energy
                    best_partner_id = partner.id
        
        if best_partner_id:
            partner_choices.append([task_id, best_partner_id])

    return partner_choices

def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually'''
    bids = []

    for task_id, task in enumerate(community.tasks):
        energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(len(task))])
        # Volunteering logic
        if player.energy - energy_cost >= 0:
            bids.append(task_id)

    # Find extremely difficult tasks
    diff_tasks = findDifficultTasks(community)
    if len(diff_tasks) > 0:
        print("Difficult Tasks:")
        for task_id, difficulty in diff_tasks:
            pass
            #print(f"Task ID: {task_id}, Difficulty: {difficulty}")
    
    return bids

def findDifficultTasks(community):
    difficult_tasks = []

    for task_id, task in enumerate(community.tasks):
    
        # Calculate the energy cost for the worst-case player (lowest abilities)
        max_energy_loss = max([
            sum([max(task[j] - player.abilities[j], 0) for j in range(len(task))])
            for player in community.members
        ])

        # Check if this task would reduce all players' energy below -20
        if all(player.energy - max_energy_loss < -20 for player in community.members):
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