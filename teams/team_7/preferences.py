def phaseIpreferences(player, community, global_random):
    # Returns a list of task indices and partner IDs
    preferences = []
    if player.energy < 0:
        return preferences
    
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)

    for task_index, task in sorted_tasks:
        # Find a partner with complementary skills and sufficient energy
        best_partner = None
        min_energy_cost = float('inf')
        for partner in community.members:
            if partner.id == player.id or partner.energy < 0:
                continue
            combined_abilities = [max(player.abilities[i], partner.abilities[i]) for i in range(len(task))]
            energy_cost = sum(max(task[i] - combined_abilities[i], 0) for i in range(len(task))) / 2
            if energy_cost < min_energy_cost and energy_cost <= player.energy and energy_cost <= partner.energy:
                min_energy_cost = energy_cost
                best_partner = partner.id
        
        if best_partner is not None:
            preferences.append([task_index, best_partner])

    return preferences

def phaseIIpreferences(player, community, global_random):
    # Returns a list of task indices for the player to do individually 
    bids = []
    if player.energy < 2: # Set a threshold
        return bids
    
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)

    for task_index, task in sorted_tasks:
        energy_cost = sum(max(task[i] - player.abilities[i], 0) for i in range(len(task)))
        if energy_cost <= player.energy:
            bids.append(task_index)

    return bids

