def phaseIpreferences(player, community, global_random):
    '''Return a list of task index and the partner id for the particular player.'''
    preferences = []
    if player.energy <= 0:  # Skip if the player has no energy
        return preferences

    # Sort tasks by total difficulty (descending)
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)

    for task_id, task in sorted_tasks:
        best_partner = None
        best_remaining_energy = -float('inf')  # Track the best post-task energy state
        for partner in community.members:
            if partner.id == player.id or partner.incapacitated or partner.energy <= 0:
                continue  # Skip self, incapacitated, or exhausted players

            # Calculate energy cost for both players
            energy_cost = sum(max(task[i] - max(player.abilities[i], partner.abilities[i]), 0) for i in range(len(task))) / 2
            player_remaining_energy = player.energy - energy_cost
            partner_remaining_energy = partner.energy - energy_cost

            # Skip pairings that result in incapacitation
            if player_remaining_energy <= 0 or partner_remaining_energy <= 0:
                continue

            # Choose the partner that maximizes the minimum remaining energy
            if min(player_remaining_energy, partner_remaining_energy) > best_remaining_energy:
                best_partner = partner.id
                best_remaining_energy = min(player_remaining_energy, partner_remaining_energy)

        # Add the task and partner to preferences if a valid partner is found
        if best_partner is not None:
            preferences.append([task_id, best_partner])

    return preferences

def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually.'''
    preferences = []
    if player.energy <= 0:  # Skip if the player has no energy
        return preferences

    # Evaluate tasks for individual completion
    for task_id, task in enumerate(community.tasks):
        energy_cost = sum(max(task[i] - player.abilities[i], 0) for i in range(len(task)))
        remaining_energy = player.energy - energy_cost

        # Only consider tasks that the player can complete without incapacitation
        if remaining_energy > 0:
            preferences.append((task_id, energy_cost, remaining_energy))

    # Sort tasks by a combination of low energy cost and high remaining energy
    preferences.sort(key=lambda x: (x[1], -x[2]))  # Sort by energy cost, then remaining energy

    # Return task IDs in preferred order
    return [task_id for task_id, _, _ in preferences]