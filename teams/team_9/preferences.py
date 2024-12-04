import math

def phaseIpreferences(player, community, global_random):
    '''Return a list of task index and the partner ids for the particular player.'''
    preferences = []

    # Compute averages to determine dynamic energy limits
    task_avg_difficulty = sum(sum(task) for task in community.tasks) / len(community.tasks)
    player_avg_skill = sum(sum(member.abilities) for member in community.members) / len(community.members)

    primary_energy_limit = 0  # Allow energy to drop but not too far into the negatives

    # Calculate the ratio of difficulty to skill
    difficulty_ratio = task_avg_difficulty / max(player_avg_skill, 1e-6)  # Avoid division by zero

    # Map difficulty ratio to the range [-10, primary_energy_limit]
    if difficulty_ratio <= 1:
        secondary_energy_limit = primary_energy_limit  # Tasks are manageable
    else:
        # The higher the difficulty_ratio, the closer the limit gets to -10
        secondary_energy_limit = primary_energy_limit - (10 * (1 - math.exp(-(difficulty_ratio - 1))))
        secondary_energy_limit = max(secondary_energy_limit, -10)  # Ensure it doesn't drop below -10

    print(difficulty_ratio, secondary_energy_limit)

    # Sort tasks by total difficulty (descending)
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)

    for task_id, task in sorted_tasks:
        potential_partners = []  # To store potential partners and their remaining energies

        # Check if the player can complete the task alone
        energy_needed = sum(max(task_i - ability_i, 0) for task_i, ability_i in zip(task, player.abilities))
        if energy_needed <= 3:
            continue  # Skip tasks that the player can handle alone or with very little energy

        for partner in community.members:
            if partner.id == player.id or partner.incapacitated:
                continue  # Skip self or incapacitated players

            # Skip tasks that the partner can handle alone
            energy_needed = sum(max(task_i - ability_i, 0) for task_i, ability_i in zip(task, partner.abilities))
            if energy_needed <= 3:
                continue

            # Calculate energy cost for both players
            energy_cost = sum(max(task[i] - max(player.abilities[i], partner.abilities[i]), 0) for i in range(len(task))) / 2
            player_remaining_energy = player.energy - energy_cost
            partner_remaining_energy = partner.energy - energy_cost

            # Allow partnering even if energy dips below the primary limit
            if (
                player_remaining_energy > secondary_energy_limit
                and partner_remaining_energy > secondary_energy_limit
            ):
                # Add potential partner and their remaining energy for sorting
                potential_partners.append((partner.id, min(player_remaining_energy, partner_remaining_energy)))

        # Sort potential partners by their effectiveness (descending by energy)
        potential_partners.sort(key=lambda x: -x[1])  # Sort by remaining energy (descending)

        # Append top 3 partners for the task to preferences
        for partner_id, _ in potential_partners[:1]:  # Take at most the top 3 partners
            preferences.append([task_id, partner_id])

    return preferences

def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually.'''
    preferences = []

    task_avg_difficulty = sum(sum(task) for task in community.tasks) / max(len(community.tasks), 1e-6) 
    player_avg_skill = sum(sum(member.abilities) for member in community.members) / max(len(community.members), 1e-6)

    primary_energy_limit = 0  # Allow energy to drop but not too far into the negatives

    # Calculate the ratio of difficulty to skill
    difficulty_ratio = task_avg_difficulty / max(player_avg_skill, 1e-6)  # Avoid division by zero

    # Map difficulty ratio to the range [-10, primary_energy_limit]
    if difficulty_ratio <= 1:
        secondary_energy_limit = primary_energy_limit  # Tasks are manageable
    else:
        # The higher the difficulty_ratio, the closer the limit gets to -10
        # secondary_energy_limit = primary_energy_limit - (difficulty_ratio - 1) * 10
        secondary_energy_limit = primary_energy_limit - (10 * (1 - math.exp(-(difficulty_ratio - 1))))
        secondary_energy_limit = max(secondary_energy_limit, -10)  # Ensure it doesn't drop below -10

    print(difficulty_ratio, secondary_energy_limit)

    # Evaluate tasks for individual completion
    for task_id, task in enumerate(community.tasks):
        energy_cost = sum(max(task[i] - player.abilities[i], 0) for i in range(len(task)))
        remaining_energy = player.energy - energy_cost

        # Consider tasks that leave the player with energy above the secondary limit
        if remaining_energy > secondary_energy_limit:
            preferences.append((task_id, energy_cost, remaining_energy))

    # Sort tasks by a combination of low energy cost and high remaining energy
    preferences.sort(key=lambda x: (x[1], -x[2]))  # Sort by energy cost, then remaining energy

    impossible_tasks = findImpossibleTasks(community)
    sacrificee_ids = getWeakestMembers(community, len(impossible_tasks))

    if impossible_tasks and sacrificee_ids:
        for task_id in impossible_tasks:
            if player.id in sacrificee_ids:
                preferences.append(task_id)

    # Return task IDs in preferred order
    return [task_id for task_id, _, _ in preferences]

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
                continue  # Skip incapacitated players
            energy_cost = sum(max(task[j] - player.abilities[j], 0) for j in range(num_abilities))
            if 10 - energy_cost > -10:
                individual_fail = False
                break

        # Check whether any partnership can complete the task without incapacitation on full energy
        partnership_fail = True
        for i, player1 in enumerate(community.members):
            if player1.incapacitated:
                continue  # Skip incapacitated players
            for j, player2 in enumerate(community.members):
                if i >= j or player2.incapacitated:  # Avoid self-pairing and incapacitated players
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

def getWeakestMembers(community, num_sacrifices):
    """
    Finds the weakest members of the community for sacrifice on impossible tasks.
    Returns a list of member IDs equal to the number of sacrifices needed.
    """
    # Filter out incapacitated members
    active_members = [member for member in community.members if not member.incapacitated]

    if not active_members:
        return []

    # Sort members by (total ability, energy), ascending
    sorted_members = sorted(
        active_members,
        key=lambda member: (sum(member.abilities), member.energy)
    )

    # Return the IDs of the weakest members
    return [member.id for member in sorted_members[:num_sacrifices]]