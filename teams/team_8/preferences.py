import pulp

def phaseIpreferences(player, community, global_random):
    """
    Phase I Preferences: Determine task and partner preferences for the player.
    Returns a list of [task_index, partner_id] pairs, indicating tasks the player
    is willing to collaborate on with specific partners.
    """

    # If the player is too tired, pass
    if player.energy < 0:
        return []

    preferences = []

    # Sort tasks by descending difficulty (sum of task difficulty values)
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)

    for task_index, task in sorted_tasks:
        # solo energy cost
        W_solo = sum(max(task[i] - player.abilities[i], 0) for i in range(len(task)))

        # Evaluate potential teamwork options
        best_partner = None
        min_W_team = float('inf')  # Initialize with a very high value

        for partner in community.members:
            # Skip invalid partners (the player themselves or exhausted partners)
            if partner.id == player.id or partner.energy < 0:
                continue

            # Calculate the teamwork energy cost with this partner
            combined_abilities = [
                max(player.abilities[i], partner.abilities[i]) for i in range(len(task))
            ]
            W_team = sum(max(task[i] - combined_abilities[i], 0) for i in range(len(task))) / 2

            # Update best partner if teamwork is more efficient
            if W_team < min_W_team and player.energy >= W_team and partner.energy >= W_team:
                min_W_team = W_team
                best_partner = partner.id

        # Choose the team work option if:
        # The solo energy cost is at least 1.5 times the teamwork energy cost,
        # which means solo work is prohibitively expensive, meaing teamwork is more favorable.
        if (
            W_solo >= 1.5 * min_W_team
            ):
            preferences.append([task_index, best_partner])
        else:
            return []
        
    #print("Player ", player.id, "'s preference is: ", preferences)
    return preferences



def phaseIIpreferences(player, community, global_random):
    """
    Phase II Preferences: Determine task preferences for solo work.
    Returns a list of task indices the player is willing to complete individually.
    """

    # If the player is too tired, pass
    if player.energy < 0:
        return [] 

    bids = []

    # Sort tasks by descending difficulty (sum of task difficulty values); greedy approach
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)

    for task_index, task in sorted_tasks:
        # Calculate the solo energy cost for the player to complete this task
        W_solo = sum(max(task[i] - player.abilities[i], 0) for i in range(len(task)))

        if W_solo <= player.energy:
            bids.append(task_index)
            #break

    return bids