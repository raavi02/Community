'''def phaseIpreferences(player, community, global_random):
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
            # and player.energy - min_W_team >= 0
            # and partner.energy - min_W_team >= 0
            ):
            preferences.append([task_index, best_partner])

        
    print("Player ", player.id, "'s preference is: ", preferences)
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

    return bids'''


# Considers all partner pairs at once, rather than comparing the best partner for a particular player
# Aims to tackle hardest tasks first
# Attempts to find single best partner pair given lowest energy cost and remaining energy
def create_connections(community):

    connections = {}
    
    partnerships = []
    for i in range(len(community.members)):
        for j in range(i+1, len(community.members)):
            partnerships.append((community.members[i], community.members[j]))

    members = list(community.members)
    member_energy = {}
    for member in members:
        member_energy[member.id] = member.energy
    print(member_energy)
    tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)
    for task in tasks:

        partnerships = []
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                partnerships.append((members[i], members[j]))
        options = []
        energy_left = []

        for member in members:
            cost = [max(0, req - max_val) for max_val, req in zip(member.abilities, task[1])]
            energy_left.append(member.energy - sum(cost))
            options.append(member.id)

        for partner in partnerships:
            abilities = [max(i, j) for i, j in zip(partner[0].abilities, partner[1].abilities)]
            energy = (partner[0].energy + partner[1].energy) / 2
            cost = [max(0, req - max_val) for max_val, req in zip(abilities, task[1])]
            energy_left.append(energy - sum(cost))
            options.append(partner)

        if energy_left:
            #print(options)
            best_idx = energy_left.index(max(energy_left))
            connections[task[0]] = options[best_idx]
            if type(options[best_idx]) == tuple:
                for member in options[best_idx]:
                    members.remove(member)

    return connections



def phaseIpreferences(player, community, global_random):
    """
    Phase I Preferences: Determine task and partner preferences for the player.
    Returns a list of [task_index, partner_id] pairs, indicating tasks the player
    is willing to collaborate on with specific partners.
    """
    preferences = []
    
    connects = create_connections(community)
    for key, value in connects.items():
        if player in value:
            for v in value:
                if player != v:
                    preferences.append([key, v.id])

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


