import itertools

global strong_players 
global remaining

strong_players = set()

def phaseIpreferences(player, community, global_random):
    global strong_players
    '''Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id'''

    all_players = set(range(len(community.members)))
    
    for p in community.members:
        # assume the player can do all tasks perfectly until proven otherwise
        is_strong_player = True
        for t in community.tasks:
            # check if the player can perform the task perfectly
            if not all(p.abilities[i] >= t[i] for i in range(len(t))):
                is_strong_player = False
                break
        if is_strong_player:
            strong_players.add(p.id)

    # Form all possible partnerships from the remaining players
    remaining_players = all_players - strong_players
    partnerships = list(itertools.combinations(remaining_players, 2))
    partner_choices = []

    # Find good partnerships for the particular player
    for p1_id, p2_id in partnerships:
        if player.id in (p1_id, p2_id):
            p1 = community.members[p1_id]
            p2 = community.members[p2_id]
            joint_abilities = [max(a1, a2) for a1, a2 in zip(p1.abilities, p2.abilities)]
            
            for task_id, task in enumerate(community.tasks):
                energy_cost = calculate_energy_cost(joint_abilities, task)
                if energy_cost <= 0:
                    partner_choices.append([task_id, p2_id] if player.id == p1_id else [task_id, p1_id])
    
    return partner_choices

def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually'''
    global strong_players 
    bids = []

    # If a player is strong, bid for all tasks as they can perform all
    if player.id in strong_players:
        for task_id, task in enumerate(community.tasks):
            bids.append(task_id)
        return bids
    
    num_abilities = len(player.abilities)

    for task_id, task in enumerate(community.tasks):
        energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(num_abilities)])
        # volunteering logic
        if player.energy - energy_cost  >= -10:
            bids.append(task_id)
    
    return bids

def calculate_energy_cost(abilities, task):
    return sum(max(0, task[i] - abilities[i]) for i in range(len(task)))
