import traceback
ENERGY_THRESH = 3
NUM_TASK_OPTIONS = 5
NUM_PAIR_OPTIONS = 1

def get_energy_cost(player_abilities, task):
    return sum(max(task[i]- player_abilities[i],0) for i in range(len(task)))

def will_incapcitate(player_energy, cost):
    return player_energy - cost <= -10

def get_zero_energy_partners(player, community):
    preferences  = []
    for id, task in enumerate(community.tasks):
        ind_cost = get_energy_cost(player.abilities, task)
        if ind_cost == 0:
            continue
        for partner in community.members:
            if partner.id == player.id:
                continue
            partner_ind_cost = get_energy_cost(partner.abilities, task)
            if partner_ind_cost == 0:
                continue
            partnerhip_cost = get_energy_cost([max(player.abilities[i], partner.abilities[i]) for i in range(len(player.abilities))], task)
            if partnerhip_cost == 0:
                preferences.append([id, partner.id])
    return preferences
    

def phaseIpreferences(player, community, global_random):
    '''Return a list of task index and the partner id for the particular player.'''
    preferences = []
    try:
        # If low on energy just do partnerships with zero energy
        if player.energy <= 3: 
            return get_zero_energy_partners(player, community)

        # Rest if low energy
        if player.energy <= 0:
            return []

        sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)
        tasks ={}
        for task_id, task in sorted_tasks:
            ind_cost = get_energy_cost(player.abilities, task)
            tasks[task_id] = [{ind_cost: player.id}]
            # If you can do it yourself with no energy spent, do it
            if ind_cost > 0:
                for partner in community.members:
                    if partner.id == player.id or partner.incapacitated:
                        continue
                    partnership_cost = get_energy_cost([max(player.abilities[i], partner.abilities[i]) for i in range(len(player.abilities))], task)
                    partner_ind_cost = get_energy_cost(partner.abilities, task)
                    
                    # If partner can do the task individually with no energy don't pair up
                    if partner_ind_cost == 0:
                        continue
                    
                    # If less tasks left partner up and save energy
                    if len(community.tasks) < len(community.members):
                        # Partner as long as it doesn't incapcitate them or us
                        if not will_incapcitate(partner.energy, (partnership_cost/2)) and not(will_incapcitate(player.energy, (partnership_cost/2))):
                            tasks[task_id].append({partnership_cost/2: (player.id, partner.id)})
                    else:
                        #If doing it with a partner is better than doing it alone then pair up
                        if ind_cost > partnership_cost/2 and ind_cost - (partnership_cost/2) > ENERGY_THRESH:
                            if not will_incapcitate(partner.energy, (partnership_cost/2)) and not(will_incapcitate(player.energy, (partnership_cost/2))):
                                tasks[task_id].append({partnership_cost/2: (player.id, partner.id)})
        
        # Take top n for each task
        best_options = {
            task: sorted([(list(option.keys())[0], list(option.values())[0]) for option in options], key=lambda x: x[0])[:NUM_PAIR_OPTIONS]
            for task, options in tasks.items()
        }
        print("Best options for each task:", best_options)

        # Take top m tasks
        top_tasks = sorted(((task, min(option[0] for option in options)) for task, options in best_options.items()), key=lambda x: x[1])[:NUM_TASK_OPTIONS]
        print("Top tasks", top_tasks)

        preferences.extend(
            [[task_id, next(value for value in players if value != player.id)] 
             for task_id, _cost in top_tasks 
             for cost, players in best_options[task_id] 
             if isinstance(players, tuple)]
        )
        return preferences
    
    except Exception as e:
        print(e)
        traceback.print_exc()

def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually.'''
    preferences = []
    if player.energy <= 0:  # Rest if the player has no energy
        return preferences

    # Evaluate tasks for individual completion
    for task_id, task in enumerate(community.tasks):
        energy_cost = sum(max(task[i] - player.abilities[i], 0) for i in range(len(task)))
        remaining_energy = player.energy - energy_cost

        # Only consider tasks that the player can complete without getting tired
        if remaining_energy > 0:
            preferences.append((task_id, energy_cost, remaining_energy))

    # Sort tasks by a combination of low energy cost and high remaining energy
    preferences.sort(key=lambda x: (x[1], -x[2]))  # Sort by energy cost, then remaining energy

    # Return task IDs in preferred order
    return [task_id for task_id, _, _ in preferences][:NUM_TASK_OPTIONS]