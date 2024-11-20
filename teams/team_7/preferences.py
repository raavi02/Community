# Partnership Round
def phaseIpreferences(player, community, global_random):
    # Too tired to be a good teammate
    if player.energy < 0:
        return []

    # Returns a list of [task_index, partner_id]
    preferences = []
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)

    # Iterate over all tasks
    for task_index, task in sorted_tasks:
        # Get best partner for the task
        preferences = get_best_partner(preferences, task, task_index, player, community)

    return preferences

# Individual Round
def phaseIIpreferences(player, community, global_random):

    bids = []
    sorted_tasks = sorted(enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True)
    bids = get_all_possible_tasks(bids, sorted_tasks, player)

    return bids


def get_all_possible_tasks(bids, sorted_tasks, player):
    for task_index, task in sorted_tasks:
        # Volunteer for all tasks that will keep your energy above 0.
        energy_cost = sum(max(task[i] - player.abilities[i], 0) for i in range(len(task)))
        if energy_cost <= player.energy:
            bids.append(task_index)

    return bids


def get_best_partner(preferences, task, task_index, player, community):
    # Find a partner with complementary skills and sufficient energy
    best_partner = None
    min_energy_cost = float('inf')

    for partner in community.members:
        # Skip invalid partners
        if partner.id == player.id or partner.energy < 0:
            continue

        # Compute combined abilities
        combined_abilities = [
            max(player.abilities[i], partner.abilities[i]) for i in range(len(task))
        ]

        # Compute energy cost per partner for the task
        energy_cost = sum(
            max(task[i] - combined_abilities[i], 0) for i in range(len(task))
        ) / 2

        # Finding best partner for the task.
        if (
                energy_cost < min_energy_cost
                and energy_cost <= player.energy
                and energy_cost <= partner.energy
        ):
            min_energy_cost = energy_cost
            best_partner = partner.id

    if best_partner is not None:
        preferences.append([task_index, best_partner])

    return preferences
