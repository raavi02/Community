import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[
        logging.FileHandler("log-results/team4_log.log", mode='w'),
        logging.StreamHandler()
    ]
    )

def phaseIpreferences(player, community, global_random):
    '''Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id'''
    list_choices = []

    num_members = len(community.members)

    personal_threshold = 0 # vary this in later stages

    for i, task in enumerate(community.tasks):
        for partner_id in range(num_members):
            if partner_id == player.id:
                continue
            partner = community.members[partner_id]
            energy_cost = sum([max(task[j] - max(player.abilities[j], partner.abilities[j]), 0) for j in range(len(player.abilities))])

            if (energy_cost / 2) <= player.energy - personal_threshold:
                list_choices.append([i, partner_id])

    return list_choices

# def phaseIIpreferences(player, community, global_random):
#     '''Return a list of tasks for the particular player to do individually'''
#     bids = []
#     if player.energy < 0:
#         return bids
#     num_abilities = len(player.abilities)
#     for i, task in enumerate(community.tasks):
#         energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(num_abilities)])
#         if energy_cost >= 10:
#             continue
#         bids.append(i)
#     return bids


def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually'''
    # logging.debug(f"Player energy: {player.energy}")
    bids = []
    pain = 2
    if player.energy < pain:
        return bids
    
    num_abilities = len(player.abilities)

    for i, task in enumerate(community.tasks):
        energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(num_abilities)])
        # logging.debug(f"Energy cost: {energy_cost}")
        if energy_cost <= pain:
            bids.append(i)
            # logging.debug(f"Task {i} is being bid on")
            # logging.debug(f"Player energy: {player.energy}")
            # logging.debug(f"Energy cost: " + str(energy_cost))

    return bids
