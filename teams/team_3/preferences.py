def phaseIpreferences(player, community, global_random):
    """Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id
    """
    list_choices = []

    # if energy is low, skip partnering
    if player.energy < 2:  # TODO improve this logic
        return list_choices

    num_abilities = len(player.abilities)

    # iterate over all tasks
    for task_index, task in enumerate(community.tasks):
        best_partner = None
        min_energy_cost = float("inf")

        # find the best partner to minimize energy cost
        for partner in community.members:
            if partner.id == player.id or partner.energy < 2:
                continue

            # calculate potential energy cost with this partner
            energy_cost = (
                sum(
                    max(task[i] - max(player.abilities[i], partner.abilities[i]), 0)
                    for i in range(num_abilities)
                )
                / 2
            )  # TODO figure out better potential energy function

            if energy_cost < min_energy_cost:
                min_energy_cost = energy_cost
                best_partner = partner.id

        # add this task-partner combo if it's beneficial
        if best_partner is not None and min_energy_cost < player.energy:
            list_choices.append([task_index, best_partner])

    return list_choices


def phaseIIpreferences(player, community, global_random):
    """Return a list of tasks for the particular player to do individually"""
    bids = []

    # if energy is low, skip volunteering
    if player.energy < 3:
        return bids

    num_abilities = len(player.abilities)

    # evaluate each task for solo completion
    for i, task in enumerate(community.tasks):
        energy_cost = sum(
            max(task[j] - player.abilities[j], 0) for j in range(num_abilities)
        )

        if energy_cost < 4:
            bids.append(i)

    return bids
