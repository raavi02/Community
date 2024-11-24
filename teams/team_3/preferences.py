import sys
from community import Member, Community


def community_statistics(community: Community):
    num_members = len(community.members)
    num_abilities = len(community.members[0].abilities)

    total_energy = 0
    avg_abilties = [0] * num_abilities

    for member in community.members:
        total_energy += member.energy
        for i, ability in enumerate(member.abilities):
            avg_abilties[i] += ability

    avg_abilties = [ab / num_members for ab in avg_abilties]
    avg_energy = total_energy / num_members
    return avg_abilties, avg_energy


def player_score(community: Community) -> dict[int:int]:
    members = {member.id: 0 for member in community.members}

    for member in community.members:
        for task in community.tasks:
            cost = sum(
                max(task[i] - member.abilities[i], 0)
                for i in range(len(member.abilities))
            )
            members[member.id] += cost

    return sorted(members, key=members.get)


def calculate_minimum_delta_individual(player: Member, community: Community):
    num_abilities = len(player.abilities)

    minimum_delta = float("inf")
    best_task = -1

    for task_index, task in enumerate(community.tasks):
        cost = sum(max(task[i] - player.abilities[i], 0) for i in range(num_abilities))
        waste = sum(max(player.abilities[i] - task[i], 0) for i in range(num_abilities))

        if minimum_delta >= cost + waste:
            minimum_delta = cost + waste
            best_task = task_index

    return (best_task, minimum_delta)


# Delta = min(energy + waste)
def calculate_minimum_delta_pair(
    player: Member, best_partner: int, community: Community
):
    members = {member.id: member for member in community.members}
    partner = members[best_partner]
    num_abilities = len(player.abilities)

    minimum_delta = float("inf")
    best_task = 0

    for task_index, task in enumerate(community.tasks):
        pair_cost = (
            sum(
                max(task[i] - max(player.abilities[i], partner.abilities[i]), 0)
                for i in range(num_abilities)
            )
            / 2
        )
        pair_waste = sum(
            max(max(player.abilities[i], partner.abilities[i]) - task[i], 0)
            for i in range(num_abilities)
        )

        if minimum_delta >= pair_cost + pair_waste:
            minimum_delta = pair_cost + pair_waste
            best_task = task_index

    return (best_task, minimum_delta)


def phaseIpreferences(player: Member, community: Community, global_random):
    """Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id
    """
    list_choices = []

    if player.energy < 2:
        return list_choices

    members = player_score(community)
    index = 0

    for i, mem in enumerate(members):
        if mem == player.id:
            index = i
            break

    best_partner = members[len(members) - 1 - index]
    best_task, pair_delta = calculate_minimum_delta_pair(
        player, best_partner, community
    )

    player_cost = sum(
        max(community.tasks[best_task][i] - player.abilities[i], 0)
        for i in range(len(player.abilities))
    )
    player_waste = sum(
        max(player.abilities[i] - community.tasks[best_task][i], 0)
        for i in range(len(player.abilities))
    )

    player_delta = player_cost + player_waste

    list_choices.append([best_task, best_partner])
    # if pair_delta < player_delta:
    #     list_choices.append([best_task, best_partner])

    return list_choices


def phaseIIpreferences(player, community, global_random):
    """Return a list of tasks for the particular player to do individually"""
    bids = []
    num_abilities = len(player.abilities)
    best_task, _ = calculate_minimum_delta_individual(player, community)

    if best_task == -1 or player.energy < 2:
        return []

    return [best_task]

    # if delta < 3:
    #     return [best_task]

    return []
    # evaluate each task for solo completion
    for i, task in enumerate(community.tasks):

        energy_cost = sum(
            max(task[j] - player.abilities[j], 0) for j in range(num_abilities)
        )

        if energy_cost < player.energy:
            bids.append((i, energy_cost))

    bids.sort(key=lambda x: x[1])
    bids = [bid[0] for bid in bids]

    return bids
