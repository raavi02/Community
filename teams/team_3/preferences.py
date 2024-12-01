from community import Member, Community


def player_score(community: Community) -> list[int]:
    """
    Calculate the score of every player sort from best to worst
    """
    members = {member.id: 0 for member in community.members}

    for member in community.members:
        if member.energy <= -10:
            continue
        for task in community.tasks:
            cost = sum(
                max(task[i] - member.abilities[i], 0)
                for i in range(len(member.abilities))
            )
            members[member.id] += cost

    return sorted(members, key=members.get)


def sacrifice(player: Member, community: Community):
    """
    Sacrifice the weakest community member, only when
    all tasks are impossible and when no other task can be done
    """
    members = player_score(community)

    if player.id != members[-1] or len(community.tasks) == 0:
        return []

    impossible_tasks = []

    # NOTE: Sacrificing is a last resort and has negative impact on community performance
    #       Thus, only consider sacrificying a player if a task cannot be completed by any
    #       pair with full energy.
    full_energy = 10

    for idx, task in enumerate(community.tasks):
        for p1 in community.members:
            for p2 in community.members:
                if p1 == p2:
                    continue

                energy_cost = (
                    sum(
                        max(task[i] - max(p1.abilities[i], p2.abilities[i]), 0)
                        for i in range(len(player.abilities))
                    )
                    * 0.5
                )

                if full_energy - energy_cost > -10:
                    return []

        impossible_tasks.append(idx)

    return [impossible_tasks[0]]


def calculate_minimum_delta_individual(player: Member, community: Community):
    """
    Find the best task for an individual, the task that minimizes waste + energy

    delta = min(energy + waste)
    """
    num_abilities = len(player.abilities)

    minimum_delta = float("inf")
    best_task = -1

    for task_index, task in enumerate(community.tasks):
        cost = sum(max(task[i] - player.abilities[i], 0) for i in range(num_abilities))
        waste = (
            sum(max(player.abilities[i] - task[i], 0) for i in range(num_abilities))
            * 0.5
        )

        if minimum_delta >= cost + waste:
            minimum_delta = cost + waste
            best_task = task_index

    return (best_task, minimum_delta)


# Delta = min(energy + waste)
def calculate_minimum_delta_pair(
    player: Member, best_partner: int, community: Community
):
    """
    Find the best task for a pair, the task that minimizes waste + energy

    delta = min(energy + waste)
    """
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
            * 0.5
        )
        pair_waste = (
            sum(
                max(max(player.abilities[i], partner.abilities[i]) - task[i], 0)
                for i in range(num_abilities)
            )
            * 0.5
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

    # Members are stored from best to worst (best player index: 0)
    members = player_score(community)
    index = members.index(player.id)

    best_partner = members[len(members) - 1 - index]
    best_task, pair_delta = calculate_minimum_delta_pair(
        player, best_partner, community
    )

    # Cost of performing the task alone
    player_cost = sum(
        max(community.tasks[best_task][i] - player.abilities[i], 0)
        for i in range(len(player.abilities))
    )
    # How much the player is wasting his abilities
    # Player has ability 8, the task costs 5. waste = 3
    player_waste = sum(
        max(player.abilities[i] - community.tasks[best_task][i], 0)
        for i in range(len(player.abilities))
    )

    player_delta = player_cost + player_waste

    # Energy management (players don't kill themselves)
    if pair_delta >= player.energy + community.members[best_partner].energy:
        return list_choices

    # Only pair up if it is more benificial
    if pair_delta < player_delta * 1.5:
        list_choices.append([best_task, best_partner])

    return list_choices


def phaseIIpreferences(player: Member, community: Community, global_random):
    """Return a list of tasks for the particular player to do individually"""

    if impossible := sacrifice(player, community):
        return impossible

    bids = []
    num_abilities = len(player.abilities)
    best_task, _ = calculate_minimum_delta_individual(player, community)

    if best_task == -1:
        return []

    energy_cost = sum(
        max(community.tasks[best_task][j] - player.abilities[j], 0)
        for j in range(num_abilities)
    )

    if energy_cost <= player.energy:
        bids.append(best_task)

    return bids
