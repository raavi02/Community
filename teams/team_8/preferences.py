def phaseIpreferences(player, community, global_random):
    """
    Phase I Preferences: Determine task and partner preferences for the player.
    Returns a list of [task_index, partner_id] pairs.
    Also stores the list of available players after Phase I in player.available_players.
    """

    # If the player is incapacitated or has zero or negative energy, pass
    if player.incapacitated or player.energy <= 0:
        player.available_players = []
        return []

    # All players have the same data and code, so we can simulate the optimal assignment
    # Initialize active players (not incapacitated and with energy > 0)
    active_players = [
        p for p in community.members if not p.incapacitated and p.energy > 0
    ]

    # **Sort active players from least to most useful**
    active_players.sort(key=lambda p: sum(p.abilities))

    # Sort tasks from hardest to easiest
    sorted_tasks = sorted(
        enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True
    )

    # Keep track of assigned players
    assigned_players = set()
    paired_players = set()

    # Assignment results: task_index -> (single_player_id) or (player1_id, player2_id)
    task_assignments = {}

    # Simulate the assignment
    for task_index, task in sorted_tasks:
        min_energy_cost_single = float("inf")
        min_energy_cost_pair = float("inf")
        best_single = None
        best_pair = None

        # **Find the best single player for the task, prioritizing less useful players**
        for p in active_players:
            if p.id in assigned_players:
                continue

            # Calculate solo energy cost
            energy_cost = sum(
                max(task[i] - p.abilities[i], 0) for i in range(len(task))
            )

            # Check if the player has enough energy
            if p.energy - energy_cost >= 0:
                # **In case of tie in energy cost, prioritize less useful player**
                if energy_cost < min_energy_cost_single or (
                    energy_cost == min_energy_cost_single
                    and sum(p.abilities) < sum(best_single.abilities)
                ):
                    min_energy_cost_single = energy_cost
                    best_single = p

        # **Find the best pair for the task, prioritizing pairs with less total usefulness**
        for i, p1 in enumerate(active_players):
            if p1.id in assigned_players:
                continue
            for p2 in active_players[i + 1 :]:
                if p2.id in assigned_players or p1.id == p2.id:
                    continue

                # Calculate pair energy cost
                combined_abilities = [
                    max(p1.abilities[j], p2.abilities[j]) for j in range(len(task))
                ]
                energy_cost = (
                    sum(
                        max(task[j] - combined_abilities[j], 0)
                        for j in range(len(task))
                    )
                    / 2
                )

                # Check if both players have enough energy
                if p1.energy - energy_cost >= 0 and p2.energy - energy_cost >= 0:
                    total_usefulness = sum(p1.abilities) + sum(p2.abilities)
                    # **In case of tie in energy cost, prioritize pair with less total usefulness**
                    if energy_cost < min_energy_cost_pair or (
                        energy_cost == min_energy_cost_pair
                        and total_usefulness
                        < sum(best_pair[0].abilities) + sum(best_pair[1].abilities)
                    ):
                        min_energy_cost_pair = energy_cost
                        best_pair = (p1, p2)

        # Decide whether to assign to single or pair
        if best_single and (
            min_energy_cost_single <= 1.5 * min_energy_cost_pair or not best_pair
        ):
            # Assign to single
            task_assignments[task_index] = (best_single,)
            assigned_players.add(best_single.id)
        elif best_pair:
            # Assign to pair
            task_assignments[task_index] = (best_pair[0], best_pair[1])
            assigned_players.add(best_pair[0].id)
            assigned_players.add(best_pair[1].id)
            paired_players.add(best_pair[0].id)
            paired_players.add(best_pair[1].id)
        else:
            # No valid assignment for this task
            continue

    # Generate preferences for this player based on the assignments
    preferences = []

    for task_index, assignees in task_assignments.items():
        if player.id in [p.id for p in assignees]:
            if len(assignees) == 2:
                # It's a partnership
                partner_id = [p.id for p in assignees if p.id != player.id][0]
                preferences.append([task_index, partner_id])
            else:
                # Single assignment, will be handled in Phase II
                pass  # Do nothing here

    # Store the available players for Phase II
    player.available_players = [p for p in active_players if p.id not in paired_players]
    return preferences


def phaseIIpreferences(player, community, global_random):
    """
    Phase II Preferences: Determine solo task preferences for the player.
    Returns a list of task indices.
    """
    if player.incapacitated:
        return []

    # Retrieve the list of available players from Phase I
    active_players = getattr(player, "available_players", [])
    if not active_players:
        active_players = [p for p in community.members if not p.incapacitated]

    # Sort active players from least to most useful (sum of abilities)
    active_players.sort(key=lambda p: sum(p.abilities))

    # Tasks not assigned in Phase I
    tasks_ordered = sorted(
        enumerate(community.tasks), key=lambda x: sum(x[1]), reverse=True
    )

    assigned_tasks = set()
    preferences = []

    # First, assign tasks that players can complete without dropping below zero energy
    for task_index, task in tasks_ordered:
        min_energy_cost = float("inf")
        best_player = None

        for p in active_players:
            if p.id in assigned_tasks:
                continue

            energy_needed = sum(
                max(task[i] - p.abilities[i], 0) for i in range(len(task))
            )

            if p.energy - energy_needed >= 0:
                if energy_needed < min_energy_cost or (
                    energy_needed == min_energy_cost
                    and sum(p.abilities) < sum(best_player.abilities)
                ):
                    min_energy_cost = energy_needed
                    best_player = p

        if best_player:
            assigned_tasks.add(task_index)
            if best_player.id == player.id:
                preferences.append(task_index)
            # Remove assigned player from active players
            active_players = [p for p in active_players if p.id != best_player.id]

    # Identify impossible tasks (no individual can complete without dropping below -10 energy)
    impossible_tasks = identify_impossible_tasks(community)

    # If there are impossible tasks, sacrifice the weakest player
    if impossible_tasks:
        if is_weakest_player(player, community):
            for task_index in impossible_tasks:
                if task_index in assigned_tasks:
                    continue

                task = community.tasks[task_index]
                energy_needed = sum(
                    max(task[i] - player.abilities[i], 0) for i in range(len(task))
                )

                # Allow energy to drop below -10 for impossible tasks
                # Assuming the game allows energy to drop to some minimum value (e.g., -20)
                if player.energy - energy_needed >= -20:
                    preferences.append(task_index)
                    assigned_tasks.add(task_index)
                    break  # Assign only one impossible task

    return preferences


def identify_impossible_tasks(community):
    """
    Identifies tasks that cannot be completed by any individual without their energy dropping below -10.
    Returns a list of indices of impossible tasks.
    """
    impossible_tasks = []

    for task_index, task in enumerate(community.tasks):
        task_possible = False

        # Check if any individual can perform the task without dropping below -10 energy
        for p in community.members:
            if p.incapacitated:
                continue

            energy_needed = sum(
                max(task[i] - p.abilities[i], 0) for i in range(len(task))
            )

            if p.energy - energy_needed >= -10:
                task_possible = True
                break  # Task is possible for an individual

        if not task_possible:
            impossible_tasks.append(task_index)

    return impossible_tasks


def is_weakest_player(player, community):
    """
    Determines if the player is the weakest based on the sum of their abilities.
    Returns True if the player is the weakest, False otherwise.
    """
    total_abilities = [
        sum(p.abilities) for p in community.members if not p.incapacitated
    ]
    player_total = sum(player.abilities)
    return player_total == min(total_abilities)
