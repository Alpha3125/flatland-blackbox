import heapq
import itertools
from collections import deque

import networkx as nx


def shortest_path_with_constraints(G, start, goal, constraints):
    """
    Find a shortest path for one agent using BFS in space-time.

    Vertex constraint: (cell, time)
    Edge constraint: (from_cell, to_cell, arrival_time)

    An agent may wait in its current cell for one timestep.
    """
    vertex_constraints = set()
    edge_constraints = set()
    latest_constraint_time = 0

    for constraint in constraints:
        if constraint["type"] == "vertex":
            vertex_constraints.add(
                (constraint["cell"], constraint["time"])
            )
            latest_constraint_time = max(
                latest_constraint_time, constraint["time"]
            )
        elif constraint["type"] == "edge":
            edge_constraints.add(
                (
                    constraint["from"],
                    constraint["to"],
                    constraint["time"],
                )
            )
            latest_constraint_time = max(
                latest_constraint_time, constraint["time"]
            )

    if (start, 0) in vertex_constraints:
        return None

    # After all constraints expire, a shortest simple route needs
    # at most |V| additional moves.
    horizon = latest_constraint_time + len(G) + 1

    queue = deque([(start, 0)])
    parent = {(start, 0): None}
    goal_state = None

    while queue:
        cell, time = queue.popleft()

        # Paths are assumed to remain at the goal after arrival.
        # Requiring time >= latest_constraint_time ensures that
        # no later constraint invalidates that assumption.
        if cell == goal and time >= latest_constraint_time:
            goal_state = (cell, time)
            break

        if time >= horizon:
            continue

        next_time = time + 1

        # Moving to a neighbor, or waiting in the same cell.
        candidates = list(G.neighbors(cell)) + [cell]

        for next_cell in candidates:
            if (next_cell, next_time) in vertex_constraints:
                continue

            if (cell, next_cell, next_time) in edge_constraints:
                continue

            state = (next_cell, next_time)
            if state not in parent:
                parent[state] = (cell, time)
                queue.append(state)

    if goal_state is None:
        return None

    path = []
    state = goal_state

    while state is not None:
        path.append(state[0])
        state = parent[state]

    path.reverse()
    return path


def find_first_conflict(paths):
    """Return the first vertex or edge conflict, if one exists."""
    if not paths:
        return None

    max_time = max(len(path) for path in paths)

    def position(path, t):
        return path[min(t, len(path) - 1)]

    for t in range(max_time):
        # Vertex conflicts.
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                pi = position(paths[i], t)
                pj = position(paths[j], t)

                if pi == pj:
                    return {
                        "type": "vertex",
                        "time": t,
                        "agents": (i, j),
                        "cell": pi,
                    }

        # Edge conflicts.
        if t == 0:
            continue

        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                i_prev = position(paths[i], t - 1)
                i_curr = position(paths[i], t)
                j_prev = position(paths[j], t - 1)
                j_curr = position(paths[j], t)

                if (
                    i_prev == j_curr
                    and j_prev == i_curr
                    and i_prev != i_curr
                ):
                    return {
                        "type": "edge",
                        "time": t,
                        "agents": (i, j),
                        "from": i_prev,
                        "to": i_curr,
                    }

    return None


def cbs(G, agents, max_high_level_nodes=10000):
    """
    Conflict-Based Search for agents represented as dictionaries:
    {"id": integer, "start": (row, col), "goal": (row, col)}

    Returns a list of paths, one per agent, or None if no solution
    is found within the search limit.
    """
    if not agents:
        return []

    # Each high-level node contains a set of constraints per agent
    # and the individually optimal paths satisfying those constraints.
    root_constraints = [[] for _ in agents]
    root_paths = []

    for agent, constraints in zip(agents, root_constraints):
        path = shortest_path_with_constraints(
            G, agent["start"], agent["goal"], constraints
        )
        if path is None:
            return None
        root_paths.append(path)

    counter = itertools.count()

    def cost(paths):
        # Sum of individual path lengths, measured in moves.
        return sum(len(path) - 1 for path in paths)

    open_list = []
    heapq.heappush(
        open_list,
        (
            cost(root_paths),
            next(counter),
            root_constraints,
            root_paths,
        ),
    )

    expanded = 0

    while open_list and expanded < max_high_level_nodes:
        _, _, constraints, paths = heapq.heappop(open_list)
        expanded += 1

        conflict = find_first_conflict(paths)

        if conflict is None:
            return paths

        # Resolve the conflict by constraining one of its two agents
        # in each child node.
        for agent_index in conflict["agents"]:
            new_constraints = [list(c) for c in constraints]

            if conflict["type"] == "vertex":
                new_constraints[agent_index].append({
                    "type": "vertex",
                    "cell": conflict["cell"],
                    "time": conflict["time"],
                })
            else:
                # For the second agent, reverse the edge direction.
                if agent_index == conflict["agents"][0]:
                    from_cell = conflict["from"]
                    to_cell = conflict["to"]
                else:
                    from_cell = conflict["to"]
                    to_cell = conflict["from"]

                new_constraints[agent_index].append({
                    "type": "edge",
                    "from": from_cell,
                    "to": to_cell,
                    "time": conflict["time"],
                })

            new_paths = list(paths)
            agent = agents[agent_index]

            replacement = shortest_path_with_constraints(
                G,
                agent["start"],
                agent["goal"],
                new_constraints[agent_index],
            )

            if replacement is None:
                continue

            new_paths[agent_index] = replacement

            heapq.heappush(
                open_list,
                (
                    cost(new_paths),
                    next(counter),
                    new_constraints,
                    new_paths,
                ),
            )

    return None
