
def detect_conflicts(paths):
    """
    Detect vertex and edge conflicts in multi-agent paths.

    Each path is a list of (row, column) coordinates.
    Agents remain at their final cell after reaching their goals.
    """
    if not paths:
        return []

    max_time = max(len(path) for path in paths) - 1
    conflicts = []

    def position(path, t):
        return path[min(t, len(path) - 1)]

    for t in range(max_time + 1):
        # Vertex conflicts: two agents occupy the same cell.
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                pi = position(paths[i], t)
                pj = position(paths[j], t)

                if pi == pj:
                    conflicts.append({
                        "type": "vertex",
                        "time": t,
                        "agents": (i, j),
                        "cell": pi,
                    })

        # Edge conflicts: two agents swap cells in one timestep.
        if t == 0:
            continue

        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                i_prev = position(paths[i], t - 1)
                i_curr = position(paths[i], t)
                j_prev = position(paths[j], t - 1)
                j_curr = position(paths[j], t)

                if i_prev == j_curr and j_prev == i_curr:
                    # Ignore the degenerate case where both agents
                    # are stationary at the same cell; vertex conflict
                    # detection already handles that case.
                    if i_prev != i_curr:
                        conflicts.append({
                            "type": "edge",
                            "time": t,
                            "agents": (i, j),
                            "edge": (i_prev, i_curr),
                        })

    return conflicts
