from pathlib import Path
import random
import networkx as nx

def load_movingai_map(map_path):
    """Load a Moving AI map into a NetworkX graph."""
    with open(map_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    map_index = lines.index("map")

    header = {}
    for line in lines[:map_index]:
        parts = line.split()
        if len(parts) == 2:
            header[parts[0]] = parts[1]

    height = int(header["height"])
    width = int(header["width"])
    grid = lines[map_index + 1:map_index + 1 + height]

    if len(grid) != height or any(len(row) != width for row in grid):
        raise ValueError("Map dimensions do not match header.")

    free_cells = {
        (r, c)
        for r, row in enumerate(grid)
        for c, char in enumerate(row)
        if char in {".", "G"}
    }

    G = nx.Graph()
    G.add_nodes_from(free_cells)

    for r, c in free_cells:
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (r + dr, c + dc)
            if neighbor in free_cells:
                G.add_edge((r, c), neighbor)

    return grid, G


def sample_grid_agents(G, num_agents, seed=0):
    """Generate agents with distinct, reachable start and goal cells."""
    if num_agents < 1:
        raise ValueError("num_agents must be at least 1.")

    # Find connected components of the traversable-cell graph.
    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)

    # Use the largest component so every start can reach every goal.
    largest_component = list(components[0]) if components else []

    if len(largest_component) < 2 * num_agents:
        raise ValueError(
            f"Need at least {2 * num_agents} cells in the largest "
            f"connected component, but found {len(largest_component)}."
        )

    rng = random.Random(seed)
    selected_cells = rng.sample(largest_component, 2 * num_agents)

    agents = []
    for i in range(num_agents):
        agents.append({
            "id": i,
            "start": selected_cells[i],
            "goal": selected_cells[num_agents + i],
        })

    return agents
