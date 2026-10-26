import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import ListedColormap

from movingai import load_movingai_map, sample_grid_agents
from mapf_cbs import cbs, find_first_conflict

grid, G = load_movingai_map("data/random-32-32-20.map")
agents = sample_grid_agents(G, num_agents=6, seed=0)

# Independent shortest paths.
independent_paths = [
    nx.shortest_path(G, agent["start"], agent["goal"])
    for agent in agents
]

print("Running CBS...")
cbs_paths = cbs(G, agents)

if cbs_paths is None:
    raise RuntimeError("CBS did not find a solution within its search limit.")

assert find_first_conflict(cbs_paths) is None

independent_cost = sum(len(p) - 1 for p in independent_paths)
cbs_cost = sum(len(p) - 1 for p in cbs_paths)

print("\nComparison")
print("Independent total moves:", independent_cost)
print("CBS total moves:", cbs_cost)
print("Additional moves due to coordination:", cbs_cost - independent_cost)

# Plot the map.
height, width = len(grid), len(grid[0])
map_image = [
    [0 if char not in {".", "G"} else 1 for char in row]
    for row in grid
]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
cmap = ListedColormap(["black", "white"])
colors = ["red", "deepskyblue", "limegreen", "orange", "magenta"]

for ax, paths, title in [
    (axes[0], independent_paths, "Independent shortest paths"),
    (axes[1], cbs_paths, "CBS collision-free paths"),
]:
    ax.imshow(map_image, cmap=cmap, origin="upper")

    for i, (agent, path) in enumerate(zip(agents, paths)):
        color = colors[i % len(colors)]
        rows = [cell[0] for cell in path]
        cols = [cell[1] for cell in path]

        ax.plot(cols, rows, color=color, linewidth=2, label=f"Agent {i}")
        ax.scatter(
            agent["start"][1], agent["start"][0],
            color=color, marker="o", s=70, edgecolors="black"
        )
        ax.scatter(
            agent["goal"][1], agent["goal"][0],
            color=color, marker="*", s=160, edgecolors="black"
        )

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.grid(True, linewidth=0.25, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig("outputs/path_comparison.png", dpi=200)
plt.show()
