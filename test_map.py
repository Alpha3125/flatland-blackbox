from movingai import load_movingai_map, sample_grid_agents
from mapf_conflicts import detect_conflicts
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

grid, G = load_movingai_map("data/random-32-32-20.map")
agents = sample_grid_agents(G, num_agents=5, seed=10)

print("Map dimensions:", len(grid), "x", len(grid[0]))
print("Traversable cells:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# Compute each agent's individual shortest path.
paths = []
for agent in agents:
    path = nx.shortest_path(
        G, source=agent["start"], target=agent["goal"]
    )
    paths.append(path)

    print(f"\nAgent {agent['id']}:")
    print("  Start:", agent["start"])
    print("  Goal:", agent["goal"])
    print("  Path length (moves):", len(path) - 1)
    print("  Path:", path)

    conflicts = detect_conflicts(paths)

print("\nConflict report:")

if not conflicts:
    print("No conflicts detected.")
else:
    for conflict in conflicts:
        print(conflict)

    print(f"\nTotal conflicts: {len(conflicts)}")

# Convert the map into an image:
# 0 = obstacle, 1 = traversable cell.
height, width = len(grid), len(grid[0])
map_image = [
    [0 if cell not in {".", "G"} else 1 for cell in row]
    for row in grid
]

fig, ax = plt.subplots(figsize=(10, 10))
cmap = ListedColormap(["black", "white"])

ax.imshow(map_image, cmap=cmap, origin="upper")

# Draw each agent's path.
colors = ["red", "deepskyblue", "limegreen", "orange", "magenta"]

for i, (agent, path) in enumerate(zip(agents, paths)):
    color = colors[i % len(colors)]
    rows = [cell[0] for cell in path]
    cols = [cell[1] for cell in path]

    ax.plot(cols, rows, color=color, linewidth=2, label=f"Agent {i}")
    ax.scatter(
        *agent["start"][::-1], color=color, marker="o",
        s=100, edgecolors="black", zorder=3
    )
    ax.scatter(
        *agent["goal"][::-1], color=color, marker="*",
        s=200, edgecolors="black", zorder=3
    )

ax.set_title("Moving AI Map: Individual Shortest Paths")
ax.set_xlabel("Column")
ax.set_ylabel("Row")
ax.set_xticks(range(width))
ax.set_yticks(range(height))
ax.grid(True, linewidth=0.25, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()