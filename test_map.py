
from movingai import load_movingai_map, sample_grid_agents

grid, G = load_movingai_map("data/random-32-32-20.map")

print("Map dimensions:", len(grid), "x", len(grid[0]))
print("Traversable cells:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

agents = sample_grid_agents(G, num_agents=3, seed=0)

print("\nGenerated agents:")
for agent in agents:
    print(
        f"Agent {agent['id']}: "
        f"start={agent['start']}, goal={agent['goal']}"
    )
