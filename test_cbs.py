
from movingai import load_movingai_map, sample_grid_agents
from mapf_cbs import cbs, find_first_conflict

grid, G = load_movingai_map("data/random-32-32-20.map")
agents = sample_grid_agents(G, num_agents=3, seed=0)

print("Agents:")
for agent in agents:
    print(agent)

print("\nRunning CBS...")
paths = cbs(G, agents)

if paths is None:
    print("No solution found within the search limit.")
else:
    print("\nSolution found!")

    for i, path in enumerate(paths):
        print(f"\nAgent {i}:")
        print("  Start:", path[0])
        print("  Goal:", path[-1])
        print("  Moves:", len(path) - 1)
        print("  Path:", path)

    conflict = find_first_conflict(paths)
    print("\nRemaining conflict:", conflict)

    print(
        "Total moves:",
        sum(len(path) - 1 for path in paths)
    )
