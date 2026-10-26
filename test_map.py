from flatland_blackbox.utils import load_movingai_map

grid, G = load_movingai_map("data/random-32-32-20.map")

print("Map dimensions:", len(grid), "x", len(grid[0]))
print("Number of traversable cells:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Example nodes:", list(G.nodes)[:10])
