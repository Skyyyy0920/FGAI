import matplotlib.pyplot as plt
import networkx as nx

G = nx.star_graph(200)
pos = nx.spring_layout(G, seed=63)  # Seed layout for reproducibility
colors = range(200)
options = {
    "node_color": "#A0CBE2",
    "edge_color": colors,
    "width": 4,
    "edge_cmap": plt.cm.Blues,
    "with_labels": False,
}
nx.draw(G, pos, **options)
plt.show()
