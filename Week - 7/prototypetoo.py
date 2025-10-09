import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

L = 5
np.random.seed(42)

G = nx.grid_2d_graph(L, L)

for (u, v) in G.edges():
    G[u][v]['weight'] = np.random.rand()

def draw_graph(G, title, edges=None, pos=None):
    if pos is None:
        pos = {(x, y): (x, -y) for x, y in G.nodes()}
    plt.figure(figsize=(6,6))
    nx.draw(G, pos, node_color='lightblue', with_labels=False, node_size=100)
    nx.draw_networkx_edges(G, pos, width=1)
    if edges:
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2.5)
    plt.title(title)
    plt.axis('off')
    plt.show()

mst = nx.minimum_spanning_tree(G, algorithm='prim')
mst_weight = sum(d['weight'] for (_,_,d) in mst.edges(data=True))
print(f"(i) Sum of MST edge weights: {mst_weight:.4f}")
draw_graph(G, f"Prim's Minimum Spanning Tree (sum={mst_weight:.4f})", edges=mst.edges())

root = (0, 0)
spt = nx.DiGraph()
path_tree_edges = []

distances, paths = nx.single_source_dijkstra(G, root, weight='weight')

for target in paths:
    path = paths[target]
    if len(path) > 1:
        for i in range(len(path)-1):
            spt.add_edge(path[i], path[i+1], weight=G[path[i]][path[i+1]]['weight'])
            path_tree_edges.append((path[i], path[i+1]))

spt_weight = sum(d['weight'] for (_,_,d) in spt.edges(data=True))
print(f"(ii) Sum of SPT edge weights: {spt_weight:.4f}")
print(f"SPT vs MST difference: {spt_weight - mst_weight:.4f}")
draw_graph(G, f"Dijkstra's Shortest Path Tree (sum={spt_weight:.4f})", edges=path_tree_edges)

flow_network = nx.DiGraph()
flow_network.add_nodes_from(G.nodes())

for u, v, d in G.edges(data=True):
    flow_network.add_edge(u, v, capacity=d['weight'])
    flow_network.add_edge(v, u, capacity=d['weight'])

source = 's'
sink = 't'
flow_network.add_node(source)
flow_network.add_node(sink)

for y in range(L):
    flow_network.add_edge(source, (0, y), capacity=5.0)
    flow_network.add_edge((L-1, y), sink, capacity=5.0)

flow_value, flow_dict = nx.maximum_flow(flow_network, source, sink)
cut_value, (S, T) = nx.minimum_cut(flow_network, source, sink)
print(f"(iii) Maximum flow value: {flow_value:.4f}")
print(f"Minimum cut value: {cut_value:.4f}")

cut_edges = []
for u in S:
    for v in flow_network[u]:
        if v in T:
            cut_edges.append((u, v))

lattice_edges = [e for e in cut_edges if isinstance(e[0], tuple) and isinstance(e[1], tuple)]
draw_graph(G, f"Minimum Cut Edges (flow={flow_value:.4f})", edges=lattice_edges)
