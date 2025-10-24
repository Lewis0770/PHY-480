import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

L = 50
pc = 0.593
p_values = np.linspace(0.60, 0.95, 10)
n_realizations = 20

def make_grid_site_percolation(L, p):
    G = nx.Graph()
    occupied = np.random.rand(L, L) < p
    
    for i in range(L):
        for j in range(L):
            if occupied[i, j]:
                G.add_node((i, j))
    
    for i in range(L):
        for j in range(L):
            if occupied[i, j]:
                if i < L - 1 and occupied[i + 1, j]:
                    G.add_edge((i, j), (i + 1, j))
                j_next = (j + 1) % L
                if occupied[i, j_next]:
                    G.add_edge((i, j), (i, j_next))
    
    return G, occupied

def spanning_cluster(G, L):
    clusters = list(nx.connected_components(G))
    for cluster in clusters:
        xs = [x for (x, y) in cluster]
        if 0 in xs and (L - 1) in xs:
            return cluster
    return set()

def compute_conductivity(G, L, spanning):
    if len(spanning) == 0:
        return 0.0
    
    nodes_in_spanning = list(spanning)
    node_index = {node: i for i, node in enumerate(nodes_in_spanning)}
    N = len(nodes_in_spanning)
    
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    for node in nodes_in_spanning:
        idx = node_index[node]
        x, y = node
        
        if x == 0:
            A[idx, :] = 0
            A[idx, idx] = 1
            b[idx] = 1.0
        elif x == L - 1:
            A[idx, :] = 0
            A[idx, idx] = 1
            b[idx] = 0.0
        else:
            neighbors = []
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx_coord = x + dx
                ny_coord = (y + dy) % L
                neighbor = (nx_coord, ny_coord)
                if neighbor in spanning:
                    neighbors.append(neighbor)
            
            A[idx, idx] = len(neighbors)
            for neighbor in neighbors:
                nidx = node_index[neighbor]
                A[idx, nidx] = -1
    
    try:
        V = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return 0.0
    
    total_current = 0.0
    for node in nodes_in_spanning:
        x, y = node
        if x < L - 1:
            neighbor = (x + 1, y)
            if neighbor in spanning:
                idx = node_index[node]
                nidx = node_index[neighbor]
                current = V[idx] - V[nidx]
                total_current += current
    
    return total_current / L

P_inf = []
Sigma = []

for p in p_values:
    P_inf_vals = []
    Sigma_vals = []
    
    for _ in range(n_realizations):
        G, occupied = make_grid_site_percolation(L, p)
        cluster = spanning_cluster(G, L)
        
        P_inf_val = len(cluster) / (L * L)
        P_inf_vals.append(P_inf_val)
        
        Sigma_val = compute_conductivity(G, L, cluster)
        Sigma_vals.append(Sigma_val)
    
    P_inf.append(np.mean(P_inf_vals))
    Sigma.append(np.mean(Sigma_vals))
    
    print(f"p = {p:.2f} | P_inf = {np.mean(P_inf_vals):.3f} | Sigma = {np.mean(Sigma_vals):.3f}")

plt.figure(figsize=(8, 6))
plt.plot(p_values, P_inf, 'o-', label='P∞ (Infinite Cluster Probability)')
plt.plot(p_values, Sigma, 's--', label='Σ (Conductivity)')
plt.xlabel('p (Occupation Probability)')
plt.ylabel('Value')
plt.title(f'Percolation Results for {L}×{L} Random Resistor Network')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
