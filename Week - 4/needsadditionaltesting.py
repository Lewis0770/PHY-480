import numpy as np
import matplotlib.pyplot as plt

L = 20              # Lattice size (LxL grid)
steps = 100000      # Total Monte Carlo steps
J = 1.0             
Tc = 2.27           
temperatures = [1.5, 2.27, 3.5] 

def initialize_spins(L):
    """Start in fully magnetized state (all spins up)."""
    return np.ones((L, L), dtype=int)

def get_neighbors(i, j, L):
    """Periodic boundary conditions: return nearest neighbors of (i,j)."""
    return [((i-1) % L, j),
            ((i+1) % L, j),
            (i, (j-1) % L),
            (i, (j+1) % L)]

def delta_energy(spins, i, j, J):
    """Energy change for flipping spin (i,j)."""
    s = spins[i, j]
    neighbors = get_neighbors(i, j, L)
    interaction = sum(spins[x, y] for x, y in neighbors)
    return 2 * J * s * interaction

# ---------- Monte Carlo Simulation ----------
def metropolis(L, T, steps):
    spins = initialize_spins(L)
    M_values = []

    for step in range(steps):
        # pick a random site
        i, j = np.random.randint(0, L), np.random.randint(0, L)
        dE = delta_energy(spins, i, j, J)

        # Metropolis acceptance criterion
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            spins[i, j] *= -1

        if step % (L*L) == 0:  # one sweep = L^2 steps
            M = np.sum(spins) / (L*L)
            M_values.append(abs(M))

    return np.array(M_values)

# ---------- Run simulations for 3 temperatures ----------
plt.figure(figsize=(12, 5))

for T in temperatures:
    M_t = metropolis(L, T, steps)
    plt.plot(M_t, label=f"T = {T:.2f}")

plt.xlabel("Monte Carlo sweeps")
plt.ylabel("Magnetization per spin")
plt.title("Magnetization vs Time for different temperatures")
plt.legend()
plt.show()
