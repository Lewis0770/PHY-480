import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, exp

num_walks = 10000
N_values = [10, 50, 100]  

def analytic_P(N, x):
    """Analytic Gaussian result from Eq. (10)."""
    return (1 / sqrt(2 * pi * N)) * np.exp(-x**2 / (2*N))

for N in N_values:
    steps = np.random.choice([-1, 1], size=(num_walks, N))
    walks = np.cumsum(steps, axis=1)
    final_positions = walks[:, -1]

    counts, bins = np.histogram(final_positions, bins=range(-N, N+2), density=True)
    centers = (bins[:-1] + bins[1:]) / 2

    plt.bar(centers, counts, width=1.0, alpha=0.6, label=f"Simulation (N={N})")
    plt.plot(centers, analytic_P(N, centers), 'r--', linewidth=2, label="Analytic Gaussian")
    plt.title(f"Random Walks vs Analytic Result (N={N})")
    plt.xlabel("Position x")
    plt.ylabel("Probability P(N,x)")
    plt.legend()
    plt.show()
