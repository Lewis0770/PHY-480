import numpy as np
import matplotlib.pyplot as plt

# Logistic map function
def logistic_map(x, lam):
    return lam * x * (1 - x)

# Parameters
lambdas = [2.5, 3.5, 3.95]   # λ values
n_iterations = 100           # number of iterations
x0 = 0.5                     # initial condition

# Plot
plt.figure(figsize=(10, 6))

for lam in lambdas:
    x = np.zeros(n_iterations)
    x[0] = x0
    for n in range(1, n_iterations):
        x[n] = logistic_map(x[n-1], lam)
    plt.plot(range(n_iterations), x, label=f"$\lambda = {lam}$")

plt.xlabel("n (iteration)")
plt.ylabel("$x_n$")
plt.title("Logistic Map Iterations")
plt.legend()
plt.grid(True)
plt.show()
