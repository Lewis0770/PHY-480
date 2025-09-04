import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
steps = 5000
lambda_ = 1.0   
np.random.seed(42)

sigma = np.sqrt(1.0 / dt)

def langevin(case="random_only"):
    v = np.zeros(steps)
    x = np.zeros(steps)

    if case == "relaxation":
        v[0] = 10.0  

    for i in range(steps - 1):
        eta = np.random.normal(0, sigma)
        if case == "random_only":
            v[i+1] = v[i] + eta * dt  # λ = 0
        elif case == "relaxation":
            v[i+1] = v[i] - lambda_ * v[i] * dt + eta * dt
        x[i+1] = x[i] + v[i] * dt

    return x, v

x1, v1 = langevin("random_only")
x2, v2 = langevin("relaxation")

fig, axs = plt.subplots(2, 2, figsize=(12, 6))

axs[0,0].plot(v1)
axs[0,0].set_title("Case (i): Velocity (random walk)")
axs[1,0].plot(x1)
axs[1,0].set_title("Case (i): Position (random walk)")

axs[0,1].plot(v2)
axs[0,1].set_title("Case (ii): Velocity relaxation")
axs[1,1].plot(x2)
axs[1,1].set_title("Case (ii): Position")

plt.tight_layout()
plt.show()
