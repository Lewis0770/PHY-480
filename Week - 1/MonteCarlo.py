import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return np.exp(-(x**2 + y**2))

def monte_carlo_integral(N):
    x = np.random.uniform(-1, 1, N)
    y = np.random.uniform(-1, 1, N)

    inside = x**2 + y**2 <= 1

    f_values = f(x[inside], y[inside])
    avg_f = np.mean(f_values)

    return np.pi * avg_f

trials = np.logspace(2, 6, 20, dtype=int) 
estimates = []

for N in trials:
    I_est = monte_carlo_integral(N)
    estimates.append(I_est)

plt.plot(1/trials, estimates, 'o-', label="MC estimate")
plt.axhline(y=np.pi*(1-np.exp(-1)), color='r', linestyle='--', label="Reference (analytic)")
plt.xlabel("1/N")
plt.ylabel("Integral Estimate I")
plt.legend()
plt.show()
