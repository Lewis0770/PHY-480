import numpy as np
import matplotlib.pyplot as plt

def logistic_map(x, lam):
    return lam * x * (1 - x)

lambdas = np.linspace(1, 4, 1000)  # 1000 values of λ in [1,4]
n_iterations = 1000                 # total iterations
last = 100                          # number of points to keep after transients

x0 = 0.5  # initial condition

lam_list = []
x_list = []

for lam in lambdas:
    x = x0
    for i in range(n_iterations - last):
        x = logistic_map(x, lam)
    for i in range(last):
        x = logistic_map(x, lam)
        lam_list.append(lam)
        x_list.append(x)

plt.figure(figsize=(10, 6))
plt.plot(lam_list, x_list, ',k', alpha=0.25)  
plt.title("Bifurcation diagram of the Logistic Map")
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$x_n$")
plt.show()
