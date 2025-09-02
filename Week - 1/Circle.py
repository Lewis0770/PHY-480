import numpy as np
import matplotlib.pyplot as plt

trials = np.logspace(2, 6, 20, dtype=int)  
pi_estimates = []

for N in trials:
    x = np.random.rand(N) 
    y = np.random.rand(N)  
    
    inside = (x**2 + y**2) <= 1  
    pi_est = 4 * np.sum(inside) / N
    pi_estimates.append(pi_est)

errors = np.abs(np.array(pi_estimates) - np.pi)

plt.loglog(trials, errors, 'o-', label="MC error")
plt.loglog(trials, 1/np.sqrt(trials), '--', label="1/sqrt(N) slope")
plt.xlabel("Number of trials (N)")
plt.ylabel("Error in π estimate")
plt.legend()
plt.show()
