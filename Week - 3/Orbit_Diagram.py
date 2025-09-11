import numpy as np
import matplotlib.pyplot as plt

b = 0.3
a_values = np.linspace(1.0, 1.5, 200)  
n_iterations = 1000                    
n_transient = 200                      

x_vals, a_vals = [], []

for a in a_values:
    x, y = 0.0, 0.0   
    
    for i in range(n_iterations):
        x_new = 1 - a * x**2 + y
        y_new = b * x
        x, y = x_new, y_new
        
        if i >= n_transient:
            if -1.5 <= x <= 1.5:  
                x_vals.append(x)
                a_vals.append(a)

plt.figure(figsize=(10,6))
plt.scatter(a_vals, x_vals, s=0.1, color="black")
plt.xlabel("a", fontsize=14)
plt.ylabel("x (steady-state values)", fontsize=14)
plt.title("Orbit Diagram of the Hénon Map (b = 0.3)", fontsize=16)
plt.show()
