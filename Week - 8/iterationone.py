import numpy as np
import matplotlib.pyplot as plt


N = 50              
lam = 1.0           
eta = 0.01         
steps = 5000       
np.random.seed(42)  

J = np.random.normal(0, 1/np.sqrt(N), (N, N))
J = (J + J.T) / 2  

x = np.random.uniform(-1, 1, N)

def H(x):
    """Continuous spin glass Hamiltonian (h = 0)."""
    return x.T @ J @ x + lam * np.sum(1 - 2*x**2 + x**4)

def grad(x):
    """Gradient of H with respect to x_i."""
    return 2 * J @ x + lam * (-4*x + 4*x**3)

energies = []
for k in range(steps):
    x -= eta * grad(x)
    x = np.clip(x, -1, 1)   
    energies.append(H(x))

print("Final approximate minimum energy:", energies[-1])
print("Mean spin magnitude:", np.mean(np.abs(x)))
print("First 10 spin values:", np.round(x[:10], 3))

plt.figure(figsize=(7, 5))
plt.plot(energies, linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Energy', fontsize=12)
plt.title('Gradient Descent on Continuous Ising Spin Glass', fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
