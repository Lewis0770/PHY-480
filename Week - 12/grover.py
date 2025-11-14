import numpy as np
import matplotlib.pyplot as plt

def grover_matrix(N, target_index):
    s = np.ones(N) / np.sqrt(N)
    projector = 2 * np.outer(s, s)
    I = np.eye(N)
    
    O = np.eye(N)
    O[target_index, target_index] = -1
    
    G = (projector - I) @ O
    return G

def run_grover(n, target_index):
    N = 2**n
    psi = np.ones(N) / np.sqrt(N)
    G = grover_matrix(N, target_index)
    
    T = 200
    probs = np.zeros(T)
    
    for t in range(T):
        probs[t] = np.abs(psi[target_index])**2
        psi = G @ psi
    
    return probs

n = 10
N = 2**n
target_index = N - 1  

probs = run_grover(n, target_index)

peak_t = np.argmax(probs)
peak_prob = probs[peak_t]

print(f"For n = {n}, N = {N}")
print(f"Target state |tar> = |{target_index}> (binary: {bin(target_index)})")
print(f"Peak iteration t* = {peak_t}")
print(f"Peak probability P(t*) = {peak_prob:.6f}")
print(f"Probability of finding |tar> at t* = {peak_prob:.6f}")

plt.figure(figsize=(10,5))
plt.plot(probs, linewidth=2)
plt.xlabel("t (Grover iterations)", fontsize=12)
plt.ylabel("Probability of measuring target state |tar>", fontsize=12)
plt.title(f"Grover probability vs time for n={n}, N={N}", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('grover_probability.png', dpi=150)
plt.show()
