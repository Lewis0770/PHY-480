import numpy as np
from scipy.linalg import expm, eigh
import matplotlib.pyplot as plt

X = np.array([[0, 1],
              [1, 0]], dtype=float)
Z = np.array([[1, 0],
              [0, -1]], dtype=float)
I = np.eye(2)

def kron_n(ops):
    out = ops[0]
    for A in ops[1:]:
        out = np.kron(out, A)
    return out

def X_i(i, N):
    ops = [I] * N
    ops[i] = X
    return kron_n(ops)

def Z_iZ_j(i, j, N):
    ops = [I] * N
    ops[i] = Z
    ops[j] = Z
    return kron_n(ops)

def SK_couplings(N):
    J = np.random.normal(0, 1/np.sqrt(N), size=(N, N))
    return np.triu(J, 1)

def H_SK(N, J):
    H = np.zeros((2**N, 2**N))
    for i in range(N):
        for j in range(i+1, N):
            H -= J[i, j] * Z_iZ_j(i, j, N)
    return H

def H1_transverse(N):
    H = np.zeros((2**N, 2**N))
    for i in range(N):
        H += X_i(i, N)
    return H

def H_t(t, a, Hnpc, H1):
    return a*t*Hnpc + (1 - a*t)*H1

def evolve_state(psi, H, dt):
    U = expm(-1j * H * dt)
    return U @ psi

N = 8
np.random.seed(42)  # Fix seed for reproducibility
J = SK_couplings(N)
Hnpc = H_SK(N, J)
H1 = H1_transverse(N)

psi = np.ones(2**N) / np.sqrt(2**N)

# VERY slow annealing
steps = 500      # Increased from 200
T = 50.0         # Increased from 10.0
dt = T / steps
a = 1.0 / T

plot_steps = [0, steps//4, steps//2, 3*steps//4, steps-1]
stored_psi = []
stored_ground_states = []
times = []

print("Running quantum annealing...")
for k in range(steps):
    t = k * dt
    H = H_t(t, a, Hnpc, H1)
    
    if k in plot_steps:
        evals, evecs = eigh(H)
        ground_state_t = evecs[:, np.argmin(evals)]
        stored_ground_states.append(ground_state_t)
        stored_psi.append(psi.copy())
        times.append(t)
        print(f"  Step {k}/{steps}, t={t:.2f}")
    
    psi = evolve_state(psi, H, dt)

print("Computing final ground state...")
evals, evecs = eigh(Hnpc)
ground_state = evecs[:, np.argmin(evals)]
ground_energy = np.min(evals)

# Compute energy of annealed state
annealed_energy = np.real(np.vdot(psi, Hnpc @ psi))

# Plot wavefunction amplitudes at series of time steps
fig, axes = plt.subplots(len(plot_steps), 1, figsize=(10, 12))
for idx, (ax, psi_t, t) in enumerate(zip(axes, stored_psi, times)):
    amplitudes = np.abs(psi_t)**2
    ax.bar(range(2**N), amplitudes, width=1.0)
    ax.set_xlabel('Basis State')
    ax.set_ylabel('|ψ|²')
    ax.set_title(f'Wavefunction at t={t:.2f}')
plt.tight_layout()
plt.savefig('annealing_series.png', dpi=150)
plt.show()

# Plot final comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

annealed_amplitudes = np.abs(psi)**2
ground_amplitudes = np.abs(ground_state)**2

ax1.bar(range(2**N), ground_amplitudes, width=1.0, color='blue', alpha=0.7)
ax1.set_xlabel('Basis State')
ax1.set_ylabel('|ψ|²')
ax1.set_title('True Ground State Wavefunction')

ax2.bar(range(2**N), annealed_amplitudes, width=1.0, color='red', alpha=0.7)
ax2.set_xlabel('Basis State')
ax2.set_ylabel('|ψ|²')
ax2.set_title('Annealed Wavefunction')

plt.tight_layout()
plt.savefig('final_comparison.png', dpi=150)
plt.show()

overlap = np.abs(np.vdot(ground_state, psi))
print("\n" + "="*60)
print("QUANTUM ANNEALING RESULTS")
print("="*60)
print(f"Overlap with true ground state: {overlap:.6f}")
print(f"Ground state energy: {ground_energy:.6f}")
print(f"Annealed state energy: {annealed_energy:.6f}")
print(f"Energy difference: {annealed_energy - ground_energy:.6f}")
print("\nComparison of wavefunctions:")
print(f"Annealed state has {np.sum(annealed_amplitudes > 0.01)} significant basis states")
print(f"Ground state has {np.sum(ground_amplitudes > 0.01)} significant basis states")
print(f"Peak annealed amplitude at state {np.argmax(annealed_amplitudes)}")
print(f"Peak ground amplitude at state {np.argmax(ground_amplitudes)}")

if overlap > 0.9:
    print("\n✓ High overlap - QA successfully found ground state")
elif overlap > 0.5:
    print("\n~ Moderate overlap - QA found good approximation")
else:
    print("\n✗ Low overlap - QA trapped in local minimum")
    print("  This demonstrates the difficulty of QA for NP-complete problems")
    print("  SK spin glasses have rugged energy landscapes with many local minima")
print("="*60)
