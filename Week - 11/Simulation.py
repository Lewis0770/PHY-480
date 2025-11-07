# ising_time_dependent.py
# PHY-480 In-Class Work 21
# Time-dependent magnetization of the transverse-field Ising model
# λ(t) = λ0 * cos(ωt)

import numpy as np
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt

# Pauli matrices
sx = csr_matrix(np.array([[0, 1], [1, 0]]))
sz = csr_matrix(np.array([[1, 0], [0, -1]]))
I = csr_matrix(np.eye(2))

# Build the Hamiltonian at a given time t for given λ(t)
def build_H(N, J, lam):
    H = csr_matrix((2**N, 2**N), dtype=float)
    for i in range(N):
        ZZ = 1
        for j in range(N):
            ZZ = kron(ZZ, sz if j in [i, (i + 1) % N] else I)
        H -= J * ZZ
        X = 1
        for j in range(N):
            X = kron(X, sx if j == i else I)
        H -= lam * X
    return H

# Calculate magnetization m = <ψ|ΣZ_i|ψ>/N
def magnetization(psi, N):
    Z_ops = [kron(identity(2**i), kron(sz, identity(2**(N - i - 1)))) for i in range(N)]
    m = np.real(sum((psi.conj().T @ (Z @ psi)) for Z in Z_ops)) / N
    return np.abs(m)

# Simulation parameters
N = 8
J = 1.0
n_steps = 100
t_total = 10.0
dt = t_total / n_steps
times = np.linspace(0, t_total, n_steps)

# λ0 values: below, at, and above λc = 1
lambda0_values = [0.5, 1.0, 1.5]
# Driving frequencies (between 1/dt and 1/t_total)
frequencies = [1 / t_total, 3 / t_total, 5 / t_total]

# Initial state: all spins up (|000...0>)
psi0 = np.zeros(2**N)
psi0[0] = 1.0

for lam0 in lambda0_values:
    plt.figure(figsize=(8, 5))
    for w in frequencies:
        psi = psi0.copy()
        mags = []

        for t in times:
            lam_t = lam0 * np.cos(w * t)
            H_t = build_H(N, J, lam_t)
            # Evolve ψ(t+dt) = exp(-iHΔt) ψ(t)
            psi = expm_multiply((-1j * H_t * dt), psi)
            mags.append(magnetization(psi, N))

        plt.plot(times, mags, label=f'ω={w:.3f}')

    plt.title(f'Magnetization vs Time (λ₀={lam0})')
    plt.xlabel('Time')
    plt.ylabel('Magnetization m(t)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
