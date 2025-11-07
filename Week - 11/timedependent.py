
import numpy as np
from scipy.sparse import kron, csr_matrix
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt

sx = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
sz = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
I = csr_matrix(np.eye(2, dtype=complex))

def kron_list(op_list):
    result = op_list[0]
    for op in op_list[1:]:
        result = kron(result, op)
    return result

def build_single_site_op(N, site, op):
    op_list = [I] * N
    op_list[site] = op
    return kron_list(op_list)

def build_two_site_op(N, site1, site2, op1, op2):
    op_list = [I] * N
    op_list[site1] = op1
    op_list[site2] = op2
    return kron_list(op_list)

def build_H(N, J, lam):
    H = csr_matrix((2**N, 2**N), dtype=complex)
    
    for i in range(N):
        i_next = (i + 1) % N
        ZZ = build_two_site_op(N, i, i_next, sz, sz)
        H -= J * ZZ
    
    for i in range(N):
        X = build_single_site_op(N, i, sx)
        H -= lam * X
    
    return H

def magnetization(psi, N):
    m_total = 0.0
    for i in range(N):
        X = build_single_site_op(N, i, sx)
        expectation = np.vdot(psi, X.dot(psi))
        m_total += np.real(expectation)
    
    return m_total / N

N = 8
J = 1.0
n_steps = 100
t_total = 10.0
dt = t_total / n_steps
times = np.linspace(0, t_total, n_steps)

lambda0_values = [0.5, 1.0, 1.5]
frequencies = [1 / t_total, 3 / t_total, 5 / t_total]

psi0 = np.zeros(2**N, dtype=complex)
psi0[0] = 1.0

for lam0 in lambda0_values:
    plt.figure(figsize=(8, 5))
    for w in frequencies:
        psi = psi0.copy()
        mags = []
        for t in times:
            lam_t = lam0 * np.cos(w * t)
            H_t = build_H(N, J, lam_t)
            psi = expm_multiply(-1j * H_t * dt, psi)
            psi = psi / np.linalg.norm(psi)  # Normalize
            mags.append(magnetization(psi, N))
        plt.plot(times, mags, label=f'ω={w:.3f}')
    plt.title(f'Transverse Magnetization vs Time (λ₀={lam0})')
    plt.xlabel('Time')
    plt.ylabel('Magnetization ⟨X⟩/N')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
