import numpy as np
from scipy.sparse import kron, csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

sx = csr_matrix([[0, 1], [1, 0]], dtype=float)
sz = csr_matrix([[1, 0], [0, -1]], dtype=float)
I2 = csr_matrix(np.eye(2), dtype=float)

def kron_list(op_list):
    """Take Kronecker product of a list of operators"""
    result = op_list[0]
    for op in op_list[1:]:
        result = kron(result, op)
    return result

def build_single_site_op(N, site, op):
    """Build operator that acts on single site with identity elsewhere"""
    op_list = [I2] * N
    op_list[site] = op
    return kron_list(op_list)

def build_two_site_op(N, site1, site2, op1, op2):
    """Build operator that acts on two sites with identity elsewhere"""
    op_list = [I2] * N
    op_list[site1] = op1
    op_list[site2] = op2
    return kron_list(op_list)

def build_H(N, J, lam):
    """Build Hamiltonian H = -J * sum(ZiZi+1) - lambda * sum(Xi)"""
    H = csr_matrix((2**N, 2**N), dtype=float)
    
    for i in range(N):
        i_next = (i + 1) % N
        ZZ = build_two_site_op(N, i, i_next, sz, sz)
        H -= J * ZZ
    
    for i in range(N):
        X = build_single_site_op(N, i, sx)
        H -= lam * X
    
    return H

def magnetization(psi, N):
    """Calculate |<sum_i Zi> / N|"""
    m_total = 0.0
    for i in range(N):
        Z = build_single_site_op(N, i, sz)
        expectation = np.vdot(psi, Z.dot(psi))
        m_total += np.real(expectation)
    
    return np.abs(m_total / N)

J = 1.0
lambdas = np.linspace(0.4, 1.4, 20) 
sizes = [6, 8, 10]

plt.figure(figsize=(10, 6))
for N in sizes:
    m_list = []
    print(f"Calculating for N={N}...")
    for lam in lambdas:
        H = build_H(N, J, lam)
        E, psi = eigsh(H, k=1, which='SA')
        m = magnetization(psi[:, 0], N)
        m_list.append(m)
    
    plt.plot(lambdas, m_list, marker='o', label=f'N={N}')

plt.xlabel('λ', fontsize=12)
plt.ylabel('Ground-State Magnetization $m_G$', fontsize=12)
plt.title('Ground State Magnetization vs λ', fontsize=14)
plt.axvline(x=1.0, color='k', linestyle='--', alpha=0.3, label='λ_c = 1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
