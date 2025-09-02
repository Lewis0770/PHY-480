import numpy as np

def matrix_walk(N, steps):
    size = 2*N + 1   
    M = np.zeros((size, size))

    for i in range(1, size-1):
        M[i, i-1] = 0.5
        M[i, i+1] = 0.5

    v = np.zeros(size)
    v[N] = 1.0

    for _ in range(steps):
        v = M @ v
    return v

prob = matrix_walk(N=10, steps=10)
print("Matrix method probabilities:", prob)

from math import comb

def combinatorial_P(N, x):
    n_plus = (N + x) // 2
    if (N + x) % 2 != 0:  
        return 0
    return comb(N, n_plus) / (2**N)

for x in range(-10, 11, 2): 
    print(f"x={x}, Matrix={prob[x+10]:.5f}, Comb={combinatorial_P(10, x):.5f}")
