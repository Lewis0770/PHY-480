import numpy as np
import cvxpy as cp
import itertools

def ising_to_maxcut(J, spins):
    return 0.5 * np.sum(J * (1 - np.outer(spins, spins)))

def gw_sdp_maxcut(J, rounds=30):
    N = J.shape[0]
    X = cp.Variable((N, N), PSD=True)
    objective = cp.Maximize(0.5 * cp.sum(cp.multiply(J, (1 - X))))
    constraints = [cp.diag(X) == 1]
    cp.Problem(objective, constraints).solve(solver=cp.SCS, verbose=False)
    X_val = X.value
    best_cut = 0
    for _ in range(rounds):
        r = np.random.randn(N)
        cut = np.sign(X_val @ r)
        best_cut = max(best_cut, 0.5 * np.sum(J * (1 - np.outer(cut, cut))))
    return best_cut

def ising_ground_state_exact(J):
    N = len(J)
    best_E = float('inf')
    best_s = None
    for config in itertools.product([-1, 1], repeat=N):
        s = np.array(config)
        E = np.sum(J * np.outer(s, s))
        if E < best_E:
            best_E, best_s = E, s
    return best_E / 2, best_s

np.random.seed(42)
for N in [10, 15, 20, 30]:
    print(f"\n--- N = {N} ---")
    J = np.random.normal(0, 1/np.sqrt(N), (N, N))
    J = (J + J.T) / 2

    cut_gw = gw_sdp_maxcut(J)
    print(f"GW-SDP Approx. Max-Cut: {cut_gw:.4f}")
    print(f"GW Lower Bound (0.87856×): {0.87856 * cut_gw:.4f}")

    if N <= 18:
        E_exact, s_exact = ising_ground_state_exact(J)
        cut_exact = ising_to_maxcut(J, s_exact)
        print(f"Exact Max-Cut: {cut_exact:.4f}")
        print(f"Ratio GW/Exact = {cut_gw / cut_exact:.4f}")
    else:
        print("Exact skipped (too large).")
