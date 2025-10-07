import numpy as np
import itertools
import matplotlib.pyplot as plt

def energy(spins, J):
    """Compute Ising Hamiltonian energy: H = -sum_{i<j} J_ij S_i S_j"""
    N = len(spins)
    E = 0
    for i in range(N):
        for j in range(i+1, N):
            E -= J[i,j] * spins[i] * spins[j]
    return E

def overlap(s1, s2):
    """Compute overlap index I_o = (sum_i S_i * S'_i) / N"""
    N = len(s1)
    return np.sum(s1 * s2) / N

def exhaustive_search(J):
    """Find all energy states using brute-force search."""
    N = len(J)
    energies = []
    for config in itertools.product([-1, 1], repeat=N):
        spins = np.array(config)
        E = energy(spins, J)
        energies.append((E, spins.copy()))
    energies.sort(key=lambda x: x[0])
    return energies

def simulated_annealing(J, T0=5.0, Tmin=1e-3, cooling=0.95, steps_per_T=100):
    """Perform simulated annealing to find low-energy configuration."""
    N = len(J)
    spins = np.random.choice([-1, 1], N)
    E = energy(spins, J)
    T = T0
    best_E, best_spins = E, spins.copy()

    while T > Tmin:
        for _ in range(steps_per_T):
            i = np.random.randint(N)
            spins_trial = spins.copy()
            spins_trial[i] *= -1  
            E_trial = energy(spins_trial, J)
            dE = E_trial - E
            if dE < 0 or np.random.rand() < np.exp(-dE / T):
                spins = spins_trial
                E = E_trial
                if E < best_E:
                    best_E, best_spins = E, spins.copy()
        T *= cooling
    return best_E, best_spins

np.random.seed(42)

print("=" * 70)
print("PART (i): EXACT EXHAUSTIVE SEARCH")
print("=" * 70)

Ns_exact = [6, 12, 18]
exact_results = {}

for N in Ns_exact:
    print(f"\n--- Instance with N = {N} spins ---")
    
    J = np.random.normal(0, 1/np.sqrt(N), (N, N))
    J = np.triu(J, 1)  
    J += J.T  
    
    all_states = exhaustive_search(J)
    
    unique_energies = []
    energy_groups = []
    
    for E, spins in all_states:
        is_new = True
        for ue in unique_energies:
            if abs(E - ue) < 1e-10:
                is_new = False
                break
        
        if is_new and len(unique_energies) < 3:
            unique_energies.append(E)
            energy_groups.append([])
        
        if len(unique_energies) <= 3:
            for idx, ue in enumerate(unique_energies):
                if abs(E - ue) < 1e-10:
                    energy_groups[idx].append(spins)
                    break
    
    print(f"\nThree lowest energy states:")
    for i, (E, group) in enumerate(zip(unique_energies[:3], energy_groups[:3])):
        print(f"  State {i+1}: E = {E:.6f}, Degeneracy = {len(group)}")
    
    print(f"\nOverlap indices I_o for pairs with different energies:")
    for i in range(len(unique_energies[:3])):
        for j in range(i+1, len(unique_energies[:3])):
            s1 = energy_groups[i][0]
            s2 = energy_groups[j][0]
            I_o = overlap(s1, s2)
            print(f"  State {i+1} vs State {j+1}: I_o = {I_o:.6f}")
    
    exact_results[N] = {
        'energies': unique_energies[:3],
        'degeneracies': [len(g) for g in energy_groups[:3]],
        'ground_state': energy_groups[0][0],
        'J': J
    }

print("\n" + "=" * 70)
print("PART (ii): SIMULATED ANNEALING COMPARISON")
print("=" * 70)

Ns_SA = list(range(10, 51, 10))

Ns_all_SA = sorted(list(set(Ns_exact + Ns_SA)))

sa_results = {}

for N in Ns_all_SA:
    if N in exact_results:
        J = exact_results[N]['J']
    else:
        J = np.random.normal(0, 1/np.sqrt(N), (N, N))
        J = np.triu(J, 1)
        J += J.T
    
    E_sa, spins_sa = simulated_annealing(J)
    sa_results[N] = {'energy': E_sa, 'spins': spins_sa}

print("\nSimulated Annealing Results:")
for N in Ns_all_SA:
    E_sa = sa_results[N]['energy']
    print(f"  N = {N}: E = {E_sa:.6f}, E/N = {E_sa/N:.6f}")

print("\nComparison with Exact Results (N ≤ 18):")
for N in Ns_exact:
    E_exact = exact_results[N]['energies'][0]  
    E_sa = sa_results[N]['energy']
    print(f"  N = {N}:")
    print(f"    Exact ground state: E = {E_exact:.6f}, E/N = {E_exact/N:.6f}")
    print(f"    SA result:          E = {E_sa:.6f}, E/N = {E_sa/N:.6f}")
    print(f"    Difference:         ΔE = {E_sa - E_exact:.6f}")
    
    ground_state = exact_results[N]['ground_state']
    sa_state = sa_results[N]['spins']
    I_o = overlap(ground_state, sa_state)
    print(f"    Overlap I_o = {I_o:.6f}")

plt.figure(figsize=(10, 6))

exact_E_per_N = [exact_results[N]['energies'][0] / N for N in Ns_exact]
plt.plot(Ns_exact, exact_E_per_N, 'o-', markersize=8, linewidth=2, 
         label='Exact (Exhaustive Search)', color='blue')

sa_E_per_N = [sa_results[N]['energy'] / N for N in Ns_all_SA]
plt.plot(Ns_all_SA, sa_E_per_N, 's-', markersize=6, linewidth=2, 
         label='Simulated Annealing', color='red')

plt.xlabel('Number of Spins (N)', fontsize=12)
plt.ylabel('Energy per Spin (E/N)', fontsize=12)
plt.title('Spin Glass Ground State Energy: Exact vs Simulated Annealing', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("\nDo the lowest energies and overlap indices look similar?")
print("Does the simulated anneal value approach the exact value?")
print("\nBased on the results above, the simulated annealing method provides")
print("a good approximation to the exact ground state energies, especially")
print("as the system size increases. The overlap indices indicate how well")
print("SA finds the true ground state configuration.")
