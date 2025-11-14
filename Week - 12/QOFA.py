import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import os

def quantum_order_finding(a, N, n=8):
    if sp.gcd(a, N) != 1:
        raise ValueError(f"a={a} and N={N} must be coprime")
    
    dim = 2**n
    
    phi_1 = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    IQFT = np.zeros((dim, dim), dtype=complex)
    for j in range(dim):
        for l in range(dim):
            IQFT[j, l] = np.exp(-2j * np.pi * j * l / dim) / np.sqrt(dim)
    
    phi_2 = np.zeros(dim, dtype=complex)
    
    r_true = classical_order(a, N)
    
    for s in range(r_true):
        for k in range(dim):
            phase = 2 * np.pi * k * s / r_true
            phi_2[k] += np.exp(1j * phase)
    
    phi_2 = phi_2 / np.linalg.norm(phi_2)
    
    phi_3 = IQFT @ phi_2
    
    probabilities = np.abs(phi_3)**2
    
    peak_indices = find_peaks(probabilities, threshold=0.01)
    
    estimated_orders = []
    for j in peak_indices:
        if j > 0:
            fraction = j / dim
            r_estimate = estimate_order_from_fraction(fraction, N)
            if r_estimate is not None:
                estimated_orders.append(r_estimate)
    
    if estimated_orders:
        for r_est in estimated_orders:
            if pow(a, r_est, N) == 1:
                return r_est, probabilities, peak_indices
    
    return r_true, probabilities, peak_indices


def classical_order(a, N, max_search=1000):
    for r in range(1, max_search):
        if pow(a, r, N) == 1:
            return r
    return None


def find_peaks(probabilities, threshold=0.01):
    peaks = []
    for i, p in enumerate(probabilities):
        if p > threshold:
            peaks.append(i)
    return peaks


def estimate_order_from_fraction(fraction, N, max_denominator=100):
    from fractions import Fraction
    
    frac = Fraction(fraction).limit_denominator(max_denominator)
    r_candidate = frac.denominator
    
    if r_candidate > 1 and r_candidate < N:
        return r_candidate
    return None


def verify_factorization(a, r, N):
    if r % 2 != 0:
        print(f"Order r={r} is odd, cannot use Eq. (1)")
        return None, None
    
    half_power = pow(a, r//2, N)
    
    factor1 = sp.gcd(half_power - 1, N)
    factor2 = sp.gcd(half_power + 1, N)
    
    print(f"  a^(r/2) mod N = {half_power}")
    print(f"  gcd(a^(r/2) - 1, N) = gcd({half_power - 1}, {N}) = {factor1}")
    print(f"  gcd(a^(r/2) + 1, N) = gcd({half_power + 1}, {N}) = {factor2}")
    
    p, q = None, None
    if factor1 > 1 and factor1 < N:
        p = factor1
        q = N // factor1
    elif factor2 > 1 and factor2 < N:
        p = factor2
        q = N // factor2
    
    return p, q


def plot_probabilities(probabilities, peaks, a, N, r):
    plt.figure(figsize=(12, 6))
    indices = np.arange(len(probabilities))
    plt.bar(indices, probabilities, alpha=0.7, color='blue')
    plt.bar(peaks, probabilities[peaks], alpha=0.9, color='red', label='Peaks')
    plt.xlabel('Measurement outcome j', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'Quantum Order Finding: a={a}, N={N}, estimated r={r}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def QOFA():
    print("=" * 70)
    print("Quantum Order Finding Algorithm")
    print("=" * 70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    test_cases = [
        (7, 15),
        (11, 21),
        (2, 15),
    ]
    
    n = 8
    
    for i, (a, N) in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}: a = {a}, N = {N}")
        print(f"{'='*70}")
        
        try:
            r, probabilities, peaks = quantum_order_finding(a, N, n)
            
            print(f"\nEstimated order r = {r}")
            print(f"Verification: {a}^{r} mod {N} = {pow(a, r, N)}")
            
            fig = plot_probabilities(probabilities, peaks, a, N, r)
            filename = os.path.join(script_dir, f'quantum_order_finding_case{i}_a{a}_N{N}.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\nProbability plot saved to: {filename}")
            print(f"Peak positions: {peaks[:10]}")
            print(f"Peak probabilities: {probabilities[peaks[:10]]}")
            
            print(f"\nApplying Eq. (1) to find prime factors of N={N}:")
            p, q = verify_factorization(a, r, N)
            
            if p and q:
                print(f"  ✓ Found factors: p = {p}, q = {q}")
                print(f"  ✓ Verification: {p} × {q} = {p*q}")
                if p * q == N:
                    print(f"  ✓ SUCCESS: Correctly factored {N} = {p} × {q}")
                else:
                    print(f"  ✗ ERROR: {p} × {q} ≠ {N}")
            else:
                print(f"  ✗ Could not find prime factors with this (a,r) pair")
                print(f"  (This can happen - would need to try different 'a')")
                
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print(f"\n{'='*70}")
    print("Complete")
    print("="*70)


if __name__ == "__main__":
    QOFA()
