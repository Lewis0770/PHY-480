import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
nu = 2
N = 4 * nu**3
rho = 0.8442
L = (N / rho)**(1.0/3.0)
dt = 0.004
nsteps_equil = 2000  # Equilibration steps
nsteps_prod = 3000   # Production steps
epsilon = 1.0
sigma = 1.0
rcut = 2.5 * sigma

def create_fcc(nu, L):
    a = L / nu
    positions = []
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]])
    for x in range(nu):
        for y in range(nu):
            for z in range(nu):
                for b in basis:
                    positions.append((np.array([x, y, z]) + b) * a)
    return np.array(positions)

def initialize_velocities(N, T):
    v = np.random.normal(0.0, np.sqrt(T), (N, 3))
    v -= v.mean(axis=0)
    return v

def apply_pbc(r, L):
    return r % L

def minimum_image(dr, L):
    dr -= L * np.round(dr / L)
    return dr

def compute_forces(r, L):
    forces = np.zeros_like(r)
    U = 0.0
    virial = 0.0
    
    for i in range(N):
        for j in range(i+1, N):
            dr = r[i] - r[j]
            dr = minimum_image(dr, L)
            r2 = np.dot(dr, dr)
            if r2 < rcut**2:
                inv_r2 = 1.0 / r2
                inv_r6 = inv_r2**3
                inv_r12 = inv_r6**2
                Uij = 4 * epsilon * (inv_r12 - inv_r6)
                U += Uij
                Fij = 48 * epsilon * inv_r2 * (inv_r12 - 0.5 * inv_r6) * dr
                forces[i] += Fij
                forces[j] -= Fij
                virial += np.dot(dr, Fij)
    
    return forces, U, virial

def velocity_rescale(v, T_target):
    """Rescale velocities to achieve target temperature"""
    KE_current = 0.5 * np.sum(v**2)
    T_current = 2 * KE_current / (3 * N)
    if T_current > 0:
        scale = np.sqrt(T_target / T_current)
        v *= scale
    return v

def compute_pair_correlation(r, L, dr=0.1, rmax=None):
    """Compute pair correlation function g(r)"""
    if rmax is None:
        rmax = L / 2
    
    nbins = int(rmax / dr)
    hist = np.zeros(nbins)
    
    for i in range(N):
        for j in range(i+1, N):
            dr_vec = r[i] - r[j]
            dr_vec = minimum_image(dr_vec, L)
            dist = np.linalg.norm(dr_vec)
            if dist < rmax:
                bin_idx = int(dist / dr)
                if bin_idx < nbins:
                    hist[bin_idx] += 2  # Count both i-j and j-i
    
    # Normalize
    r_vals = np.arange(nbins) * dr + dr/2
    volume_shell = 4 * np.pi * r_vals**2 * dr
    number_density = N / L**3
    g_r = hist / (N * volume_shell * number_density)
    
    return r_vals, g_r

def run_simulation_at_temperature(T_target, nsteps_equil, nsteps_prod, rescale_freq=10):
    """Run simulation at specified temperature"""
    print(f"\nRunning simulation at T = {T_target:.3f}")
    
    # Initialize
    r = create_fcc(nu, L)
    v = initialize_velocities(N, T_target)
    
    # Storage for production run
    KE_list, PE_list, TE_list, T_list, P_list = [], [], [], [], []
    r_init = r.copy()
    msd_list = []
    
    # Equilibration phase
    F, U, virial = compute_forces(r, L)
    for step in range(nsteps_equil):
        r += v * dt + 0.5 * F * dt**2
        r = apply_pbc(r, L)
        
        F_new, U, virial = compute_forces(r, L)
        v += 0.5 * (F + F_new) * dt
        F = F_new
        
        # Rescale velocities periodically
        if step % rescale_freq == 0:
            v = velocity_rescale(v, T_target)
    
    print(f"  Equilibration complete")
    
    # Save initial positions for MSD calculation
    r_init = r.copy()
    
    # Production phase
    F, U, virial = compute_forces(r, L)
    for step in range(nsteps_prod):
        r += v * dt + 0.5 * F * dt**2
        r = apply_pbc(r, L)
        
        F_new, U, virial = compute_forces(r, L)
        v += 0.5 * (F + F_new) * dt
        F = F_new
        
        # Calculate properties
        KE = 0.5 * np.sum(v**2)
        PE = U
        TE = KE + PE
        T_inst = 2 * KE / (3 * N)
        
        # Pressure using virial theorem
        P = N * T_inst / L**3 + virial / (3 * L**3)
        
        # MSD calculation (accounting for PBC)
        dr_total = r - r_init
        msd = np.mean(np.sum(dr_total**2, axis=1))
        
        KE_list.append(KE)
        PE_list.append(PE)
        TE_list.append(TE)
        T_list.append(T_inst)
        P_list.append(P)
        msd_list.append(msd)
    
    print(f"  Production complete")
    
    # Compute pair correlation function
    r_vals, g_r = compute_pair_correlation(r, L)
    
    # Compute heat capacity using fluctuation formula
    KE_mean = np.mean(KE_list)
    KE2_mean = np.mean(np.array(KE_list)**2)
    KE_fluct = (KE2_mean - KE_mean**2) / KE_mean**2
    Cv = (2.0 / (3*N)) * (1 - (3*N) / (2) * KE_fluct)
    
    results = {
        'T_target': T_target,
        'KE': KE_list,
        'PE': PE_list,
        'TE': TE_list,
        'T': T_list,
        'P': P_list,
        'MSD': msd_list,
        'r_vals': r_vals,
        'g_r': g_r,
        'Cv': Cv,
        'T_avg': np.mean(T_list),
        'E_avg': np.mean(TE_list),
        'P_avg': np.mean(P_list)
    }
    
    return results

# Run simulations at multiple temperatures
temperatures = [0.5, 0.8, 1.0, 1.2]
all_results = []

for T in temperatures:
    results = run_simulation_at_temperature(T, nsteps_equil, nsteps_prod)
    all_results.append(results)

# Create comprehensive plots
fig = plt.figure(figsize=(16, 12))

# 1. Total Energy vs Temperature
ax1 = fig.add_subplot(3, 3, 1)
T_targets = [r['T_avg'] for r in all_results]
E_avgs = [r['E_avg'] for r in all_results]
ax1.plot(T_targets, E_avgs, 'o-', linewidth=2, markersize=8)
ax1.set_xlabel('Temperature (reduced units)', fontsize=10)
ax1.set_ylabel('Total Energy (reduced units)', fontsize=10)
ax1.set_title('(i) Total Energy vs Temperature', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Pressure vs Temperature
ax2 = fig.add_subplot(3, 3, 2)
P_avgs = [r['P_avg'] for r in all_results]
ax2.plot(T_targets, P_avgs, 's-', linewidth=2, markersize=8, color='orange')
ax2.set_xlabel('Temperature (reduced units)', fontsize=10)
ax2.set_ylabel('Pressure (reduced units)', fontsize=10)
ax2.set_title('(ii) Pressure vs Temperature', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Specific Heat vs Temperature
ax3 = fig.add_subplot(3, 3, 3)
Cv_vals = [r['Cv'] for r in all_results]
ax3.plot(T_targets, Cv_vals, '^-', linewidth=2, markersize=8, color='green')
ax3.set_xlabel('Temperature (reduced units)', fontsize=10)
ax3.set_ylabel('Cv (reduced units)', fontsize=10)
ax3.set_title('(iii) Heat Capacity vs Temperature', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. MSD vs Time for different temperatures
ax4 = fig.add_subplot(3, 3, 4)
time_vals = np.arange(nsteps_prod) * dt
for results in all_results:
    ax4.plot(time_vals, results['MSD'], label=f"T={results['T_target']:.2f}", linewidth=2)
ax4.set_xlabel('Time (reduced units)', fontsize=10)
ax4.set_ylabel('MSD (reduced units)', fontsize=10)
ax4.set_title('(iv) Mean Square Displacement vs Time', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. Pair correlation at low temperature
ax5 = fig.add_subplot(3, 3, 5)
ax5.plot(all_results[0]['r_vals'], all_results[0]['g_r'], linewidth=2)
ax5.set_xlabel('r (reduced units)', fontsize=10)
ax5.set_ylabel('g(r)', fontsize=10)
ax5.set_title(f"(v) Pair Correlation at T={all_results[0]['T_target']:.2f}", fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, 5)

# 6. Pair correlation at high temperature
ax6 = fig.add_subplot(3, 3, 6)
ax6.plot(all_results[-1]['r_vals'], all_results[-1]['g_r'], linewidth=2, color='red')
ax6.set_xlabel('r (reduced units)', fontsize=10)
ax6.set_ylabel('g(r)', fontsize=10)
ax6.set_title(f"(v) Pair Correlation at T={all_results[-1]['T_target']:.2f}", fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0, 5)

# 7. Energy conservation check
ax7 = fig.add_subplot(3, 3, 7)
results_check = all_results[1]  # Use middle temperature
ax7.plot(results_check['TE'], linewidth=1, alpha=0.7)
ax7.set_xlabel('Timestep', fontsize=10)
ax7.set_ylabel('Total Energy', fontsize=10)
ax7.set_title('Energy Conservation Check', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3)

# 8. Temperature evolution
ax8 = fig.add_subplot(3, 3, 8)
for results in all_results:
    ax8.plot(results['T'], label=f"T={results['T_target']:.2f}", linewidth=1, alpha=0.7)
ax8.set_xlabel('Timestep', fontsize=10)
ax8.set_ylabel('Instantaneous Temperature', fontsize=10)
ax8.set_title('Temperature Evolution', fontsize=11, fontweight='bold')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)

# 9. Pressure evolution
ax9 = fig.add_subplot(3, 3, 9)
for results in all_results:
    ax9.plot(results['P'], label=f"T={results['T_target']:.2f}", linewidth=1, alpha=0.7)
ax9.set_xlabel('Timestep', fontsize=10)
ax9.set_ylabel('Instantaneous Pressure', fontsize=10)
ax9.set_title('Pressure Evolution', fontsize=11, fontweight='bold')
ax9.legend(fontsize=9)
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('md_analysis_results.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)
for results in all_results:
    print(f"\nTemperature: {results['T_target']:.3f}")
    print(f"  Average T: {results['T_avg']:.4f}")
    print(f"  Average E: {results['E_avg']:.4f}")
    print(f"  Average P: {results['P_avg']:.4f}")
    print(f"  Cv: {results['Cv']:.4f}")
    print(f"  Final MSD: {results['MSD'][-1]:.4f}")
print("="*60)
