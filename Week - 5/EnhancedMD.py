import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

R = 4.0  
N = 200  
dt = 0.02  
nt = 20 

mass = 9.109e-31 
charge = -1.602e-19 
k_coulomb = 8.99e9 

positions = np.zeros((N, 3))
min_distance = R * 0.01 
max_attempts = 1000

print("Initializing particle positions...")
for i in range(N):
    attempts = 0
    while attempts < max_attempts:
        # Generate random position in sphere
        phi = np.random.uniform(0, 2*np.pi)
        costheta = np.random.uniform(-1, 1)
        u = np.random.uniform(0, 1)
        
        theta = np.arccos(costheta)
        r = R * (u**(1/3))  # Correct radial distribution for sphere
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        new_pos = np.array([x, y, z])
        
        # Check minimum distance constraint
        too_close = False
        for j in range(i):
            distance = np.linalg.norm(new_pos - positions[j])
            if distance < min_distance:
                too_close = True
                break
        
        if not too_close:
            positions[i] = new_pos
            break
        
        attempts += 1
    
    if attempts >= max_attempts:
        print(f"Warning: Could not place particle {i} with minimum distance constraint")
        positions[i] = new_pos

initial_positions = positions.copy()

com_pos = np.mean(positions, axis=0)
positions -= com_pos

velocities = np.zeros((N, 3))

def compute_forces(pos):
    """Compute Coulomb forces between all particles"""
    forces = np.zeros((N, 3))
    min_dist = R * 1e-6  
    
    for i in range(N):
        for j in range(i+1, N):
            r_vec = pos[i] - pos[j]
            r_mag = np.linalg.norm(r_vec)
            
            if r_mag < min_dist:
                r_mag = min_dist
            
            # Repulsive force
            force_mag = k_coulomb * charge**2 / r_mag**2
            r_unit = r_vec / r_mag
            force_vec = force_mag * r_unit
            
            forces[i] += force_vec
            forces[j] -= force_vec
    
    return forces

def compute_radial_quantities(pos, vel):
    """Compute radial positions, velocities, and forces"""
    radial_pos = np.linalg.norm(pos, axis=1)
    
    # Radial velocity: v_r = (r · v) / |r|
    radial_vel = np.zeros(N)
    for i in range(N):
        if radial_pos[i] > 1e-12:
            radial_vel[i] = np.dot(pos[i], vel[i]) / radial_pos[i]
    
    return radial_pos, radial_vel

position_history = np.zeros((nt+1, N, 3))
velocity_history = np.zeros((nt+1, N, 3))
energy_history = {'kinetic': [], 'potential': [], 'total': []}
radial_history = {'positions': [], 'velocities': [], 'avg_radius': []}

position_history[0] = positions.copy()
velocity_history[0] = velocities.copy()

print("Starting MD simulation...")

forces = compute_forces(positions)
accelerations = forces / mass

for step in range(nt):
    positions = positions + velocities * dt + 0.5 * accelerations * dt**2
    
    com_pos = np.mean(positions, axis=0)
    positions -= com_pos
    
    new_forces = compute_forces(positions)
    new_accelerations = new_forces / mass
    
    velocities = velocities + 0.5 * (accelerations + new_accelerations) * dt
    
    total_momentum = np.sum(velocities, axis=0)
    velocities -= total_momentum / N
    
    accelerations = new_accelerations
    
    position_history[step+1] = positions.copy()
    velocity_history[step+1] = velocities.copy()
    
    kinetic_energy = 0.5 * mass * np.sum(velocities**2)
    
    potential_energy = 0
    for i in range(N):
        for j in range(i+1, N):
            r_mag = np.linalg.norm(positions[i] - positions[j])
            if r_mag > 1e-12:
                potential_energy += k_coulomb * charge**2 / r_mag
    
    total_energy = kinetic_energy + potential_energy
    
    energy_history['kinetic'].append(kinetic_energy)
    energy_history['potential'].append(potential_energy)
    energy_history['total'].append(total_energy)
    
    radial_pos, radial_vel = compute_radial_quantities(positions, velocities)
    radial_history['positions'].append(radial_pos.copy())
    radial_history['velocities'].append(radial_vel.copy())
    radial_history['avg_radius'].append(np.mean(radial_pos))
    
    if step % 5 == 0:
        print(f"Step {step+1}/{nt} completed")

print("Simulation completed. Creating plots...")

fig = plt.figure(figsize=(20, 18))

# 3D scatter plot of initial and final positions
ax1 = fig.add_subplot(3, 3, 1, projection='3d')
ax1.scatter(initial_positions[:, 0], initial_positions[:, 1], initial_positions[:, 2], 
           c='blue', alpha=0.6, s=30, label='Initial')
ax1.scatter(position_history[-1, :, 0], position_history[-1, :, 1], position_history[-1, :, 2], 
           c='red', alpha=0.6, s=30, label='Final')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Initial vs Final Positions')
ax1.legend()

# Energy vs time step
ax2 = fig.add_subplot(3, 3, 2)
time_steps = range(nt)
ax2.plot(time_steps, energy_history['kinetic'], label='Kinetic Energy', linewidth=2)
ax2.plot(time_steps, energy_history['potential'], label='Potential Energy', linewidth=2)
ax2.plot(time_steps, energy_history['total'], label='Total Energy', linewidth=2)
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Energy (J)')
ax2.set_title('Energy Conservation')
ax2.legend()
ax2.grid(True)

# Histogram of particle positions vs r at different time steps
ax3 = fig.add_subplot(3, 3, 3)
plot_steps = [5, 10, 15, min(19, nt-1)]  # Adjust based on available steps
colors = ['blue', 'green', 'orange', 'red']
for i, step in enumerate(plot_steps):
    if step < len(radial_history['positions']):
        r_values = radial_history['positions'][step]
        ax3.hist(r_values, bins=20, alpha=0.7, color=colors[i], 
                label=f'Step {step+1}', density=True)
ax3.set_xlabel('Radial Distance r')
ax3.set_ylabel('Probability Density')
ax3.set_title('Radial Distribution at Different Times')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Radial force scatter plot (at intermediate time)
ax4 = fig.add_subplot(3, 3, 4)
mid_step = min(10, len(radial_history['positions'])-1)
if mid_step < len(radial_history['positions']):
    r_values = radial_history['positions'][mid_step]
    positions_mid = position_history[mid_step+1]
    
    forces_mid = compute_forces(positions_mid)
    radial_forces = np.zeros(N)
    
    for i in range(N):
        r_mag = np.linalg.norm(positions_mid[i])
        if r_mag > 1e-12:
            radial_forces[i] = np.dot(positions_mid[i], forces_mid[i]) / r_mag
    
    ax4.scatter(r_values, radial_forces, alpha=0.7)
    ax4.set_xlabel('Radial Distance r')
    ax4.set_ylabel('Radial Force F_r')
    ax4.set_title(f'Radial Force vs Position (Step {mid_step+1})')
    ax4.grid(True, alpha=0.3)

# Radial velocity scatter plot
ax5 = fig.add_subplot(3, 3, 5)
if mid_step < len(radial_history['positions']):
    r_values = radial_history['positions'][mid_step]
    vr_values = radial_history['velocities'][mid_step]
    ax5.scatter(r_values, vr_values, alpha=0.7)
    ax5.set_xlabel('Radial Distance r')
    ax5.set_ylabel('Radial Velocity v_r')
    ax5.set_title(f'Radial Velocity vs Position (Step {mid_step+1})')
    ax5.grid(True, alpha=0.3)
    
    correlation = np.corrcoef(r_values, vr_values)[0, 1]
    ax5.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax5.transAxes, verticalalignment='top')

# Time dependence of cloud properties
ax6 = fig.add_subplot(3, 3, 6)
time_array = np.arange(nt) * dt

avg_radii = radial_history['avg_radius']
ax6.plot(time_array, avg_radii, 'b-', linewidth=2, label='Avg Radius')
ax6.set_xlabel('Time')
ax6.set_ylabel('Average Radius')
ax6.set_title('(a) Average Radius of Cloud vs Time')
ax6.grid(True, alpha=0.3)
ax6.legend()

ax7 = fig.add_subplot(3, 3, 7)

radial_velocity_cloud = np.gradient(avg_radii, dt)
ax7.plot(time_array, radial_velocity_cloud, 'g-', linewidth=2, label='Radial Velocity')
ax7.set_xlabel('Time')
ax7.set_ylabel('Radial Velocity of Cloud')
ax7.set_title('(b) Cloud Radial Velocity')
ax7.grid(True, alpha=0.3)
ax7.legend()

ax8 = fig.add_subplot(3, 3, 8)
velocity_fluctuations = []
for step in range(len(radial_history['velocities'])):
    vr_rms = np.sqrt(np.mean(np.array(radial_history['velocities'][step])**2))
    velocity_fluctuations.append(vr_rms)

ax8.plot(time_array, velocity_fluctuations, 'r-', linewidth=2, label='v_r RMS')
ax8.set_xlabel('Time')
ax8.set_ylabel('Velocity Fluctuations v²_r')
ax8.set_title('(c) RMS Radial Velocity Fluctuations')
ax8.grid(True, alpha=0.3)
ax8.legend()

# Combined plot for easy comparison - ALL THREE QUANTITIES ON ONE GRAPH
ax9 = fig.add_subplot(3, 3, 9)

ax9_twin1 = ax9.twinx()
ax9_twin2 = ax9.twinx()

ax9_twin2.spines['right'].set_position(('outward', 60))

line1 = ax9.plot(time_array, avg_radii, 'b-', linewidth=3, label='(a) Avg Radius of Cloud')
line2 = ax9_twin1.plot(time_array, radial_velocity_cloud, 'g-', linewidth=3, label='(b) Radial Velocity of Cloud')
line3 = ax9_twin2.plot(time_array, velocity_fluctuations, 'r-', linewidth=3, label='(c) Velocity Fluctuations v²_r')

ax9.set_xlabel('Time', fontsize=12, fontweight='bold')
ax9.set_ylabel('Average Radius', color='blue', fontsize=11, fontweight='bold')
ax9_twin1.set_ylabel('Radial Velocity', color='green', fontsize=11, fontweight='bold')
ax9_twin2.set_ylabel('Velocity Fluctuations', color='red', fontsize=11, fontweight='bold')

ax9.tick_params(axis='y', labelcolor='blue')
ax9_twin1.tick_params(axis='y', labelcolor='green')
ax9_twin2.tick_params(axis='y', labelcolor='red')

ax9.set_title('(vi) Time Dependence: Three Cloud Quantities', fontsize=12, fontweight='bold')

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax9.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=True, fancybox=True, shadow=True)

ax9.grid(True, alpha=0.3)

plt.tight_layout(pad=5.0)
plt.subplots_adjust(hspace=0.6, wspace=0.4, top=0.93, bottom=0.10)
plt.show()

print("\n=== Simulation Results ===")
print(f"Parameters: R={R}, N={N}, dt={dt}, nt={nt}")
print(f"Initial total energy: {energy_history['total'][0]:.6e} J")
print(f"Final total energy: {energy_history['total'][-1]:.6e} J")
if len(energy_history['total']) > 1:
    energy_change = abs(energy_history['total'][-1] - energy_history['total'][0])
    energy_error = energy_change / abs(energy_history['total'][0]) * 100
    print(f"Energy conservation error: {energy_error:.4f}%")

print(f"Initial average radius: {avg_radii[0]:.6e}")
print(f"Final average radius: {avg_radii[-1]:.6e}")
print(f"Average cloud expansion rate: {np.mean(radial_velocity_cloud):.6e}")

print("\nSimulation completed successfully!")
print("For production runs, change parameters to: N=1000, nt=50")
