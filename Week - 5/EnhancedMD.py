import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 200
dt = 0.02
num_iterations = 20
mass = 9.109e-31
charge = -1.602e-19
k_coulomb = 8.99e9

radius = 4.0
positions = np.zeros((N, 3))
min_distance = 0.1
max_attempts = 1000

# Generate initial positions within sphere
for i in range(N):
    attempts = 0
    while attempts < max_attempts:
        x = np.random.uniform(-radius, radius)
        y = np.random.uniform(-radius, radius)
        z = np.random.uniform(-radius, radius)
        
        if x**2 + y**2 + z**2 <= radius**2:
            new_pos = np.array([x, y, z])
            
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
        print(f"Regenerating positions due to overcrowding...")
        i = 0
        positions = np.zeros((N, 3))

com_pos = np.mean(positions, axis=0)
positions -= com_pos

velocities = np.zeros((N, 3))

def compute_forces(pos):
    forces = np.zeros((N, 3))
    min_dist = 1e-3
    
    for i in range(N):
        for j in range(i+1, N):
            r_vec = pos[i] - pos[j]
            r_mag = np.linalg.norm(r_vec)
            
            if r_mag < min_dist:
                r_mag = min_dist
            
            force_mag = k_coulomb * charge**2 / r_mag**2
            r_unit = r_vec / r_mag
            force_vec = force_mag * r_unit
            
            forces[i] += force_vec
            forces[j] -= force_vec
    
    return forces

# Storage arrays
all_positions = []
all_velocities = []
all_forces = []
kinetic_energies = []
potential_energies = []
total_energies = []
cloud_radius = []
radial_velocities = []
velocity_fluctuations = []

forces = compute_forces(positions)
accelerations = forces / mass

print("Starting MD simulation...")

# Main simulation loop
for iteration in range(num_iterations):
    all_positions.append(positions.copy())
    all_velocities.append(velocities.copy())
    all_forces.append(forces.copy())
    
    positions = positions + velocities * dt + 0.5 * accelerations * dt**2
    
    com_pos = np.mean(positions, axis=0)
    positions -= com_pos
    
    new_forces = compute_forces(positions)
    new_accelerations = new_forces / mass
    
    velocities = velocities + 0.5 * (accelerations + new_accelerations) * dt
    
    total_momentum = np.sum(velocities, axis=0)
    velocities -= total_momentum / N
    
    accelerations = new_accelerations
    
    # Calculate energies
    kinetic_energy = 0.5 * mass * np.sum(velocities**2)
    
    potential_energy = 0
    for i in range(N):
        for j in range(i+1, N):
            r_mag = np.linalg.norm(positions[i] - positions[j])
            if r_mag > 1e-12:
                potential_energy += k_coulomb * charge**2 / r_mag
    
    total_energy = kinetic_energy + potential_energy
    
    kinetic_energies.append(kinetic_energy)
    potential_energies.append(potential_energy)
    total_energies.append(total_energy)
    
    distances_from_com = np.linalg.norm(positions, axis=1)
    rms_radius = np.sqrt(np.mean(distances_from_com**2))
    cloud_radius.append(rms_radius)
    
    radial_vel = []
    for i in range(N):
        r_vec = positions[i]
        v_vec = velocities[i]
        r_mag = np.linalg.norm(r_vec)
        if r_mag > 1e-12:
            r_unit = r_vec / r_mag
            v_radial = np.dot(v_vec, r_unit)
            radial_vel.append(v_radial)
    
    if radial_vel:
        avg_radial_vel = np.mean(radial_vel)
        radial_velocities.append(avg_radial_vel)
    else:
        radial_velocities.append(0)
    
    v_magnitudes = np.linalg.norm(velocities, axis=1)
    v_fluctuation = np.std(v_magnitudes)
    velocity_fluctuations.append(v_fluctuation)
    
    if iteration % 5 == 0:
        print(f"Iteration {iteration}/{num_iterations}")

print("Simulation completed.")

# Convert to arrays
all_positions = np.array(all_positions)
all_velocities = np.array(all_velocities)
all_forces = np.array(all_forces)
time_array = np.arange(num_iterations) * dt

# Create plots
fig = plt.figure(figsize=(24, 18))
fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.07, hspace=0.4, wspace=0.3)

# (i) 3D scatter plot
ax1 = fig.add_subplot(3, 4, 1, projection='3d')
ax1.scatter(all_positions[0, :, 0], all_positions[0, :, 1], all_positions[0, :, 2], 
           c='blue', alpha=0.6, s=20, label='Initial')
ax1.scatter(all_positions[-1, :, 0], all_positions[-1, :, 1], all_positions[-1, :, 2], 
           c='red', alpha=0.6, s=20, label='Final')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title('Initial vs Final Positions')
ax1.legend()

# (ii) Energy plot
ax2 = fig.add_subplot(3, 4, 2)
ax2.plot(time_array, kinetic_energies, label='Kinetic Energy', color='blue')
ax2.plot(time_array, potential_energies, label='Potential Energy', color='red')
ax2.plot(time_array, total_energies, label='Total Energy', color='green')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Energy (J)')
ax2.set_title('Energy vs Time')
ax2.legend()
ax2.grid(True)

# (iii) Histogram of positions
ax3 = fig.add_subplot(3, 4, 3)
times_to_plot = [0, len(all_positions)//4, len(all_positions)//2, -1]
time_labels = ['t=0', 't=T/4', 't=T/2', 't=T']
colors = ['blue', 'green', 'orange', 'red']

for i, (time_idx, label, color) in enumerate(zip(times_to_plot, time_labels, colors)):
    distances = np.linalg.norm(all_positions[time_idx], axis=1)
    ax3.hist(distances, bins=15, alpha=0.6, label=label, color=color, density=True)

ax3.set_xlabel('Distance from COM (m)')
ax3.set_ylabel('Probability Density')
ax3.set_title('Position Distribution vs Time')
ax3.legend()
ax3.grid(True)

# (iv) Force analysis
ax4 = fig.add_subplot(3, 4, 4)
r_sample = np.linspace(0.1, 5.0, 100)
force_sample = k_coulomb * charge**2 / r_sample**2
ax4.plot(r_sample, force_sample, 'k-', linewidth=2, label='Theoretical F=kq²/r²')

if len(all_forces) > 10:
    mid_time_idx = min(10, len(all_forces)-1)
    distances_sim = []
    forces_sim = []
    
    for i in range(N):
        r_vec = all_positions[mid_time_idx, i]
        f_vec = all_forces[mid_time_idx, i]
        r_mag = np.linalg.norm(r_vec)
        f_mag = np.linalg.norm(f_vec)
        if r_mag > 0.01 and f_mag > 0:
            distances_sim.append(r_mag)
            forces_sim.append(f_mag)
    
    if distances_sim:
        ax4.scatter(distances_sim, forces_sim, alpha=0.6, s=20, c='red', 
                   label=f'Simulation data (t={mid_time_idx*dt:.3f}s)')

ax4.set_xlabel('Distance (m)')
ax4.set_ylabel('Force Magnitude (N)')
ax4.set_title('Force vs Distance: Theory vs Simulation')
ax4.legend()
ax4.grid(True)
ax4.set_yscale('log')
ax4.set_xscale('log')

# (v) Radial velocity scatter
ax5 = fig.add_subplot(3, 4, 5)
final_distances = np.linalg.norm(all_positions[-1], axis=1)
final_radial_vels = []
for i in range(N):
    r_vec = all_positions[-1, i]
    v_vec = all_velocities[-1, i]
    r_mag = np.linalg.norm(r_vec)
    if r_mag > 1e-12:
        r_unit = r_vec / r_mag
        v_radial = np.dot(v_vec, r_unit)
        final_radial_vels.append(v_radial)
    else:
        final_radial_vels.append(0)

ax5.scatter(final_distances, final_radial_vels, alpha=0.6)
ax5.set_xlabel('Distance from COM (m)')
ax5.set_ylabel('Radial Velocity (m/s)')
ax5.set_title('Radial Velocity vs Distance')
ax5.grid(True)

# (vi) Time dependence plots
ax6 = fig.add_subplot(3, 4, 6)
ax6.plot(time_array, cloud_radius, 'b-', linewidth=2, label='Cloud Radius')
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('RMS Radius (m)')
ax6.set_title('Cloud Radius vs Time')
ax6.grid(True)
ax6.legend()

ax7 = fig.add_subplot(3, 4, 7)
ax7.plot(time_array, radial_velocities, 'r-', linewidth=2, label='Avg Radial Velocity')
ax7.set_xlabel('Time (s)')
ax7.set_ylabel('Radial Velocity (m/s)')
ax7.set_title('Average Radial Velocity vs Time')
ax7.grid(True)
ax7.legend()

ax8 = fig.add_subplot(3, 4, 8)
ax8.plot(time_array, velocity_fluctuations, 'g-', linewidth=2, label='Velocity Fluctuations')
ax8.set_xlabel('Time (s)')
ax8.set_ylabel('Velocity Std Dev (m/s)')
ax8.set_title('Velocity Fluctuations vs Time')
ax8.grid(True)
ax8.legend()

# Additional plots
ax9 = fig.add_subplot(3, 4, 9)
# COM position drift
com_positions = np.array([np.mean(pos, axis=0) for pos in all_positions])
ax9.plot(time_array, np.linalg.norm(com_positions, axis=1), 'k-', linewidth=2)
ax9.set_xlabel('Time (s)')
ax9.set_ylabel('COM Displacement (m)')
ax9.set_title('Center of Mass Drift')
ax9.grid(True)

ax10 = fig.add_subplot(3, 4, 10)
energy_error = (np.array(total_energies) - total_energies[0]) / abs(total_energies[0]) * 100
ax10.plot(time_array, energy_error, 'r-', linewidth=2)
ax10.set_xlabel('Time (s)')
ax10.set_ylabel('Energy Error (%)')
ax10.set_title('Energy Conservation Error')
ax10.grid(True)

ax11 = fig.add_subplot(3, 4, 11)
if len(cloud_radius) > 1:
    expansion_rate = np.diff(cloud_radius) / dt
    ax11.plot(time_array[1:], expansion_rate, 'purple', linewidth=2)
ax11.set_xlabel('Time (s)')
ax11.set_ylabel('dR/dt (m/s)')
ax11.set_title('Cloud Expansion Rate')
ax11.grid(True)

ax12 = fig.add_subplot(3, 4, 12)
temperature_like = np.array(kinetic_energies) / (1.5 * N * 1.38e-23)
ax12.plot(time_array, temperature_like, 'orange', linewidth=2)
ax12.set_xlabel('Time (s)')
ax12.set_ylabel('Temperature-like (K)')
ax12.set_title('Effective Temperature')
ax12.grid(True)

plt.show()

print(f"Initial total energy: {total_energies[0]:.6e} J")
print(f"Final total energy: {total_energies[-1]:.6e} J")
print(f"Initial radius: {cloud_radius[0]:.3f} m")
print(f"Final radius: {cloud_radius[-1]:.3f} m")
