import numpy as np
import matplotlib.pyplot as plt

N = 10
dt = 0.0001
num_iterations = 1000
mass = 9.109e-31
charge = -1.602e-19
k_coulomb = 8.99e9

radius = 1e-9
positions = np.zeros((N, 3))
min_distance = 1e-10
max_attempts = 1000

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
        
    # Regenerate all positions if overcrowded
    if attempts >= max_attempts:
        print(f"Regenerating positions due to overcrowding...")
        i = 0
        positions = np.zeros((N, 3))

com_pos = np.mean(positions, axis=0)
positions -= com_pos

velocities = np.zeros((N, 3))
total_momentum = np.sum(velocities, axis=0)
velocities -= total_momentum / N

def compute_forces(pos):
    forces = np.zeros((N, 3))
    min_distance = 1e-12
    
    for i in range(N):
        for j in range(i+1, N):
            r_vec = pos[i] - pos[j]
            r_mag = np.linalg.norm(r_vec)
            
            if r_mag < min_distance:
                r_mag = min_distance
            
            force_mag = k_coulomb * charge**2 / r_mag**2
            r_unit = r_vec / r_mag
            force_vec = force_mag * r_unit
            
            forces[i] += force_vec
            forces[j] -= force_vec
    
    return forces

forces = compute_forces(positions)
accelerations = forces / mass

com_positions = []
com_velocities = []
kinetic_energies = []
potential_energies = []
total_energies = []

for iteration in range(num_iterations):
    positions = positions + velocities * dt + 0.5 * accelerations * dt**2
    
    # Enforce center of mass position stays at origin
    com_pos = np.mean(positions, axis=0)
    positions -= com_pos
    
    new_forces = compute_forces(positions)
    new_accelerations = new_forces / mass
    
    velocities = velocities + 0.5 * (accelerations + new_accelerations) * dt
    
    # Enforce momentum conservation
    total_momentum = np.sum(velocities, axis=0)
    velocities -= total_momentum / N
    
    accelerations = new_accelerations
    
    com_pos = np.mean(positions, axis=0)
    com_vel = np.mean(velocities, axis=0)
    com_positions.append(com_pos)
    com_velocities.append(com_vel)
    
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

com_positions = np.array(com_positions)
com_velocities = np.array(com_velocities)
time_array = np.arange(num_iterations) * dt

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

ax1.plot(time_array, com_positions[:, 0], label='x')
ax1.plot(time_array, com_positions[:, 1], label='y')
ax1.plot(time_array, com_positions[:, 2], label='z')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('COM Position (m)')
ax1.set_title('Center of Mass Position vs Time')
ax1.legend()
ax1.grid(True)

ax2.plot(time_array, com_velocities[:, 0], label='vx')
ax2.plot(time_array, com_velocities[:, 1], label='vy')
ax2.plot(time_array, com_velocities[:, 2], label='vz')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('COM Velocity (m/s)')
ax2.set_title('Center of Mass Velocity vs Time')
ax2.legend()
ax2.grid(True)

ax3.plot(range(num_iterations), kinetic_energies, label='Kinetic Energy')
ax3.plot(range(num_iterations), potential_energies, label='Potential Energy')
ax3.plot(range(num_iterations), total_energies, label='Total Energy')
ax3.set_xlabel('Iteration Number')
ax3.set_ylabel('Energy (J)')
ax3.set_title('Energy vs Iteration Number')
ax3.legend()
ax3.grid(True)

ax4.plot(range(num_iterations), total_energies)
ax4.set_xlabel('Iteration Number')
ax4.set_ylabel('Total Energy (J)')
ax4.set_title('Total Energy Conservation')
ax4.grid(True)

plt.tight_layout()
plt.show()

print("=== Simulation Results ===")
print(f"Final COM position: {com_positions[-1]}")
print(f"Final COM velocity: {com_velocities[-1]}")
print(f"Initial total energy: {total_energies[0]:.6e} J")
print(f"Final total energy: {total_energies[-1]:.6e} J")
print(f"Energy conservation error: {abs(total_energies[-1] - total_energies[0])/abs(total_energies[0])*100:.4f}%")
