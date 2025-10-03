import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
nu = 2
N = 4 * nu**3
rho = 0.8442
L = (N / rho)**(1.0/3.0)
dt = 0.004
nsteps = 500
T_init = 1.0
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
    for i in range(N):
        for j in range(i+1, N):
            dr = r[i] - r[j]
            dr = minimum_image(dr, L)
            r2 = np.dot(dr, dr)
            if r2 < rcut**2:
                inv_r2 = 1.0 / r2
                inv_r6 = inv_r2**3
                inv_r12 = inv_r6**2
                Uij = 4 * (inv_r12 - inv_r6)
                U += Uij
                Fij = 48 * inv_r2 * (inv_r12 - 0.5 * inv_r6) * dr
                forces[i] += Fij
                forces[j] -= Fij
    return forces, U

def velocity_verlet(r, v, L, dt, nsteps):
    KE_list, PE_list, TE_list = [], [], []
    F, U = compute_forces(r, L)
    
    for step in range(nsteps):
        r += v * dt + 0.5 * F * dt**2
        r = apply_pbc(r, L)
        
        F_new, U = compute_forces(r, L)
        v += 0.5 * (F + F_new) * dt
        F = F_new
        
        KE = 0.5 * np.sum(v**2)
        PE = U
        TE = KE + PE
        
        KE_list.append(KE)
        PE_list.append(PE)
        TE_list.append(TE)
    
    return r, v, KE_list, PE_list, TE_list

r = create_fcc(nu, L)
v = initialize_velocities(N, T_init)

# Plot initial positions
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(r[:,0], r[:,1], r[:,2], s=40, alpha=0.7)
ax.set_xlim(0,L)
ax.set_ylim(0,L)
ax.set_zlim(0,L)
ax.set_title("Initial Positions of Atoms")
plt.show()

# Run simulation
r, v, KE_list, PE_list, TE_list = velocity_verlet(r, v, L, dt, nsteps)

# Plot energies
plt.figure(figsize=(8,6))
plt.plot(KE_list, label='Kinetic Energy')
plt.plot(PE_list, label='Potential Energy')
plt.plot(TE_list, label='Total Energy')
plt.xlabel('Timestep')
plt.ylabel('Energy (reduced units)')
plt.legend()
plt.title('Energy vs Time')
plt.show()
