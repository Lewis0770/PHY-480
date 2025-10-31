import numpy as np
import matplotlib.pyplot as plt

Lx = 50.0
Nx = 500
dx = Lx / Nx
x = np.linspace(0, Lx, Nx)

dt = 0.0001
n_steps = 50000
plot_steps = [0, 10000, 20000, 30000, 40000]

h0 = np.zeros(Nx)
center = Nx // 2
half_width = int(5 / dx / 2)
h0[center - half_width:center + half_width] = 1.0

def evolve(h, a1, a2):
    h_profiles = []
    h_current = h.copy()
    
    for n in range(n_steps + 1):
        hxx = (np.roll(h_current, -1) - 2 * h_current + np.roll(h_current, 1)) / dx**2
        hxxxx = (np.roll(hxx, -1) - 2 * hxx + np.roll(hxx, 1)) / dx**2
        h_current = h_current + dt * (a1 * hxx - a2 * hxxxx)
        
        if n in plot_steps:
            h_profiles.append(h_current.copy())
            print(f"Step {n}: min={h_current.min():.3f}, max={h_current.max():.3f}")
    
    return h_profiles

print("Simulating non-conserved case (a1=0.1, a2=0)...")
profiles_non = evolve(h0, a1=0.1, a2=0.0)

print("\nSimulating conserved case (a1=0, a2=0.1)...")
profiles_con = evolve(h0, a1=0.0, a2=0.1)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

for i, h in enumerate(profiles_non):
    t_val = plot_steps[i] * dt
    axs[0].plot(x, h, label=f't={t_val:.2f}')
axs[0].set_title('Non-conserved (a1=0.1, a2=0)')
axs[0].set_xlabel('x')
axs[0].set_ylabel('h(x,t)')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

for i, h in enumerate(profiles_con):
    t_val = plot_steps[i] * dt
    axs[1].plot(x, h, label=f't={t_val:.2f}')
axs[1].set_title('Conserved (a1=0, a2=0.1)')
axs[1].set_xlabel('x')
axs[1].set_ylabel('h(x,t)')
axs[1].legend()
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('interconnect_profiles.png', dpi=150)
print("\nPlot saved to interconnect_profiles.png")
plt.show()
