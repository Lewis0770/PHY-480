
import numpy as np
import matplotlib.pyplot as plt


N = 50
J = 1.0
steps = 50000
temps = np.linspace(1.5, 2.5, 10)  # Focus around critical temperature Tc = 2J = 2.0
warmup = 10000  
h = 0.0  


def delta_energy(spins, i, J, N, total_spin):
    """
    Compute change in energy if spin[i] is flipped
    for the complete graph Ising model with Jij = -2J/N.
    """
    dE = -(4.0 * J / N) * spins[i] * (total_spin - spins[i])
    return dE

avg_magnetizations = []

time_series_data = {}
selected_temps = [1.6, 2.0, 2.4]  # Below, at, and above Tc

for T in temps:
    spins = np.random.choice([-1, 1], size=N)
    total_spin = np.sum(spins)  
    magnetizations = []
    
    for step in range(steps):
        i = np.random.randint(0, N)
        
        dE = delta_energy(spins, i, J, N, total_spin)
        
        # Metropolis acceptance rule
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            total_spin -= 2 * spins[i]  
            spins[i] *= -1  
        
        M = total_spin / N
        magnetizations.append(M)
    
    # Store time series for selected temperatures
    if any(abs(T - temp) < 0.05 for temp in selected_temps):
        if abs(T - 1.6) < 0.05:
            time_series_data['low'] = (T, magnetizations.copy())
        elif abs(T - 2.0) < 0.05:
            time_series_data['critical'] = (T, magnetizations.copy())
        elif abs(T - 2.4) < 0.05:
            time_series_data['high'] = (T, magnetizations.copy())
    
    equilibrium_data = magnetizations[warmup:]
    avg_magnetizations.append(np.mean(np.abs(equilibrium_data)))


plt.figure(figsize=(12, 6))
for key, (temp, mag_data) in time_series_data.items():
    if key == 'low':
        label = f"T = {temp:.2f} (T < Tc)"
        color = 'blue'
    elif key == 'critical':
        label = f"T = {temp:.2f} (T ≈ Tc)"
        color = 'orange'
    else:
        label = f"T = {temp:.2f} (T > Tc)"
        color = 'green'
    
    steps_plot = range(0, len(mag_data), 50)
    mag_plot = [mag_data[i] for i in steps_plot]
    plt.plot(steps_plot, mag_plot, label=label, alpha=0.7, color=color, linewidth=1)

plt.axvline(x=warmup, color='red', linestyle='--', alpha=0.5, label=f'Warmup ends')
plt.xlabel("Monte Carlo steps")
plt.ylabel("Magnetization per spin M")
plt.title("Magnetization vs MC time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(temps, avg_magnetizations, "o-", linewidth=2, markersize=6)
plt.axvline(x=2.0, color='red', linestyle='--', alpha=0.7, label='Tc = 2J = 2.0')
plt.xlabel("Temperature T")
plt.ylabel("Average |M|")
plt.title("Equilibrium Magnetization vs Temperature")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Critical temperature (theoretical): Tc = 2J = {2*J}")
print(f"Temperature range simulated: {temps[0]:.2f} to {temps[-1]:.2f}")
print(f"Warmup steps: {warmup}")
print(f"Equilibrium steps: {steps - warmup}")
