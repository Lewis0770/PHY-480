import numpy as np
import matplotlib.pyplot as plt

beta2 = -1.0e-26
gamma = 2.0e-3
alpha = 0.0
L = 2.0
dz = 0.001
nt = 2048
Tmax = 50e-12
tau = 10e-12
S0 = np.sqrt(abs(beta2) / (gamma * tau**2))

T = np.linspace(-Tmax, Tmax, nt)
dt = T[1] - T[0]
freq = np.fft.fftfreq(nt, dt) * 2 * np.pi
A0 = S0 * (1 / np.cosh(T / tau))

n_steps = int(L / dz)
A = A0.copy()

for _ in range(n_steps):
    A_freq = np.fft.fft(A)
    linear_phase = np.exp(0.5j * beta2 * freq**2 * dz - alpha * dz / 2)
    A = np.fft.ifft(A_freq * linear_phase)
    A = A * np.exp(1j * gamma * np.abs(A)**2 * dz)
    A_freq = np.fft.fft(A)
    A = np.fft.ifft(A_freq * linear_phase)

plt.figure(figsize=(8, 5))
plt.plot(T * 1e12, np.abs(A0)**2 / np.max(np.abs(A0)**2), '--', label='Input')
plt.plot(T * 1e12, np.abs(A)**2 / np.max(np.abs(A0)**2), label='Output')
plt.xlabel('Time (ps)')
plt.ylabel('Normalized |A|^2')
plt.legend()
plt.tight_layout()
plt.show()
