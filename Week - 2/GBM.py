import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
N, x0 = 1000, 1.0

r_vals = np.random.uniform(0.5, 1.5, (5000, N))
gbm_final = x0 * np.prod(r_vals, axis=1)

ln_r_mean = np.mean(np.log(np.random.uniform(0.5, 1.5, 10000)))
ln_r_var = np.var(np.log(np.random.uniform(0.5, 1.5, 10000)))
mu_theory = N * ln_r_mean
sigma2_theory = N * ln_r_var

print(f"Theoretical: μ={mu_theory:.3f}, σ²={sigma2_theory:.3f}")
print(f"Empirical: μ={np.mean(np.log(gbm_final)):.3f}, σ²={np.var(np.log(gbm_final)):.3f}")

r_shared = np.random.uniform(0.5, 1.5, size=(3, 500))
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
for i in range(3):
    r = r_shared[i]                      
    abm = np.concatenate([[x0], x0 + np.cumsum(r)])
    plt.plot(abm, alpha=0.7)
plt.title('ABM (Additive)')
plt.xlabel('Step'); plt.ylabel('x')

plt.subplot(1, 2, 2)
for i in range(3):
    r = r_shared[i]                      
    gbm = np.concatenate([[x0], x0 * np.cumprod(r)])
    plt.plot(gbm, alpha=0.7)
plt.title('GBM (Multiplicative)')
plt.xlabel('Step'); plt.ylabel('x')

plt.tight_layout()
plt.show()
