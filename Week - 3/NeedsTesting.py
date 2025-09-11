import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

a, b = 1.4, 0.3
n_points = 1_100_000
transient = 100_000
x, y = 0.0, 0.0
xs, ys = [], []

for i in range(n_points):
    x_new = 1 - a*x**2 + y
    y_new = b*x
    x, y = x_new, y_new
    if i >= transient:
        xs.append(x)
        ys.append(y)

xs, ys = np.array(xs), np.array(ys)

def box_counting_hist(xs, ys, eps_values):
    counts = []
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    xs_scaled = (xs - x_min) / (x_max - x_min)
    ys_scaled = (ys - y_min) / (y_max - y_min)
    for eps in eps_values:
        n = int(1/eps) + 1
        H, _, _ = np.histogram2d(xs_scaled, ys_scaled, bins=n)
        counts.append(np.count_nonzero(H))
    return counts

eps_values = np.logspace(-1.5, -3, 20)
counts = box_counting_hist(xs, ys, eps_values)

plt.loglog(eps_values, counts, 'o-')
plt.xlabel("ε")
plt.ylabel("N(ε)")
plt.title("Box-counting with histogram2d")
plt.show()

log_eps = np.log(1/eps_values)
log_counts = np.log(counts)
low, high = 5, 15
slope, intercept, _, _, _ = linregress(log_eps[low:high], log_counts[low:high])

print("Estimated fractal dimension ≈", slope)

